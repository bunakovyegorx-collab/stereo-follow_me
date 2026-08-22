"""ROS 2 image detector backed by a Hailo-8 accelerated YOLOv8 HEF.

NMS runs on-chip (HAILO_NMS_BY_CLASS output format): the Hailo output is a
list of one numpy array per COCO class, each shaped (num_detections, 5) with
columns [y_min, x_min, y_max, x_max, score] normalized to [0, 1] in the
640x640 letterboxed input space -- confirmed empirically against
/usr/share/hailo-models/yolov8s_h8.hef via `hailortcli parse-hef` and a
standalone smoke test (see docs referenced in the plan). There is therefore
no iou_threshold/device parameter here, unlike yolo_person_car's NCNN
detector -- NMS and the execution device are both fixed by the HEF.

Preprocessing/postprocessing/detection-message helpers are intentionally
duplicated from yolo_person_car/detector_node.py rather than imported,
mirroring the precedent hailo_stereo_bringup/stereonet_node.py already set
for yuyv_to_bgr()/sensor_qos() -- this keeps the two packages (NCNN vs
Hailo backends) independently deployable.

Uses an exclusive VDevice (multi_process_service=False): the HailoRT Python
client for this HEF reliably segfaults/bus-errors at process teardown after
a successful inference (confirmed empirically, independent of
multi_process_service). With multi_process_service=True that crash also
takes down the shared hailort_service daemon (observed via journalctl:
SIGSEGV, systemd auto-restart); with it False, only this node's own process
is affected on its own shutdown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from hailo_yolo_detector.letterbox import letterbox
from hailo_yolo_detector.letterbox import undo_letterbox_box


COCO_CLASS_NAMES = {
    0: 'person',
    2: 'car',
}


@dataclass(frozen=True)
class Box:
    """A YOLO bounding box in source-image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float


def sensor_qos() -> QoSProfile:
    """Use one best-effort sample to prevent latency from growing."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def validate_allowed_classes(class_ids: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize the supported COCO class filter."""
    normalized = tuple(int(class_id) for class_id in class_ids)
    unknown = set(normalized).difference(COCO_CLASS_NAMES)
    if unknown:
        raise ValueError(
            f'allowed_classes supports only {sorted(COCO_CLASS_NAMES)}, '
            f'got unsupported values {sorted(unknown)}'
        )
    if not normalized:
        raise ValueError('allowed_classes must contain at least one class')
    return normalized


def detection_message(box: Box, header: Header) -> Detection2D:
    """Convert one source-image box into the standard ROS detection format."""
    message = Detection2D()
    message.header = header
    message.bbox.center.position.x = (box.x1 + box.x2) / 2.0
    message.bbox.center.position.y = (box.y1 + box.y2) / 2.0
    message.bbox.size_x = max(0.0, box.x2 - box.x1)
    message.bbox.size_y = max(0.0, box.y2 - box.y1)

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = COCO_CLASS_NAMES[box.class_id]
    hypothesis.hypothesis.score = box.confidence
    message.results.append(hypothesis)
    return message


def image_to_bgr(bridge: CvBridge, message: Image) -> Any:
    """Convert the encodings emitted by camera_ros into BGR for YOLO."""
    encoding = message.encoding.lower()
    # cv_bridge in this Jazzy build does not map yuv422_yuy2 to an OpenCV
    # matrix. Decode the two-byte YUYV pixels directly from the ROS buffer.
    if encoding == 'yuv422_yuy2':
        image = np.ndarray(
            shape=(message.height, message.width, 2),
            dtype=np.uint8,
            buffer=message.data,
            strides=(message.step, 2, 1),
        )
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    image = bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
    if encoding == 'bgr8':
        return image
    if encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == 'bgra8':
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ('mono8', '8uc1'):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported input image encoding {message.encoding!r}')


def bgr_image_message(image: Any, source_message: Image) -> Image:
    """Create a BGR ROS image without cv_bridge's incomplete encoding table."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('debug image must be a uint8 HxWx3 BGR array')
    message = Image()
    message.header = source_message.header
    message.height, message.width = image.shape[:2]
    message.encoding = 'bgr8'
    message.is_bigendian = False
    message.step = message.width * 3
    message.data = image.tobytes()
    return message


class HailoYoloEngine:
    """HailoRT VDevice/InferModel lifecycle for a single-input YOLO HEF with
    on-chip NMS (HAILO_NMS_BY_CLASS output format).
    """

    def __init__(
        self,
        hef_path: str,
        input_name: str,
        output_name: str,
        group_id: str = 'camera_ws_hailo_yolo',
        multi_process_service: bool = False,
    ) -> None:
        from hailo_platform import HailoSchedulingAlgorithm
        from hailo_platform import VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = group_id
        params.multi_process_service = multi_process_service
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        self._configured = self._infer_model.configure()
        self._input_name = input_name
        self._output_name = output_name
        self._raw_output_shape = tuple(self._infer_model.output(output_name).shape)

    def infer(self, input_hwc_uint8: np.ndarray, timeout_ms: int = 500) -> list[np.ndarray]:
        """Run inference and return the on-chip NMS result: a list of one
        (num_detections, 5) array per COCO class, columns
        [y_min, x_min, y_max, x_max, score], normalized [0, 1].
        """
        raw_output_buffer = np.empty(self._raw_output_shape, dtype=np.float32)
        bindings = self._configured.create_bindings()
        bindings.input(self._input_name).set_buffer(np.ascontiguousarray(input_hwc_uint8))
        bindings.output(self._output_name).set_buffer(raw_output_buffer)
        self._configured.run([bindings], timeout_ms)
        return bindings.output(self._output_name).get_buffer()

    def close(self) -> None:
        del self._configured
        self._vdevice.release()


class DetectorNode(Node):
    """Keep only the newest camera image and infer it with Hailo-8 YOLOv8."""

    def __init__(self) -> None:
        super().__init__('hailo_detector')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('hef_path', '/usr/share/hailo-models/yolov8s_h8.hef')
        self.declare_parameter('input_name', 'yolov8s/input_layer1')
        self.declare_parameter('output_name', 'yolov8s/yolov8_nms_postprocess')
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('allowed_classes', [0, 2])
        self.declare_parameter('enable_debug_image', True)
        self.declare_parameter('inference_period_sec', 0.01)
        self.declare_parameter('max_inference_fps', 0.0)
        self.declare_parameter('group_id', 'camera_ws_hailo_yolo')
        self.declare_parameter('multi_process_service', False)

        hef_path = Path(
            self.get_parameter('hef_path').get_parameter_value().string_value
        ).expanduser()
        if not hef_path.is_file():
            raise RuntimeError(f'hef_path does not exist: {hef_path}')

        self._confidence = self._bounded_float('confidence_threshold', 0.0, 1.0)
        self._imgsz = self._positive_int('imgsz')
        self._period = self._positive_float('inference_period_sec')
        self._max_inference_fps = self._nonnegative_float('max_inference_fps')
        self._minimum_inference_interval = (
            1.0 / self._max_inference_fps
            if self._max_inference_fps > 0.0 else 0.0
        )
        self._next_inference_at = 0.0
        self._allowed_classes = validate_allowed_classes(
            self.get_parameter('allowed_classes').value
        )
        self._publish_debug = self.get_parameter('enable_debug_image').value
        self._bridge = CvBridge()
        self._latest_image: Image | None = None
        self._image_lock = Lock()
        self._last_error_at = 0.0

        self.get_logger().info(f'loading HEF: {hef_path}')
        self._engine = HailoYoloEngine(
            str(hef_path),
            self.get_parameter('input_name').value,
            self.get_parameter('output_name').value,
            group_id=self.get_parameter('group_id').value,
            multi_process_service=bool(
                self.get_parameter('multi_process_service').value
            ),
        )

        self._detections_publisher = self.create_publisher(
            Detection2DArray, 'detections', sensor_qos()
        )
        self._debug_publisher = self.create_publisher(
            Image, 'debug_image', sensor_qos()
        )
        image_topic = self.get_parameter('image_topic').value
        self._subscription = self.create_subscription(
            Image, image_topic, self._image_callback, sensor_qos()
        )
        self._timer = self.create_timer(self._period, self._process_latest_image)
        self.get_logger().info(
            f'Hailo YOLO detector ready: hef={hef_path}, input={image_topic}, '
            f'imgsz={self._imgsz}, classes={self._allowed_classes}, '
            f'max_fps={self._max_inference_fps or "unlimited"}'
        )

    def _bounded_float(self, name: str, minimum: float, maximum: float) -> float:
        value = float(self.get_parameter(name).value)
        if not minimum <= value <= maximum:
            raise ValueError(f'{name} must be between {minimum} and {maximum}')
        return value

    def _positive_float(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f'{name} must be greater than zero')
        return value

    def _nonnegative_float(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value < 0.0:
            raise ValueError(f'{name} must be zero or greater')
        return value

    def _positive_int(self, name: str) -> int:
        value = int(self.get_parameter(name).value)
        if value <= 0:
            raise ValueError(f'{name} must be greater than zero')
        return value

    def _image_callback(self, message: Image) -> None:
        with self._image_lock:
            self._latest_image = message

    def _process_latest_image(self) -> None:
        now = monotonic()
        if now < self._next_inference_at:
            return
        with self._image_lock:
            message = self._latest_image
            self._latest_image = None
        if message is None:
            return
        if self._minimum_inference_interval > 0.0:
            self._next_inference_at = now + self._minimum_inference_interval
        try:
            bgr = image_to_bgr(self._bridge, message)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            canvas, transform = letterbox(rgb, self._imgsz)
            per_class_detections = self._engine.infer(canvas)
            boxes = self._boxes_from_hailo_output(per_class_detections, transform)
            self._publish(message, bgr, boxes)
        except Exception as exc:  # Keep the live pipeline running after one bad frame.
            error_time = monotonic()
            if error_time - self._last_error_at >= 5.0:
                self.get_logger().error(
                    f'inference failed ({type(exc).__name__}): {exc}'
                )
                self._last_error_at = error_time

    def _boxes_from_hailo_output(
        self, per_class_detections: list[np.ndarray], transform,
    ) -> list[Box]:
        boxes = []
        for class_id in self._allowed_classes:
            detections = per_class_detections[class_id]
            for detection in detections:
                score = float(detection[4])
                if score < self._confidence:
                    continue
                y1, x1, y2, x2 = (float(v) * self._imgsz for v in detection[:4])
                sx1, sy1, sx2, sy2 = undo_letterbox_box(x1, y1, x2, y2, transform)
                boxes.append(Box(sx1, sy1, sx2, sy2, class_id, score))
        return boxes

    def _publish(self, source_message: Image, image: Any, boxes: list[Box]) -> None:
        detections = Detection2DArray()
        detections.header = source_message.header
        detections.detections = [
            detection_message(box, source_message.header) for box in boxes
        ]
        self._detections_publisher.publish(detections)

        if not self._publish_debug:
            return
        debug = image.copy()
        for box in boxes:
            label = f'{COCO_CLASS_NAMES[box.class_id]} {box.confidence:.2f}'
            cv2.rectangle(
                debug, (round(box.x1), round(box.y1)),
                (round(box.x2), round(box.y2)), (0, 255, 0), 2,
            )
            cv2.putText(
                debug, label, (round(box.x1), max(18, round(box.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
        self._debug_publisher.publish(bgr_image_message(debug, source_message))

    def destroy_node(self) -> bool:
        try:
            self._engine.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
