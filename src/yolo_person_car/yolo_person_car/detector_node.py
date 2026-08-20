"""ROS 2 image detector backed by an explicitly exported NCNN YOLO model."""

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


class DetectorNode(Node):
    """Keep only the newest camera image and infer it with NCNN YOLO."""

    def __init__(self) -> None:
        super().__init__('detector')
        self.declare_parameter('image_topic', 'image_raw')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('allowed_classes', [0, 2])
        self.declare_parameter('enable_debug_image', True)
        self.declare_parameter('inference_period_sec', 0.01)
        self.declare_parameter('max_inference_fps', 0.0)

        model_path = Path(
            self.get_parameter('model_path').get_parameter_value().string_value
        ).expanduser()
        if not model_path.is_dir():
            raise RuntimeError(
                'model_path must point to an exported NCNN model directory; '
                f'got {model_path}'
            )
        self._confidence = self._bounded_float('confidence_threshold', 0.0, 1.0)
        self._iou = self._bounded_float('iou_threshold', 0.0, 1.0)
        self._imgsz = self._positive_int('imgsz')
        self._period = self._positive_float('inference_period_sec')
        self._max_inference_fps = self._nonnegative_float('max_inference_fps')
        self._minimum_inference_interval = (
            1.0 / self._max_inference_fps
            if self._max_inference_fps > 0.0 else 0.0
        )
        self._next_inference_at = 0.0
        self._device = self.get_parameter('device').value
        self._allowed_classes = validate_allowed_classes(
            self.get_parameter('allowed_classes').value
        )
        self._publish_debug = self.get_parameter('enable_debug_image').value
        self._bridge = CvBridge()
        self._latest_image: Image | None = None
        self._image_lock = Lock()
        self._last_error_at = 0.0

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'ultralytics is unavailable. Launch this node through the '
                'configured Python virtual environment.'
            ) from exc
        self._model = YOLO(str(model_path), task='detect')

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
            f'NCNN detector ready: model={model_path}, input={image_topic}, '
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
            source = image_to_bgr(self._bridge, message)
            result = self._model.predict(
                source=source,
                imgsz=self._imgsz,
                conf=self._confidence,
                iou=self._iou,
                classes=list(self._allowed_classes),
                device=self._device,
                verbose=False,
            )[0]
            boxes = self._boxes_from_result(result)
            self._publish(message, source, boxes)
        except Exception as exc:  # Keep the live pipeline running after one bad frame.
            error_time = monotonic()
            if error_time - self._last_error_at >= 5.0:
                self.get_logger().error(
                    f'inference failed ({type(exc).__name__}): {exc}'
                )
                self._last_error_at = error_time

    @staticmethod
    def _boxes_from_result(result: Any) -> list[Box]:
        if result.boxes is None:
            return []
        coordinates = result.boxes.xyxy.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        return [
            Box(*xyxy, int(class_id), float(confidence))
            for xyxy, class_id, confidence in zip(
                coordinates, class_ids, confidences, strict=True
            )
        ]

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
