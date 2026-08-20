"""ROS 2 node running Hailo-8 StereoNet as a parallel depth pilot.

Subscribes directly to raw camera_ros topics (YUYV image_raw + camera_info)
for both stereo sides, does its own rectification+resize (see
rectification.py), runs HailoRT inference, and publishes a
stereo_msgs/DisparityImage compatible with the existing
drone_stereo_bringup/disparity_viz_node.py visualizer.

See docs/HAILO_OFFLOAD_PLAN.md and the approved plan for the full design
rationale, especially the honest handling of StereoNet's uncalibrated raw
output (no invented metric scale).
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import monotonic

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage

from hailo_stereo_bringup.disparity_message import build_disparity_message
from hailo_stereo_bringup.rectification import RESIZE_STRATEGIES
from hailo_stereo_bringup.rectification import SideRectifier


def sensor_qos() -> QoSProfile:
    """Match yolo_person_car/detector_node.py: drop stale frames, never block."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def disparity_qos() -> QoSProfile:
    """Match drone_stereo_bringup/disparity_viz_node.py's RELIABLE subscriber."""
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


def yuyv_to_bgr(message: Image) -> np.ndarray:
    """Same decode as yolo_person_car/detector_node.py's image_to_bgr()."""
    encoding = message.encoding.lower()
    if encoding == 'yuv422_yuy2':
        image = np.ndarray(
            shape=(message.height, message.width, 2),
            dtype=np.uint8,
            buffer=message.data,
            strides=(message.step, 2, 1),
        )
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f'unsupported input image encoding {message.encoding!r}')


class StereoNetEngine:
    """HailoRT VDevice/InferModel lifecycle for a two-input StereoNet HEF."""

    def __init__(
        self,
        hef_path: str,
        left_input_name: str,
        right_input_name: str,
        output_name: str,
        group_id: str = 'camera_ws_hailo',
    ) -> None:
        from hailo_platform import HailoSchedulingAlgorithm
        from hailo_platform import VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = group_id
        params.multi_process_service = True
        self._vdevice = VDevice(params)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)
        self._configured = self._infer_model.configure()
        self._left_name = left_input_name
        self._right_name = right_input_name
        self._output_name = output_name
        self._output_shape = tuple(self._infer_model.output(output_name).shape)

    def infer(
        self, left_hwc: np.ndarray, right_hwc: np.ndarray, timeout_ms: int = 500,
    ) -> np.ndarray:
        output_buffer = np.empty(self._output_shape, dtype=np.uint8)
        bindings = self._configured.create_bindings(
            input_buffers={
                self._left_name: np.ascontiguousarray(left_hwc),
                self._right_name: np.ascontiguousarray(right_hwc),
            },
            output_buffers={self._output_name: output_buffer},
        )
        self._configured.run([bindings], timeout_ms)
        return output_buffer

    def close(self) -> None:
        del self._configured
        self._vdevice.release()


class StereoSideState:
    """Per-side (left/right) latest-frame buffering, mirrors detector_node.py."""

    def __init__(self) -> None:
        self.rectifier = SideRectifier()
        self._latest_image: Image | None = None
        self._lock = Lock()

    def on_image(self, message: Image) -> None:
        with self._lock:
            self._latest_image = message

    def take_latest(self) -> Image | None:
        with self._lock:
            message, self._latest_image = self._latest_image, None
        return message

    def on_camera_info(self, message: CameraInfo) -> None:
        if not self.rectifier.ready:
            self.rectifier.update(message)


class StereoNetNode(Node):

    def __init__(self) -> None:
        super().__init__('stereonet')
        self.declare_parameter('left_image_topic', '/stereo/left/image_raw')
        self.declare_parameter('right_image_topic', '/stereo/right/image_raw')
        self.declare_parameter('left_camera_info_topic', '/stereo/left/camera_info')
        self.declare_parameter('right_camera_info_topic', '/stereo/right/camera_info')
        self.declare_parameter(
            'hef_path', str(Path.home() / '.local/share/hailo-models/stereonet.hef'),
        )
        self.declare_parameter('left_input_name', 'stereonet/input_layer1')
        self.declare_parameter('right_input_name', 'stereonet/input_layer2')
        self.declare_parameter('output_name', 'stereonet/conv53')
        self.declare_parameter('input_width', 1232)
        self.declare_parameter('input_height', 368)
        # StereoNet's calibration YAML normalizes in RGB channel order
        # (mean/std lists match the standard ImageNet RGB convention) --
        # camera_ros gives us YUYV -> BGR by default, so swap by default.
        self.declare_parameter('swap_rb', True)
        self.declare_parameter('resize_strategy', 'stretch')
        self.declare_parameter('output_topic', 'disparity_hailo')
        self.declare_parameter('inference_period_sec', 0.1)
        # Raw StereoNet output is an uncalibrated UINT8 tensor (see module
        # docstring) -- default bounds simply span the native byte range.
        # Adjust once real scene output stats are observed on hardware.
        self.declare_parameter('min_disparity', 0.0)
        self.declare_parameter('max_disparity', 255.0)

        hef_path = Path(
            self.get_parameter('hef_path').get_parameter_value().string_value
        ).expanduser()
        if not hef_path.is_file():
            raise RuntimeError(
                f'hef_path does not exist: {hef_path}. Run Phase 0 of the '
                'plan (hailo-apps depth_estimation_stereo example) first.'
            )

        self._input_width = self._positive_int('input_width')
        self._input_height = self._positive_int('input_height')
        self._swap_rb = bool(self.get_parameter('swap_rb').value)
        strategy_name = self.get_parameter('resize_strategy').value
        if strategy_name not in RESIZE_STRATEGIES:
            raise ValueError(f'unknown resize_strategy {strategy_name!r}')
        self._resize = RESIZE_STRATEGIES[strategy_name]
        self._min_disparity = float(self.get_parameter('min_disparity').value)
        self._max_disparity = float(self.get_parameter('max_disparity').value)
        self._period = self._positive_float('inference_period_sec')

        self._left = StereoSideState()
        self._right = StereoSideState()
        self._last_error_at = 0.0
        self._last_stats_at = 0.0

        self.get_logger().info(f'loading HEF: {hef_path}')
        self._engine = StereoNetEngine(
            str(hef_path),
            self.get_parameter('left_input_name').value,
            self.get_parameter('right_input_name').value,
            self.get_parameter('output_name').value,
        )

        output_topic = self.get_parameter('output_topic').value
        self._publisher = self.create_publisher(
            DisparityImage, output_topic, disparity_qos(),
        )
        self.create_subscription(
            Image, self.get_parameter('left_image_topic').value,
            self._left.on_image, sensor_qos(),
        )
        self.create_subscription(
            Image, self.get_parameter('right_image_topic').value,
            self._right.on_image, sensor_qos(),
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('left_camera_info_topic').value,
            self._left.on_camera_info, sensor_qos(),
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('right_camera_info_topic').value,
            self._right.on_camera_info, sensor_qos(),
        )
        self.create_timer(self._period, self._process_latest_pair)
        self.get_logger().info(
            f'StereoNet pilot ready: input={self._input_width}x{self._input_height}, '
            f'resize_strategy={strategy_name}, output_topic={output_topic}'
        )

    def _positive_int(self, name: str) -> int:
        value = int(self.get_parameter(name).value)
        if value <= 0:
            raise ValueError(f'{name} must be greater than zero')
        return value

    def _positive_float(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if value <= 0.0:
            raise ValueError(f'{name} must be greater than zero')
        return value

    def _prepare_side(self, side: StereoSideState, message: Image) -> np.ndarray:
        bgr = yuyv_to_bgr(message)
        rectified = side.rectifier.rectify(bgr)
        resized = self._resize(rectified, self._input_width, self._input_height)
        if self._swap_rb:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return resized

    def _process_latest_pair(self) -> None:
        left_msg = self._left.take_latest()
        right_msg = self._right.take_latest()
        if left_msg is None or right_msg is None:
            return
        if not (self._left.rectifier.ready and self._right.rectifier.ready):
            return
        try:
            left_input = self._prepare_side(self._left, left_msg)
            right_input = self._prepare_side(self._right, right_msg)
            raw_output = self._engine.infer(left_input, right_input)
            self._log_output_stats(raw_output)

            left_fx = self._left.rectifier.focal_length_px()
            scale = self._input_width / float(left_msg.width)
            message = build_disparity_message(
                raw_output,
                header=left_msg.header,
                f=left_fx * scale,
                baseline_m=self._right.rectifier.baseline_m(),
                min_disparity=self._min_disparity,
                max_disparity=self._max_disparity,
            )
            self._publisher.publish(message)
        except Exception as exc:  # Keep the pilot running after one bad frame.
            now = monotonic()
            if now - self._last_error_at >= 5.0:
                self.get_logger().error(f'inference failed ({type(exc).__name__}): {exc}')
                self._last_error_at = now

    def _log_output_stats(self, raw_output: np.ndarray) -> None:
        now = monotonic()
        if now - self._last_stats_at < 5.0:
            return
        self._last_stats_at = now
        values = raw_output.astype(np.float32)
        self.get_logger().info(
            f'raw StereoNet output stats: min={values.min():.1f} '
            f'max={values.max():.1f} mean={values.mean():.1f} '
            '(uncalibrated -- see docs/HAILO_OFFLOAD_PLAN.md)'
        )

    def destroy_node(self) -> bool:
        try:
            self._engine.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = StereoNetNode()
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
