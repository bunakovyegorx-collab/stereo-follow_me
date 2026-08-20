"""ROS 2 node that fuses person boxes with timestamp-matched disparity."""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np
import rclpy
from foxglove_msgs.msg import ImageAnnotations
from foxglove_msgs.msg import SceneUpdate
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from person_range_fusion.foxglove_viz import build_image_annotations
from person_range_fusion.foxglove_viz import build_scene_update
from person_range_fusion.fusion import BBox
from person_range_fusion.fusion import DepthEstimate
from person_range_fusion.fusion import RangeCandidate
from person_range_fusion.fusion import RangeTracker
from person_range_fusion.fusion import estimate_person_depth


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def disparity_array(message: DisparityImage) -> np.ndarray:
    image = message.image
    if image.encoding.lower() not in ('32fc1', '32fc'):
        raise ValueError(f'expected 32FC1 disparity, got {image.encoding!r}')
    if image.height <= 0 or image.width <= 0 or image.step < image.width * 4:
        raise ValueError('invalid disparity image dimensions')
    row_values = image.step // 4
    expected_size = int(image.height) * int(image.step)
    if len(image.data) != expected_size:
        raise ValueError('invalid disparity buffer size')
    result = np.frombuffer(image.data, dtype=np.float32).reshape(
        int(image.height), row_values,
    )[:, :int(image.width)]
    if image.is_bigendian:
        result = result.byteswap()
    return result


def detection_bbox(detection: Detection2D) -> BBox:
    center = detection.bbox.center.position
    half_width = float(detection.bbox.size_x) / 2.0
    half_height = float(detection.bbox.size_y) / 2.0
    return BBox(
        float(center.x) - half_width,
        float(center.y) - half_height,
        float(center.x) + half_width,
        float(center.y) + half_height,
    )


def hypothesis(detection: Detection2D) -> tuple[str, float] | None:
    if not detection.results:
        return None
    best = max(detection.results, key=lambda item: item.hypothesis.score)
    return best.hypothesis.class_id, float(best.hypothesis.score)


def image_to_bgr(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding in ('mono8', '8uc1'):
        mono = np.ndarray(
            shape=(message.height, message.width), dtype=np.uint8,
            buffer=message.data, strides=(message.step, 1),
        )
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if encoding == 'bgr8':
        return np.ndarray(
            shape=(message.height, message.width, 3), dtype=np.uint8,
            buffer=message.data, strides=(message.step, 3, 1),
        ).copy()
    raise ValueError(f'unsupported debug image encoding {message.encoding!r}')


def bgr_message(image: np.ndarray, source: Image) -> Image:
    result = Image()
    result.header = source.header
    result.height, result.width = image.shape[:2]
    result.encoding = 'bgr8'
    result.is_bigendian = False
    result.step = result.width * 3
    result.data = image.tobytes()
    return result


class PersonRangeFusion(Node):
    """Fuse timestamp-aligned person detections and disparity images."""

    def __init__(self) -> None:
        super().__init__('fusion')
        self.declare_parameter('detections_topic', '/person_detector/detections')
        self.declare_parameter('disparity_topic', '/stereo/disparity')
        self.declare_parameter('camera_info_topic', '/stereo/left/camera_info')
        self.declare_parameter('image_topic', '/stereo/left/image_rect')
        self.declare_parameter('enable_debug_image', False)
        self.declare_parameter('enable_image_annotations', True)
        self.declare_parameter('enable_scene', True)
        # REP-103 output frame (X forward, Y left, Z up). Marker size is
        # natural in that frame: [width, depth, height] — not optical axes.
        self.declare_parameter('person_marker_size', [0.5, 0.5, 1.7])
        # Scene marker style: red wireframe edges, transparent body by default.
        self.declare_parameter('person_marker_wireframe', True)
        self.declare_parameter('person_marker_edge_thickness', 0.03)
        self.declare_parameter('person_marker_fill_alpha', 0.0)
        self.declare_parameter('output_frame', 'stereo_left_up')
        self.declare_parameter('scene_lifetime_sec', 1.0)
        self.declare_parameter('buffer_size', 16)
        self.declare_parameter('max_sync_delta_sec', 0.05)
        self.declare_parameter('min_depth', 0.5)
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('roi_width_fraction', 0.5)
        self.declare_parameter('roi_y_start', 0.2)
        self.declare_parameter('roi_y_end', 0.7)
        self.declare_parameter('min_valid_pixels', 20)
        self.declare_parameter('min_valid_fraction', 0.1)
        self.declare_parameter('ema_alpha', 0.4)
        self.declare_parameter('track_iou_threshold', 0.3)
        self.declare_parameter('track_timeout_sec', 0.7)

        buffer_size = int(self.get_parameter('buffer_size').value)
        if buffer_size <= 0:
            raise ValueError('buffer_size must be positive')
        self._max_sync_delta_ns = int(
            float(self.get_parameter('max_sync_delta_sec').value) * 1_000_000_000
        )
        self._min_depth = float(self.get_parameter('min_depth').value)
        self._max_depth = float(self.get_parameter('max_depth').value)
        if not 0.0 < self._min_depth < self._max_depth:
            raise ValueError('expected 0 < min_depth < max_depth')
        self._roi_width_fraction = float(
            self.get_parameter('roi_width_fraction').value
        )
        self._roi_y_start = float(self.get_parameter('roi_y_start').value)
        self._roi_y_end = float(self.get_parameter('roi_y_end').value)
        self._min_valid_pixels = int(
            self.get_parameter('min_valid_pixels').value
        )
        self._min_valid_fraction = float(
            self.get_parameter('min_valid_fraction').value
        )
        self._enable_debug = bool(
            self.get_parameter('enable_debug_image').value
        )
        self._enable_annotations = bool(
            self.get_parameter('enable_image_annotations').value
        )
        self._enable_scene = bool(self.get_parameter('enable_scene').value)
        marker = list(self.get_parameter('person_marker_size').value)
        if len(marker) != 3 or any(float(v) <= 0.0 for v in marker):
            raise ValueError('person_marker_size must be three positive values')
        self._person_marker_size = (
            float(marker[0]), float(marker[1]), float(marker[2]),
        )
        self._person_marker_wireframe = bool(
            self.get_parameter('person_marker_wireframe').value
        )
        self._person_marker_edge_thickness = float(
            self.get_parameter('person_marker_edge_thickness').value
        )
        if self._person_marker_edge_thickness <= 0.0:
            raise ValueError('person_marker_edge_thickness must be positive')
        fill_alpha = float(self.get_parameter('person_marker_fill_alpha').value)
        if not 0.0 <= fill_alpha <= 1.0:
            raise ValueError('person_marker_fill_alpha must be in [0, 1]')
        self._person_marker_fill_alpha = fill_alpha
        self._output_frame = str(self.get_parameter('output_frame').value).strip()
        self._scene_lifetime_sec = float(
            self.get_parameter('scene_lifetime_sec').value
        )
        if self._scene_lifetime_sec <= 0.0:
            raise ValueError('scene_lifetime_sec must be positive')
        self._disparities: deque[DisparityImage] = deque(maxlen=buffer_size)
        self._images: deque[Image] = deque(maxlen=buffer_size)
        self._pending_detections: deque[Detection2DArray] = deque(
            maxlen=buffer_size
        )
        self._camera_info: CameraInfo | None = None
        self._tracker = RangeTracker(
            alpha=float(self.get_parameter('ema_alpha').value),
            iou_threshold=float(
                self.get_parameter('track_iou_threshold').value
            ),
            timeout_sec=float(self.get_parameter('track_timeout_sec').value),
        )
        # Projection stays in the optical frame; final 3D outputs are
        # transformed into output_frame (REP-103) via the static TF tree.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = sensor_qos()
        self._detections_pub = self.create_publisher(
            Detection3DArray, 'detections_3d', qos,
        )
        self._nearest_pub = self.create_publisher(
            PointStamped, 'nearest_point', qos,
        )
        self._annotations_pub = self.create_publisher(
            ImageAnnotations, 'image_annotations', qos,
        )
        self._scene_pub = self.create_publisher(SceneUpdate, 'scene', qos)
        # Legacy pixel debug is opt-in only: no publisher/sub unless enabled.
        self._debug_pub = (
            self.create_publisher(Image, 'debug_image', qos)
            if self._enable_debug else None
        )
        self.create_subscription(
            DisparityImage,
            self.get_parameter('disparity_topic').value,
            self._disparity_callback,
            qos,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_callback,
            qos,
        )
        if self._enable_debug:
            self.create_subscription(
                Image,
                self.get_parameter('image_topic').value,
                self._image_callback,
                qos,
            )
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('detections_topic').value,
            self._detections_callback,
            qos,
        )
        self.get_logger().info(
            'person range fusion ready: depth=%.1f..%.1fm, sync=%dms, '
            'output_frame=%r, annotations=%s, scene=%s, legacy_debug=%s'
            % (
                self._min_depth, self._max_depth,
                self._max_sync_delta_ns // 1_000_000,
                self._output_frame or '(source optical)',
                self._enable_annotations, self._enable_scene, self._enable_debug,
            )
        )

    def _disparity_callback(self, message: DisparityImage) -> None:
        self._disparities.append(message)
        self._process_pending_detections()

    def _image_callback(self, message: Image) -> None:
        self._images.append(message)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _closest(self, messages: deque[Any], target_ns: int) -> Any | None:
        if not messages:
            return None
        closest = min(messages, key=lambda message: abs(stamp_ns(message) - target_ns))
        if abs(stamp_ns(closest) - target_ns) > self._max_sync_delta_ns:
            return None
        return closest

    def _detections_callback(self, message: Detection2DArray) -> None:
        people = [
            detection for detection in message.detections
            if hypothesis(detection) is not None
            and hypothesis(detection)[0] == 'person'
        ]
        if not people:
            self._publish(message, [], [], stamp_ns(message))
            return
        self._pending_detections.append(message)
        self._process_pending_detections()

    def _process_pending_detections(self) -> None:
        if self._camera_info is None or not self._disparities:
            return
        latest_disparity_ns = stamp_ns(self._disparities[-1])
        remaining: deque[Detection2DArray] = deque(
            maxlen=self._pending_detections.maxlen
        )
        while self._pending_detections:
            message = self._pending_detections.popleft()
            target_ns = stamp_ns(message)
            disparity_message = self._closest(self._disparities, target_ns)
            if disparity_message is not None:
                self._fuse(message, disparity_message, target_ns)
            elif latest_disparity_ns - target_ns > self._max_sync_delta_ns:
                people = [
                    detection for detection in message.detections
                    if hypothesis(detection) is not None
                    and hypothesis(detection)[0] == 'person'
                ]
                self.get_logger().warning(
                    'discarding detection without timestamp-matched disparity',
                    throttle_duration_sec=5.0,
                )
                self._publish(
                    message, people, [None] * len(people), target_ns,
                )
            else:
                remaining.append(message)
        self._pending_detections = remaining

    def _fuse(
        self,
        message: Detection2DArray,
        disparity_message: DisparityImage,
        target_ns: int,
    ) -> None:
        people = [
            detection for detection in message.detections
            if hypothesis(detection) is not None
            and hypothesis(detection)[0] == 'person'
        ]
        try:
            disparity = disparity_array(disparity_message)
            estimates = [
                estimate_person_depth(
                    disparity,
                    detection_bbox(detection),
                    f=float(disparity_message.f),
                    baseline=float(disparity_message.t),
                    min_disparity=float(disparity_message.min_disparity),
                    max_disparity=float(disparity_message.max_disparity),
                    min_depth=self._min_depth,
                    max_depth=self._max_depth,
                    roi_width_fraction=self._roi_width_fraction,
                    roi_y_start=self._roi_y_start,
                    roi_y_end=self._roi_y_end,
                    min_valid_pixels=self._min_valid_pixels,
                    min_valid_fraction=self._min_valid_fraction,
                )
                for detection in people
            ]
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            estimates = [None] * len(people)
        self._publish(message, people, estimates, target_ns)

    def _publish(
        self,
        source: Detection2DArray,
        people: list[Detection2D],
        estimates: list[DepthEstimate | None],
        target_ns: int,
    ) -> None:
        valid_indices = [
            index for index, estimate in enumerate(estimates)
            if estimate is not None
        ]
        candidates = [
            RangeCandidate(detection_bbox(people[index]), estimates[index].z)
            for index in valid_indices
        ]
        tracked = self._tracker.update(candidates, target_ns)
        tracked_by_index = dict(zip(valid_indices, tracked, strict=True))

        # 3D projection is optical; published products use output_frame.
        source_frame = (
            self._camera_info.header.frame_id
            if self._camera_info is not None and self._camera_info.header.frame_id
            else source.header.frame_id
        )
        output = Detection3DArray()
        output.header = source.header
        output_frame = source_frame
        nearest_point: PointStamped | None = None
        nearest_range = float('inf')
        for index in valid_indices:
            detection = people[index]
            estimate = estimates[index]
            track = tracked_by_index[index]
            position_optical = self._project(
                detection_bbox(detection), track.filtered_z,
            )
            position, frame_id = self._to_output_frame(
                position_optical, source.header.stamp, source_frame,
            )
            output_frame = frame_id
            detection_3d = self._detection_3d(
                source, detection, track.track_id, position, estimate.sigma_z,
                frame_id=frame_id,
            )
            output.detections.append(detection_3d)
            # Nearest by metric stereo range (optical Z / filtered depth),
            # not by a single axis of the output frame.
            if track.filtered_z < nearest_range:
                nearest_range = track.filtered_z
                nearest_point = PointStamped()
                nearest_point.header = source.header
                nearest_point.header.frame_id = frame_id
                nearest_point.point.x = position[0]
                nearest_point.point.y = position[1]
                nearest_point.point.z = position[2]
        output.header.frame_id = output_frame
        self._detections_pub.publish(output)
        if nearest_point is not None:
            self._nearest_pub.publish(nearest_point)
        if self._enable_annotations:
            self._annotations_pub.publish(
                build_image_annotations(
                    people, estimates, tracked_by_index, source.header,
                    detection_bbox_fn=detection_bbox,
                    hypothesis_fn=hypothesis,
                )
            )
        if self._enable_scene:
            self._scene_pub.publish(
                build_scene_update(
                    output,
                    marker_size=self._person_marker_size,
                    lifetime_sec=self._scene_lifetime_sec,
                    wireframe=self._person_marker_wireframe,
                    edge_thickness=self._person_marker_edge_thickness,
                    fill_alpha=self._person_marker_fill_alpha,
                )
            )
        if self._enable_debug:
            self._publish_debug(source, people, estimates, tracked_by_index, target_ns)

    def _project(self, bbox: BBox, z: float) -> tuple[float, float, float]:
        """Project pixel + optical-axis depth into the camera optical frame."""
        projection = self._camera_info.p
        fx, cx = float(projection[0]), float(projection[2])
        fy, cy = float(projection[5]), float(projection[6])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError('camera projection matrix has invalid focal lengths')
        u = (bbox.x1 + bbox.x2) / 2.0
        v = (bbox.y1 + bbox.y2) / 2.0
        return (u - cx) * z / fx, (v - cy) * z / fy, z

    def _to_output_frame(
        self,
        position_optical: tuple[float, float, float],
        stamp: Any,
        source_frame: str,
    ) -> tuple[tuple[float, float, float], str]:
        """Map an optical-frame point into ``output_frame`` (REP-103).

        Projection math stays optical; only the finished 3D point is
        transformed. On TF failure, fall back to the source optical frame.
        """
        if not self._output_frame or self._output_frame == source_frame:
            return position_optical, source_frame
        try:
            # Static optical→REP-103; Time() avoids stamp extrapolation gaps.
            transform = self._tf_buffer.lookup_transform(
                self._output_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            point = PointStamped()
            point.header.frame_id = source_frame
            point.header.stamp = stamp
            point.point.x = position_optical[0]
            point.point.y = position_optical[1]
            point.point.z = position_optical[2]
            out = do_transform_point(point, transform)
            return (out.point.x, out.point.y, out.point.z), self._output_frame
        except TransformException as exc:
            self.get_logger().warning(
                'TF %s→%s unavailable (%s); publishing in optical frame'
                % (source_frame, self._output_frame, exc),
                throttle_duration_sec=5.0,
            )
            return position_optical, source_frame

    @staticmethod
    def _detection_3d(
        source: Detection2DArray,
        detection: Detection2D,
        track_id: str,
        position: tuple[float, float, float],
        sigma_z: float,
        *,
        frame_id: str,
    ) -> Detection3D:
        class_id, score = hypothesis(detection)
        result = Detection3D()
        result.header = source.header
        result.header.frame_id = frame_id
        result.id = track_id
        result.bbox.center.position.x = position[0]
        result.bbox.center.position.y = position[1]
        result.bbox.center.position.z = position[2]
        result.bbox.center.orientation.w = 1.0
        object_hypothesis = ObjectHypothesisWithPose()
        object_hypothesis.hypothesis.class_id = class_id
        object_hypothesis.hypothesis.score = score
        object_hypothesis.pose.pose.position.x = position[0]
        object_hypothesis.pose.pose.position.y = position[1]
        object_hypothesis.pose.pose.position.z = position[2]
        object_hypothesis.pose.pose.orientation.w = 1.0
        # Optical-axis depth variance: zz in optical frames, xx in REP-103
        # (X forward == former optical Z).
        variance = sigma_z ** 2
        if 'optical' in frame_id:
            object_hypothesis.pose.covariance[14] = variance
        else:
            object_hypothesis.pose.covariance[0] = variance
        result.results.append(object_hypothesis)
        return result

    def _publish_debug(
        self,
        source: Detection2DArray,
        people: list[Detection2D],
        estimates: list[DepthEstimate | None],
        tracked_by_index: dict[int, Any],
        target_ns: int,
    ) -> None:
        image_message = self._closest(self._images, target_ns)
        if image_message is None:
            return
        try:
            image = image_to_bgr(image_message)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        for index, detection in enumerate(people):
            bbox = detection_bbox(detection)
            estimate = estimates[index]
            class_id, score = hypothesis(detection)
            if estimate is None:
                label = f'{class_id} {score:.2f} | depth unavailable'
                color = (0, 165, 255)
            else:
                filtered = tracked_by_index[index].filtered_z
                label = (
                    f'{class_id} {score:.2f} | Z={filtered:.2f}m '
                    f'(raw {estimate.z:.2f}m)'
                )
                color = (0, 255, 0)
            cv2.rectangle(
                image, (round(bbox.x1), round(bbox.y1)),
                (round(bbox.x2), round(bbox.y2)), color, 2,
            )
            cv2.putText(
                image, label,
                (round(bbox.x1), max(18, round(bbox.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
            )
        if self._debug_pub is not None:
            self._debug_pub.publish(bgr_message(image, image_message))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PersonRangeFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
