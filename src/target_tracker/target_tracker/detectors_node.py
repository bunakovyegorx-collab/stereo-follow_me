"""Two geometric detectors over one depth image, side by side.

Both detectors live in one node on purpose. /stereo/depth is 32FC1 640x480 =
1.23 MB per frame, so a second subscription would double the deserialisation
cost on the Pi; and the point of this stage is to compare the two on the very
same frame, which two nodes could not guarantee.

Nothing heavy goes on the wire: boxes as Detection3DArray + SceneUpdate, and
the U-map only as a rate-limited JPEG. No PointCloud2, no raw images -- eth0
here is 100 Mb/s and raw depth already proved it cannot take the load.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue
from foxglove_msgs.msg import SceneUpdate
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray

from target_tracker.clustering import cluster_voxels
from target_tracker.geometry import depth_to_points
from target_tracker.geometry import optical_to_base
from target_tracker.geometry import voxel_filter
from target_tracker.udepth import build_u_map
from target_tracker.udepth import extract_u_boxes
from target_tracker.udepth import u_box_to_3d
from target_tracker.viz import CLUSTER_BLUE
from target_tracker.viz import UDEPTH_YELLOW
from target_tracker.viz import build_scene_update
from target_tracker.viz import clusters_to_detections
from target_tracker.viz import voxel_cloud
from target_tracker.viz import u_boxes_to_detections


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def depth_array(message: Image) -> np.ndarray:
    """View a 32FC1 Image as a 2D float32 array without copying.

    Mirrors ``person_range_fusion.fusion_node.disparity_array``: rows may be
    padded, so the stride comes from ``step`` rather than from ``width``.
    """
    if message.encoding.lower() not in ('32fc1', '32fc'):
        raise ValueError(f'expected 32FC1 depth, got {message.encoding!r}')
    if message.height <= 0 or message.width <= 0 or message.step < message.width * 4:
        raise ValueError('invalid depth image dimensions')
    row_values = int(message.step) // 4
    if len(message.data) != int(message.height) * int(message.step):
        raise ValueError('invalid depth buffer size')
    result = np.frombuffer(message.data, dtype=np.float32).reshape(
        int(message.height), row_values,
    )[:, :int(message.width)]
    if message.is_bigendian:
        result = result.byteswap()
    return result


class Detectors(Node):
    """Voxel clustering and U-depth over the same depth frame."""

    def __init__(self) -> None:
        super().__init__('detectors')
        self.declare_parameter('depth_topic', '/stereo/depth')
        self.declare_parameter('camera_info_topic', '/stereo/left/camera_info')
        self.declare_parameter('output_frame', 'stereo_left_up')
        self.declare_parameter('optical_frame', 'stereo_left_optical_frame')
        self.declare_parameter('camera_height', 0.35)
        self.declare_parameter('enable_cluster', True)
        self.declare_parameter('enable_udepth', True)
        self.declare_parameter('stride', 3)
        self.declare_parameter('z_min', 1.2)
        self.declare_parameter('z_max', 5.5)
        self.declare_parameter('voxel_resolution', 0.08)
        self.declare_parameter('min_points_per_voxel', 4)
        self.declare_parameter('ground_z', 0.15)
        self.declare_parameter('min_cluster_points', 25)
        self.declare_parameter('min_cluster_voxels', 4)
        self.declare_parameter('udepth_bins', 64)
        self.declare_parameter('udepth_col_scale', 0.5)
        self.declare_parameter('udepth_min_count', 6)
        self.declare_parameter('udepth_min_run', 8)
        self.declare_parameter('publish_umap_jpeg', True)
        self.declare_parameter('umap_jpeg_fps', 2.0)
        self.declare_parameter('umap_jpeg_quality', 80)
        self.declare_parameter('scene_lifetime_sec', 0.5)
        self.declare_parameter('publish_cloud', True)

        self._output_frame = str(self.get_parameter('output_frame').value).strip()
        self._optical_frame = str(
            self.get_parameter('optical_frame').value
        ).strip()
        self._camera_height = float(self.get_parameter('camera_height').value)
        self._enable_cluster = bool(self.get_parameter('enable_cluster').value)
        self._enable_udepth = bool(self.get_parameter('enable_udepth').value)
        self._stride = int(self.get_parameter('stride').value)
        if self._stride < 1:
            raise ValueError('stride must be >= 1')
        self._z_min = float(self.get_parameter('z_min').value)
        self._z_max = float(self.get_parameter('z_max').value)
        if not 0.0 < self._z_min < self._z_max:
            raise ValueError('expected 0 < z_min < z_max')
        self._voxel_resolution = float(
            self.get_parameter('voxel_resolution').value
        )
        if self._voxel_resolution <= 0.0:
            raise ValueError('voxel_resolution must be positive')
        self._min_points_per_voxel = int(
            self.get_parameter('min_points_per_voxel').value
        )
        self._ground_z = float(self.get_parameter('ground_z').value)
        self._min_cluster_points = int(
            self.get_parameter('min_cluster_points').value
        )
        self._min_cluster_voxels = int(
            self.get_parameter('min_cluster_voxels').value
        )
        self._udepth_bins = int(self.get_parameter('udepth_bins').value)
        self._udepth_col_scale = float(
            self.get_parameter('udepth_col_scale').value
        )
        self._udepth_min_count = int(
            self.get_parameter('udepth_min_count').value
        )
        self._udepth_min_run = int(self.get_parameter('udepth_min_run').value)
        self._scene_lifetime_sec = float(
            self.get_parameter('scene_lifetime_sec').value
        )
        if self._scene_lifetime_sec <= 0.0:
            raise ValueError('scene_lifetime_sec must be positive')

        self._publish_cloud = bool(self.get_parameter('publish_cloud').value)
        self._publish_umap = bool(self.get_parameter('publish_umap_jpeg').value)
        self._umap_quality = int(self.get_parameter('umap_jpeg_quality').value)
        umap_fps = float(self.get_parameter('umap_jpeg_fps').value)
        # The JPEG exists for a human eye, not for the pipeline: a couple of
        # frames a second is plenty and keeps the link essentially free.
        self._umap_min_period_ns = int(1e9 / umap_fps) if umap_fps > 0.0 else 0
        self._last_umap_ns: int | None = None

        self._camera_info: CameraInfo | None = None

        qos = sensor_qos()
        self._cluster_boxes_pub = self.create_publisher(
            Detection3DArray, 'cluster/boxes', qos,
        )
        self._cluster_scene_pub = self.create_publisher(
            SceneUpdate, 'cluster/scene', qos,
        )
        self._cloud_pub = (
            self.create_publisher(PointCloud2, 'cluster/cloud', qos)
            if self._publish_cloud else None
        )
        self._udepth_boxes_pub = self.create_publisher(
            Detection3DArray, 'udepth/boxes', qos,
        )
        self._udepth_scene_pub = self.create_publisher(
            SceneUpdate, 'udepth/scene', qos,
        )
        self._umap_pub = (
            self.create_publisher(CompressedImage, 'udepth/umap/compressed', qos)
            if self._publish_umap else None
        )
        self._timing_pub = self.create_publisher(DiagnosticArray, 'timing', qos)

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self._camera_info_callback,
            qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self._depth_callback,
            qos,
        )
        self.get_logger().info(
            'detectors ready: cluster=%s udepth=%s stride=%d z=[%.2f, %.2f]'
            % (
                self._enable_cluster, self._enable_udepth,
                self._stride, self._z_min, self._z_max,
            )
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _intrinsics(self) -> tuple[float, float, float, float]:
        """fx, fy, cx, cy from the rectified projection matrix."""
        projection = self._camera_info.p
        fx, cx = float(projection[0]), float(projection[2])
        fy, cy = float(projection[5]), float(projection[6])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError('camera projection matrix has invalid focal lengths')
        return fx, fy, cx, cy

    def _depth_callback(self, message: Image) -> None:
        if self._camera_info is None:
            self.get_logger().warning(
                'waiting for camera_info', throttle_duration_sec=5.0,
            )
            return
        try:
            depth = depth_array(message)
            fx, fy, cx, cy = self._intrinsics()
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return

        timings: dict[str, float] = {}
        started = time.perf_counter()
        if self._enable_cluster:
            self._run_cluster(depth, message.header, fx, fy, cx, cy, timings)
        if self._enable_udepth:
            self._run_udepth(depth, message.header, fx, fy, cx, cy, timings)
        timings['total'] = (time.perf_counter() - started) * 1000.0
        self._publish_timing(message.header, timings)

    def _run_cluster(
        self,
        depth: np.ndarray,
        header: Header,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        timings: dict[str, float],
    ) -> None:
        mark = time.perf_counter()
        points = depth_to_points(
            depth, fx=fx, fy=fy, cx=cx, cy=cy,
            stride=self._stride, z_min=self._z_min, z_max=self._z_max,
        )
        base = optical_to_base(points, camera_height=self._camera_height)
        timings['to_points'] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        centroids, index, counts = voxel_filter(
            base,
            resolution=self._voxel_resolution,
            min_points_per_voxel=self._min_points_per_voxel,
            ground_z=self._ground_z,
            max_range=self._z_max,
        )
        timings['voxel'] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        try:
            clusters = cluster_voxels(
                centroids, index, counts,
                min_cluster_points=self._min_cluster_points,
                min_cluster_voxels=self._min_cluster_voxels,
            )
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        timings['cluster'] = (time.perf_counter() - mark) * 1000.0
        timings['cluster_count'] = float(len(clusters))
        timings['voxel_count'] = float(centroids.shape[0])

        out_header = Header()
        out_header.stamp = header.stamp
        out_header.frame_id = self._output_frame or header.frame_id
        if self._cloud_pub is not None:
            mark = time.perf_counter()
            self._cloud_pub.publish(voxel_cloud(centroids, counts, out_header))
            timings['cloud'] = (time.perf_counter() - mark) * 1000.0

        detections = clusters_to_detections(clusters, out_header)
        self._cluster_boxes_pub.publish(detections)
        self._cluster_scene_pub.publish(
            build_scene_update(
                detections,
                color=CLUSTER_BLUE,
                lifetime_sec=self._scene_lifetime_sec,
                label_prefix='C ',
            )
        )

    def _run_udepth(
        self,
        depth: np.ndarray,
        header: Header,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        timings: dict[str, float],
    ) -> None:
        mark = time.perf_counter()
        u_map = build_u_map(
            depth,
            z_min=self._z_min, z_max=self._z_max,
            bins=self._udepth_bins, col_scale=self._udepth_col_scale,
        )
        timings['umap'] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        rects = extract_u_boxes(u_map, min_count=self._udepth_min_count)
        timings['u_boxes'] = (time.perf_counter() - mark) * 1000.0

        mark = time.perf_counter()
        boxes = []
        for rect in rects:
            box = u_box_to_3d(
                depth, rect, fx=fx, fy=fy, cx=cx, cy=cy,
                z_min=self._z_min, z_max=self._z_max,
                bins=self._udepth_bins, col_scale=self._udepth_col_scale,
                min_run=self._udepth_min_run,
            )
            if box is not None:
                boxes.append(box)
        timings['u_to_3d'] = (time.perf_counter() - mark) * 1000.0
        timings['udepth_count'] = float(len(boxes))

        # U-depth stays in the optical frame: the height search is a pixel
        # operation, so there is nothing to gain from rotating it here.
        out_header = Header()
        out_header.stamp = header.stamp
        out_header.frame_id = self._optical_frame or header.frame_id
        detections = u_boxes_to_detections(boxes, out_header)
        self._udepth_boxes_pub.publish(detections)
        self._udepth_scene_pub.publish(
            build_scene_update(
                detections,
                color=UDEPTH_YELLOW,
                lifetime_sec=self._scene_lifetime_sec,
                label_prefix='U ',
            )
        )
        self._publish_umap_jpeg(u_map, rects, header)

    def _publish_umap_jpeg(
        self,
        u_map: np.ndarray,
        rects: list[tuple[int, int, int, int]],
        header: Header,
    ) -> None:
        if self._umap_pub is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._umap_min_period_ns > 0 and self._last_umap_ns is not None:
            if now_ns - self._last_umap_ns < self._umap_min_period_ns:
                return

        peak = int(u_map.max())
        scale = 255.0 / peak if peak > 0 else 0.0
        mono8 = np.clip(u_map.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        colour = cv2.applyColorMap(mono8, cv2.COLORMAP_JET)
        for x, y, w, h in rects:
            cv2.rectangle(colour, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
        ok, buffer = cv2.imencode(
            '.jpg', colour, [cv2.IMWRITE_JPEG_QUALITY, self._umap_quality],
        )
        if not ok:
            self.get_logger().warning(
                'U-map JPEG encoding failed', throttle_duration_sec=10.0,
            )
            return

        self._last_umap_ns = now_ns
        output = CompressedImage()
        output.header = header
        output.format = 'jpeg'
        output.data = buffer.tobytes()
        self._umap_pub.publish(output)

    def _publish_timing(self, header: Header, timings: dict[str, float]) -> None:
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = 'target_tracker: detectors'
        status.hardware_id = 'stereo'
        status.message = 'total %.1f ms' % timings.get('total', 0.0)
        for key in sorted(timings):
            status.values.append(
                KeyValue(key=key, value='%.3f' % timings[key])
            )

        message = DiagnosticArray()
        message.header.stamp = header.stamp
        message.status.append(status)
        self._timing_pub.publish(message)

        self.get_logger().info(
            ' '.join('%s=%.2f' % (k, timings[k]) for k in sorted(timings)),
            throttle_duration_sec=5.0,
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Detectors()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
