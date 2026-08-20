import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2


PREVIEW_STRIDE = 4
PREVIEW_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
)


def validate_preview_stride(value):
    stride = int(value)
    if stride < 1:
        raise ValueError('stride must be at least 1')
    return stride


def decimate_organized_data(data, height, width, point_step, stride):
    if stride < 1:
        raise ValueError('stride must be at least 1')
    expected = height * width * point_step
    if len(data) != expected:
        raise ValueError(
            f'expected {expected} point bytes, got {len(data)}'
        )
    source = np.frombuffer(data, np.uint8).reshape(
        height, width, point_step,
    )
    preview = source[::stride, ::stride].copy()
    return preview.tobytes(), preview.shape[0], preview.shape[1]


def decimate_mono8_data(data, height, width, step, stride):
    if stride < 1:
        raise ValueError('stride must be at least 1')
    if step < width or len(data) != height * step:
        raise ValueError('invalid mono8 dimensions or row step')
    source = np.frombuffer(data, np.uint8).reshape(height, step)[:, :width]
    preview = source[::stride, ::stride].copy()
    return preview.tobytes(), preview.shape[0], preview.shape[1]


class PointCloudPreview(Node):
    def __init__(self):
        super().__init__('stereo_point_cloud_preview')
        self.preview_stride = validate_preview_stride(
            self.declare_parameter('stride', PREVIEW_STRIDE).value
        )
        self.publisher = self.create_publisher(
            PointCloud2, '/stereo/points2_preview', PREVIEW_QOS,
        )
        self.left_publisher = self.create_publisher(
            Image, '/stereo/left/image_rect_preview', PREVIEW_QOS,
        )
        self.depth_publisher = self.create_publisher(
            Image, '/stereo/depth_viz_preview', PREVIEW_QOS,
        )
        self.subscription = self.create_subscription(
            PointCloud2, '/stereo/points2', self.callback, PREVIEW_QOS,
        )
        self.left_subscription = self.create_subscription(
            Image,
            '/stereo/left/image_rect',
            lambda msg: self.image_callback(msg, self.left_publisher),
            PREVIEW_QOS,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            '/stereo/depth_viz',
            lambda msg: self.image_callback(msg, self.depth_publisher),
            PREVIEW_QOS,
        )
        self.get_logger().info(
            'publishing /stereo/points2_preview with stride '
            f'{self.preview_stride}'
        )

    def callback(self, msg):
        packed_row_step = msg.width * msg.point_step
        source = np.frombuffer(msg.data, np.uint8).reshape(
            msg.height, msg.row_step,
        )[:, :packed_row_step].copy()
        data, height, width = decimate_organized_data(
            source.tobytes(),
            msg.height,
            msg.width,
            msg.point_step,
            self.preview_stride,
        )
        preview = PointCloud2()
        preview.header = msg.header
        preview.height = height
        preview.width = width
        preview.fields = msg.fields
        preview.is_bigendian = msg.is_bigendian
        preview.point_step = msg.point_step
        preview.row_step = width * msg.point_step
        preview.data = data
        preview.is_dense = msg.is_dense
        self.publisher.publish(preview)

    def image_callback(self, msg, publisher):
        if msg.encoding.lower() != 'mono8':
            self.get_logger().warning(
                f'preview requires mono8, got {msg.encoding}',
                throttle_duration_sec=10.0,
            )
            return
        data, height, width = decimate_mono8_data(
            msg.data,
            msg.height,
            msg.width,
            msg.step,
            self.preview_stride,
        )
        preview = Image()
        preview.header = msg.header
        preview.height = height
        preview.width = width
        preview.encoding = 'mono8'
        preview.is_bigendian = msg.is_bigendian
        preview.step = width
        preview.data = data
        publisher.publish(preview)


def main():
    rclpy.init()
    node = PointCloudPreview()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
