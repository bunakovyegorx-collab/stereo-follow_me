"""Publish metric depth Image from stereo DisparityImage (vectorized)."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage

from person_range_fusion.depth_convert import disparity_to_depth_m
from person_range_fusion.fusion_node import disparity_array


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def depth_image_message(depth: np.ndarray, source: DisparityImage) -> Image:
    """Pack float32 depth (metres) into sensor_msgs/Image 32FC1."""
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise ValueError('depth must be a float32 HxW array')
    message = Image()
    message.header = source.header
    message.height, message.width = depth.shape
    message.encoding = '32FC1'
    message.is_bigendian = False
    message.step = message.width * 4
    message.data = depth.tobytes()
    return message


class DisparityToDepthNode(Node):
    """Convert each disparity frame to metric depth with one numpy divide."""

    def __init__(self) -> None:
        super().__init__('depth')
        self.declare_parameter('disparity_topic', '/stereo/disparity')
        self.declare_parameter('depth_topic', '/stereo/depth')

        qos = sensor_qos()
        depth_topic = str(self.get_parameter('depth_topic').value)
        disparity_topic = str(self.get_parameter('disparity_topic').value)
        self._pub = self.create_publisher(Image, depth_topic, qos)
        self.create_subscription(
            DisparityImage, disparity_topic, self._callback, qos,
        )
        self.get_logger().info(
            'disparity→depth ready: in=%s out=%s (32FC1 metres)'
            % (disparity_topic, depth_topic)
        )

    def _callback(self, message: DisparityImage) -> None:
        try:
            disparity = disparity_array(message)
            depth = disparity_to_depth_m(
                disparity,
                f=float(message.f),
                baseline=float(message.t),
                min_disparity=float(message.min_disparity),
            )
            self._pub.publish(depth_image_message(depth, message))
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DisparityToDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
