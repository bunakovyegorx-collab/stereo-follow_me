"""Publish a rate-limited JPEG view of /stereo/disparity for Foxglove.

Raw 32FC1 disparity is 4 bytes per pixel: 1.23 MB per frame at 640x480, or
~13.5 MB/s at 11 Hz. eth0 here negotiates 100 Mb/s (12.5 MB/s), so that one
topic alone cannot fit down the link -- the websocket send queue grows without
bound and Foxglove falls tens of seconds behind.

This node keeps the full-precision disparity on the Pi (depth_node and fusion
still consume it untouched) and sends the host a colourised JPEG instead:
~36 KB per frame at 5 Hz is ~0.18 MB/s, roughly 1% of the link. It also
restores foxglove_bridge's send_buffer_limit as a working backpressure valve --
at 1.23 MB a single frame exceeded the 1 MB limit, so stale frames could never
be dropped.
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from stereo_msgs.msg import DisparityImage

from drone_stereo_bringup.disparity_viz_node import disparity_bounds
from drone_stereo_bringup.disparity_viz_node import disparity_to_mono8


# Keep only the newest frame: this node exists to shed load, so queueing
# disparity frames here would just move the backlog upstream of the network.
STREAM_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def encode_jpeg(mono8, quality, colormap=cv2.COLORMAP_TURBO):
    """Colourise a mono8 disparity view and return JPEG bytes.

    Colour costs ~10 KB more per frame than greyscale but makes depth banding
    readable at a glance, which is the whole point of the Foxglove view.
    """
    image = mono8 if colormap is None else cv2.applyColorMap(mono8, colormap)
    ok, buffer = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise ValueError('JPEG encoding failed')
    return buffer.tobytes()


class DisparityJpeg(Node):
    """Throttle + compress disparity into sensor_msgs/CompressedImage."""

    def __init__(self):
        super().__init__('disparity_jpeg')
        self.declare_parameter('disparity_topic', '/stereo/disparity')
        self.declare_parameter(
            'jpeg_topic', '/stereo/disparity_viz/compressed',
        )
        # 5 Hz is well under the ~11 Hz SGBM rate: a human watching a depth
        # view cannot use more, and every skipped frame is 36 KB saved.
        self.declare_parameter('max_fps', 5.0)
        self.declare_parameter('jpeg_quality', 60)
        self.declare_parameter('colour', True)

        self._quality = int(self.get_parameter('jpeg_quality').value)
        self._colormap = (
            cv2.COLORMAP_TURBO
            if bool(self.get_parameter('colour').value)
            else None
        )
        max_fps = float(self.get_parameter('max_fps').value)
        self._min_period_ns = int(1e9 / max_fps) if max_fps > 0.0 else 0
        self._last_stamp_ns = None

        jpeg_topic = str(self.get_parameter('jpeg_topic').value)
        disparity_topic = str(self.get_parameter('disparity_topic').value)
        self._publisher = self.create_publisher(
            CompressedImage, jpeg_topic, STREAM_QOS,
        )
        self.create_subscription(
            DisparityImage, disparity_topic, self.callback, STREAM_QOS,
        )
        self.get_logger().info(
            'disparity→JPEG ready: in=%s out=%s (%.1f Hz max, q%d, %s)'
            % (
                disparity_topic, jpeg_topic, max_fps, self._quality,
                'colour' if self._colormap is not None else 'mono',
            )
        )

    def _due(self, now_ns):
        """True when enough time has passed to publish another frame."""
        if self._min_period_ns <= 0:
            return True
        if self._last_stamp_ns is None:
            return True
        return now_ns - self._last_stamp_ns >= self._min_period_ns

    def callback(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        if not self._due(now_ns):
            return

        image = msg.image
        if image.encoding.lower() not in ('32fc1', '32fc'):
            self.get_logger().warning(
                f'expected 32FC1 disparity, got {image.encoding}',
                throttle_duration_sec=10.0,
            )
            return
        try:
            minimum, maximum = disparity_bounds(msg)
            mono8 = disparity_to_mono8(
                image.data,
                int(image.height),
                int(image.width),
                int(image.step),
                minimum,
                maximum,
            )
            payload = encode_jpeg(mono8, self._quality, self._colormap)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=10.0)
            return

        self._last_stamp_ns = now_ns
        output = CompressedImage()
        output.header = image.header
        output.format = 'jpeg'
        output.data = payload
        self._publisher.publish(output)


def main():
    rclpy.init()
    node = DisparityJpeg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
