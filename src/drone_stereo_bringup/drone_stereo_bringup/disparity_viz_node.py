import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage


RVIZ_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)


def disparity_to_mono8(data, height, width, step, minimum, maximum):
    """Convert a padded 32FC1 disparity image into an RViz-friendly mono8."""
    if height <= 0 or width <= 0 or step < width * 4 or step % 4:
        raise ValueError('invalid 32FC1 disparity dimensions')
    if len(data) != height * step:
        raise ValueError('invalid 32FC1 disparity buffer size')

    row_floats = step // 4
    disparity = np.frombuffer(data, dtype=np.float32).reshape(
        height, row_floats,
    )[:, :width]
    valid = np.isfinite(disparity) & (disparity >= minimum)
    span = float(maximum) - float(minimum)
    if span <= 0:
        raise ValueError('maximum disparity must exceed minimum disparity')

    result = np.zeros((height, width), dtype=np.uint8)
    scaled = 1.0 + 254.0 * np.clip(
        (disparity - float(minimum)) / span, 0.0, 1.0,
    )
    result[valid] = scaled[valid].astype(np.uint8)
    return result


class DisparityViz(Node):
    def __init__(self):
        super().__init__('disparity_viz')
        self.publisher = self.create_publisher(
            Image, '/stereo/disparity_viz', RVIZ_QOS,
        )
        self.subscription = self.create_subscription(
            DisparityImage,
            '/stereo/disparity',
            self.callback,
            RVIZ_QOS,
        )
        self.get_logger().info(
            'publishing /stereo/disparity_viz (mono8) from '
            '/stereo/disparity'
        )

    def callback(self, msg):
        image = msg.image
        if image.encoding.lower() not in ('32fc1', '32fc'):
            self.get_logger().warning(
                f'expected 32FC1 disparity, got {image.encoding}',
                throttle_duration_sec=10.0,
            )
            return
        try:
            maximum = float(msg.min_disparity) + float(msg.max_disparity)
            # max_disparity in stereo_msgs is normally the absolute upper
            # bound. Keep compatibility with publishers that expose it so.
            if float(msg.max_disparity) > float(msg.min_disparity):
                maximum = float(msg.max_disparity)
            visualization = disparity_to_mono8(
                image.data,
                int(image.height),
                int(image.width),
                int(image.step),
                float(msg.min_disparity),
                maximum,
            )
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=10.0)
            return

        output = Image()
        output.header = image.header
        output.height = image.height
        output.width = image.width
        output.encoding = 'mono8'
        output.is_bigendian = False
        output.step = image.width
        output.data = visualization.tobytes()
        self.publisher.publish(output)


def main():
    rclpy.init()
    node = DisparityViz()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
