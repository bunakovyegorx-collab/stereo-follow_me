#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image

class DepthViz(Node):
    def __init__(self):
        super().__init__('stereo_depth_viz')
        self.sub = self.create_subscription(DisparityImage, '/stereo/disparity', self.cb, qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, '/stereo/depth_meters', 10)
        self.pub_viz = self.create_publisher(Image, '/stereo/depth_viz', 10)
        self.get_logger().info('publishing /stereo/depth_meters 32FC1 and /stereo/depth_viz mono8 from /stereo/disparity')

    def cb(self, msg):
        img = msg.image
        h, w = int(img.height), int(img.width)
        if h <= 0 or w <= 0 or img.encoding.lower() not in ('32fc1', '32fc'):
            return
        row_floats = int(img.step // 4)
        try:
            disp = np.frombuffer(img.data, dtype=np.float32).reshape((h, row_floats))[:, :w]
        except Exception as exc:
            self.get_logger().warn(f'bad disparity buffer: {exc}')
            return

        baseline_f = abs(float(msg.f) * float(msg.t))
        valid = np.isfinite(disp) & (disp > 0.01) & (baseline_f > 0.0)
        depth = np.zeros((h, w), dtype=np.float32)
        depth[valid] = baseline_f / disp[valid]

        out = Image()
        out.header = msg.header
        out.height = h
        out.width = w
        out.encoding = '32FC1'
        out.is_bigendian = False
        out.step = w * 4
        out.data = depth.tobytes()
        self.pub_depth.publish(out)

        viz = np.zeros((h, w), dtype=np.uint8)
        finite = valid & (depth > 0.02) & (depth < 20.0)
        if np.any(finite):
            vals = depth[finite]
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98))
            if hi <= lo:
                hi = lo + 1.0
            scaled = 255.0 * (1.0 - np.clip((depth - lo) / (hi - lo), 0.0, 1.0))
            viz[finite] = scaled[finite].astype(np.uint8)
        vmsg = Image()
        vmsg.header = msg.header
        vmsg.height = h
        vmsg.width = w
        vmsg.encoding = 'mono8'
        vmsg.is_bigendian = False
        vmsg.step = w
        vmsg.data = viz.tobytes()
        self.pub_viz.publish(vmsg)

def main():
    rclpy.init()
    node = DepthViz()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
