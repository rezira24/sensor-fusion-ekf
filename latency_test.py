#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from carla_perception_msgs.msg import DetectedObjectArray
import time

class LatencyTester(Node):
    def __init__(self):
        super().__init__('latency_tester')
        self.subscription = self.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            self.latency_callback,
            10)
        
    def latency_callback(self, msg):
        current_time = self.get_clock().now()
        msg_time = msg.header.stamp
        
        latency_ns = (current_time.nanoseconds - 
                      (msg_time.sec * 1e9 + msg_time.nanosec))
        latency_ms = latency_ns / 1e6
        
        self.get_logger().info(f'Latency: {latency_ms:.2f} ms')

def main(args=None):
    rclpy.init(args=args)
    tester = LatencyTester()
    rclpy.spin(tester)
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()