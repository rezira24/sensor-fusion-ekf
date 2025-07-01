#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from carla_perception_msgs.msg import DetectedObjectArray

class PerceptionSubscriber(Node):
    def __init__(self):
        super().__init__('perception_subscriber')
        self.subscription = self.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'Received {len(msg.objects)} objects')
        for obj in msg.objects:
            self.get_logger().info(f'ID: {obj.id}, Label: {obj.label}, Distance: {obj.euclidean_distance:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
