#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from carla_perception_msgs.msg import DetectedObjectArray
import numpy as np
from collections import defaultdict
import time

class PerceptionAnalyzer(Node):
    def __init__(self):
        super().__init__('perception_analyzer')
        self.subscription = self.create_subscription(
            DetectedObjectArray,
            '/perception/detected_objects',
            self.analyze_callback,
            10)
        
        # Statistics
        self.object_counts = defaultdict(int)
        self.total_frames = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        
    def analyze_callback(self, msg):
        self.total_frames += 1
        
        # Count objects by type
        vehicles = sum(1 for obj in msg.objects if obj.label == 'Vehicle')
        pedestrians = sum(1 for obj in msg.objects if obj.label == 'Pedestrian')
        
        # Calculate statistics
        distances = [obj.euclidean_distance for obj in msg.objects]
        
        # Print every 2 seconds
        current_time = time.time()
        if current_time - self.last_print_time >= 2.0:
            runtime = current_time - self.start_time
            fps = self.total_frames / runtime
            
            self.get_logger().info(f'\n{"="*60}')
            self.get_logger().info(f'CARLA PERCEPTION ANALYSIS')
            self.get_logger().info(f'{"="*60}')
            self.get_logger().info(f'Runtime: {runtime:.1f}s | FPS: {fps:.1f}')
            self.get_logger().info(f'Total frames processed: {self.total_frames}')
            self.get_logger().info(f'\nCurrent Frame:')
            self.get_logger().info(f'  Vehicles: {vehicles}')
            self.get_logger().info(f'  Pedestrians: {pedestrians}')
            self.get_logger().info(f'  Total objects: {len(msg.objects)}')
            
            if distances:
                self.get_logger().info(f'\nDistance Statistics:')
                self.get_logger().info(f'  Min: {min(distances):.2f}m')
                self.get_logger().info(f'  Max: {max(distances):.2f}m')
                self.get_logger().info(f'  Mean: {np.mean(distances):.2f}m')
                
            # Show closest objects
            if msg.objects:
                self.get_logger().info(f'\nClosest Objects:')
                sorted_objects = sorted(msg.objects, key=lambda x: x.euclidean_distance)
                for obj in sorted_objects[:3]:
                    self.get_logger().info(
                        f'  {obj.label} ID:{obj.id} - '
                        f'Distance: {obj.euclidean_distance:.2f}m '
                        f'(x:{obj.position.x:.2f}, y:{obj.position.y:.2f})'
                    )
            
            self.last_print_time = current_time

def main(args=None):
    rclpy.init(args=args)
    analyzer = PerceptionAnalyzer()
    rclpy.spin(analyzer)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()