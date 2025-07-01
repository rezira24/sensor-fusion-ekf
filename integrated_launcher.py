#!/usr/bin/env python

"""
integrated_launcher.py - Unified Timeline Multi-Sensor Fusion System
Integrates traffic generation and sensor fusion into a single synchronized system
Ensures all components use the same timeline for consistent results
FIXED: Now properly records ALL timing metrics including frame_time and visualization
ENHANCED: Added ROS 2 Publisher for detected objects
"""

import carla
import numpy as np
import cv2
import time
import random
import argparse
import logging
from queue import Queue, Empty
from collections import deque
import os
from datetime import datetime
import threading

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from carla_perception_msgs.msg import DetectedObject, DetectedObjectArray

# Import existing modules
from config1 import *
from utils import get_intrinsic_matrix, process_image, MAPPING_COLORS
from object6_detector import (
    euclidean_clustering, project_3d_bounding_boxes, 
    preprocess_lidar, print_preprocessing_stats, 
    draw_lidar_projection, get_object_bboxes_from_segmentation, 
    draw_bboxes_on_image
)
from distance_calculator import check_ground_truth_distance
from segmentation import visualize_semantic_segmentation
from lane_detection import estimate_lane_lines, merge_lane_lines
from sensors import setup_sensors, get_sensor_data, cleanup_sensors, sensor_callback

# Import 4D state EKF fusion module
from sensor11_fusion import (
    MultiSensorTracker,
    MultiSensorBEVVisualizer, 
    convert_to_multi_sensor_format,
    PerformanceMonitor
)

# Import enhanced evaluation system
from eval_system import (
    EnhancedEvaluationMetrics,
    EnhancedPerformanceMonitor,
    generate_final_report_and_plots
)

# Import visualization classes from main12
from main12enhancedcopy import (
    EnhancedBEVVisualizerWithFusion,
    preprocess_camera_segmentation,
    enhance_multi_sensor_tracker_with_tiered_lifetime,
    create_info_panel,
    create_performance_panel,
    print_periodic_report,
    process_lidar_data,
    process_camera_data,
    perform_clustering,
    perform_sensor_fusion
)


class IntegratedSensorFusionSystem:
    """
    Integrated system that handles both traffic generation and sensor fusion
    in a single synchronized timeline
    ENHANCED: Now includes ROS 2 publishing capability
    """
    
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.ego_vehicle = None
        self.sensors = {}
        self.sensor_queues = {}
        
        # Traffic vehicles and walkers
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        
        # Sensor fusion components
        self.multi_sensor_tracker = None
        self.bev_visualizer = None
        self.evaluator = None
        self.performance_monitor = None
        
        # Timing and synchronization
        self.synchronous_master = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_print_time = time.time()

        self.depth_map_counter = 0
        self.segmentation_counter = 0
        self.combined_counter = 0
        self.ekf_counter = 0
        
        # ROS 2 components
        self.ros2_node = None
        self.ros2_publisher = None
        self.ros2_executor = None
        self.ros2_thread = None
        
        # Configure logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        
    def initialize_ros2(self):
        """Initialize ROS 2 node and publisher"""
        try:
            # Initialize rclpy if not already initialized
            if not rclpy.ok():
                rclpy.init()
            
            # Create ROS 2 node
            self.ros2_node = Node('carla_perception_node')
            
            # Create QoS profile with depth 10
            qos_profile = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE
            )
            
            # Create publisher
            self.ros2_publisher = self.ros2_node.create_publisher(
                DetectedObjectArray,
                '/perception/detected_objects',
                qos_profile
            )
            
            # Create executor
            self.ros2_executor = SingleThreadedExecutor()
            self.ros2_executor.add_node(self.ros2_node)
            
            # Start executor in separate thread
            self.ros2_thread = threading.Thread(
                target=self._ros2_spin,
                daemon=True
            )
            self.ros2_thread.start()
            
            self.ros2_node.get_logger().info('ROS 2 perception node initialized')
            print("✓ ROS 2 node 'carla_perception_node' initialized")
            print("✓ Publishing to '/perception/detected_objects'")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize ROS 2: {e}")
            print("  Continuing without ROS 2 publishing...")
            self.ros2_node = None
            self.ros2_publisher = None
            
    def _ros2_spin(self):
        """ROS 2 executor spin function for separate thread"""
        try:
            self.ros2_executor.spin()
        except Exception as e:
            print(f"ROS 2 executor error: {e}")
            
    def publish_ros2_data(self, tracks):
        """
        Publish detected objects to ROS 2 topic
        
        Args:
            tracks: List of tracks from sensor fusion (all_tracks)
        """
        if self.ros2_node is None or self.ros2_publisher is None:
            return
            
        try:
            # Create DetectedObjectArray message
            msg = DetectedObjectArray()
            
            # Set header
            msg.header = Header()
            msg.header.stamp = self.ros2_node.get_clock().now().to_msg()
            msg.header.frame_id = 'ego_vehicle'
            
            # Process each track
            for track in tracks:
                # Only publish tracks with labels
                label = track.get('label')
                if label is None:
                    continue
                    
                # Create DetectedObject
                obj = DetectedObject()
                
                # Set ID
                obj.id = track['id']
                
                # Set label - convert to string if needed
                if label == 10 or label == 'Vehicle':
                    obj.label = 'Vehicle'
                elif label == 4 or label == 'Pedestrian':
                    obj.label = 'Pedestrian'
                else:
                    continue  # Skip unknown labels
                
                # Get relative position from track
                relative_pos = track['position']  # [forward, lateral]
                
                # Set position (x: forward, y: right, z: up)
                obj.position = Point()
                obj.position.x = float(relative_pos[0])  # Forward distance
                obj.position.y = float(relative_pos[1])  # Lateral distance (right positive)
                obj.position.z = 0.0  # Ground level
                
                # Calculate Euclidean distance
                obj.euclidean_distance = float(np.linalg.norm(relative_pos))
                
                # Add to array
                msg.objects.append(obj)
            
            # Publish message (even if empty)
            self.ros2_publisher.publish(msg)
            
        except Exception as e:
            if self.ros2_node:
                self.ros2_node.get_logger().error(f'Failed to publish ROS 2 data: {e}')
                
    def connect_to_carla(self):
        """Connect to CARLA server and configure world settings"""
        print("Connecting to CARLA server...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        
        self.world = self.client.get_world()
        
        # Configure Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        if self.args.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
            
        if self.args.seed is not None:
            self.traffic_manager.set_random_device_seed(self.args.seed)
            random.seed(self.args.seed)
        
        # Set synchronous mode for the entire system
        print("Setting up synchronous mode...")
        settings = self.world.get_settings()
        self.traffic_manager.set_synchronous_mode(True)
        
        if not settings.synchronous_mode:
            self.synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            print("✓ Synchronous mode enabled (20 FPS)")
        else:
            self.synchronous_master = False
            print("✓ Using existing synchronous mode")
            
    def spawn_ego_vehicle(self):
        """Spawn ego vehicle with sensors"""
        print("\nSpawning ego vehicle...")
        
        # Clean up any existing ego vehicles
        vehicles = self.world.get_actors().filter("vehicle.*")
        for vehicle in vehicles:
            if vehicle.attributes.get('role_name') == 'hero':
                print(f"Removing existing ego vehicle (ID: {vehicle.id})")
                vehicle.destroy()
        
        # Get blueprint
        blueprint_library = self.world.get_blueprint_library()
        
        # Try Tesla Model 3 first
        tesla_model3 = blueprint_library.filter('vehicle.tesla.model3')
        if tesla_model3:
            vehicle_bp = tesla_model3[0]
            print("Selected vehicle: Tesla Model 3")
        else:
            # Fallback to Audi A2
            audi_a2 = blueprint_library.filter('vehicle.audi.a2')
            if audi_a2:
                vehicle_bp = audi_a2[0]
                print("Using fallback: Audi A2")
            else:
                raise RuntimeError("No suitable vehicle blueprint found!")
        
        # Set as hero vehicle
        vehicle_bp.set_attribute('role_name', 'hero')
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '255,0,0')  # Red color
        
        # Find suitable spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available!")
            
        # Try to spawn at first available position
        for i, spawn_point in enumerate(spawn_points[:3]):
            try:
                self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                print(f"✓ Spawned ego vehicle at position {i}")
                break
            except:
                continue
                
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle!")
            
        # Enable autopilot
        self.ego_vehicle.set_autopilot(True)
        
        # Configure Traffic Manager for ego vehicle
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, -30)
        self.traffic_manager.distance_to_leading_vehicle(self.ego_vehicle, 3.0)
        
        print(f"✓ Ego vehicle ready (ID: {self.ego_vehicle.id})")
        
    def setup_ego_sensors(self):
        """Setup sensors on ego vehicle"""
        print("\nSetting up sensors...")
        
        # Camera parameters
        camera_matrix = get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)
        
        # Setup sensors using existing function
        camera, semantic_cam, lidar, image_queue, semantic_queue, lidar_queue = setup_sensors(
            self.world, self.ego_vehicle, IMAGE_WIDTH, IMAGE_HEIGHT, FOV)
        
        self.sensors = {
            'camera': camera,
            'semantic': semantic_cam,
            'lidar': lidar
        }
        
        self.sensor_queues = {
            'image': image_queue,
            'semantic': semantic_queue,
            'lidar': lidar_queue
        }
        
        # Initialize tracking and evaluation systems
        self.multi_sensor_tracker = MultiSensorTracker(
            camera_matrix, 
            max_age=FUSION_MAX_AGE, 
            min_hits=FUSION_MIN_HITS
        )
        self.multi_sensor_tracker = enhance_multi_sensor_tracker_with_tiered_lifetime(
            self.multi_sensor_tracker
        )
        
        self.bev_visualizer = EnhancedBEVVisualizerWithFusion(
            bev_range=50, 
            image_size=600
        )
        
        self.evaluator = EnhancedEvaluationMetrics(history_size=30)
        self.performance_monitor = EnhancedPerformanceMonitor()
        
        print("✓ Sensors and tracking systems initialized")
        
    def spawn_traffic_vehicles(self):
        """Spawn traffic vehicles"""
        print(f"\nSpawning {self.args.number_of_vehicles} traffic vehicles...")
        
        blueprints = self.world.get_blueprint_library().filter(self.args.filterv)
        
        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        
        if self.args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points
        
        # Spawn vehicles
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            
            # Skip ego vehicle spawn point
            if self.ego_vehicle and transform.location.distance(self.ego_vehicle.get_location()) < 5.0:
                continue
                
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            blueprint.set_attribute('role_name', 'autopilot')
            
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
        
        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)
        
        print(f"✓ Spawned {len(self.vehicles_list)} traffic vehicles")
        
    def spawn_walkers(self):
        """Spawn pedestrians"""
        print(f"\nSpawning {self.args.number_of_walkers} pedestrians...")
        
        blueprintsWalkers = self.world.get_blueprint_library().filter(self.args.filterw)
        
        percentagePedestriansRunning = 0.0
        percentagePedestriansCrossing = 0.0
        
        if self.args.seedw:
            self.world.set_pedestrians_seed(self.args.seedw)
            random.seed(self.args.seedw)
        
        # Get spawn points
        spawn_points = []
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # Spawn walker actors
        SpawnActor = carla.command.SpawnActor
        batch = []
        walker_speed = []
        
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                if random.random() > percentagePedestriansRunning:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        
        # Spawn walker controllers
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        
        # Put together walkers and controllers
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        
        self.all_actors = self.world.get_actors(self.all_id)
        
        # Initialize walker behavior
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].start()
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print(f"✓ Spawned {len(self.walkers_list)} pedestrians")
        
    def process_sensor_data(self):
        """Process one frame of sensor data - ENHANCED with ROS 2 publishing"""
        frame_start = time.time()
        
        # Get sensor data (blocking with timeout) - MEASURE FRAME TIME
        sensor_acquisition_start = time.time()
        try:
            image_data = self.sensor_queues['image'].get(timeout=0.1)
            semantic_data = self.sensor_queues['semantic'].get(timeout=0.1)
            lidar_data = self.sensor_queues['lidar'].get(timeout=0.1)
        except Empty:
            return False
        
        # RECORD FRAME TIME (sensor acquisition)
        frame_time = time.time() - sensor_acquisition_start
        self.performance_monitor.add_metric('frame_time', frame_time)
            
        # Process timestamps
        current_timestamp = self.frame_count * 0.05  # Based on fixed delta time
        
        # Process image
        im_array = process_image(image_data)
        
        # Process LiDAR
        filtered_points, intensities, raw_points, non_ground_mask = process_lidar_data(
            lidar_data, self.performance_monitor)
        
        if filtered_points.shape[0] == 0:
            return True
            
        # Clustering
        clusters = perform_clustering(filtered_points, self.performance_monitor)
        
        # Process semantic segmentation
        semantic_raw, semantic_cleaned = process_camera_data(
            semantic_data, self.performance_monitor)
        
        # Get camera bounding boxes
        bboxes = get_object_bboxes_from_segmentation(semantic_cleaned, target_labels=[4, 10])
        
        # Multi-sensor fusion
        all_tracks = perform_sensor_fusion(
            clusters, filtered_points, bboxes, 
            self.sensors['lidar'], self.sensors['camera'],
            self.multi_sensor_tracker, current_timestamp, 
            self.performance_monitor)
        
        # PUBLISH TO ROS 2
        self.publish_ros2_data(all_tracks)
        
        # Update fusion events
        self.bev_visualizer.update_fusion_events(all_tracks, self.frame_count)
        
        # Enhanced evaluation
        self.evaluator.update_timestamp()
        
        # Metric 1: Camera Semantic IoU
        camera_iou = self.evaluator.evaluate_camera_semantic_iou(
            semantic_cleaned, semantic_data, target_labels=[4, 10])
        
        # Metric 2: LiDAR Distance Error
        lidar_error = self.evaluator.evaluate_lidar_distance_error(
            clusters, filtered_points, self.sensors['lidar'].get_transform(), 
            self.world, self.ego_vehicle)
        
        # Metric 3: EKF Position Error
        ekf_error = self.evaluator.evaluate_ekf_position_error(
            all_tracks, self.world, self.ego_vehicle)
        
        # MEASURE VISUALIZATION TIME
        viz_start = time.time()
        
        # Visualization
        if self.frame_count % 1 == 0:
            self.update_visualization(all_tracks, filtered_points, 0,  # fps will be calculated later
                                    im_array, clusters, intensities,
                                    semantic_cleaned, semantic_raw)
        
        # RECORD VISUALIZATION TIME
        viz_time = time.time() - viz_start
        self.performance_monitor.add_metric('visualization', viz_time)
        
        # Calculate TOTAL frame time and FPS
        frame_time_total = time.time() - frame_start
        fps = 1.0 / frame_time_total if frame_time_total > 0 else 0
        
        # Record performance
        self.performance_monitor.record_frame(fps)
        self.performance_monitor.add_metric('total', frame_time_total)
        
        # Periodic reporting
        current_time = time.time()
        if current_time - self.last_print_time >= 5:
            avg_fps = self.performance_monitor.get_average_fps()
            cpu_usage = self.performance_monitor.get_average_cpu_usage()
            print_periodic_report(self.frame_count, self.evaluator, fps, avg_fps, cpu_usage)
            self.last_print_time = current_time
            
        return True
        
    def update_visualization(self, all_tracks, filtered_points, fps, 
                        im_array, clusters, intensities,
                        semantic_cleaned, semantic_raw):
        """Update all visualization windows with new 2x2 grid layout - OPTIMIZED VERSION"""
        
        # Get current FPS for display
        if self.performance_monitor.fps_history:
            fps = self.performance_monitor.fps_history[-1]
        else:
            fps = 0.0
        
        # Increment counters with new update frequencies
        self.depth_map_counter = (self.depth_map_counter + 1) % 5  # Update every 5 frames
        self.segmentation_counter = (self.segmentation_counter + 1) % 5  # Update every 5 frames
        self.combined_counter = (self.combined_counter + 1) % 3  # Update every 3 frames
        self.ekf_counter = (self.ekf_counter + 1) % 3  # Update every 3 frames
        
        # BEV visualization (always update)
        viz_points = filtered_points
        if len(filtered_points) > 5000:
            indices = np.random.choice(len(filtered_points), 5000, replace=False)
            viz_points = filtered_points[indices]
        
        bev_image = self.bev_visualizer.draw_bev(all_tracks, viz_points)
        
        # Create info panel
        current_metrics = self.evaluator.get_metrics()
        avg_fps = self.performance_monitor.get_average_fps()
        cpu_usage = self.performance_monitor.get_average_cpu_usage()
        
        info_panel = create_info_panel(
            all_tracks, current_metrics, fps, avg_fps, cpu_usage, 
            self.frame_count, self.start_time)
        
        # Combine panels for main BEV window
        bev_with_info = np.vstack([info_panel, bev_image])
        cv2.imshow("Integrated Multi-Sensor Fusion System", bev_with_info)
        
        # Create 2x2 grid visualization - OPTIMIZED
        # Reduced size for each panel: 400x400
        panel_width = 400
        panel_height = 300
        
        # 1. Top-left: Semantic Segmentation (update every 5 frames)
        if self.segmentation_counter == 0 or not hasattr(self, 'last_seg_vis'):
            seg_vis = visualize_semantic_segmentation(
                semantic_raw, IMAGE_HEIGHT, IMAGE_WIDTH, MAPPING_COLORS)
            # Use INTER_NEAREST for faster resize
            self.last_seg_vis = cv2.resize(seg_vis, (panel_width, panel_height), 
                                        interpolation=cv2.INTER_NEAREST)
        seg_panel = self.last_seg_vis
        
        # 2. Top-right: Combined Camera View (update every 3 frames)
        if self.combined_counter == 0 or not hasattr(self, 'last_combined_vis'):
            im_combined = im_array.copy()
            
            # Reduce LiDAR points for projection (sample 50%)
            sampled_points = filtered_points[::2] if len(filtered_points) > 1000 else filtered_points
            sampled_intensities = intensities[::2] if len(intensities) > 1000 else intensities
            
            im_combined = draw_lidar_projection(
                sampled_points, sampled_intensities, self.sensors['lidar'].get_transform(),
                self.sensors['camera'].get_transform(), im_combined, 
                get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV), 
                IMAGE_WIDTH, IMAGE_HEIGHT)
            
            im_combined = project_3d_bounding_boxes(
                im_combined, clusters, filtered_points, self.sensors['lidar'],
                self.sensors['camera'].get_transform(), 
                get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV), 
                self.world, IMAGE_WIDTH, IMAGE_HEIGHT)
            
            bboxes_viz = get_object_bboxes_from_segmentation(semantic_cleaned, target_labels=[4, 10])
            draw_bboxes_on_image(im_combined, bboxes_viz)
            self.last_combined_vis = cv2.resize(im_combined, (panel_width, panel_height),
                                            interpolation=cv2.INTER_LINEAR)
        combined_panel = self.last_combined_vis
        
        # 3. Bottom-left: LiDAR Depth Map (update every 5 frames)
        if self.depth_map_counter == 0 or not hasattr(self, 'last_depth_map'):
            from object6_detector import create_lidar_depth_map
            
            # Sample points for faster processing
            depth_points = filtered_points[::3] if len(filtered_points) > 2000 else filtered_points
            
            depth_map = create_lidar_depth_map(
                depth_points,
                self.sensors['lidar'].get_transform(),
                self.sensors['camera'].get_transform(),
                get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV),
                IMAGE_WIDTH, IMAGE_HEIGHT
            )
            self.last_depth_map = cv2.resize(depth_map, (panel_width, panel_height),
                                        interpolation=cv2.INTER_NEAREST)
        depth_panel = self.last_depth_map
        
        # 4. Bottom-right: EKF Tracks on Camera (update every 3 frames)
        if self.ekf_counter == 0 or not hasattr(self, 'last_ekf_vis'):
            from object6_detector import draw_ekf_tracks_on_camera
            
            ekf_vis = draw_ekf_tracks_on_camera(
                im_array.copy(),
                all_tracks,
                self.ego_vehicle.get_transform(),
                self.sensors['camera'].get_transform(),
                get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)
            )
            self.last_ekf_vis = cv2.resize(ekf_vis, (panel_width, panel_height),
                                        interpolation=cv2.INTER_LINEAR)
        ekf_panel = self.last_ekf_vis
        
        # Add labels to each panel
        label_color = (255, 255, 255)
        label_bg_color = (0, 0, 0)
        
        # Add semi-transparent background for labels
        def add_label(img, text, position=(10, 25)):
            cv2.rectangle(img, (position[0]-5, position[1]-20), 
                        (position[0]+len(text)*7+5, position[1]+5), 
                        label_bg_color, -1)
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, label_color, 1)
        
        add_label(seg_panel, "Semantic Segmentation")
        add_label(combined_panel, "Combined Sensor View")
        add_label(depth_panel, "LiDAR Depth Map")
        add_label(ekf_panel, "EKF Fusion Tracking")
        
        # Combine into 2x2 grid
        top_row = np.hstack([seg_panel, combined_panel])
        bottom_row = np.hstack([depth_panel, ekf_panel])
        grid_2x2 = np.vstack([top_row, bottom_row])
        
        # Display the 2x2 grid (800x800 total instead of 1200x900)
        cv2.imshow("Tampilan Sensor Lengkap", grid_2x2)
        
        # Performance panel (update less frequently)
        if self.frame_count % 10 == 0:
            timing_summary = self.performance_monitor.get_summary()
            perf_panel = create_performance_panel(timing_summary, avg_fps, cpu_usage)
            cv2.imshow("System Performance", perf_panel)
            
    def run(self):
        """Main execution loop"""
        try:
            # Initialize CARLA connection
            self.connect_to_carla()
            
            # Initialize ROS 2
            self.initialize_ros2()
            
            # Spawn ego vehicle and sensors
            self.spawn_ego_vehicle()
            self.setup_ego_sensors()
            
            # Spawn traffic
            self.spawn_traffic_vehicles()
            self.spawn_walkers()
            
            # Configure Traffic Manager
            self.traffic_manager.global_percentage_speed_difference(30.0)
            
            print(f"\n{'='*60}")
            print("INTEGRATED SENSOR FUSION SYSTEM RUNNING")
            print(f"{'='*60}")
            print(f"Synchronous Mode: {self.synchronous_master}")
            print(f"Fixed Delta Time: 0.05s (20 FPS)")
            print(f"Traffic Vehicles: {len(self.vehicles_list)}")
            print(f"Pedestrians: {len(self.walkers_list)}")
            if self.ros2_node:
                print(f"ROS 2 Publishing: Enabled")
            else:
                print(f"ROS 2 Publishing: Disabled")
            print(f"{'='*60}\n")
            
            # Main loop
            while True:
                # Tick the world (master clock)
                if self.synchronous_master:
                    self.world.tick()
                
                # Process sensor data
                if self.process_sensor_data():
                    self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('p'):  # Pause
                    cv2.waitKey(0)
                    
        except KeyboardInterrupt:
            print("\nTerminated by user")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up all resources - ENHANCED with ROS 2 cleanup"""
        print(f"\n{'='*60}")
        print("CLEANUP PHASE")
        print(f"{'='*60}")
        
        # Generate final report
        if self.evaluator and self.performance_monitor:
            try:
                print("\nGenerating final evaluation report...")
                report_path = generate_final_report_and_plots(
                    self.evaluator, self.performance_monitor)
                print(f"✅ Evaluation report saved to: {report_path}")
            except Exception as e:
                print(f"❌ Error generating report: {e}")
        
        # Clean up ROS 2
        if self.ros2_node:
            try:
                print("\nCleaning up ROS 2...")
                if self.ros2_executor:
                    self.ros2_executor.shutdown()
                if self.ros2_node:
                    self.ros2_node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                print("✓ ROS 2 cleaned up")
            except Exception as e:
                print(f"⚠️ Error during ROS 2 cleanup: {e}")
        
        # Restore world settings
        if self.synchronous_master and self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            print("✓ Restored asynchronous mode")
        
        # Clean up sensors
        if self.sensors:
            for sensor in self.sensors.values():
                if sensor is not None:
                    sensor.destroy()
            print("✓ Sensors destroyed")
        
        # Clean up ego vehicle
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            print("✓ Ego vehicle destroyed")
        
        # Clean up traffic vehicles
        if self.vehicles_list and self.client:
            print(f"\nDestroying {len(self.vehicles_list)} traffic vehicles...")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
            print("✓ Traffic vehicles destroyed")
        
        # Clean up walkers
        if self.all_actors and self.all_id and self.client:
            # Stop walker controllers
            for i in range(0, len(self.all_id), 2):
                try:
                    self.all_actors[i].stop()
                except:
                    pass
            
            print(f"\nDestroying {len(self.walkers_list)} pedestrians...")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
            print("✓ Pedestrians destroyed")
        
        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print("CLEANUP COMPLETE - System terminated successfully")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    argparser = argparse.ArgumentParser(description='Integrated Multi-Sensor Fusion System for CARLA')
    
    # Connection settings
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    
    # Traffic settings
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=30,
        type=int,
        help='Number of walkers (default: 30)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    
    # Traffic Manager settings
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    
    # Random seeds
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    
    args = argparser.parse_args()
    
    # Create and run integrated system
    system = IntegratedSensorFusionSystem(args)
    system.run()


if __name__ == '__main__':
    main()