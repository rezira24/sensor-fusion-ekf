"""
main12_enhanced.py - ENHANCED EVALUATION VERSION
Integrates comprehensive evaluation system with existing multi-sensor fusion
Well-organized code structure for better readability and maintenance
"""

import carla
import numpy as np
import cv2
import time
from queue import Empty
from collections import deque
import os
from datetime import datetime

# Import existing modules
from config import *
from utils import get_intrinsic_matrix, process_image, MAPPING_COLORS
from object_detector import (
    euclidean_clustering, project_3d_bounding_boxes, 
    preprocess_lidar, print_preprocessing_stats, 
    draw_lidar_projection, get_object_bboxes_from_segmentation, 
    draw_bboxes_on_image
)
from distance_calculator import check_ground_truth_distance
from segmentation import visualize_semantic_segmentation
from lane_detection import estimate_lane_lines, merge_lane_lines
from sensors import setup_sensors, get_sensor_data, cleanup_sensors

# Import 4D state EKF fusion module (v2.0)
from sensor_fusion import (
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

# ========== EGO VEHICLE MANAGEMENT FUNCTIONS ==========

def spawn_new_ego_vehicle(world):
    """
    Always spawn a new ego vehicle (remove existing hero vehicles first)
    Priority: Position 1 -> Position 2 -> Position 3 -> Random available position
    """
    # First, clean up any existing hero vehicles
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if vehicle.attributes.get('role_name') == 'hero':
            print(f"Removing existing ego vehicle (ID: {vehicle.id})")
            vehicle.destroy()
    
    print("Spawning new ego vehicle...")
    
    # Get blueprint
    blueprint_library = world.get_blueprint_library()
    
    # Try Tesla Model 3 first (primary choice)
    tesla_model3 = blueprint_library.filter('vehicle.tesla.model3')
    if tesla_model3:
        vehicle_bp = tesla_model3[0]
        print("Selected vehicle: Tesla Model 3")
    else:
        # Fallback to Audi A2 (only if Model 3 blueprint not in this CARLA version)
        audi_a2 = blueprint_library.filter('vehicle.audi.a2')
        if audi_a2:
            vehicle_bp = audi_a2[0]
            print("Tesla Model 3 blueprint not found. Using: Audi A2")
        else:
            raise RuntimeError("Neither Tesla Model 3 nor Audi A2 blueprints found!")
    
    # Set as hero vehicle
    vehicle_bp.set_attribute('role_name', 'hero')
    
    # Set a distinct color for easy identification
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', '255,0,0')  # Red color for ego vehicle
    
    # Get all spawn points
    spawn_points = world.get_map().get_spawn_points()
    
    if not spawn_points:
        raise RuntimeError("No spawn points available in the map!")
    
    # Refresh vehicle list after cleanup
    vehicles = world.get_actors().filter("vehicle.*")
    
    # Function to check if a spawn point is occupied
    def is_spawn_point_occupied(spawn_point, margin=5.0):
        """Check if there's a vehicle near the spawn point"""
        spawn_location = spawn_point.location
        
        for vehicle in vehicles:
            vehicle_location = vehicle.get_location()
            distance = spawn_location.distance(vehicle_location)
            
            if distance < margin:
                return True
        return False
    
    # Try spawn points in order of preference
    vehicle = None
    spawn_attempts = [
        ("Position 1", 0),
        ("Position 2", 1), 
        ("Position 3", 2)
    ]
    
    # Try preferred positions first
    for position_name, index in spawn_attempts:
        if index < len(spawn_points):
            spawn_point = spawn_points[index]
            
            if not is_spawn_point_occupied(spawn_point):
                try:
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    print(f"✓ Spawned ego vehicle at {position_name} (index {index})")
                    print(f"  Location: x={spawn_point.location.x:.1f}, y={spawn_point.location.y:.1f}")
                    print(f"  Vehicle: {vehicle.type_id} (ID: {vehicle.id})")
                    return vehicle
                except Exception as e:
                    print(f"✗ Failed to spawn at {position_name}: {e}")
            else:
                print(f"✗ {position_name} is occupied")
    
    # If all preferred positions are occupied, try random unoccupied positions
    print("All preferred positions occupied. Trying random positions...")
    
    # Shuffle spawn points for random selection
    import random
    available_spawn_points = [sp for sp in spawn_points if not is_spawn_point_occupied(sp)]
    
    if available_spawn_points:
        random.shuffle(available_spawn_points)
        
        for spawn_point in available_spawn_points[:5]:  # Try up to 5 random positions
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                spawn_index = spawn_points.index(spawn_point)
                print(f"✓ Spawned ego vehicle at random position (index {spawn_index})")
                print(f"  Location: x={spawn_point.location.x:.1f}, y={spawn_point.location.y:.1f}")
                print(f"  Vehicle: {vehicle.type_id} (ID: {vehicle.id})")
                return vehicle
            except Exception as e:
                print(f"Failed to spawn at random position: {e}")
    
    # Last resort: force spawn at first position
    print("WARNING: All spawn points seem occupied. Force spawning at Position 1...")
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"⚠️  Force spawned ego vehicle at Position 1")
        print(f"  Vehicle: {vehicle.type_id} (ID: {vehicle.id})")
        return vehicle
    except Exception as e:
        raise RuntimeError(f"Failed to spawn ego vehicle anywhere: {e}")


def cleanup_ego_vehicle(world):
    """
    Clean up ego vehicle when exiting
    """
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if vehicle.attributes.get('role_name') == 'hero':
            print(f"Destroying ego vehicle (ID: {vehicle.id})")
            vehicle.destroy()
            return True
    return False

# ==================== VISUALIZATION CLASSES ====================

class EnhancedBEVVisualizerWithFusion(MultiSensorBEVVisualizer):
    """Enhanced BEV visualizer that shows fusion events"""
    
    def __init__(self, bev_range=50, image_size=600):
        super().__init__(bev_range, image_size)
        self.fusion_events = {}
        self.current_frame = 0
        
    def update_fusion_events(self, tracks, current_frame):
        """Track when objects get fused"""
        self.current_frame = current_frame
        
        for track in tracks:
            track_id = track['id']
            label = track.get('label')
            
            if label is not None and track_id not in self.fusion_events:
                ekf = track.get('ekf')
                if ekf and len(ekf.sensor_history) > 1:
                    self.fusion_events[track_id] = current_frame

    def draw_bev(self, tracks, raw_lidar_points=None):
        """Override to add fusion visualization and sensor range indicators"""
        # Create base BEV image
        bev_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Draw grid first
        self._draw_grid(bev_img)
        
        # Draw sensor ranges (before other elements for transparency effect)
        self._draw_sensor_ranges(bev_img)
        
        # Draw ego vehicle
        self._draw_ego_vehicle(bev_img)
        
        # Draw LiDAR points if provided
        if raw_lidar_points is not None:
            self._draw_lidar_points(bev_img, raw_lidar_points)
        
        # Draw tracks
        for track in tracks:
            self._draw_track_with_velocity(bev_img, track)
        
        # Draw fusion indicators
        center_x, center_y = self.image_size // 2, self.image_size // 2
        
        for track in tracks:
            track_id = track['id']
            
            if track_id in self.fusion_events:
                frames_since_fusion = self.current_frame - self.fusion_events[track_id]
                
                if frames_since_fusion < 10:
                    pos = track['position']
                    px = int(center_x + pos[1] * self.scale)
                    py = int(center_y - pos[0] * self.scale)
                    
                    if 0 <= px < self.image_size and 0 <= py < self.image_size:
                        radius = int(20 + 10 * np.sin(frames_since_fusion * 0.5))
                        opacity = int(255 * (1 - frames_since_fusion / 10))
                        
                        overlay = bev_img.copy()
                        cv2.circle(overlay, (px, py), radius, (255, 255, 0), 3)
                        cv2.addWeighted(overlay, opacity/255, bev_img, 1-opacity/255, 0, bev_img)
                        
                        if frames_since_fusion < 5:
                            cv2.putText(bev_img, "FUSED!", (px - 25, py - radius - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                elif frames_since_fusion > 20:
                    del self.fusion_events[track_id]
        
        # Draw legend
        self._draw_legend(bev_img)
        
        return bev_img

    def _draw_sensor_ranges(self, img):
        """Draw LiDAR range circle and camera FOV cone"""
        center_x, center_y = self.image_size // 2, self.image_size // 2
        
        # Create overlay for transparency
        overlay = img.copy()
        
        # 1. Draw LiDAR range circle
        lidar_range_meters = 100  # From sensors.py: lidar_bp.set_attribute('range', '100')
        lidar_range_pixels = int(lidar_range_meters * self.scale)
        
        # Draw filled circle with transparency
        cv2.circle(overlay, (center_x, center_y), lidar_range_pixels, 
                (80, 80, 80), -1)  # Gray color
        
        # Add the overlay with transparency (alpha = 0.1 for subtle effect)
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        
        # Draw circle outline
        cv2.circle(img, (center_x, center_y), lidar_range_pixels, 
                (100, 100, 100), 2)  # Slightly brighter gray for outline
        
        # 2. Draw Camera FOV cone
        camera_fov_degrees = 90  # From config1.py
        camera_range_meters = 60  # Reasonable camera range
        camera_range_pixels = int(camera_range_meters * self.scale)
        
        # Calculate cone points
        # Camera looks forward (along negative Y in BEV, which is up in image)
        # Convert FOV to radians and calculate half angle
        half_fov_rad = np.radians(camera_fov_degrees / 2)
        
        # Cone vertices (ego vehicle position + two corners of FOV)
        # In BEV: forward is negative Y (up in image), right is positive X (right in image)
        cone_points = np.array([
            [center_x, center_y],  # Ego vehicle position
            [center_x + int(camera_range_pixels * np.sin(half_fov_rad)), 
            center_y - int(camera_range_pixels * np.cos(half_fov_rad))],  # Right corner
            [center_x - int(camera_range_pixels * np.sin(half_fov_rad)), 
            center_y - int(camera_range_pixels * np.cos(half_fov_rad))]   # Left corner
        ], np.int32)
        
        # Draw filled cone with transparency
        overlay2 = img.copy()
        cv2.fillPoly(overlay2, [cone_points], (255, 200, 100))  # Light blue
        cv2.addWeighted(overlay2, 0.15, img, 0.85, 0, img)
        
        # Draw cone outline
        cv2.polylines(img, [cone_points], True, (255, 220, 150), 2)
        
        # Draw FOV angle arc
        start_angle = 270 - camera_fov_degrees/2  # 270 is up in OpenCV
        end_angle = 270 + camera_fov_degrees/2
        cv2.ellipse(img, (center_x, center_y), 
                    (camera_range_pixels, camera_range_pixels),
                    0, start_angle, end_angle, (255, 220, 150), 2)
        
        # Add labels
        # LiDAR label
        lidar_label_y = center_y + lidar_range_pixels + 15
        if lidar_label_y < self.image_size - 20:
            cv2.putText(img, f"LiDAR: {lidar_range_meters}m", 
                        (center_x - 40, lidar_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # Camera label
        camera_label_y = center_y - camera_range_pixels - 5
        if camera_label_y > 20:
            cv2.putText(img, f"Camera FOV: {camera_fov_degrees}°", 
                        (center_x - 50, camera_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 150), 1)

    # Also add this to the __init__ method of EnhancedBEVVisualizerWithFusion if needed:
    # (This assumes the class already inherits the necessary attributes from parent)
    # self.lidar_range = 100  # meters
    # self.camera_fov = 90    # degrees
    # self.camera_range = 60  # meters


# ==================== HELPER FUNCTIONS ====================

def preprocess_camera_segmentation(semantic_raw):
    """Apply morphological operations to clean up segmentation"""
    vehicle_mask = (semantic_raw == 10).astype(np.uint8) * 255
    pedestrian_mask = (semantic_raw == 4).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, kernel)
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_OPEN, kernel)
    
    pedestrian_mask = cv2.morphologyEx(pedestrian_mask, cv2.MORPH_CLOSE, kernel)
    pedestrian_mask = cv2.morphologyEx(pedestrian_mask, cv2.MORPH_OPEN, kernel)
    
    cleaned_seg = semantic_raw.copy()
    cleaned_seg[vehicle_mask > 0] = 10
    cleaned_seg[pedestrian_mask > 0] = 4
    
    return cleaned_seg


def get_sensor_data_with_timestamps(image_queue, semantic_queue, lidar_queue, timeout=1.0):
    """Get sensor data with timestamps"""
    try:
        image_data = image_queue.get(timeout=timeout)
        semantic_data = semantic_queue.get(timeout=timeout)
        lidar_data = lidar_queue.get(timeout=timeout)
        
        current_time = time.time()
        
        image_timestamp = current_time
        semantic_timestamp = current_time
        lidar_timestamp = current_time
        
        if hasattr(image_data, 'frame'):
            frame_number = image_data.frame
            image_timestamp = frame_number * 0.05
            semantic_timestamp = frame_number * 0.05
            lidar_timestamp = frame_number * 0.05
        
        return (image_data, semantic_data, lidar_data, 
                image_timestamp, semantic_timestamp, lidar_timestamp)
    except Empty:
        return None, None, None, None, None, None


def enhance_multi_sensor_tracker_with_tiered_lifetime(tracker):
    """Enhance the tracker to support two-tier lifetime management"""
    if not hasattr(tracker, 'original_max_age'):
        tracker.original_max_age = tracker.max_age
        tracker.probationary_max_age = 5
    
    original_update = tracker.update
    
    def update_with_tiered_lifetime(lidar_detections, camera_detections, lidar_transform, 
                                   camera_transform, timestamp):
        tracks = original_update(lidar_detections, camera_detections, lidar_transform, 
                               camera_transform, timestamp)
        
        for track_id, ekf in tracker.tracks.items():
            if ekf.label is not None:
                ekf.max_age = tracker.original_max_age
                if not hasattr(ekf, 'fusion_time') and hasattr(ekf, 'creation_timestamp'):
                    ekf.fusion_time = timestamp - ekf.creation_timestamp
            else:
                ekf.max_age = tracker.probationary_max_age
        
        return tracks
    
    tracker.update = update_with_tiered_lifetime
    return tracker


def create_info_panel(tracks, metrics, fps, avg_fps, cpu_usage, frame_count, start_time):
    """Create information panel for BEV visualization"""
    panel = np.zeros((350, 600, 3), dtype=np.uint8)
    
    # Track statistics
    total_tracks = len(tracks)
    labeled_tracks = len([t for t in tracks if t.get('label') is not None])
    unlabeled_tracks = total_tracks - labeled_tracks
    fused_tracks = sum(1 for t in tracks if t.get('sensor_mode') == 'fused')
    vehicles = sum(1 for t in tracks if t.get('label') in [10, 'Vehicle'])
    pedestrians = sum(1 for t in tracks if t.get('label') in [4, 'Pedestrian'])
    
    # Display info
    y_offset = 20
    cv2.putText(panel, "=== ENHANCED EVALUATION SYSTEM ===", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 35
    cv2.putText(panel, f"Total Tracks: {total_tracks} (Labeled: {labeled_tracks}, Unlabeled: {unlabeled_tracks})",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 20
    cv2.putText(panel, f"Vehicles: {vehicles} | Pedestrians: {pedestrians} | Fused: {fused_tracks}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Enhanced evaluation metrics
    y_offset += 35
    cv2.putText(panel, "=== REAL-TIME METRICS (Smoothed) ===", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Metric 1: Camera IoU
    y_offset += 30
    cam_color = (0, 255, 0) if metrics['camera_iou'] > 0.7 else (255, 255, 0) if metrics['camera_iou'] > 0.5 else (0, 0, 255)
    cv2.putText(panel, f"1. Camera Semantic IoU: {metrics['camera_iou']:.3f}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam_color, 2)
    
    # Metric 2: LiDAR MAE
    y_offset += 30
    lidar_color = (0, 255, 0) if metrics['lidar_mae'] < 0.5 else (255, 255, 0) if metrics['lidar_mae'] < 1.0 else (0, 0, 255)
    cv2.putText(panel, f"2. LiDAR MAE: {metrics['lidar_mae']:.2f} m", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, lidar_color, 2)
    
    # Metric 3: EKF MAE
    y_offset += 30
    if metrics['ekf_mae'] == float('inf'):
        cv2.putText(panel, f"3. EKF MAE: No Data", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    else:
        ekf_color = (0, 255, 0) if metrics['ekf_mae'] < 0.3 else (255, 255, 0) if metrics['ekf_mae'] < 0.6 else (0, 0, 255)
        cv2.putText(panel, f"3. EKF MAE: {metrics['ekf_mae']:.2f} m", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ekf_color, 2)
    
    # Performance
    y_offset += 35
    cv2.putText(panel, f"Performance: {fps:.1f} FPS (Avg: {avg_fps:.1f} FPS)", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if fps >= 20 else (255, 255, 0), 1)
    
    # CPU usage
    y_offset += 20
    cv2.putText(panel, f"CPU Usage: {cpu_usage:.1f}%", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if cpu_usage < 70 else (255, 255, 0), 1)
    
    # Status
    y_offset += 25
    cv2.putText(panel, f"Frame: {frame_count} | Runtime: {time.time() - start_time:.1f}s", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    y_offset += 20
    cv2.putText(panel, "Collecting data for final report & plots...", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return panel


def create_performance_panel(timing_summary, avg_fps, cpu_usage, width=400, height=400):
    """Create enhanced performance panel"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.putText(panel, "=== Enhanced Performance Metrics (ms) ===", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset = 50
    
    for name, stats in timing_summary.items():
        text = f"{name}: {stats['mean']:.1f} ± {stats['std']:.1f}"
        
        if stats['mean'] < 10:
            color = (0, 255, 0)
        elif stats['mean'] < 20:
            color = (255, 255, 0)
        else:
            color = (0, 0, 255)
            
        cv2.putText(panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 25
    
    y_offset += 20
    cv2.putText(panel, f"Average FPS: {avg_fps:.1f}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(panel, f"CPU Usage: {cpu_usage:.1f}%", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    y_offset += 30
    avg_total = timing_summary.get('total', {}).get('mean', 0)
    if avg_total < 20:
        classification = "REAL-TIME CAPABLE"
        color = (0, 255, 0)
    elif avg_total < 50:
        classification = "NEAR REAL-TIME"
        color = (255, 255, 0)
    else:
        classification = "OFFLINE PROCESSING"
        color = (0, 0, 255)
        
    cv2.putText(panel, f"System: {classification}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add enhanced features info
    y_offset += 30
    cv2.putText(panel, "=== Enhanced Features ===", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 20
    cv2.putText(panel, "✓ RMSE & Relative Error tracking", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    y_offset += 15
    cv2.putText(panel, "✓ CPU usage monitoring", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    y_offset += 15
    cv2.putText(panel, "✓ Frame-by-frame history", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    y_offset += 15
    cv2.putText(panel, "✓ Automated plot generation", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    y_offset += 15
    cv2.putText(panel, "✓ Comprehensive final report", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return panel


def print_periodic_report(frame_count, evaluator, fps, avg_fps, cpu_usage):
    """Print periodic evaluation report to console - FIXED to handle inf values"""
    print(f"\n{'='*60}")
    print(f"=== ENHANCED EVALUATION REPORT (Frame {frame_count}) ===")
    print(f"{'='*60}")
    
    # Get latest per-frame metrics with inf handling
    latest_lidar_mae = evaluator.lidar_mae_per_frame[-1] if evaluator.lidar_mae_per_frame else 0
    latest_lidar_rmse = evaluator.lidar_rmse_per_frame[-1] if evaluator.lidar_rmse_per_frame else 0
    latest_lidar_rel = evaluator.lidar_relative_error_per_frame[-1] if evaluator.lidar_relative_error_per_frame else 0
    
    # FIXED: Handle inf values for EKF
    latest_ekf_mae = evaluator.ekf_mae_per_frame[-1] if evaluator.ekf_mae_per_frame else float('inf')
    latest_ekf_rmse = evaluator.ekf_rmse_per_frame[-1] if evaluator.ekf_rmse_per_frame else float('inf')
    latest_ekf_rel = evaluator.ekf_relative_error_per_frame[-1] if evaluator.ekf_relative_error_per_frame else float('inf')
    
    current_metrics = evaluator.get_metrics()
    
    print(f"\n[Current Frame Metrics]")
    print(f"Camera IoU: {current_metrics['camera_iou']:.3f}")
    print(f"LiDAR: MAE={latest_lidar_mae:.3f}m, RMSE={latest_lidar_rmse:.3f}m, RelErr={latest_lidar_rel:.1f}%")
    
    # FIXED: Special formatting for EKF inf values
    if np.isinf(latest_ekf_mae) or np.isnan(latest_ekf_mae):
        print(f"EKF: MAE=No Data, RMSE=No Data, RelErr=No Data")
    else:
        print(f"EKF: MAE={latest_ekf_mae:.3f}m, RMSE={latest_ekf_rmse:.3f}m, RelErr={latest_ekf_rel:.1f}%")
    
    print(f"\n[System Performance]")
    print(f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})")
    print(f"CPU Usage: {cpu_usage:.1f}%")
    print(f"{'='*60}\n")


# ==================== SENSOR PROCESSING FUNCTIONS ====================

def process_lidar_data(lidar_data, performance_monitor):
    """Process LiDAR data and return filtered points"""
    lidar_start = time.time()
    
    filtered_points, intensities, raw_points, non_ground_mask = preprocess_lidar(
        lidar_data, GROUND_THRESHOLD, VOXEL_SIZE)
    
    lidar_time = time.time() - lidar_start
    performance_monitor.add_metric('lidar_preprocessing', lidar_time)
    
    return filtered_points, intensities, raw_points, non_ground_mask


def process_camera_data(semantic_data, performance_monitor):
    """Process semantic segmentation data"""
    seg_start = time.time()
    
    semantic_array = np.frombuffer(semantic_data.raw_data, dtype=np.uint8)
    semantic_array = semantic_array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    semantic_raw = semantic_array[:, :, 2]
    
    semantic_cleaned = preprocess_camera_segmentation(semantic_raw)
    
    seg_time = time.time() - seg_start
    performance_monitor.add_metric('segmentation', seg_time)
    
    return semantic_raw, semantic_cleaned


def perform_clustering(filtered_points, performance_monitor):
    """Perform LiDAR clustering"""
    clustering_start = time.time()
    
    clusters = euclidean_clustering(
        filtered_points, CLUSTERING_TOLERANCE, MIN_CLUSTER_POINTS, MAX_CLUSTER_POINTS)
    
    clustering_time = time.time() - clustering_start
    performance_monitor.add_metric('clustering', clustering_time)
    
    return clusters


def perform_sensor_fusion(clusters, filtered_points, bboxes, lidar, camera, 
                         multi_sensor_tracker, current_timestamp, performance_monitor):
    """Perform multi-sensor fusion and tracking"""
    fusion_start = time.time()
    
    # Extract LiDAR detections    
    # Extract LiDAR detections
    lidar_detections = []
    for cluster in clusters:
        if len(cluster) >= 5:
            cluster_points = filtered_points[cluster]
            # UPDATED: Use center of bounding box instead of centroid
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            center = (min_coords + max_coords) / 2  # Center of bounding box
            bbox_3d = np.concatenate([min_coords, max_coords])
            
            detection_info = (center, bbox_3d, len(cluster))
            lidar_detections.append(detection_info)

    # Convert to multi-sensor format
    lidar_dets, camera_dets = convert_to_multi_sensor_format(
        lidar_detections, bboxes, current_timestamp)
    
    # Update tracker
    all_tracks = multi_sensor_tracker.update(
        lidar_dets, camera_dets,
        lidar.get_transform(),
        camera.get_transform(),
        current_timestamp
    )
    
    fusion_time = time.time() - fusion_start
    performance_monitor.add_metric('fusion', fusion_time)
    
    return all_tracks


# ==================== MAIN FUNCTION ====================

def main():
    """Main function with organized structure"""
    
    # ========== INITIALIZATION ==========
    # Connect to CARLA
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CARLA_TIMEOUT)
    world = client.get_world()

    # Get vehicle and setup sensors
    vehicle = spawn_new_ego_vehicle(world)  # Selalu spawn baru

    # Enable autopilot for ego vehicle
    vehicle.set_autopilot(True)
    print("✓ Autopilot enabled for ego vehicle")

    # Simple Traffic Manager config (compatible dengan semua versi)
    try:
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -30)  # 30% slower
        traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)  # 3m distance
        print("✓ Traffic Manager configured")
    except Exception as e:
        print(f"⚠️ Traffic Manager configuration partially failed: {e}")
        print("  Continuing with basic autopilot...")

    # Wait a moment for vehicle to settle
    time.sleep(0.5)

    camera, semantic_cam, lidar, image_queue, semantic_queue, lidar_queue = setup_sensors(
        world, vehicle, IMAGE_WIDTH, IMAGE_HEIGHT, FOV)
    camera_matrix = get_intrinsic_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, FOV)

    # Initialize multi-sensor fusion system
    multi_sensor_tracker = MultiSensorTracker(camera_matrix, max_age=FUSION_MAX_AGE, min_hits=FUSION_MIN_HITS)
    multi_sensor_tracker = enhance_multi_sensor_tracker_with_tiered_lifetime(multi_sensor_tracker)
    
    # Initialize visualization
    bev_visualizer = EnhancedBEVVisualizerWithFusion(bev_range=50, image_size=600)

    # Initialize enhanced evaluation system
    enhanced_evaluator = EnhancedEvaluationMetrics(
        history_size=EVALUATION_HISTORY_SIZE
    )
    enhanced_performance_monitor = EnhancedPerformanceMonitor()

    # Configuration
    skip_lane_detection = False
    downsample_visualization = 1
    
    # Timing variables
    frame_count = 0
    start_time = time.time()
    last_timestamp = 0.0
    first_frame = True
    last_print_time = time.time()

    # ========== MAIN LOOP ==========
    try:
        while True:
            frame_start = time.time()
            
            # ===== SENSOR DATA ACQUISITION =====
            sensor_start = time.time()
            (image_data, semantic_data, lidar_data, 
             image_timestamp, semantic_timestamp, lidar_timestamp) = get_sensor_data_with_timestamps(
                image_queue, semantic_queue, lidar_queue)
                
            if image_data is None or lidar_data is None or semantic_data is None:
                time.sleep(0.01)
                continue

            # Process timestamps
            current_timestamp = max(filter(lambda x: x is not None, 
                                         [image_timestamp, semantic_timestamp, lidar_timestamp]))
            
            if current_timestamp is None:
                current_timestamp = time.time()
            
            if first_frame:
                last_timestamp = current_timestamp
                first_frame = False
                
            im_array = process_image(image_data)
            
            frame_time = time.time() - sensor_start
            enhanced_performance_monitor.add_metric('frame_time', frame_time)

            # ===== SENSOR DATA PROCESSING =====
            # Process LiDAR
            filtered_points, intensities, raw_points, non_ground_mask = process_lidar_data(
                lidar_data, enhanced_performance_monitor, GROUND_THRESHOLD, VOXEL_SIZE)
            
            last_print_time = print_preprocessing_stats(
                raw_points, non_ground_mask, filtered_points, last_print_time)
            
            if filtered_points.shape[0] == 0:
                print("No points after preprocessing.")
                continue

            # Perform clustering
            clusters = perform_clustering(filtered_points, enhanced_performance_monitor, CLUSTERING_TOLERANCE, MIN_CLUSTER_POINTS, MAX_CLUSTER_POINTS)

            # Process camera semantic segmentation
            semantic_raw, semantic_cleaned = process_camera_data(
                semantic_data, enhanced_performance_monitor)

            # Get camera bounding boxes
            bboxes = get_object_bboxes_from_segmentation(semantic_cleaned, target_labels=[4, 10])
            
            # ===== OPTIONAL VISUALIZATIONS =====
            # Lane detection (optional)
            if not skip_lane_detection and frame_count % 5 == 0:
                rgb_lane = im_array.copy()
                lane_lines = estimate_lane_lines(semantic_raw)
                if lane_lines is not None and lane_lines.size > 0:
                    merged_lane_lines = merge_lane_lines(lane_lines)
                    if merged_lane_lines.size > 0:
                        if merged_lane_lines.ndim == 1:
                            merged_lane_lines = merged_lane_lines.reshape(1, 4)
                        for line in merged_lane_lines:
                            x1, y1, x2, y2 = map(int, line)
                            if (0 <= x1 < IMAGE_WIDTH and 0 <= y1 < IMAGE_HEIGHT and
                                0 <= x2 < IMAGE_WIDTH and 0 <= y2 < IMAGE_HEIGHT):
                                cv2.line(rgb_lane, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                if downsample_visualization > 1:
                    rgb_lane = cv2.resize(rgb_lane, 
                                        (IMAGE_WIDTH // downsample_visualization, 
                                         IMAGE_HEIGHT // downsample_visualization))
                cv2.imshow("Lane Detection", rgb_lane)

            # Combined visualization (update every few frames)
            if frame_count % 1 == 0:
                im_combined = im_array.copy()
                im_combined = draw_lidar_projection(
                    filtered_points, intensities, lidar.get_transform(),
                    camera.get_transform(), im_combined, camera_matrix, IMAGE_WIDTH, IMAGE_HEIGHT)

                im_combined = project_3d_bounding_boxes(
                    im_combined, clusters, filtered_points, lidar,
                    camera.get_transform(), camera_matrix, world, IMAGE_WIDTH, IMAGE_HEIGHT)

                bboxes_viz = get_object_bboxes_from_segmentation(semantic_cleaned, target_labels=[4, 10])
                draw_bboxes_on_image(im_combined, bboxes_viz)
                
                if downsample_visualization > 1:
                    im_combined = cv2.resize(im_combined,
                                           (IMAGE_WIDTH // downsample_visualization,
                                            IMAGE_HEIGHT // downsample_visualization))
                cv2.imshow("Combined Object Detection", im_combined)

            # ===== MULTI-SENSOR FUSION =====
            all_tracks = perform_sensor_fusion(
                clusters, filtered_points, bboxes, lidar, camera,
                multi_sensor_tracker, current_timestamp, enhanced_performance_monitor)
            
            # Update fusion events
            bev_visualizer.update_fusion_events(all_tracks, frame_count)

            # ===== ENHANCED EVALUATION =====
            # Update timestamp
            enhanced_evaluator.update_timestamp()
            
            # Metric 1: Camera Semantic IoU
            camera_iou = enhanced_evaluator.evaluate_camera_semantic_iou(
                semantic_cleaned, semantic_data, target_labels=[4, 10])
            
            # Metric 2: LiDAR Distance Error (with RMSE and Relative Error)
            lidar_error = enhanced_evaluator.evaluate_lidar_distance_error(
                clusters, filtered_points, lidar.get_transform(), world, vehicle)
            
            # Metric 3: EKF Position Error (with RMSE and Relative Error)
            ekf_error = enhanced_evaluator.evaluate_ekf_position_error(
                all_tracks, world, vehicle)

            # ===== PERFORMANCE METRICS =====
            # Calculate FPS
            frame_time_total = time.time() - frame_start
            fps = 1.0 / frame_time_total if frame_time_total > 0 else 0
            
            # Record frame performance
            enhanced_performance_monitor.record_frame(fps)
            enhanced_performance_monitor.add_metric('total', frame_time_total)

            # ===== BEV VISUALIZATION WITH INFO PANEL =====
            viz_start = time.time()
            
            # Subsample points for visualization
            viz_points = filtered_points
            if len(filtered_points) > 5000:
                indices = np.random.choice(len(filtered_points), 5000, replace=False)
                viz_points = filtered_points[indices]
            
            # Draw BEV
            bev_image = bev_visualizer.draw_bev(all_tracks, viz_points)
            
            # Create info panel
            current_metrics = enhanced_evaluator.get_metrics()
            avg_fps = enhanced_performance_monitor.get_average_fps()
            cpu_usage = enhanced_performance_monitor.get_average_cpu_usage()
            
            info_panel = create_info_panel(
                all_tracks, current_metrics, fps, avg_fps, cpu_usage, 
                frame_count, start_time)
            
            # Combine panels
            bev_with_info = np.vstack([info_panel, bev_image])
            cv2.imshow("Enhanced Multi-Sensor Fusion Evaluation", bev_with_info)
            
            viz_time = time.time() - viz_start
            enhanced_performance_monitor.add_metric('visualization', viz_time)
            
            # ===== PERFORMANCE MONITORING PANEL =====
            # Update performance panel less frequently
            if frame_count % 10 == 0:
                timing_summary = enhanced_performance_monitor.get_summary()
                perf_panel = create_performance_panel(timing_summary, avg_fps, cpu_usage)
                cv2.imshow("Enhanced Performance Metrics", perf_panel)

            # ===== PERIODIC CONSOLE REPORTING =====
            current_time = time.time()
            if current_time - last_print_time >= 5:
                print_periodic_report(frame_count, enhanced_evaluator, fps, avg_fps, cpu_usage)
                last_print_time = current_time

            # ===== FRAME COUNTER & INPUT HANDLING =====
            frame_count += 1
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('p'):  # Pause
                cv2.waitKey(0)

    # ========== EXCEPTION HANDLING ==========
    except KeyboardInterrupt:
        print("\nTerminated by user")
    except Exception as e:
        print(f"\nError in main loop: {e}")
        import traceback
        traceback.print_exc()
        
    # ========== FINALIZATION ==========
    finally:
        print(f"\n{'='*70}")
        print("=== GENERATING FINAL EVALUATION REPORT AND PLOTS ===")
        print(f"{'='*70}")
        
        # Generate comprehensive evaluation plots and report
        try:
            report_path = generate_final_report_and_plots(
                enhanced_evaluator, enhanced_performance_monitor)
            print(f"\n✅ Evaluation complete! Report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
        
        # Clean up resources
        cleanup_sensors(camera, semantic_cam, lidar)
        
        # Clean up ego vehicle
        try:
            if cleanup_ego_vehicle(world):
                print("✓ Ego vehicle cleaned up")
        except Exception as e:
            print(f"Error cleaning up ego vehicle: {e}")
        
        cv2.destroyAllWindows()
        print("\nCleanup complete.")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    main()