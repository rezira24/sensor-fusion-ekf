"""
enhanced_evaluation_system.py - Comprehensive Evaluation and Visualization System
For CARLA Multi-Sensor Fusion with EKF
FIXED: Complete reporting of ALL timing components
"""

import carla 
import numpy as np
import cv2
import time
import os
from datetime import datetime
from collections import deque
import psutil
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from config import (
    EVALUATION_DISTANCE_THRESHOLD,
    EVALUATION_HISTORY_SIZE
)

# Use seaborn style for professional looking plots
try:
    mplstyle.use('seaborn-v0_8-darkgrid')
except:
    mplstyle.use('seaborn-darkgrid')  # Fallback for older versions


class EnhancedEvaluationMetrics:
    """
    Enhanced evaluation system with comprehensive metrics
    Extends SimpleEvaluationMetrics with RMSE, Relative Error, and detailed history tracking
    """
    
    def __init__(self, history_size=None):

        if history_size is None:
            history_size = EVALUATION_HISTORY_SIZE

        # Basic metrics history (smoothed)
        self.camera_iou_history = deque(maxlen=history_size)
        self.lidar_error_history = deque(maxlen=history_size)
        self.ekf_error_history = deque(maxlen=history_size)
        
        # Frame-by-frame history for visualization (no limit)
        self.camera_iou_per_frame = []
        self.lidar_mae_per_frame = []
        self.lidar_rmse_per_frame = []
        self.lidar_relative_error_per_frame = []
        self.ekf_mae_per_frame = []
        self.ekf_rmse_per_frame = []
        self.ekf_relative_error_per_frame = []
        self.timestamps = []
        
        # Aggregated errors for RMSE calculation
        self.lidar_errors_squared = []
        self.ekf_errors_squared = []
        self.lidar_relative_errors = []
        self.ekf_relative_errors = []
        
        # Thresholds
        self.iou_threshold = 0.5
        self.distance_threshold = EVALUATION_DISTANCE_THRESHOLD  # 2.5 meters
        
        # Frame counter and timing
        self.frame_count = 0
        self.start_time = time.time()
        
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def get_all_ground_truth_positions(self, world, vehicle, max_distance=50.0):
        """Get all ground truth object positions"""
        ego_transform = vehicle.get_transform()
        ego_location = ego_transform.location
        ego_pos = np.array([ego_location.x, ego_location.y, ego_location.z])
        
        vehicles = world.get_actors().filter('vehicle.*')
        walkers = world.get_actors().filter('walker.pedestrian.*')
        
        ground_truths = []
        
        for actor in list(vehicles) + list(walkers):
            if actor.id == vehicle.id:
                continue
            
            # Determine label
            is_pedestrian = 'walker' in actor.type_id.lower() or 'pedestrian' in actor.type_id.lower()
            label = 4 if is_pedestrian else 10
            
            # Get world position
            actor_loc = actor.get_transform().location
            actor_pos = np.array([actor_loc.x, actor_loc.y, actor_loc.z])
            
            # Calculate distance
            distance = np.linalg.norm(actor_pos[:2] - ego_pos[:2])
            
            if distance <= max_distance:
                ground_truths.append({
                    'position': actor_pos,
                    'label': label,
                    'id': actor.id,
                    'distance': distance  # Store ground truth distance
                })
        
        return ground_truths
    
    def evaluate_camera_semantic_iou(self, predicted_seg, semantic_data, target_labels=[4, 10]):
        """
        Metric 1: Pixel-wise IoU for semantic segmentation
        """
        # Get ground truth segmentation from semantic camera
        gt_array = np.frombuffer(semantic_data.raw_data, dtype=np.uint8)
        gt_array = gt_array.reshape((predicted_seg.shape[0], predicted_seg.shape[1], 4))
        gt_seg = gt_array[:, :, 2]  # Red channel contains labels
        
        total_iou = 0.0
        valid_classes = 0
        
        # Calculate IoU for each target class
        for label in target_labels:
            # Create binary masks
            pred_mask = (predicted_seg == label)
            gt_mask = (gt_seg == label)
            
            # Calculate intersection and union
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            # Calculate IoU
            if union > 0:
                iou = intersection / union
                total_iou += iou
                valid_classes += 1
        
        # Calculate mean IoU
        mean_iou = total_iou / valid_classes if valid_classes > 0 else 0.0
        
        # Add to history
        self.camera_iou_history.append(mean_iou)
        self.camera_iou_per_frame.append(mean_iou)
        
        # Return smoothed average
        return np.mean(self.camera_iou_history) if self.camera_iou_history else 0.0
    
    def evaluate_lidar_distance_error(self, clusters, filtered_points, lidar_transform, world, vehicle):
        """
        Metric 2: Enhanced distance error for LiDAR detections with RMSE and Relative Error
        FIXED: Skip evaluation when no data available instead of appending 0.0
        """
        # Get ground truth positions
        ground_truths = self.get_all_ground_truth_positions(world, vehicle)
        
        # FIXED: Skip evaluation if no data to evaluate
        if not clusters or not ground_truths:
            # Don't append anything - skip this frame completely
            # Return previous smoothed value for real-time display
            if self.lidar_error_history:
                return np.mean(self.lidar_error_history)
            else:
                return 0.0  # Only for very first frame
        
        # Transform matrices
        lidar_world = lidar_transform.get_matrix()
        
        errors = []
        squared_errors = []
        relative_errors = []
        
        for cluster in clusters:
            if len(cluster) < 5:  # Skip very small clusters
                continue
            
            # Get cluster center (bounding box center) in LiDAR space
            cluster_points = filtered_points[cluster]
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            center_lidar = (min_coords + max_coords) / 2

            # Transform to world space
            center_hom = np.array([center_lidar[0], center_lidar[1], 
                                center_lidar[2] if len(center_lidar) > 2 else 0, 1])
            center_world = lidar_world @ center_hom
            center_pos = center_world[:3]

            # Determine cluster label based on size (for label matching)
            width = abs(max_coords[0] - min_coords[0])
            length = abs(max_coords[1] - min_coords[1])
            height = abs(max_coords[2] - min_coords[2]) if len(max_coords) > 2 else 0

            # Simple heuristic: small objects are likely pedestrians
            if width < 1.0 and length < 1.0 and height < 2.5:
                estimated_label = 4  # Pedestrian
            else:
                estimated_label = 10  # Vehicle

            # Find closest GT object WITH SAME LABEL
            min_distance = float('inf')
            closest_gt_distance = None

            for gt in ground_truths:
                # Check label compatibility
                if gt['label'] != estimated_label:
                    continue
                    
                gt_pos = gt['position']
                distance = np.linalg.norm(center_pos[:2] - gt_pos[:2])
                
                if distance < min_distance:
                    min_distance = distance
                    closest_gt_distance = gt['distance']
            
            # Count as match if within threshold
            if min_distance < self.distance_threshold and closest_gt_distance is not None:
                errors.append(min_distance)
                squared_errors.append(min_distance ** 2)
                # Absolute relative error
                rel_error = abs(min_distance) / closest_gt_distance * 100
                relative_errors.append(rel_error)
        
        # FIXED: Skip evaluation if no matches found
        if not errors:
            # No matches found - skip this frame completely
            # Return previous smoothed value for real-time display
            if self.lidar_error_history:
                return np.mean(self.lidar_error_history)
            else:
                return 0.0  # Only for very first frame
        
        # Calculate metrics only if we have valid matches
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mean_relative_error = np.mean(relative_errors)
        
        # Store frame-by-frame data (only when we have actual data)
        self.lidar_mae_per_frame.append(mae)
        self.lidar_rmse_per_frame.append(rmse)
        self.lidar_relative_error_per_frame.append(mean_relative_error)
        
        # Add to smoothed history
        self.lidar_error_history.append(mae)
        
        # Store for aggregated calculations
        self.lidar_errors_squared.extend(squared_errors)
        self.lidar_relative_errors.extend(relative_errors)
        
        # Return smoothed MAE
        return np.mean(self.lidar_error_history)
    
    def evaluate_ekf_position_error(self, tracks, world, vehicle):
        """
        Metric 3: Enhanced position error for EKF fusion tracks with RMSE and Relative Error
        FIXED: Skip evaluation when no data available instead of appending previous values
        """
        # Get ground truth positions
        ground_truths = self.get_all_ground_truth_positions(world, vehicle)
        
        # Get ego vehicle transform
        ego_transform = vehicle.get_transform()
        
        # Filter labeled tracks
        labeled_tracks = [t for t in tracks if t.get('label') is not None]
        
        # FIXED: Skip evaluation if no data to evaluate
        if not labeled_tracks or not ground_truths:
            # Don't append anything - skip this frame completely
            # Return previous smoothed value for real-time display
            if self.ekf_error_history:
                return np.mean(self.ekf_error_history)
            else:
                return float('inf')  # Indicate no data available
        
        errors = []
        squared_errors = []
        relative_errors = []
        
        for track in labeled_tracks:
            # Track position is in vehicle-relative coordinates [forward, lateral]
            track_pos_rel = track['position']
            
            # Create a relative location (x=forward, y=lateral in CARLA's right-handed system)
            relative_location = carla.Location(
                x=track_pos_rel[0],  # Forward
                y=track_pos_rel[1],  # Lateral (right positive)
                z=0.0  # Ground level
            )
            
            # Transform to world coordinates using CARLA's built-in method
            world_location = ego_transform.transform(relative_location)
            track_pos_world = np.array([world_location.x, world_location.y])
            
            # Get track label
            track_label = track['label']
            
            # Find closest GT object with same label
            min_distance = float('inf')
            closest_gt_distance = None
            
            for gt in ground_truths:
                if gt['label'] == track_label:
                    gt_pos = gt['position']
                    distance = np.linalg.norm(track_pos_world - gt_pos[:2])
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_gt_distance = gt['distance']
            
            # Count as match if within threshold
            if min_distance < self.distance_threshold and closest_gt_distance is not None:
                errors.append(min_distance)
                squared_errors.append(min_distance ** 2)
                # Absolute relative error
                rel_error = abs(min_distance) / closest_gt_distance * 100
                relative_errors.append(rel_error)
        
        # FIXED: Skip evaluation if no matches found
        if not errors:
            # No matches found - skip this frame completely
            # Return previous smoothed value for real-time display
            if self.ekf_error_history:
                return np.mean(self.ekf_error_history)
            else:
                return float('inf')  # Indicate no data available
        
        # Calculate metrics only if we have valid matches
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))
        mean_relative_error = np.mean(relative_errors)
        
        # Store frame-by-frame data (only when we have actual data)
        self.ekf_mae_per_frame.append(mae)
        self.ekf_rmse_per_frame.append(rmse)
        self.ekf_relative_error_per_frame.append(mean_relative_error)
        
        # Add to smoothed history
        self.ekf_error_history.append(mae)
        
        # Store for aggregated calculations
        self.ekf_errors_squared.extend(squared_errors)
        self.ekf_relative_errors.extend(relative_errors)
        
        # Return smoothed MAE
        return np.mean(self.ekf_error_history)
    
    def update_timestamp(self):
        """Update timestamp for current frame"""
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        self.frame_count += 1
    
    def get_metrics(self):
        """Get current smoothed metrics - FIXED to handle empty evaluation lists"""
        # Handle case where some metrics might be empty due to skipped frames
        camera_iou = np.mean(self.camera_iou_history) if self.camera_iou_history else 0.0
        lidar_mae = np.mean(self.lidar_error_history) if self.lidar_error_history else 0.0
        
        # For EKF, return inf if no data available
        if self.ekf_error_history:
            ekf_mae = np.mean(self.ekf_error_history)
        else:
            ekf_mae = float('inf')  # No EKF data available
        
        return {
            'camera_iou': camera_iou,
            'lidar_mae': lidar_mae,
            'ekf_mae': ekf_mae
        }
    

    def get_final_metrics(self):
        """Get comprehensive final metrics for report - FIXED to handle skipped frames"""
        
        # Debug prints
        print(f"\n=== DEBUG FINAL METRICS ===")
        print(f"EKF MAE per frame count: {len(self.ekf_mae_per_frame)}")
        print(f"EKF RMSE per frame count: {len(self.ekf_rmse_per_frame)}")
        print(f"EKF errors squared count: {len(self.ekf_errors_squared)}")
        
        if self.ekf_mae_per_frame:
            print(f"Last 5 MAE per frame: {self.ekf_mae_per_frame[-5:]}")
        if self.ekf_rmse_per_frame:
            print(f"Last 5 RMSE per frame: {self.ekf_rmse_per_frame[-5:]}")
        if self.ekf_errors_squared:
            print(f"Last 5 squared errors: {self.ekf_errors_squared[-5:]}")
            
        # Calculate different ways
        mae_from_per_frame = np.mean(self.ekf_mae_per_frame) if self.ekf_mae_per_frame else 0.0
        rmse_from_per_frame = np.mean(self.ekf_rmse_per_frame) if self.ekf_rmse_per_frame else 0.0
        rmse_from_raw_errors = np.sqrt(np.mean(self.ekf_errors_squared)) if self.ekf_errors_squared else 0.0
        
        print(f"MAE from per-frame: {mae_from_per_frame:.3f}")
        print(f"RMSE from per-frame: {rmse_from_per_frame:.3f}")
        print(f"RMSE from raw errors: {rmse_from_raw_errors:.3f}")
        print(f"=========================\n")        
        
        # Calculate overall RMSE only if we have data
        lidar_rmse = 0.0
        if self.lidar_errors_squared:
            lidar_rmse = np.sqrt(np.mean(self.lidar_errors_squared))
        
        ekf_rmse = 0.0
        if self.ekf_errors_squared:
            ekf_rmse = np.sqrt(np.mean(self.ekf_errors_squared))
        
        # Calculate overall relative errors only if we have data
        lidar_rel_error = 0.0
        if self.lidar_relative_errors:
            lidar_rel_error = np.mean(self.lidar_relative_errors)
        
        ekf_rel_error = 0.0
        if self.ekf_relative_errors:
            ekf_rel_error = np.mean(self.ekf_relative_errors)
        
        return {
            # Camera metrics
            'camera_iou': np.mean(self.camera_iou_per_frame) if self.camera_iou_per_frame else 0.0,
            
            # LiDAR metrics - ALL from per-frame data for consistency
            'lidar_mae': np.mean(self.lidar_mae_per_frame) if self.lidar_mae_per_frame else 0.0,
            'lidar_rmse': np.mean(self.lidar_rmse_per_frame) if self.lidar_rmse_per_frame else 0.0,  
            'lidar_relative_error': np.mean(self.lidar_relative_error_per_frame) if self.lidar_relative_error_per_frame else 0.0,  
            
            # EKF metrics - ALL from per-frame data for consistency  
            'ekf_mae': np.mean(self.ekf_mae_per_frame) if self.ekf_mae_per_frame else 0.0,
            'ekf_rmse': np.mean(self.ekf_rmse_per_frame) if self.ekf_rmse_per_frame else 0.0, 
            'ekf_relative_error': np.mean(self.ekf_relative_error_per_frame) if self.ekf_relative_error_per_frame else 0.0  
        }
    
    def generate_report(self):
        """Generate evaluation report string - FIXED to handle inf values"""
        final_metrics = self.get_final_metrics()
        
        report = []
        report.append("=== EVALUATION METRICS SUMMARY ===")
        report.append(f"\nCamera Semantic Segmentation:")
        report.append(f"  - Average IoU: {final_metrics['camera_iou']:.3f}")
        
        report.append(f"\nLiDAR Distance Accuracy:")
        report.append(f"  - MAE: {final_metrics['lidar_mae']:.3f} m")
        report.append(f"  - RMSE: {final_metrics['lidar_rmse']:.3f} m")
        report.append(f"  - Relative Error: {final_metrics['lidar_relative_error']:.1f}%")
        
        report.append(f"\nEKF Fusion Accuracy:")
        # FIXED: Handle inf values
        if np.isinf(final_metrics['ekf_mae']) or np.isnan(final_metrics['ekf_mae']):
            report.append(f"  - MAE: No Data Available")
            report.append(f"  - RMSE: No Data Available") 
            report.append(f"  - Relative Error: No Data Available")
            report.append(f"  - Note: No successful track-to-ground-truth matches found")
        else:
            report.append(f"  - MAE: {final_metrics['ekf_mae']:.3f} m")
            report.append(f"  - RMSE: {final_metrics['ekf_rmse']:.3f} m")
            report.append(f"  - Relative Error: {final_metrics['ekf_relative_error']:.1f}%")
        
        report.append(f"\nTotal Frames Processed: {self.frame_count}")
        report.append(f"Total Runtime: {time.time() - self.start_time:.1f} seconds")
        
        return '\n'.join(report)


class EnhancedPerformanceMonitor:
    """
    Enhanced performance monitoring with CPU usage tracking
    """
    
    def __init__(self, window_size=100):
        self.metrics = {
            'frame_time': deque(maxlen=window_size),
            'lidar_preprocessing': deque(maxlen=window_size),
            'clustering': deque(maxlen=window_size),
            'segmentation': deque(maxlen=window_size),
            'fusion': deque(maxlen=window_size),
            'visualization': deque(maxlen=window_size),
            'total': deque(maxlen=window_size)
        }
        
        # FPS tracking
        self.fps_history = []
        self.timestamps = []
        
        # CPU usage tracking
        self.cpu_usage_history = []
        self.process = psutil.Process(os.getpid())
        
        self.start_time = time.time()
        self.frame_count = 0
        
    def add_metric(self, metric_name, value):
        """Add a timing metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def record_frame(self, fps):
        """Record FPS and CPU usage for current frame"""
        self.frame_count += 1
        current_time = time.time() - self.start_time
        
        # Record FPS
        self.fps_history.append(fps)
        self.timestamps.append(current_time)
        
        # Record CPU usage (non-blocking)
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_usage_history.append(cpu_percent)
        except:
            self.cpu_usage_history.append(0.0)
    
    def get_summary(self):
        """Get timing summary statistics - FIXED to include ALL components"""
        summary = {}
        # Process ALL metrics, not just some
        for name, values in self.metrics.items():
            if values and len(values) > 0:  # Ensure we have data
                summary[name] = {
                    'mean': np.mean(values) * 1000,  # Convert to ms
                    'std': np.std(values) * 1000,
                    'max': np.max(values) * 1000,
                    'min': np.min(values) * 1000
                }
        return summary
    
    def get_average_cpu_usage(self):
        """Get average CPU usage"""
        if self.cpu_usage_history:
            # Filter out initial readings which might be 0
            valid_readings = [cpu for cpu in self.cpu_usage_history if cpu > 0]
            return np.mean(valid_readings) if valid_readings else 0.0
        return 0.0
    
    def get_average_fps(self):
        """Get average FPS"""
        return np.mean(self.fps_history) if self.fps_history else 0.0


def generate_final_report_and_plots(evaluator, performance_monitor, output_dir='evaluation_results'):
    """
    Generate comprehensive evaluation plots and final report
    FIXED: Now includes ALL timing components in both text report and pie chart
    FIXED: Handle empty datasets and inf values gracefully
    
    Args:
        evaluator: EnhancedEvaluationMetrics instance
        performance_monitor: EnhancedPerformanceMonitor instance
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get final metrics
    final_metrics = evaluator.get_final_metrics()
    timing_summary = performance_monitor.get_summary()
    
    print(f"\n{'='*60}")
    print("Generating evaluation plots and report...")
    print(f"{'='*60}")
    
    # 1. EKF Error vs Time Plot - FIXED
    plt.figure(figsize=(12, 6))
    
    # Check if we have EKF data
    if evaluator.ekf_mae_per_frame and evaluator.timestamps:
        # Filter out inf values for plotting
        ekf_timestamps = []
        ekf_mae_clean = []
        ekf_rmse_clean = []
        
        for i, (ts, mae, rmse) in enumerate(zip(evaluator.timestamps, 
                                               evaluator.ekf_mae_per_frame, 
                                               evaluator.ekf_rmse_per_frame)):
            if not (np.isinf(mae) or np.isinf(rmse) or np.isnan(mae) or np.isnan(rmse)):
                ekf_timestamps.append(ts)
                ekf_mae_clean.append(mae)
                ekf_rmse_clean.append(rmse)
        
        if ekf_timestamps:  # Have valid data points
            plt.plot(ekf_timestamps, ekf_mae_clean, 'b-', label='EKF MAE', linewidth=2)
            plt.plot(ekf_timestamps, ekf_rmse_clean, 'r--', label='EKF RMSE', linewidth=2)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Position Error (meters)', fontsize=12)
            plt.title('EKF Position Error Over Time', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        else:  # No valid data points
            plt.text(0.5, 0.5, 'No EKF Data Available\n(All values were inf/invalid)', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=14, color='red')
            plt.title('EKF Position Error Over Time - No Data', fontsize=14, fontweight='bold')
    else:  # No EKF data at all
        plt.text(0.5, 0.5, 'No EKF Data Collected', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, color='red')
        plt.title('EKF Position Error Over Time - No Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ekf_error_vs_time_{timestamp}.png'), dpi=150)
    plt.close()
    print("✓ Generated: EKF error vs time plot")
    
    # 2. LiDAR vs EKF Comparison - FIXED
    plt.figure(figsize=(12, 6))
    metrics = ['MAE', 'RMSE']
    lidar_values = [final_metrics['lidar_mae'], final_metrics['lidar_rmse']]
    ekf_values = [final_metrics['ekf_mae'], final_metrics['ekf_rmse']]
    
    # Replace inf with 0 for plotting (will show as no bar)
    ekf_values_clean = [0 if np.isinf(val) or np.isnan(val) else val for val in ekf_values]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, lidar_values, width, label='LiDAR', color='#FF7F0E')
    bars_ekf = plt.bar(x + width/2, ekf_values_clean, width, label='EKF', color='#2CA02C')
    
    # Special handling for inf values - mark bars as "No Data"
    for i, (original_val, clean_val) in enumerate(zip(ekf_values, ekf_values_clean)):
        if np.isinf(original_val) or np.isnan(original_val):
            # Make bar red and add "No Data" label
            bars_ekf[i].set_color('#FF0000')
            plt.text(i + width/2, max(lidar_values) * 0.1, 'No\nData', ha='center', va='bottom', 
                    fontsize=8, color='white', weight='bold')
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Error Value (meters)', fontsize=12)
    plt.title('LiDAR vs EKF Performance Comparison (Position Accuracy)', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars - handle inf
    for i, (lidar_val, ekf_val) in enumerate(zip(lidar_values, ekf_values)):
        plt.text(i - width/2, lidar_val + max(lidar_values) * 0.01, f'{lidar_val:.3f} m', ha='center', va='bottom')
        if not (np.isinf(ekf_val) or np.isnan(ekf_val)):
            plt.text(i + width/2, ekf_val + max(lidar_values) * 0.01, f'{ekf_val:.3f} m', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'lidar_vs_ekf_comparison_{timestamp}.png'), dpi=150)
    plt.close()
    print("✓ Generated: LiDAR vs EKF comparison plot")

    # 3. Semantic IoU Over Time - FIXED
    plt.figure(figsize=(12, 6))
    if evaluator.camera_iou_per_frame and evaluator.timestamps:
        plt.plot(evaluator.timestamps, evaluator.camera_iou_per_frame, 'g-', linewidth=2)
        plt.axhline(y=np.mean(evaluator.camera_iou_per_frame), color='r', linestyle='--', 
                    label=f'Average: {np.mean(evaluator.camera_iou_per_frame):.3f}')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('IoU Score', fontsize=12)
        plt.title('Camera Semantic Segmentation IoU Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
    else:
        plt.text(0.5, 0.5, 'No Camera IoU Data Available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, color='red')
        plt.title('Camera Semantic Segmentation IoU Over Time - No Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'semantic_iou_over_time_{timestamp}.png'), dpi=150)
    plt.close()
    print("✓ Generated: Semantic IoU over time plot")
    
    # 4. Processing Breakdown (Pie Chart) - FIXED to include ALL components
    plt.figure(figsize=(12, 6))
    
    # Calculate average times for ALL components
    avg_times = {}
    labels = []
    sizes = []
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    
    # Define the order we want to display (matching the "correct" example)
    component_order = ['frame_time', 'lidar_preprocessing', 'clustering', 'segmentation', 'fusion', 'visualization']
    
    # Process components in the defined order
    for component in component_order:
        if component in timing_summary and timing_summary[component]['mean'] > 0:
            avg_time = timing_summary[component]['mean']
            avg_times[component] = avg_time
            # Format component name for display
            display_name = component.replace('_', ' ').title()
            labels.append(f'{display_name}\n({avg_time:.1f} ms)')
            sizes.append(avg_time)
    
    # Only create pie chart if we have data
    if sizes:
        # Create pie chart
        plt.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
        plt.title('Processing Time Breakdown by Component', fontsize=14, fontweight='bold')
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Processing Time Breakdown - No Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'processing_breakdown_{timestamp}.png'), dpi=150)
    plt.close()
    print("✓ Generated: Processing breakdown plot")
    
    # 5. FPS Over Time - FIXED  
    plt.figure(figsize=(12, 6))
    if performance_monitor.timestamps and performance_monitor.fps_history:
        plt.plot(performance_monitor.timestamps, performance_monitor.fps_history, 'b-', linewidth=1, alpha=0.7)
        avg_fps = performance_monitor.get_average_fps()
        plt.axhline(y=avg_fps, color='r', linestyle='--', linewidth=2, label=f'Average: {avg_fps:.1f} FPS')
        plt.axhline(y=20, color='g', linestyle=':', linewidth=2, label='Real-time threshold (20 FPS)')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('Frame Rate Performance Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
    else:
        plt.text(0.5, 0.5, 'No FPS Data Available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, color='red')
        plt.title('Frame Rate Performance Over Time - No Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fps_over_time_{timestamp}.png'), dpi=150)
    plt.close()
    print("✓ Generated: FPS over time plot")
    
    report_content = evaluator.generate_report()
    
    # Generate Final Report - FIXED to include ALL timing components and handle inf values
    report_path = os.path.join(output_dir, f'final_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL MULTI-SENSOR FUSION EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runtime: {performance_monitor.timestamps[-1] if performance_monitor.timestamps else 0:.1f}s\n")
        f.write(f"Total Frames: {evaluator.frame_count}\n")
        avg_fps = performance_monitor.get_average_fps()
        f.write(f"Average FPS: {avg_fps:.1f}\n\n")
        
        f.write("[Positioning Accuracy]\n")
        f.write(f"- LiDAR MAE: {final_metrics['lidar_mae']:.3f} m\n")
        f.write(f"- LiDAR RMSE: {final_metrics['lidar_rmse']:.3f} m\n")
        f.write(f"- LiDAR Relative Error: {final_metrics['lidar_relative_error']:.1f}%\n")
        
        # FIXED: Handle inf values in EKF metrics
        if np.isinf(final_metrics['ekf_mae']) or np.isnan(final_metrics['ekf_mae']):
            f.write(f"- EKF MAE: No Data Available\n")
            f.write(f"- EKF RMSE: No Data Available\n") 
            f.write(f"- EKF Relative Error: No Data Available\n")
            
            # Add explanation
            f.write(f"\n[EKF Evaluation Note]\n")
            f.write(f"EKF metrics show 'No Data Available' because:\n")
            f.write(f"- No labeled tracks were successfully matched with ground truth, or\n")
            f.write(f"- All evaluation frames were skipped due to insufficient data\n")
            f.write(f"- This may indicate tracking initialization issues or very sparse ground truth\n")
        else:
            f.write(f"- EKF MAE: {final_metrics['ekf_mae']:.3f} m\n")
            f.write(f"- EKF RMSE: {final_metrics['ekf_rmse']:.3f} m\n")
            f.write(f"- EKF Relative Error: {final_metrics['ekf_relative_error']:.1f}%\n")
        
        f.write(f"\n[Camera Segmentation Accuracy]\n")
        f.write(f"- Average Semantic IoU: {final_metrics['camera_iou']:.3f}\n\n")
        
        f.write("[System Performance]\n")
        f.write(f"- Average CPU Usage: {performance_monitor.get_average_cpu_usage():.1f}%\n")
        f.write("- Latency Breakdown (ms):\n")
        
        # FIXED: Display ALL components in the defined order
        component_order = ['frame_time', 'lidar_preprocessing', 'clustering', 'segmentation', 'fusion', 'visualization']
        component_display_names = {
            'frame_time': 'Frame Time',
            'lidar_preprocessing': 'Lidar Preprocessing',
            'clustering': 'Clustering',
            'segmentation': 'Segmentation',
            'fusion': 'Fusion',
            'visualization': 'Visualization'
        }
        
        # Track the sum of individual components
        component_sum = 0.0
        
        for component in component_order:
            if component in timing_summary:
                stats = timing_summary[component]
                display_name = component_display_names.get(component, component.replace('_', ' ').title())
                f.write(f"  - {display_name}: {stats['mean']:.1f}\n")
                component_sum += stats['mean']
        
        # Write total
        if 'total' in timing_summary:
            f.write(f"  - Total Average Latency: {timing_summary['total']['mean']:.1f}\n")
            
            # Validate that components sum approximately equals total
            total_latency = timing_summary['total']['mean']
            difference = abs(total_latency - component_sum)
            if difference > 1.0:  # More than 1ms difference
                f.write(f"\n  [WARNING] Component sum ({component_sum:.1f} ms) differs from total ({total_latency:.1f} ms) by {difference:.1f} ms\n")
        else:
            # If no total, calculate from components
            f.write(f"  - Total Average Latency: {component_sum:.1f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Generated: Final report saved to {report_path}")

    # Debug information - print what components we found
    print("\n[DEBUG] Timing components found:")
    for component in ['frame_time', 'lidar_preprocessing', 'clustering', 'segmentation', 'fusion', 'visualization']:
        if component in timing_summary:
            print(f"  - {component}: {timing_summary[component]['mean']:.1f} ms")
        else:
            print(f"  - {component}: NOT FOUND")

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(report_content)
    print("="*60)

    print(f"\nAll evaluation results saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return report_path