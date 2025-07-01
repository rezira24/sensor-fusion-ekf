import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque

from config1 import (
    # EKF core parameters
    EKF_PROCESS_NOISE_POSITION, EKF_PROCESS_NOISE_VELOCITY,
    EKF_LIDAR_NOISE, EKF_CAMERA_NOISE,
    # Association parameters
    ASSOCIATION_DISTANCE_THRESHOLD,
    # Object filters
    MIN_BBOX_WIDTH, MAX_BBOX_WIDTH, MIN_BBOX_HEIGHT, MAX_BBOX_HEIGHT,
    MOTORCYCLE_MIN_WIDTH, MOTORCYCLE_MAX_WIDTH,
    MOTORCYCLE_MIN_HEIGHT, MOTORCYCLE_MAX_HEIGHT,
    # Temporal consistency
    TEMPORAL_HISTORY_WINDOW, TEMPORAL_GRID_SIZE,
    HIGH_CONFIDENCE_MIN_CLUSTER_SIZE,
    HIGH_CONFIDENCE_MIN_OBJECT_WIDTH, HIGH_CONFIDENCE_MIN_OBJECT_LENGTH
)


@dataclass
class Detection:
    """Container for sensor detections with timestamp support"""
    position: np.ndarray  # [x, y] for LiDAR
    timestamp: float  # Detection timestamp in seconds
    bbox: Optional[Tuple[int, int, int, int]] = None  # For camera (x, y, w, h)
    label: Optional[int] = None
    confidence: float = 1.0
    centroid_3d: Optional[np.ndarray] = None  # 3D centroid for LiDAR
    is_fused: bool = False  # Flag for fused detections
    bbox_3d: Optional[np.ndarray] = None  # 3D bounding box


class MultiSensorEKF4D:
    """
    Extended Kalman Filter with 4D state vector [x, y, vx, vy]
    Uses Constant Velocity motion model for better prediction
    Handles asynchronous sensor updates with timestamps
    """
    
    def __init__(self, initial_state, camera_matrix, initial_timestamp):
        """
        Initialize EKF with 4D state
        
        Args:
            initial_state: [x, y] initial position (will be extended to include zero velocity)
            camera_matrix: 3x3 camera intrinsic matrix
            initial_timestamp: Initial timestamp in seconds
        """
        # State vector [x, y, vx, vy] - Position and velocity
        self.x = np.array([initial_state[0], initial_state[1], 0.0, 0.0], dtype=np.float64)
        
        # State covariance (4x4)
        self.P = np.eye(4)
        self.P[0, 0] = self.P[1, 1] = 5.0  # Position uncertainty
        self.P[2, 2] = self.P[3, 3] = 2.0  # Velocity uncertainty
        
        # Process noise - tuned for typical vehicle/pedestrian motion
        self.q_pos = EKF_PROCESS_NOISE_POSITION  # Ganti 0.1  # Position process noise
        self.q_vel = EKF_PROCESS_NOISE_VELOCITY  # Ganti 0.5  # Velocity process noise
        
        # Measurement models
        # LiDAR observes position directly
        self.H_lidar = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float64)
        
        # LiDAR measurement noise
        self.R_lidar = np.eye(2) * EKF_LIDAR_NOISE  # Ganti 0.25  # 0.5m std dev
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        # Camera measurement noise
        self.R_camera = np.eye(2) * EKF_CAMERA_NOISE  # Ganti 25.0  # 5 pixel std dev
        
        # Dynamic height estimation parameters
        self.height_estimator_A = 1.0  # Base height
        self.height_estimator_B = 0.01  # Height per pixel scaling
        
        # Tracking info
        self.last_update_timestamp = initial_timestamp
        self.creation_timestamp = initial_timestamp
        self.age = 0
        self.hits = 0
        self.consecutive_misses = 0
        self.last_lidar_update = None
        self.sensor_history = deque(maxlen=10)
        
        # Label tracking
        self.label = None
        
    def predict(self, current_timestamp):
        """
        Prediction step with Constant Velocity model
        Dynamically calculates dt based on timestamps
        """
        # Calculate time delta
        dt = current_timestamp - self.last_update_timestamp
        
        # Clamp dt to reasonable bounds (0.001 to 1.0 seconds)
        dt = np.clip(dt, 0.001, 1.0)
        
        # Constant Velocity transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        
        # Process noise covariance
        self.Q = np.array([[dt**3/3 * self.q_pos, 0, dt**2/2 * self.q_pos, 0],
                           [0, dt**3/3 * self.q_pos, 0, dt**2/2 * self.q_pos],
                           [dt**2/2 * self.q_pos, 0, dt * self.q_vel, 0],
                           [0, dt**2/2 * self.q_pos, 0, dt * self.q_vel]], dtype=np.float64)
        
        # Predict state and covariance
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update age
        self.age += 1
        self.last_update_timestamp = current_timestamp
        
    def update_lidar(self, z_lidar, timestamp, label=None):
        """
        Update with LiDAR measurement (linear Kalman update)
        Includes input validation and numerical stability improvements
        """
        # Input validation
        if not self._validate_measurement(z_lidar):
            print(f"Warning: Invalid LiDAR measurement: {z_lidar}")
            return
            
        # Predict to current timestamp
        if timestamp > self.last_update_timestamp:
            self.predict(timestamp)
        
        # Innovation
        y = z_lidar - self.H_lidar @ self.x
        
        # Innovation covariance
        S = self.H_lidar @ self.P @ self.H_lidar.T + self.R_lidar
        
        # Kalman gain using solve for numerical stability
        try:
            # K = P * H^T * S^(-1) => K * S = P * H^T
            K = np.linalg.solve(S.T, (self.P @ self.H_lidar.T).T).T
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in LiDAR update, skipping")
            return
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H_lidar
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_lidar @ K.T
        
        # Ensure positive definite covariance
        self.P = (self.P + self.P.T) / 2
        
        # Update tracking info
        self.hits += 1
        self.consecutive_misses = 0
        self.last_lidar_update = self.x[:2].copy()
        self.sensor_history.append('lidar')
        
        # Update label if available
        if label is not None:
            self.label = label
        
    def update_camera(self, z_camera, timestamp, bbox_size=None, label=None):
        """
        Update with camera measurement (EKF update due to non-linear model)
        Includes dynamic height estimation based on bbox size
        """
        # Input validation
        if not self._validate_measurement(z_camera):
            print(f"Warning: Invalid camera measurement: {z_camera}")
            return
            
        # Cannot update with camera only without depth reference
        if self.last_lidar_update is None:
            return
            
        # Predict to current timestamp
        if timestamp > self.last_update_timestamp:
            self.predict(timestamp)
            
        # Estimate object height from bbox if available
        if bbox_size is not None:
            bbox_height = bbox_size[1]
            self.estimated_height = self.height_estimator_A + self.height_estimator_B * bbox_height
        else:
            self.estimated_height = 1.5  # Default
            
        # Non-linear measurement model
        h_x = self._h_camera(self.x)
        
        # Check if projection is valid
        if h_x is None:
            return
            
        # Jacobian of camera measurement model
        H = self._compute_camera_jacobian(self.x)
        if H is None:
            return
        
        # Innovation
        y = z_camera - h_x
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_camera
        
        # Kalman gain using solve
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in camera update, skipping")
            return
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form)
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_camera @ K.T
        
        # Ensure positive definite
        self.P = (self.P + self.P.T) / 2
        
        # Update tracking info
        self.hits += 1
        self.consecutive_misses = 0
        self.sensor_history.append('camera')
        
        # Update label if available
        if label is not None and self.label is None:
            self.label = label
        
    def update_fused(self, z_lidar, z_camera, lidar_timestamp, camera_timestamp, 
                     bbox_size=None, lidar_label=None, camera_label=None):
        """
        Optimal fusion of LiDAR and camera measurements with timestamp alignment
        """
        # Input validation
        if not self._validate_measurement(z_lidar) or not self._validate_measurement(z_camera):
            print("Warning: Invalid measurements in fused update")
            return
            
        # Use the most recent timestamp as reference
        reference_timestamp = max(lidar_timestamp, camera_timestamp)
        
        # Predict to reference timestamp
        if reference_timestamp > self.last_update_timestamp:
            self.predict(reference_timestamp)
            
        # Estimate height from bbox
        if bbox_size is not None:
            bbox_height = bbox_size[1]
            self.estimated_height = self.height_estimator_A + self.height_estimator_B * bbox_height
        else:
            self.estimated_height = 1.5
        
        # Stack measurements
        z = np.hstack([z_lidar, z_camera])
        
        # Expected measurements
        h_lidar_exp = self.H_lidar @ self.x
        h_camera_exp = self._h_camera(self.x)
        
        if h_camera_exp is None:
            # Fall back to LiDAR-only update
            self.update_lidar(z_lidar, lidar_timestamp, lidar_label)
            return
            
        h_exp = np.hstack([h_lidar_exp, h_camera_exp])
        
        # Jacobian matrix (stacked)
        H_camera = self._compute_camera_jacobian(self.x)
        if H_camera is None:
            # Fall back to LiDAR-only update
            self.update_lidar(z_lidar, lidar_timestamp, lidar_label)
            return
            
        H = np.vstack([self.H_lidar, H_camera])
        
        # Measurement noise (block diagonal)
        R = np.block([
            [self.R_lidar, np.zeros((2, 2))],
            [np.zeros((2, 2)), self.R_camera]
        ])
        
        # EKF update
        y = z - h_exp  # Innovation
        S = H @ self.P @ H.T + R  # Innovation covariance
        
        # Kalman gain using solve
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in fused update, falling back to LiDAR")
            self.update_lidar(z_lidar, lidar_timestamp, lidar_label)
            return
        
        # Update state and covariance
        self.x = self.x + K @ y
        
        # Joseph form covariance update
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure positive definite
        self.P = (self.P + self.P.T) / 2
        
        # Update tracking info
        self.hits += 1
        self.consecutive_misses = 0
        self.last_lidar_update = self.x[:2].copy()
        self.sensor_history.append('fused')
        
        # Prioritize camera label
        if camera_label is not None:
            self.label = camera_label
        elif lidar_label is not None and self.label is None:
            self.label = lidar_label
            
    def _validate_measurement(self, measurement):
        """Validate that measurement contains no NaN or inf values"""
        return np.all(np.isfinite(measurement))
        
    def _h_camera(self, state):
        """
        Non-linear camera measurement model with dynamic height
        Projects 3D position to 2D pixel coordinates
        """
        x, y, vx, vy = state
        
        # Avoid division by zero
        if abs(x) < 0.1:
            x = 0.1 if x >= 0 else -0.1
            
        # Use estimated height
        z = self.estimated_height
        
        # Project to camera image plane
        u = self.fx * (y / x) + self.cx  # Lateral in image
        v = self.fy * (z / x) + self.cy  # Vertical in image
        
        # Validate projection
        if not np.isfinite(u) or not np.isfinite(v):
            return None
            
        return np.array([u, v])
        
    def _compute_camera_jacobian(self, state):
        """
        Compute Jacobian of camera measurement model w.r.t 4D state
        Partial derivatives of h_camera w.r.t [x, y, vx, vy]
        """
        x, y, vx, vy = state
        z = self.estimated_height
        
        # Avoid division by zero
        if abs(x) < 0.1:
            x = 0.1 if x >= 0 else -0.1
            
        # Jacobian matrix 2x4 (measurement dim x state dim)
        H = np.zeros((2, 4))
        
        # ∂u/∂x, ∂u/∂y, ∂u/∂vx, ∂u/∂vy
        H[0, 0] = -self.fx * y / (x**2)
        H[0, 1] = self.fx / x
        H[0, 2] = 0
        H[0, 3] = 0
        
        # ∂v/∂x, ∂v/∂y, ∂v/∂vx, ∂v/∂vy
        H[1, 0] = -self.fy * z / (x**2)
        H[1, 1] = 0
        H[1, 2] = 0
        H[1, 3] = 0
        
        # Validate Jacobian
        if not np.all(np.isfinite(H)):
            return None
            
        return H
        
    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()
        
    def get_position(self):
        """Get current position estimate"""
        return self.x[:2].copy()
        
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.x[2:4].copy()
        
    def get_speed(self):
        """Get current speed (magnitude of velocity)"""
        velocity = self.get_velocity()
        return np.linalg.norm(velocity)
        
    def get_uncertainty_ellipse(self, n_std=2):
        """
        Get uncertainty ellipse parameters for visualization
        """
        # Extract position covariance
        cov_pos = self.P[:2, :2]
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_pos)
        
        # Ensure positive eigenvalues
        eigenvalues = np.abs(eigenvalues)
        
        # Sort by eigenvalue
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Angle of major axis
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # Width and height (n_std * sqrt(eigenvalue))
        width, height = n_std * np.sqrt(eigenvalues)
        
        return self.get_position(), width, height, np.degrees(angle)


class MultiSensorTracker4D:
    """
    Multi-object tracker using 4D state EKF for each track
    Handles asynchronous sensor data with proper timestamp management
    """
    
    def __init__(self, camera_matrix, max_age=30, min_hits=3):
        self.camera_matrix = camera_matrix
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        
        # Association parameters
        self.association_distance_threshold = ASSOCIATION_DISTANCE_THRESHOLD # meters
        
        # Noise filtering parameters
        self.max_height = 3.0  # meters
        self.max_lateral_distance = 15.0  # meters
        self.max_behind_distance = 20.0  # meters
        
        # Object dimension filters
        self.min_bbox_width = MIN_BBOX_WIDTH
        self.max_bbox_width = MAX_BBOX_WIDTH
        self.min_bbox_height = MIN_BBOX_HEIGHT
        self.max_bbox_height = MAX_BBOX_HEIGHT
        self.min_aspect_ratio = 0.2
        self.max_aspect_ratio = 6.0
        
        # Pedestrian-specific parameters
        self.pedestrian_min_width = 0.3
        self.pedestrian_max_width = 1.5
        self.pedestrian_min_height = 0.3
        self.pedestrian_max_height = 2.5
        
        # Motorcycle-specific parameters (NEW)
        self.motorcycle_min_width = MOTORCYCLE_MIN_WIDTH
        self.motorcycle_max_width = MOTORCYCLE_MAX_WIDTH
        self.motorcycle_min_height = MOTORCYCLE_MIN_HEIGHT
        self.motorcycle_max_height = MOTORCYCLE_MAX_HEIGHT
        
        # Performance tracking
        self.timing_stats = {
            'association': deque(maxlen=100),
            'update': deque(maxlen=100),
            'filtering': deque(maxlen=100)
        }
        
        # Temporal consistency with grid-based hashing
        self.detection_history = {}
        self.history_window = TEMPORAL_HISTORY_WINDOW
        self.history_cleanup_interval = 10  # Reduced from 100
        self.grid_size = TEMPORAL_GRID_SIZE  # Ganti 1.0 # 1m x 1m grid cells
        
        # Current timestamp tracking
        self.current_timestamp = 0.0
        
    def update(self, lidar_detections, camera_detections, lidar_transform, camera_transform, timestamp):
        """
        Update all tracks with new detections
        Now requires timestamp for proper temporal handling
        """
        self.frame_count += 1
        self.current_timestamp = timestamp
        
        # Frequent cleanup of detection history
        if self.frame_count % self.history_cleanup_interval == 0:
            self._cleanup_detection_history()
        
        # Filter noise detections
        filter_start = time.time()
        lidar_detections = self._filter_noise_detections(lidar_detections)
        self.timing_stats['filtering'].append(time.time() - filter_start)
        
        # Predict all existing tracks to current timestamp
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id].predict(timestamp)
            
        # Associate detections
        assoc_start = time.time()
        matched_detections, unmatched_lidar, unmatched_camera = self._associate_detections(
            lidar_detections, camera_detections, lidar_transform, camera_transform)
        self.timing_stats['association'].append(time.time() - assoc_start)
            
        # Update matched tracks
        update_start = time.time()
        updated_tracks = set()
        
        for track_id, lidar_det, camera_det in matched_detections:
            ekf = self.tracks[track_id]
            updated_tracks.add(track_id)
            
            if lidar_det is not None and camera_det is not None:
                # Fused update
                z_lidar = lidar_det.position[:2]
                bbox = camera_det.bbox
                z_camera = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                bbox_size = (bbox[2], bbox[3])
                
                ekf.update_fused(z_lidar, z_camera, 
                               lidar_det.timestamp, camera_det.timestamp,
                               bbox_size=bbox_size,
                               lidar_label=lidar_det.label,
                               camera_label=camera_det.label)
                    
            elif lidar_det is not None:
                # LiDAR only
                z_lidar = lidar_det.position[:2]
                ekf.update_lidar(z_lidar, lidar_det.timestamp, label=lidar_det.label)
                    
            elif camera_det is not None:
                # Camera only
                bbox = camera_det.bbox
                z_camera = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                bbox_size = (bbox[2], bbox[3])
                ekf.update_camera(z_camera, camera_det.timestamp, 
                                bbox_size, label=camera_det.label)
                    
        self.timing_stats['update'].append(time.time() - update_start)
        
        # Handle missed detections
        for track_id in self.tracks:
            if track_id not in updated_tracks:
                self.tracks[track_id].consecutive_misses += 1
        
        # Create new tracks from LiDAR
        for lidar_det in unmatched_lidar:
            # Check if this is a high-confidence detection
            high_confidence = False
            
            # Check cluster size (number of points)
            if hasattr(lidar_det, 'cluster_size') and lidar_det.cluster_size > HIGH_CONFIDENCE_MIN_CLUSTER_SIZE:
                high_confidence = True
            
            # Also check bbox size
            if hasattr(lidar_det, 'bbox_3d') and lidar_det.bbox_3d is not None:
                bbox = lidar_det.bbox_3d
                width = abs(bbox[3] - bbox[0])
                length = abs(bbox[4] - bbox[1])
                # Large object = high confidence
                if width > HIGH_CONFIDENCE_MIN_OBJECT_WIDTH and length > HIGH_CONFIDENCE_MIN_OBJECT_LENGTH:  # Car-sized or larger
                    high_confidence = True
            
            # For high-confidence detections, create track immediately
            if high_confidence:
                track_id = self._create_new_track(lidar_det.position[:2], 
                                                lidar_det.timestamp,
                                                label=lidar_det.label)
                # if hasattr(lidar_det, 'cluster_size'):
                #     print(f"High-confidence track created: {lidar_det.cluster_size} points at {lidar_det.position[:2]}")
            elif self._passes_temporal_consistency_check(lidar_det.position[:2]):
                track_id = self._create_new_track(lidar_det.position[:2], 
                                                lidar_det.timestamp,
                                                label=lidar_det.label)
        
        # IMPORTANT: Also create tracks from camera detections (for close/small objects)
        # This is crucial for motorcycles and pedestrians that might have few LiDAR points
        # for camera_det in unmatched_camera:
        #     # Estimate rough position from camera bbox
        #     # For close objects, we can use a simple heuristic
        #     bbox = camera_det.bbox
        #     x, y, w, h = bbox
            
        #     # Better distance estimation based on bbox height
        #     # Assuming average vehicle height in image at known distances
        #     # At 5m, a motorcycle might be ~200 pixels tall
        #     # At 10m, it might be ~100 pixels tall
        #     estimated_distance = 1000.0 / h  # Simple inverse relationship
        #     estimated_distance = np.clip(estimated_distance, 2.0, 20.0)
            
        #     # Position estimate (straight ahead for center objects)
        #     cx = x + w/2
        #     cy = y + h/2
            
        #     # Rough lateral position estimate
        #     lateral_offset = (cx - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        #     lateral_position = lateral_offset * estimated_distance
            
        #     estimated_position = np.array([estimated_distance, lateral_position])
            
        #     # Only create track if bbox is significant size and centered
        #     bbox_area = h * w
            
        #     # ONLY create camera tracks for VERY CLOSE objects (< 5m)
        #     if estimated_distance <= 5.0 and h > 100:  # Must be close AND large bbox
        #         # Create track immediately for very close objects
        #         track_id = self._create_new_track(estimated_position, 
        #                                         camera_det.timestamp,
        #                                         label=camera_det.label,
        #                                         camera_init=True)
        #         print(f"Created camera-only track for CLOSE object: dist={estimated_distance:.1f}m, bbox={w}x{h}")
                
        #     # Don't create camera-only tracks for distant objects
        
        # Remove dead tracks
        tracks_to_remove = []
        for track_id, ekf in self.tracks.items():
            if ekf.consecutive_misses > self.max_age:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Remove duplicate tracks every 10 frames
        if self.frame_count % 10 == 0:
            self._remove_duplicate_tracks()

        # Return confirmed tracks
        confirmed_tracks = []
        for track_id, ekf in self.tracks.items():
            if ekf.hits >= self.min_hits or self.frame_count <= self.min_hits:
                track_info = {
                    'id': track_id,
                    'ekf': ekf,
                    'position': ekf.get_position(),
                    'velocity': ekf.get_velocity(),
                    'speed': ekf.get_speed(),
                    'label': ekf.label,
                    'sensor_mode': ekf.sensor_history[-1] if ekf.sensor_history else 'none',
                    'confidence': min(ekf.hits / self.min_hits, 1.0),
                    'age': timestamp - ekf.creation_timestamp
                }
                confirmed_tracks.append(track_info)
                
        return confirmed_tracks
        
    def _cleanup_detection_history(self):
        """Remove old entries from detection history"""
        old_threshold = self.frame_count - self.history_window * 2
        keys_to_remove = []
        
        for key, history in self.detection_history.items():
            # Remove old frame numbers
            while history and history[0] < old_threshold:
                history.popleft()
            
            # Remove empty entries
            if not history:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.detection_history[key]
        
    def _filter_noise_detections(self, lidar_detections):
        """Filter out noise detections with special handling for motorcycles"""
        filtered = []
        
        for det in lidar_detections:
            pos = det.position
            
            # Basic position filters
            if len(pos) > 2 and pos[2] > self.max_height:
                continue
            if abs(pos[1]) > self.max_lateral_distance:
                continue
            if pos[0] < -self.max_behind_distance:
                continue
                
            # Size-based filtering
            if hasattr(det, 'bbox_3d') and det.bbox_3d is not None:
                bbox = det.bbox_3d
                width = abs(bbox[3] - bbox[0])
                height = abs(bbox[4] - bbox[1])
                
                # Check if likely pedestrian
                is_likely_pedestrian = (
                    self.pedestrian_min_width <= width <= self.pedestrian_max_width and
                    self.pedestrian_min_height <= height <= self.pedestrian_max_height
                )
                
                # Check if likely motorcycle (NEW)
                is_likely_motorcycle = (
                    self.motorcycle_min_width <= width <= self.motorcycle_max_width and
                    self.motorcycle_min_height <= height <= self.motorcycle_max_height
                )
                
                if is_likely_pedestrian or is_likely_motorcycle:
                    # Very relaxed filter for pedestrians and motorcycles
                    filtered.append(det)
                else:
                    # Standard filter for vehicles
                    if width < self.min_bbox_width or width > self.max_bbox_width:
                        continue
                    if height < self.min_bbox_height or height > self.max_bbox_height:
                        continue
                    
                    # Aspect ratio (skip for small objects)
                    if width > 1.0 and height > 1.0:  # Only check aspect ratio for larger objects
                        aspect_ratio = width / height if height > 0 else 0
                        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                            continue
                    
                    filtered.append(det)
            else:
                # If no bbox, include it (let clustering decide)
                filtered.append(det)
            
        return filtered
        
    def _associate_detections(self, lidar_detections, camera_detections, 
                            lidar_transform, camera_transform):
        """Associate detections with tracks"""
        # Match LiDAR and camera detections
        lidar_camera_matches = self._match_lidar_camera_centroid(
            lidar_detections, camera_detections, lidar_transform, camera_transform)
            
        # Create combined detections
        combined_detections = []
        used_lidar = set()
        used_camera = set()
        
        for lidar_idx, camera_idx, match_score in lidar_camera_matches:
            if match_score > 0.5:
                # Mark as fused
                lidar_detections[lidar_idx].is_fused = True
                if camera_detections[camera_idx].label is not None:
                    lidar_detections[lidar_idx].label = camera_detections[camera_idx].label
                    
                combined_detections.append((
                    lidar_detections[lidar_idx],
                    camera_detections[camera_idx]
                ))
                used_lidar.add(lidar_idx)
                used_camera.add(camera_idx)
                
        # Add unmatched
        for i, det in enumerate(lidar_detections):
            if i not in used_lidar:
                combined_detections.append((det, None))
                
        for i, det in enumerate(camera_detections):
            if i not in used_camera:
                combined_detections.append((None, det))
                
        # Associate with tracks
        if not self.tracks or not combined_detections:
            unmatched_lidar = [det for i, det in enumerate(lidar_detections) if i not in used_lidar]
            unmatched_camera = [det for i, det in enumerate(camera_detections) if i not in used_camera]
            return [], unmatched_lidar, unmatched_camera
            
        # Build cost matrix
        track_positions = np.array([track.get_position() for track in self.tracks.values()])
        detection_positions = []
        
        for lidar_det, camera_det in combined_detections:
            if lidar_det is not None:
                detection_positions.append(lidar_det.position[:2])
            else:
                detection_positions.append([100, 100])  # Far position
                
        detection_positions = np.array(detection_positions)
        
        # Calculate distances
        cost_matrix = cdist(detection_positions, track_positions)
        
        # Penalize matches between different object types
        track_ids = list(self.tracks.keys())
        for det_idx, (lidar_det, camera_det) in enumerate(combined_detections):
            det_label = None
            if camera_det is not None and camera_det.label is not None:
                det_label = camera_det.label
            elif lidar_det is not None and lidar_det.label is not None:
                det_label = lidar_det.label
                
            for track_idx, track_id in enumerate(track_ids):
                track_label = self.tracks[track_id].label
                # Penalize different labels
                if det_label is not None and track_label is not None and det_label != track_label:
                    cost_matrix[det_idx, track_idx] += 5.0
        
        # Solve assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches
        matched_detections = []
        matched_combined_indices = []
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < self.association_distance_threshold:
                lidar_det, camera_det = combined_detections[det_idx]
                track_id = track_ids[track_idx]
                matched_detections.append((track_id, lidar_det, camera_det))
                matched_combined_indices.append(det_idx)
                
        # Find unmatched
        unmatched_lidar = []
        unmatched_camera = []
        
        for i, (lidar_det, camera_det) in enumerate(combined_detections):
            if i not in matched_combined_indices:
                if lidar_det is not None:
                    unmatched_lidar.append(lidar_det)
                if camera_det is not None:
                    unmatched_camera.append(camera_det)
                    
        return matched_detections, unmatched_lidar, unmatched_camera
        
    def _match_lidar_camera_centroid(self, lidar_detections, camera_detections, 
                                   lidar_transform, camera_transform):
        """Match LiDAR and camera using centroid projection"""
        matches = []
        
        if not lidar_detections or not camera_detections:
            return matches
            
        # Transform matrices
        lidar_world = lidar_transform.get_matrix()
        camera_world_inv = camera_transform.get_inverse_matrix()
        
        for lidar_idx, lidar_det in enumerate(lidar_detections):
            # Get centroid
            if hasattr(lidar_det, 'centroid_3d') and lidar_det.centroid_3d is not None:
                centroid = lidar_det.centroid_3d
            else:
                centroid = lidar_det.position
                
            # Transform to camera
            point_lidar = np.array([centroid[0], centroid[1], 
                                   centroid[2] if len(centroid) > 2 else 0, 1])
            point_world = lidar_world @ point_lidar
            point_camera = camera_world_inv @ point_world
            
            # Camera coordinates
            cam_coords = np.array([
                point_camera[1],   # Y -> X
                -point_camera[2],  # -Z -> Y  
                point_camera[0]    # X -> Z
            ])
            
            # Skip if behind camera
            if cam_coords[2] <= 0:
                continue
                
            # Project to image
            pixel_coords = self.camera_matrix @ cam_coords
            pixel_coords = pixel_coords[:2] / pixel_coords[2]
            
            # Validate projection
            if not np.all(np.isfinite(pixel_coords)):
                continue
                
            px, py = int(pixel_coords[0]), int(pixel_coords[1])
            
            # Find best match
            best_match = None
            best_score = 0
            
            for cam_idx, cam_det in enumerate(camera_detections):
                x, y, w, h = cam_det.bbox
                
                # Bbox centroid
                bbox_cx = x + w / 2
                bbox_cy = y + h / 2
                
                # Check if inside bbox
                if x <= px <= x + w and y <= py <= y + h:
                    # Centroid distance
                    centroid_dist = np.sqrt((px - bbox_cx)**2 + (py - bbox_cy)**2)
                    
                    # Normalize
                    bbox_diag = np.sqrt(w**2 + h**2)
                    normalized_dist = centroid_dist / (bbox_diag + 1e-6)
                    
                    # Score
                    score = 1.0 - min(normalized_dist, 1.0)
                    
                    if score > best_score:
                        best_score = score
                        best_match = cam_idx
                        
            if best_match is not None:
                matches.append((lidar_idx, best_match, best_score))
                
        return matches
        
    def _passes_temporal_consistency_check(self, position):
        """Check temporal consistency using grid-based hashing"""
        # Grid-based key
        grid_x = int(position[0] / self.grid_size)
        grid_y = int(position[1] / self.grid_size)
        key = f"{grid_x},{grid_y}"
        
        if key not in self.detection_history:
            self.detection_history[key] = deque(maxlen=self.history_window)
        self.detection_history[key].append(self.frame_count)
        
        # Check for consistency
        if len(self.detection_history[key]) >= 2:
            frames = list(self.detection_history[key])
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] <= 2:
                    return True
                    
        return len(self.detection_history[key]) >= 3
        
    def _create_new_track(self, initial_position, timestamp, label=None, camera_init=False):
        """Create new track with timestamp"""
        track_id = self.next_id
        self.next_id += 1
        ekf = MultiSensorEKF4D(initial_position, self.camera_matrix, timestamp)
        
        if label is not None:
            ekf.label = label
            
        # Mark if initialized from camera (might need special handling)
        ekf.camera_initialized = camera_init
        
        self.tracks[track_id] = ekf
        return track_id
        
    def get_performance_stats(self):
        """Get timing statistics"""
        stats = {}
        for key, values in self.timing_stats.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values) * 1000,
                    'std': np.std(values) * 1000,
                    'max': np.max(values) * 1000
                }
        return stats

    def _remove_duplicate_tracks(self):
        """Remove tracks that are too close to each other"""
        tracks_to_remove = []
        track_ids = list(self.tracks.keys())
        
        for i, track_id_1 in enumerate(track_ids):
            if track_id_1 in tracks_to_remove:
                continue
            
            track_1 = self.tracks[track_id_1]
            pos_1 = track_1.get_position()
            
            for track_id_2 in track_ids[i+1:]:
                if track_id_2 in tracks_to_remove:
                    continue
                
                track_2 = self.tracks[track_id_2]
                pos_2 = track_2.get_position()
                
                distance = np.linalg.norm(pos_1 - pos_2)
                
                # If tracks are too close (same object)
                if distance < 2.0:  # 2 meter threshold
                    # Keep track with more hits
                    if track_1.hits >= track_2.hits:
                        tracks_to_remove.append(track_id_2)
                    else:
                        tracks_to_remove.append(track_id_1)
                        break
        
        # Remove duplicates
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
                print(f"Removed duplicate track {track_id}")

class EnhancedBEVVisualizer:
    """Enhanced BEV visualization with velocity vectors"""
    
    def __init__(self, bev_range=50, image_size=600):
        self.bev_range = bev_range
        self.image_size = image_size
        self.scale = image_size / (2 * bev_range)
        
        # Sensor mode colors
        self.sensor_colors = {
            'fused': (0, 255, 0),
            'lidar': (255, 255, 0),
            'camera': (255, 0, 255),
            'predicted': (128, 128, 128)
        }
        
        # Object type colors
        self.object_colors = {
            'vehicle': (0, 255, 0),      # Green for vehicles
            'pedestrian': (0, 0, 255),   # Red for pedestrians
            'unknown': (128, 128, 128)   # Gray for unknown
        }
        
        # Visualization parameters
        self.max_lidar_points = 5000
        self.velocity_scale = 5.0  # Scale factor for velocity arrows
        
    def draw_bev(self, tracks, raw_lidar_points=None):
        bev_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        self._draw_grid(bev_img)
        self._draw_ego_vehicle(bev_img)
        
        if raw_lidar_points is not None:
            self._draw_lidar_points(bev_img, raw_lidar_points)
            
        for track in tracks:
            self._draw_track_with_velocity(bev_img, track)
            
        self._draw_legend(bev_img)
        
        return bev_img
        
    def _draw_grid(self, img):
        grid_spacing = 10
        grid_color = (50, 50, 50)
        center = self.image_size // 2
        
        for i in range(-self.bev_range, self.bev_range + 1, grid_spacing):
            pos = int(center + i * self.scale)
            cv2.line(img, (pos, 0), (pos, self.image_size), grid_color, 1)
            cv2.line(img, (0, pos), (self.image_size, pos), grid_color, 1)
            
    def _draw_ego_vehicle(self, img):
        center_x, center_y = self.image_size // 2, self.image_size // 2
        
        cv2.rectangle(img, (center_x-8, center_y-15), 
                     (center_x+8, center_y+15), (255, 255, 255), -1)
        cv2.rectangle(img, (center_x-6, center_y-15), 
                     (center_x+6, center_y-10), (0, 255, 0), -1)
        cv2.putText(img, "EGO", (center_x-15, center_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
    def _draw_lidar_points(self, img, points):
        center_x, center_y = self.image_size // 2, self.image_size // 2
        
        # Subsample for performance
        if len(points) > self.max_lidar_points:
            indices = np.random.choice(len(points), self.max_lidar_points, replace=False)
            points = points[indices]
        
        for point in points:
            if abs(point[0]) < self.bev_range and abs(point[1]) < self.bev_range:
                px = int(center_x + point[1] * self.scale)
                py = int(center_y - point[0] * self.scale)
                if 0 <= px < self.image_size and 0 <= py < self.image_size:
                    cv2.circle(img, (px, py), 1, (80, 80, 80), -1)
                    
    def _draw_track_with_velocity(self, img, track):
        """Draw track with velocity vector"""
        pos = track['position']
        velocity = track['velocity']
        speed = track.get('speed', 0)
        track_id = track['id']
        sensor_mode = track.get('sensor_mode', 'predicted')
        ekf = track.get('ekf')
        label = track.get('label')
        
        center_x, center_y = self.image_size // 2, self.image_size // 2
        px = int(center_x + pos[1] * self.scale)
        py = int(center_y - pos[0] * self.scale)
        
        if not (0 <= px < self.image_size and 0 <= py < self.image_size):
            return
        
        # Determine object type and color
        if label == 10 or label == 'Vehicle':
            object_type = 'vehicle'
            color = self.object_colors['vehicle']  # Green
            label_text = f"Car_{track_id}"
        elif label == 4 or label == 'Pedestrian':
            object_type = 'pedestrian'
            color = self.object_colors['pedestrian']  # Red
            label_text = f"Ped_{track_id}"
        else:
            object_type = 'unknown'
            color = self.object_colors['unknown']  # Gray
            label_text = f"Obj_{track_id}"
        
        # Draw uncertainty ellipse
        if ekf is not None:
            try:
                center, width, height, angle = ekf.get_uncertainty_ellipse()
                ellipse_w = int(width * self.scale)
                ellipse_h = int(height * self.scale)
                
                if ellipse_w > 1 and ellipse_h > 1:
                    ellipse_w = min(ellipse_w, self.image_size // 4)
                    ellipse_h = min(ellipse_h, self.image_size // 4)
                    ellipse_color = tuple(int(c * 0.6) for c in color)
                    cv2.ellipse(img, (px, py), (ellipse_w, ellipse_h), 
                               int(angle), 0, 360, ellipse_color, 1)
            except Exception:
                pass
        
        # Draw object shape
        if object_type == 'vehicle':
            vehicle_length = int(4.5 * self.scale)
            vehicle_width = int(2.0 * self.scale)
            
            # Calculate rotation based on velocity
            if speed > 0.5:  # Only rotate if moving
                heading = np.arctan2(velocity[1], velocity[0])
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                
                # Rotated rectangle corners
                corners = np.array([
                    [-vehicle_width/2, -vehicle_length/2],
                    [vehicle_width/2, -vehicle_length/2],
                    [vehicle_width/2, vehicle_length/2],
                    [-vehicle_width/2, vehicle_length/2]
                ])
                
                # Rotate corners
                rotated = np.zeros_like(corners)
                rotated[:, 0] = corners[:, 0] * cos_h - corners[:, 1] * sin_h
                rotated[:, 1] = corners[:, 0] * sin_h + corners[:, 1] * cos_h
                
                # Translate to position
                rotated[:, 0] += px
                rotated[:, 1] += py
                
                # Draw rotated vehicle
                pts = rotated.astype(np.int32)
                cv2.fillPoly(img, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)
                
                # Draw windshield
                windshield_pts = np.array([
                    [pts[0]], [pts[1]], 
                    [(pts[0] + pts[1]) // 2 + (pts[2] - pts[1]) * 0.2]
                ]).astype(np.int32)
                windshield_color = tuple(int(c * 0.5) for c in color)
                cv2.fillPoly(img, windshield_pts, windshield_color)
            else:
                # Static vehicle (axis-aligned)
                cv2.rectangle(img,
                             (px - vehicle_width//2, py - vehicle_length//2),
                             (px + vehicle_width//2, py + vehicle_length//2),
                             color, -1)
                cv2.rectangle(img,
                             (px - vehicle_width//2, py - vehicle_length//2),
                             (px + vehicle_width//2, py + vehicle_length//2),
                             color, 2)
            
        elif object_type == 'pedestrian':
            # Draw pedestrian as circle
            radius = int(0.5 * self.scale)
            cv2.circle(img, (px, py), radius, color, -1)
            cv2.circle(img, (px, py), radius + 2, color, 2)
            
        else:
            # Unknown object as square
            size = int(1.5 * self.scale)
            cv2.rectangle(img,
                         (px - size//2, py - size//2),
                         (px + size//2, py + size//2),
                         color, -1)
            cv2.rectangle(img,
                         (px - size//2, py - size//2),
                         (px + size//2, py + size//2),
                         color, 2)
        
        # Draw velocity vector
        if speed > 0.1:  # Only draw if significant motion
            # Scale velocity for visualization
            vx_scaled = velocity[1] * self.scale * self.velocity_scale
            vy_scaled = -velocity[0] * self.scale * self.velocity_scale
            
            # Arrow end point
            end_x = int(px + vx_scaled)
            end_y = int(py + vy_scaled)
            
            # Ensure arrow stays within bounds
            end_x = np.clip(end_x, 0, self.image_size - 1)
            end_y = np.clip(end_y, 0, self.image_size - 1)
            
            # Draw arrow with bright color
            arrow_color = (255, 255, 0)  # Yellow for visibility
            cv2.arrowedLine(img, (px, py), (end_x, end_y), 
                          arrow_color, 2, tipLength=0.3)
            
            # Draw speed text
            speed_text = f"{speed:.1f}m/s"
            cv2.putText(img, speed_text, (px + 15, py - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, arrow_color, 1)
        
        # Draw labels
        cv2.putText(img, label_text, (px - 20, py + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw sensor mode
        sensor_color = self.sensor_colors.get(sensor_mode, (128, 128, 128))
        cv2.putText(img, sensor_mode.upper(), (px - 20, py + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, sensor_color, 1)
                   
    def _draw_legend(self, img):
        x, y = 10, 20
        cv2.putText(img, "Multi-Sensor Fusion (4D State with Velocity)", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Object type legend
        y += 30
        cv2.putText(img, "Object Types:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        
        # Vehicle (green)
        cv2.rectangle(img, (x + 5, y - 8), (x + 15, y - 2), self.object_colors['vehicle'], -1)
        cv2.putText(img, "Vehicle", (x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.object_colors['vehicle'], 1)
        y += 20
        
        # Pedestrian (red)
        cv2.circle(img, (x + 10, y - 5), 5, self.object_colors['pedestrian'], -1)
        cv2.putText(img, "Pedestrian", (x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.object_colors['pedestrian'], 1)
        y += 25
        
        # Velocity indicator
        cv2.arrowedLine(img, (x + 5, y), (x + 20, y), (255, 255, 0), 2, tipLength=0.3)
        cv2.putText(img, "Velocity Vector", (x + 25, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y += 25
        
        # Sensor mode legend
        cv2.putText(img, "Sensor Modes:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        
        for mode, color in self.sensor_colors.items():
            cv2.circle(img, (x + 10, y), 5, color, -1)
            cv2.putText(img, mode.capitalize(), (x + 25, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 20


def convert_to_multi_sensor_format(lidar_detections, camera_bboxes, timestamp):
    """
    Convert detection formats with timestamp support
    Now requires timestamp parameter
    """
    lidar_dets = []
    for detection in lidar_detections:
        if len(detection) == 3:  # New format with cluster size
            pos, bbox_3d, cluster_size = detection
        else:  # Old format
            pos, bbox_3d = detection
            cluster_size = None
            
        det = Detection(
            position=pos,
            timestamp=timestamp,
            centroid_3d=pos,
            label=None
        )
        if bbox_3d is not None:
            det.bbox_3d = bbox_3d
        if cluster_size is not None:
            det.cluster_size = cluster_size  # Store cluster size
        lidar_dets.append(det)
        
    camera_dets = []
    for bbox in camera_bboxes:
        x, y, w, h, label = bbox
        det = Detection(
            position=np.array([0, 0]),
            timestamp=timestamp,
            bbox=(x, y, w, h),
            label=label,
            confidence=0.8
        )
        camera_dets.append(det)
        
    return lidar_dets, camera_dets


# Aliases for compatibility (but now point to 4D versions)
MultiSensorEKF = MultiSensorEKF4D
MultiSensorTracker = MultiSensorTracker4D
MultiSensorBEVVisualizer = EnhancedBEVVisualizer


class PerformanceMonitor:
    """Performance monitoring with memory management"""
    
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
        self.start_time = time.time()
        
    def add_metric(self, metric_name, value):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
            
    def get_summary(self):
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values) * 1000,
                    'std': np.std(values) * 1000,
                    'max': np.max(values) * 1000,
                    'min': np.min(values) * 1000
                }
        return summary
        
    def create_performance_panel(self, width=400, height=350):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        cv2.putText(panel, "=== Performance Metrics (ms) ===", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 50
        summary = self.get_summary()
        
        for name, stats in summary.items():
            text = f"{name}: {stats['mean']:.1f} ± {stats['std']:.1f} (max: {stats['max']:.1f})"
            
            if stats['mean'] < 10:
                color = (0, 255, 0)
            elif stats['mean'] < 20:
                color = (255, 255, 0)
            else:
                color = (0, 0, 255)
                
            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 25
            
        runtime = time.time() - self.start_time
        total_frames = len(self.metrics['total'])
        avg_fps = total_frames / runtime if runtime > 0 else 0
        
        y_offset += 20
        cv2.putText(panel, f"Average FPS: {avg_fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(panel, f"Total Runtime: {runtime:.1f}s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        avg_total = summary.get('total', {}).get('mean', 0)
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
        
        # Add version 2.0 improvements info
        y_offset += 25
        cv2.putText(panel, "=== v2.0 Improvements ===", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(panel, "- 4D State [x, y, vx, vy]", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(panel, "- Async timestamp handling", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(panel, "- Grid-based temporal consistency", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(panel, "- Dynamic height estimation", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(panel, "- Velocity visualization", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return panel