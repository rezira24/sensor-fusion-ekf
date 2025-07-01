import numpy as np
import cv2
from scipy.spatial import cKDTree
import time
import carla

from utils import VIRIDIS, VID_RANGE
from utils import COLOR_CAR, COLOR_PED

from config import (
    VOXEL_SIZE,  # Ini tidak ada di object6_detector tapi di preprocess_lidar
    GROUND_THRESHOLD,
    CLUSTERING_TOLERANCE,
    MIN_CLUSTER_POINTS,
    MAX_CLUSTER_POINTS
)

def get_object_bboxes_from_segmentation(segmentation_array, target_labels):
    """
    Get bounding boxes from semantic segmentation for specific labels.
    Optimized for performance.
    
    Args:
        segmentation_array: Semantic segmentation label array
        target_labels: List of semantic labels to detect (e.g. [4, 10] for pedestrians and vehicles)
    
    Returns:
        List of bounding boxes as (x, y, w, h, label)
    """
    # Process at smaller size
    scale = 2
    small_seg = cv2.resize(segmentation_array, 
                          (segmentation_array.shape[1]//scale, segmentation_array.shape[0]//scale), 
                          interpolation=cv2.INTER_NEAREST)
    
    bboxes = []
    min_area = 25  # Minimum area (already scaled)

    for label in target_labels:
        mask = (small_seg == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_area:
                continue
            # Rescale bbox to original size
            bboxes.append((x*scale, y*scale, w*scale, h*scale, label))

    return bboxes

def draw_bboxes_on_image(image, bboxes):
    """
    Draw bounding boxes on an OpenCV image.
    
    Args:
        image: OpenCV image to draw on
        bboxes: List of bounding boxes as (x, y, w, h, label)
    """
    for (x, y, w, h, label) in bboxes:
        color = COLOR_CAR if label == 10 else COLOR_PED
        label_name = "Vehicle" if label == 10 else "Pedestrian"
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def preprocess_lidar(lidar_data, ground_threshold=None, voxel_size=None):
    """
    Preprocess LiDAR data by filtering ground points and downsampling
    
    Args:
        lidar_data: Raw LiDAR data from CARLA
        ground_threshold: Z-coordinate threshold for ground removal
        voxel_size: Size of voxels for downsampling (0 to skip downsampling)
    
    Returns:
        points: Downsampled non-ground points (N, 3)
        intensity: Intensity values for downsampled points (N,)
        lidar_points: Original LiDAR points (M, 4)
        non_ground_mask: Boolean mask for non-ground points (M,)
    """

    if ground_threshold is None:
        ground_threshold = GROUND_THRESHOLD  # Ganti -2.3
    if voxel_size is None:
        voxel_size = VOXEL_SIZE  # Ganti 0.1    

    # Convert raw LiDAR data to numpy array
    lidar_points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    intensity = lidar_points[:, 3]
    xyz = lidar_points[:, :3]

    # Ground removal: filter points above ground_threshold
    non_ground_mask = xyz[:, 2] > ground_threshold
    points = xyz[non_ground_mask]
    intensity = intensity[non_ground_mask]

    # Downsampling: Using voxel size for downsampling
    if voxel_size > 0:
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        points = points[unique_indices]
        intensity = intensity[unique_indices]

    # Return results
    return points, intensity, lidar_points, non_ground_mask

def print_preprocessing_stats(raw_points, non_ground_mask, filtered_points, last_print_time):
    current_time = time.time()
    if current_time - last_print_time >= 3:
        # print("Initial point count:", raw_points.shape[0])
        # print("After ground removal:", np.sum(non_ground_mask))
        # print("After voxel downsampling:", filtered_points.shape[0])
        return current_time
    return last_print_time

def draw_lidar_projection(points, intensities, lidar_transform, camera_transform, image, K, image_w, image_h):
    """
    Project LiDAR points onto camera image
    
    Args:
        points: 3D LiDAR points (N, 3)
        intensities: Intensity values (N,)
        lidar_transform: CARLA transform for LiDAR sensor
        camera_transform: CARLA transform for camera
        image: Image to draw on
        K: Camera intrinsic matrix (3x3)
        image_w: Image width
        image_h: Image height
        
    Returns:
        Modified image with projected points
    """
    local_points = points.T
    local_points = np.vstack((local_points, np.ones(local_points.shape[1])))

    lidar_world = lidar_transform.get_matrix()
    camera_world_inv = camera_transform.get_inverse_matrix()
    points_world = np.dot(lidar_world, local_points)
    points_camera = np.dot(camera_world_inv, points_world)

    pc = np.array([points_camera[1], -points_camera[2], points_camera[0]])
    points_2d = np.dot(K, pc)
    points_2d = points_2d[:2] / points_2d[2]

    mask = (points_2d[0] >= 0) & (points_2d[0] < image_w) & \
           (points_2d[1] >= 0) & (points_2d[1] < image_h) & (pc[2] > 0)

    u, v = points_2d[0, mask].astype(np.int32), points_2d[1, mask].astype(np.int32)
    intensities = intensities[mask]
    colors = np.array([
        np.interp(intensities, VID_RANGE, VIRIDIS[:, 0]) * 255,
        np.interp(intensities, VID_RANGE, VIRIDIS[:, 1]) * 255,
        np.interp(intensities, VID_RANGE, VIRIDIS[:, 2]) * 255]).astype(np.uint8).T

    for i in range(len(u)):
        if 1 <= u[i] < image_w - 1 and 1 <= v[i] < image_h - 1:
            image[v[i]-1:v[i]+1, u[i]-1:u[i]+1] = colors[i]
    return image

def euclidean_clustering(points, tolerance=None, min_points=None, max_points=None):
    """
    Perform Euclidean clustering on 3D points
    
    Args:
        points: 3D points (N, 3)
        tolerance: Distance threshold for clustering
        min_points: Minimum points per cluster
        max_points: Maximum points per cluster
        
    Returns:
        List of clusters, where each cluster is a list of point indices
    """

    if tolerance is None:
        tolerance = CLUSTERING_TOLERANCE  # Ganti 0.5
    if min_points is None:
        min_points = MIN_CLUSTER_POINTS  # Ganti 8
    if max_points is None:
        max_points = MAX_CLUSTER_POINTS  # Ganti 1000

    tree = cKDTree(points)
    clusters = []
    visited = np.zeros(points.shape[0], dtype=bool)

    for idx in range(points.shape[0]):
        if visited[idx]:
            continue
        queue = [idx]
        visited[idx] = True
        cluster = []

        while queue:
            current_idx = queue.pop()
            cluster.append(current_idx)
            neighbors = tree.query_ball_point(points[current_idx], tolerance)
            for n_idx in neighbors:
                if not visited[n_idx]:
                    visited[n_idx] = True
                    queue.append(n_idx)

        if min_points <= len(cluster) <= max_points:
            clusters.append(cluster)

    return clusters


def project_3d_bounding_boxes(image, clusters, points, lidar, camera_transform, K, world, image_w, image_h):
    """
    Project 3D bounding boxes of detected objects onto the camera image
    
    Args:
        image: Image to draw on
        clusters: List of clusters
        points: 3D LiDAR points (N, 3)
        lidar: CARLA LiDAR sensor object
        camera_transform: CARLA transform for camera
        K: Camera intrinsic matrix
        world: CARLA world object
        image_w: Image width
        image_h: Image height
        
    Returns:
        Image with projected bounding boxes and distance information
    """
    # Get the transform from lidar object
    lidar_transform = lidar.get_transform()
    
    # Get ego vehicle transform (lidar's parent)
    ego_vehicle = lidar.parent
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_pos = np.array([ego_location.x, ego_location.y, ego_location.z])
    
    # Transform matrices
    lidar_world = lidar_transform.get_matrix()
    camera_world_inv = camera_transform.get_inverse_matrix()
    
    # Get ground truth distances from distance_calculator
    from distance_calculator import check_ground_truth_distance
    gt_distances = check_ground_truth_distance(world, ego_vehicle)
    
    for cluster_idx, cluster in enumerate(clusters):
        cluster_points = points[cluster]
        
        if len(cluster_points) < 5:  # Skip very small clusters
            continue
            
        # Calculate bounding box in lidar space
        x_min, y_min, z_min = cluster_points.min(axis=0)
        x_max, y_max, z_max = cluster_points.max(axis=0)

        # Create box corners
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max]
        ])

        # Transform corners to world space
        box_hom = np.hstack((corners, np.ones((8, 1)))).T
        corners_world = np.dot(lidar_world, box_hom)
        
        # Calculate cluster centroid in world space
        cluster_center_lidar = ((x_min + x_max) / 2, 
                            (y_min + y_max) / 2, 
                            (z_min + z_max) / 2)
        centroid_lidar = np.array(cluster_center_lidar)

        centroid_hom = np.hstack((centroid_lidar.reshape(1, 3), np.ones((1, 1)))).T
        centroid_world = np.dot(lidar_world, centroid_hom)
        
        # Calculate distance from ego vehicle to cluster centroid in world space (XY plane only)
        cluster_world_pos = centroid_world[:3, 0]
        distance = np.linalg.norm(cluster_world_pos[:2] - ego_pos[:2])
        
        # Project the bounding box to camera space
        points_camera = np.dot(camera_world_inv, corners_world)
        
        # Transform coordinates for camera projection
        pc = np.array([points_camera[1], -points_camera[2], points_camera[0]])
        
        # Check if box is in front of camera
        if np.any(pc[2] <= 0):
            continue  # Skip if any point is behind camera
            
        # Project to 2D image plane
        pts_2d = np.dot(K, pc)
        pts_2d = pts_2d[:2] / pts_2d[2]
        
        # Check for NaN or infinite values before casting
        if np.any(np.isnan(pts_2d)) or np.any(np.isinf(pts_2d)):
            continue  # Skip this cluster if projection contains invalid values
            
        # Safely cast to int32
        pts = pts_2d.T.astype(np.int32)
        
        # Create dictionary to store valid points
        valid_points = {}
        for i in range(len(pts)):
            if 0 <= pts[i][0] < image_w and 0 <= pts[i][1] < image_h:
                valid_points[i] = tuple(pts[i])
        
        # Skip cluster if too few valid points
        if len(valid_points) < 4:
            continue
        
        # Draw bounding box lines only if both endpoints are valid
        box_edges = [(0,1),(1,3),(3,2),(2,0), (4,5),(5,7),(7,6),(6,4), (0,4),(1,5),(2,6),(3,7)]
        for i, j in box_edges:
            if i in valid_points and j in valid_points:
                cv2.line(image, valid_points[i], valid_points[j], color=(0,255,0), thickness=2)
        
        # Project centroid to camera space
        centroid_camera = np.dot(camera_world_inv, centroid_world)
        coord = np.array([
            centroid_camera[1, 0],
            -centroid_camera[2, 0],
            centroid_camera[0, 0]
        ])
        
        if coord[2] > 0:
            # Project centroid to 2D image plane
            center_2d = np.dot(K, coord)
            center_2d = (center_2d[:2] / center_2d[2])
            
            # Check for valid values before casting
            if not np.any(np.isnan(center_2d)) and not np.any(np.isinf(center_2d)):
                center_2d = center_2d.astype(np.int32)
                
                # Check if centroid is within frame
                if 0 <= center_2d[0] < image_w and 0 <= center_2d[1] < image_h:
                    # Draw centroid and distance
                    cv2.circle(image, tuple(center_2d), 4, (0, 0, 255), -1)
                    cv2.putText(image, f"{distance:.1f} m", tuple(center_2d),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    
                    # Find closest ground truth vehicle to this cluster
                    closest_gt = None
                    min_gt_dist = float('inf')
                    
                    # Check distance to all ground truth vehicles
                    for vehicle_id, (obj_pos, gt_distance) in gt_distances.items():
                        dist_to_cluster = np.linalg.norm(obj_pos[:2] - cluster_world_pos[:2])
                        if dist_to_cluster < min_gt_dist and dist_to_cluster < 5.0:  # 5m threshold
                            min_gt_dist = dist_to_cluster
                            closest_gt = gt_distance
                    
                    # Show ground truth distance if found
                    if closest_gt is not None:
                        cv2.putText(image, f"GT: {closest_gt:.2f} m", 
                                    (center_2d[0], center_2d[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image

# Add these functions to object6_detector.py

def create_lidar_depth_map(points, lidar_transform, camera_transform, K, image_w, image_h):
    """
    Create a depth map from LiDAR points using camera projection.
    MODIFIED: Ensures a black background for unprojected areas.
    
    Args:
        points: 3D LiDAR points (N, 3)
        lidar_transform: CARLA transform for LiDAR sensor
        camera_transform: CARLA transform for camera
        K: Camera intrinsic matrix (3x3)
        image_w: Image width
        image_h: Image height
        
    Returns:
        Grayscale depth map image with a black background.
    """
    # Create empty depth map (black background) and depth buffer
    depth_map = np.zeros((image_h, image_w), dtype=np.uint8) # Changed to uint8 for direct use
    depth_buffer = np.full((image_h, image_w), np.inf, dtype=np.float32)
    
    # Transform points from LiDAR to camera coordinates
    local_points = points.T
    local_points = np.vstack((local_points, np.ones(local_points.shape[1])))
    
    lidar_world = lidar_transform.get_matrix()
    camera_world_inv = camera_transform.get_inverse_matrix()
    points_world = np.dot(lidar_world, local_points)
    points_camera = np.dot(camera_world_inv, points_world)
    
    # Camera coordinates
    pc = np.array([points_camera[1], -points_camera[2], points_camera[0]])
    
    # Get distances (depth) before projection
    distances = points_camera[0]  # X coordinate in camera space is depth
    
    # Project to 2D
    points_2d = np.dot(K, pc)
    # Add a small epsilon to avoid division by zero
    points_2d = points_2d[:2] / (points_2d[2] + 1e-6)
    
    # Filter valid projections
    mask = (points_2d[0] >= 0) & (points_2d[0] < image_w) & \
           (points_2d[1] >= 0) & (points_2d[1] < image_h) & (pc[2] > 0)
    
    u, v = points_2d[0, mask].astype(np.int32), points_2d[1, mask].astype(np.int32)
    valid_distances = distances[mask]
    
    # Update depth buffer (keep closest point per pixel)
    for i in range(len(u)):
        if valid_distances[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = valid_distances[i]
    
    # ================= MODIFIKASI DIMULAI DI SINI =================
    
    # Create a mask for pixels that have valid depth information
    valid_mask = depth_buffer < np.inf
    
    # Get only the valid depth values
    valid_depths = depth_buffer[valid_mask]
    
    # Define normalization range
    max_range = 50.0  # Maximum range in meters
    min_range = 0.5   # Minimum range to avoid division issues
    
    # Clip and normalize only the valid depth values
    # Closer points will be brighter (white), farther points will be darker (gray)
    valid_depths = np.clip(valid_depths, min_range, max_range)
    normalized_values = 255 - ((valid_depths - min_range) / (max_range - min_range) * 255)
    
    # Place the normalized values back into the depth map using the mask
    # The background will remain black (0)
    depth_map[valid_mask] = 255 # normalized_values.astype(np.uint8)
    
    # ================== MODIFIKASI SELESAI ===================
    
    # Apply slight Gaussian blur to fill small gaps
    # depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
    
    # Convert to BGR for consistency
    depth_map_bgr = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    
    return depth_map_bgr


def draw_ekf_tracks_on_camera(image, tracks, vehicle_transform, camera_transform, K):
    """
    Draw EKF tracked objects on camera image with proper 3D projection
    
    Args:
        image: Camera RGB image to draw on
        tracks: List of EKF tracks from sensor fusion
        vehicle_transform: Ego vehicle transform
        camera_transform: Camera sensor transform
        K: Camera intrinsic matrix
        
    Returns:
        Image with EKF tracks visualized
    """
    # Copy image to draw on
    result = image.copy()
    
    # Standard object dimensions (in meters)
    object_dimensions = {
        10: {'width': 2.0, 'length': 4.5, 'height': 1.5, 'name': 'Vehicle', 'color': (0, 255, 0)},
        'Vehicle': {'width': 2.0, 'length': 4.5, 'height': 1.5, 'name': 'Vehicle', 'color': (0, 255, 0)},
        4: {'width': 0.6, 'length': 0.6, 'height': 1.8, 'name': 'Pedestrian', 'color': (0, 0, 255)},
        'Pedestrian': {'width': 0.6, 'length': 0.6, 'height': 1.8, 'name': 'Pedestrian', 'color': (0, 0, 255)},
        'unknown': {'width': 1.0, 'length': 1.0, 'height': 1.0, 'name': 'Unknown', 'color': (128, 128, 128)}
    }
    
    # Get camera world inverse transform
    camera_world_inv = camera_transform.get_inverse_matrix()
    
    for track in tracks:
        # Get track info
        track_id = track['id']
        position = track['position']  # [forward, lateral] in vehicle coordinates
        speed = track.get('speed', 0)
        label = track.get('label', 'unknown')
        
        # Get object dimensions
        if label in object_dimensions:
            dims = object_dimensions[label]
        else:
            dims = object_dimensions['unknown']
        
        # Convert track position from vehicle-relative to world coordinates
        # Track position is [forward, lateral] in vehicle frame
        relative_location = carla.Location(
            x=position[0],  # Forward
            y=position[1],  # Lateral (right positive)
            z=0.0  # Ground level
        )
        
        # Transform to world coordinates
        world_location = vehicle_transform.transform(relative_location)
        
        # Create 3D bounding box corners in world space
        half_width = dims['width'] / 2
        half_length = dims['length'] / 2
        height = dims['height']
        
        # Get vehicle yaw for rotating the box
        yaw = vehicle_transform.rotation.yaw * np.pi / 180
        
        # Box corners in local coordinates (before rotation)
        local_corners = [
            [-half_length, -half_width, 0],      # Back-left-bottom
            [half_length, -half_width, 0],       # Front-left-bottom
            [half_length, half_width, 0],        # Front-right-bottom
            [-half_length, half_width, 0],       # Back-right-bottom
            [-half_length, -half_width, height], # Back-left-top
            [half_length, -half_width, height],  # Front-left-top
            [half_length, half_width, height],   # Front-right-top
            [-half_length, half_width, height]   # Back-right-top
        ]
        
        # Rotate corners based on vehicle yaw (assuming objects aligned with road)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        world_corners = []
        for corner in local_corners:
            # Rotate around Z axis
            rotated_x = corner[0] * cos_yaw - corner[1] * sin_yaw
            rotated_y = corner[0] * sin_yaw + corner[1] * cos_yaw
            
            # Translate to world position
            world_corner = [
                world_location.x + rotated_x,
                world_location.y + rotated_y,
                world_location.z + corner[2]
            ]
            world_corners.append(world_corner)
        
        # Convert to homogeneous coordinates
        world_corners_hom = np.array([c + [1] for c in world_corners]).T
        
        # Transform to camera space
        camera_corners = camera_world_inv @ world_corners_hom
        
        # Project to image
        cam_coords = np.array([
            camera_corners[1],    # Y -> X
            -camera_corners[2],   # -Z -> Y
            camera_corners[0]     # X -> Z (depth)
        ])
        
        # Check if object is in front of camera
        if np.all(cam_coords[2] > 0):
            # Project to 2D
            image_coords = K @ cam_coords
            image_coords = image_coords[:2] / image_coords[2]
            
            # Check if any corner is visible
            u_coords = image_coords[0]
            v_coords = image_coords[1]
            
            if (np.any((u_coords >= 0) & (u_coords < image.shape[1])) and 
                np.any((v_coords >= 0) & (v_coords < image.shape[0]))):
                
                # Get 2D bounding box that encloses all corners
                u_min = int(np.clip(np.min(u_coords), 0, image.shape[1]-1))
                u_max = int(np.clip(np.max(u_coords), 0, image.shape[1]-1))
                v_min = int(np.clip(np.min(v_coords), 0, image.shape[0]-1))
                v_max = int(np.clip(np.max(v_coords), 0, image.shape[0]-1))
                
                # Draw bounding box
                color = dims['color']
                cv2.rectangle(result, (u_min, v_min), (u_max, v_max), color, 2)
                
                # Draw label with ID and speed
                label_text = f"{dims['name']}_{track_id}"
                speed_text = f"{speed:.1f} m/s"
                
                # Background for text
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(result, (u_min, v_min - 20), 
                            (u_min + text_size[0] + 4, v_min - 2), color, -1)
                
                # Draw text
                cv2.putText(result, label_text, (u_min + 2, v_min - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(result, speed_text, (u_min + 2, v_min - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw 3D box edges if close enough
                distance = np.mean(cam_coords[2])
                if distance < 30:  # Only for objects closer than 30m
                    edges = [(0,1),(1,2),(2,3),(3,0),  # Bottom edges
                            (4,5),(5,6),(6,7),(7,4),   # Top edges
                            (0,4),(1,5),(2,6),(3,7)]   # Vertical edges
                    
                    for i, j in edges:
                        pt1 = (int(image_coords[0, i]), int(image_coords[1, i]))
                        pt2 = (int(image_coords[0, j]), int(image_coords[1, j]))
                        
                        # Check if both points are within image bounds
                        if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                            0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                            cv2.line(result, pt1, pt2, color, 1)
    
    return result