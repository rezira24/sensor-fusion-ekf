# Configuration parameters for LiDAR processing - FOCUSED ON EKF ERROR REDUCTION
# Only the most important tunable parameters that directly impact EKF accuracy

# ========================================
# CARLA CONNECTION (keep existing)
# ========================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 5.0

# ========================================
# CAMERA SETTINGS (keep existing)
# ========================================
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOV = 90

# ========================================
# CRITICAL PARAMETERS FOR EKF ERROR REDUCTION
# ========================================

# 1. LIDAR PREPROCESSING - Affects detection quality
VOXEL_SIZE = 0.125  # Smaller = more points but slower. Affects detection accuracy
GROUND_THRESHOLD = -2.3  # Too low = ground points included, too high = lose low objects

# 2. CLUSTERING - Critical for object detection
CLUSTERING_TOLERANCE = 0.65  # Key for motorcycle detection. Too small = split objects
MIN_CLUSTER_POINTS = 3  # Lower = detect small objects, but more false positives
MAX_CLUSTER_POINTS = 1000

# 3. EKF PROCESS NOISE - Directly affects tracking smoothness
EKF_PROCESS_NOISE_POSITION = 0.25  # Higher = trust measurements more, lower = smoother but laggy
EKF_PROCESS_NOISE_VELOCITY = 1.0  # Higher = faster velocity adaptation

# 4. EKF MEASUREMENT NOISE - Affects measurement trust
EKF_LIDAR_NOISE = 0.1  # Lower = trust LiDAR more (usually accurate)
EKF_CAMERA_NOISE = 25.0  # Higher = trust camera less (pixel noise)

# 5. ASSOCIATION - Critical for track continuity
ASSOCIATION_DISTANCE_THRESHOLD = 1.2  # Max distance to associate detection with track
FUSION_MAX_AGE = 15  # How long to keep tracks without detection
FUSION_MIN_HITS = 1  # Confirmations needed before track is valid

# 6. TEMPORAL CONSISTENCY - Reduces false positives
TEMPORAL_HISTORY_WINDOW = 10  # Detections needed in N frames
TEMPORAL_GRID_SIZE = 2.0  # Spatial resolution for consistency check

# 7. OBJECT SIZE FILTERS - Prevent impossible detections
# Motorcycle-specific (critical for close-range detection)
MOTORCYCLE_MIN_WIDTH = 0.4
MOTORCYCLE_MAX_WIDTH = 2.0
MOTORCYCLE_MIN_HEIGHT = 0.5
MOTORCYCLE_MAX_HEIGHT = 2.0

# General vehicle
MIN_BBOX_WIDTH = 0.3
MAX_BBOX_WIDTH = 6.0
MIN_BBOX_HEIGHT = 0.3  
MAX_BBOX_HEIGHT = 6.0

# 8. HIGH CONFIDENCE THRESHOLDS - Skip temporal check for obvious objects
HIGH_CONFIDENCE_MIN_CLUSTER_SIZE = 30  # Points - large objects tracked immediately
HIGH_CONFIDENCE_MIN_OBJECT_WIDTH = 1.5  # meters
HIGH_CONFIDENCE_MIN_OBJECT_LENGTH = 2.0  # meters

# ========================================
# BEV VISUALIZATION (keep existing)
# ========================================
BEV_IMAGE_SIZE = (600, 600)
BEV_SCALE = 10
BEV_MAX_RANGE = 60

# ========================================
# EVALUATION PARAMETERS
# ========================================
EVALUATION_DISTANCE_THRESHOLD = 2.5  # For matching with ground truth
EVALUATION_HISTORY_SIZE = 30  # Smoothing window

# ========================================
# TUNING TIPS:
# ========================================
# To reduce EKF RMSE:
# 1. Decrease VOXEL_SIZE (0.2 -> 0.1) for better resolution but slower processing
# 2. Tune CLUSTERING_TOLERANCE based on object size (0.5-1.0)
# 3. Reduce EKF_PROCESS_NOISE_* for smoother tracks (but may lag)
# 4. Decrease ASSOCIATION_DISTANCE_THRESHOLD for stricter matching
# 5. Increase TEMPORAL_HISTORY_WINDOW to reduce false positives
# 6. Adjust object size filters based on your target objects