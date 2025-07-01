import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from matplotlib import cm


# Color mapping for semantic segmentation visualization
MAPPING_COLORS = {
    0: (0, 0, 0),        # None
    1: (0, 0, 255),      # Buildings
    2: (64, 64, 128),    # Fences
    3: (64, 0, 128),     # Other
    4: (128, 128, 0),    # Pedestrians
    5: (255, 255, 255),  # Poles
    6: (128, 0, 128),    # RoadLines
    7: (255, 0, 0),      # Roads
    8: (0, 255, 255),    # Sidewalks
    9: (0, 128, 0),      # Vegetation
    10: (0, 255, 0),     # Vehicles
    11: (128, 128, 128), # Walls
    12: (255, 255, 0),   # TrafficSigns
    13: (0, 0, 128),     # Sky
    14: (128, 64, 0),    # Ground
    15: (0, 128, 128),   # Bridge
    16: (128, 0, 0),     # RailTrack
    17: (64, 64, 0),     # GuardRail
    18: (255, 0, 255),   # TrafficLight
    19: (64, 0, 64),     # Static
    20: (192, 128, 128), # Dynamic
    21: (0, 64, 64),     # Water
    22: (64, 192, 0),    # Terrain
}

# Bounding box colors
COLOR_CAR = (0, 0, 255)        # Green
COLOR_PED = (0, 255, 255)      # Yellow

# ==== Global Config ====
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def get_intrinsic_matrix(image_w, image_h, fov):
    """Calculate camera intrinsic matrix based on image dimensions and field of view."""
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K

def get_vehicle(world):
    while True:
        vehicles = world.get_actors().filter("vehicle.*")
        if vehicles:
            vehicle = vehicles[0]
            print(f"Vehicle found: {vehicle.type_id}")
            return vehicle
        print("Looking for vehicle...")
        time.sleep(0.5)

def process_image(image_data):
    return np.frombuffer(image_data.raw_data, dtype=np.uint8).reshape(
        (image_data.height, image_data.width, 4))[:, :, :3][:, :, ::-1].copy()