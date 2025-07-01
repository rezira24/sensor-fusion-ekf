import carla
from queue import Queue, Empty


def setup_sensors(world, vehicle, image_w, image_h, fov):
    """
    Setup RGB camera, Semantic Segmentation camera, and LiDAR sensors attached to the vehicle
    
    Args:
        world: CARLA world object
        vehicle: Vehicle to attach sensors to
        image_w: Camera image width
        image_h: Camera image height
        fov: Camera field of view
        
    Returns:
        camera: RGB Camera sensor object
        semantic_cam: Semantic camera sensor object
        lidar: LiDAR sensor object
        image_queue: Queue for RGB camera data
        semantic_queue: Queue for semantic camera data
        lidar_queue: Queue for LiDAR data
    """
    blueprint_library = world.get_blueprint_library()
    
    # Initialize queues
    image_queue = Queue()
    semantic_queue = Queue()
    lidar_queue = Queue()
    
    # Shared transform
    sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    # RGB Camera setup
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(image_w))
    camera_bp.set_attribute("image_size_y", str(image_h))
    camera_bp.set_attribute("fov", str(fov))
    camera = world.spawn_actor(camera_bp, sensor_transform, attach_to=vehicle)
    camera.listen(lambda data: sensor_callback(data, image_queue))

    # Semantic Camera setup
    semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    semantic_bp.set_attribute("image_size_x", str(image_w))
    semantic_bp.set_attribute("image_size_y", str(image_h))
    semantic_bp.set_attribute("fov", str(fov))
    semantic_cam = world.spawn_actor(semantic_bp, sensor_transform, attach_to=vehicle)
    semantic_cam.listen(lambda data: sensor_callback(data, semantic_queue))
    
    # LiDAR setup
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '200000')
    lidar_transform = carla.Transform(carla.Location(x=1, z=2.5))  # Slightly higher
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar.listen(lambda data: sensor_callback(data, lidar_queue))
    
    return camera, semantic_cam, lidar, image_queue, semantic_queue, lidar_queue

def get_sensor_data(image_queue, semantic_queue, lidar_queue, timeout=1.0):
    try:
        image_data = image_queue.get(timeout=timeout)
        semantic_data = semantic_queue.get(timeout=timeout)
        lidar_data = lidar_queue.get(timeout=timeout)
        return image_data, semantic_data, lidar_data
    except Empty:
        return None, None, None


def sensor_callback(data, queue):
    """
    Callback function to put sensor data into queue
    
    Args:
        data: Sensor data
        queue: Queue to put data into
    """
    queue.put(data)


def cleanup_sensors(camera, semantic_cam, lidar):
    """
    Clean up sensors by destroying them
    
    Args:
        camera: Camera sensor object
        lidar: LiDAR sensor object
    """
    camera.destroy()
    semantic_cam.destroy()
    lidar.destroy()
    print("Sensors destroyed.")