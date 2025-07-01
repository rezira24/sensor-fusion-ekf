import numpy as np

def check_ground_truth_distance(world, vehicle):
    """
    Calculate ground truth distances from ego vehicle to all other vehicles in the scene
    
    Args:
        world: CARLA world object
        vehicle: Ego vehicle actor
        
    Returns:
        Dictionary of vehicle IDs with their positions and distances to ego vehicle
    """
    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_pos = np.array([ego_location.x, ego_location.y, ego_location.z])

    vehicles = world.get_actors().filter('vehicle.*')
    gt_distances = {}
    
    for v in vehicles:
        if v.id != vehicle.id:
            obj_loc = v.get_transform().location
            obj_pos = np.array([obj_loc.x, obj_loc.y, obj_loc.z])
            distance = np.linalg.norm(obj_pos[:2] - ego_pos[:2])  # XY distance only
            gt_distances[v.id] = (obj_pos, distance)
    
    return gt_distances