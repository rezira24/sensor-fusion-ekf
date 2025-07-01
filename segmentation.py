import numpy as np

def visualize_semantic_segmentation(semantic_array, height, width, mapping_colors):
    """
    Create a colored visualization of semantic segmentation results
    
    Args:
        semantic_array: The semantic segmentation label array
        height: Image height
        width: Image width
        mapping_colors: Dictionary mapping label IDs to colors
    
    Returns:
        Colored visualization of semantic segmentation
    """
    # Create empty RGB image
    seg_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set colors according to semantic labels
    for label_id, color in mapping_colors.items():
        seg_vis[semantic_array == label_id] = color
    
    return seg_vis
