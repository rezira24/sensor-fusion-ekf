import numpy as np
import cv2

def estimate_lane_lines(semantic_raw):
    """
    Detect lane lines from semantic segmentation.
    Optimized for performance.
    
    Args:
        semantic_raw: Semantic segmentation image
    
    Returns:
        An array of detected lines (x1, y1, x2, y2)
    """
    # Process on smaller image
    scale = 2
    small_sem = cv2.resize(semantic_raw, (semantic_raw.shape[1]//scale, semantic_raw.shape[0]//scale), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Step 1: Mask for lane marking labels (6 and 8)
    lane_mask = np.zeros_like(small_sem, dtype=np.uint8)
    lane_mask[(small_sem == 6) | (small_sem == 8)] = 255

    # Step 2: Edge detection
    edges = cv2.Canny(lane_mask, 50, 150)

    # Step 3: Line estimation with Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                          minLineLength=50, maxLineGap=10)
    
    # Rescale lines to original size
    if lines is not None:
        lines = np.squeeze(lines) * scale
        if lines.ndim == 1:  # If there's only one line
            lines = np.array([lines])
    
    return lines

def get_slope_intecept(lines):
    """
    Calculate slope and intercept from a set of lines.
    
    Args:
        lines: Array of lines [x1, y1, x2, y2]
    
    Returns:
        Slopes and intercepts
    """
    if lines.size == 0:
        return np.array([]), np.array([])
        
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    slopes = (y2 - y1) / (x2 - x1 + 1e-6)  # avoid division by zero
    intercepts = y1 - slopes * x1

    return slopes, intercepts


def merge_lane_lines(lines):
    """
    Merge similar lines to reduce duplicates
    
    Args:
        lines: Array of lines [x1, y1, x2, y2]
        
    Returns:
        Merged lines
    """
    if lines is None or lines.size == 0:
        return np.empty((0, 4))
        
    # Handle special case if there's only one line
    if lines.ndim == 1:
        return lines.reshape(1, 4)
        
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3

    slopes, intercepts = get_slope_intecept(lines)
    
    if slopes.size == 0:
        return np.empty((0, 4))

    valid_indices = np.where(np.abs(slopes) >= min_slope_threshold)[0]
    if len(valid_indices) == 0:
        return np.empty((0, 4))

    filtered_lines = lines[valid_indices]
    filtered_slopes = slopes[valid_indices]
    filtered_intercepts = intercepts[valid_indices]

    clusters = []
    assigned = np.zeros(len(filtered_lines), dtype=bool)

    for i in range(len(filtered_lines)):
        if assigned[i]:
            continue

        cluster = [i]
        assigned[i] = True

        for j in range(i + 1, len(filtered_lines)):
            if not assigned[j] and \
                abs(filtered_slopes[i] - filtered_slopes[j]) <= slope_similarity_threshold and \
                abs(filtered_intercepts[i] - filtered_intercepts[j]) <= intercept_similarity_threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    merged_lines = []
    for cluster in clusters:
        cluster_lines = filtered_lines[cluster]
        x1_avg = np.mean(cluster_lines[:, 0])
        y1_avg = np.mean(cluster_lines[:, 1])
        x2_avg = np.mean(cluster_lines[:, 2])
        y2_avg = np.mean(cluster_lines[:, 3])
        merged_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])

    return np.array(merged_lines) if len(merged_lines) > 0 else np.empty((0, 4))
