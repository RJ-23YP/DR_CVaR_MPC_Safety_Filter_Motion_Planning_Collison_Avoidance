"""
Mathematical utilities for the DR-CVaR safety filtering implementation.
"""
import numpy as np

def normalize_vector(vector):
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm

def is_point_in_halfspace(point, h, g):
    """
    Check if a point is in a halfspace.
    
    Halfspace is defined as {x | h·x + g ≤ 0}
    
    Args:
        point: Point to check
        h: Normal vector of the halfspace
        g: Offset of the halfspace
    
    Returns:
        True if the point is in the halfspace
    """
    return np.dot(h, point) + g <= 0

def project_point_to_halfspace(point, h, g):
    """
    Project a point onto a halfspace.
    
    Args:
        point: Point to project
        h: Normal vector of the halfspace
        g: Offset of the halfspace
    
    Returns:
        Projected point
    """
    h_norm = normalize_vector(h)
    signed_dist = np.dot(h_norm, point) + g
    
    if signed_dist <= 0:
        # Point is already in the halfspace
        return point
    
    # Project the point onto the halfspace boundary
    return point - h_norm * signed_dist