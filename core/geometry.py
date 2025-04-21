"""
Geometric utilities for collision avoidance and safety filtering.
"""
import numpy as np

def support_function_circle(direction, radius):
    """
    Compute the support function of a circle.
    
    Args:
        direction: Direction vector (normalized)
        radius: Radius of the circle
    
    Returns:
        Support function value
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return 0.0
    return radius * norm

def minkowski_difference_circle_circle(radius_A, radius_B):
    """
    Compute the Minkowski difference of two circles.
    
    Args:
        radius_A: Radius of the first circle
        radius_B: Radius of the second circle
    
    Returns:
        Combined radius
    """
    return radius_A + radius_B

def compute_separating_vector(ego_pos, obstacle_pos):
    """
    Compute a separating vector between ego and obstacle.
    
    Args:
        ego_pos: Position of the ego robot
        obstacle_pos: Position of the obstacle
    
    Returns:
        h: Normalized separating vector
    """
    diff = obstacle_pos - ego_pos
    norm = np.linalg.norm(diff)
    
    if norm < 1e-10:
        # If positions are too close, use a default direction
        return np.array([1.0, 0.0])
    
    return diff / norm

def signed_distance(ego_pos, obstacle_pos, h, g_tilde):
    """
    Compute the signed distance for collision detection.
    
    Implements Equation 3 from the paper:
    ℓ(p, h, g) = -(h · p + g - (S_O(h) + S_(-A)(h)))
    
    Args:
        ego_pos: Position of the ego robot
        obstacle_pos: Position of the obstacle
        h: Normal vector of the halfspace
        g_tilde: Adjusted offset of the halfspace
    
    Returns:
        Signed distance (negative means no collision)
    """
    # Compute h · p + g_tilde
    projection = np.dot(h, obstacle_pos) + g_tilde
    
    # Negative signed distance (as defined in the paper)
    return -projection