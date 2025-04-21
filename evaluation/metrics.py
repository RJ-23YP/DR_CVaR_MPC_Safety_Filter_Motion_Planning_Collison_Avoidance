"""
Safety evaluation metrics for the DR-CVaR safety filtering.
"""
import numpy as np

def collision_rate(distances):
    """
    Compute the collision rate from distances to collision.
    
    Args:
        distances: Array of minimum distances to collision for each run
    
    Returns:
        Collision rate (between 0 and 1)
    """
    return np.mean(distances < 0)

def expectation_of_shortfall(distances, threshold=0):
    """
    Compute the expected shortfall (average distance below threshold).
    
    Args:
        distances: Array of minimum distances to collision for each run
        threshold: Threshold for collision (typically 0)
    
    Returns:
        Expected shortfall (negative value indicates average collision depth)
    """
    shortfalls = distances[distances < threshold]
    if len(shortfalls) == 0:
        return 0.0
    return np.mean(shortfalls - threshold)

def safety_metrics(distances, threshold=0):
    """
    Compute a set of safety metrics from distances to collision.
    
    Args:
        distances: Array of minimum distances to collision for each run
        threshold: Threshold for collision (typically 0)
    
    Returns:
        Dict of safety metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = np.mean(distances)
    metrics['min'] = np.min(distances)
    metrics['max'] = np.max(distances)
    metrics['std'] = np.std(distances)
    
    # Collision rate
    metrics['collision_rate'] = collision_rate(distances)
    
    # Expected shortfall
    metrics['expected_shortfall'] = expectation_of_shortfall(distances, threshold)
    
    # Quantiles
    metrics['q10'] = np.percentile(distances, 10)  # 10th percentile
    metrics['q25'] = np.percentile(distances, 25)  # 25th percentile
    metrics['median'] = np.median(distances)       # 50th percentile
    metrics['q75'] = np.percentile(distances, 75)  # 75th percentile
    metrics['q90'] = np.percentile(distances, 90)  # 90th percentile
    
    return metrics