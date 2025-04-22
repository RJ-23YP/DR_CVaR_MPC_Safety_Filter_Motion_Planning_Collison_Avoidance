"""
Implementation of safe halfspaces for collision avoidance.
"""
import numpy as np
import cvxpy as cp
from utils.timing import timeit, Timer
from core.geometry import compute_separating_vector
from core.risk_metrics import dr_cvar_halfspace, cvar_halfspace
import json

class SafeHalfspace:
    """
    Base class for safe halfspaces.
    
    A safe halfspace is defined as {y | h·y + g ≤ 0} where:
    - h is the normal vector (pointing from ego to obstacle)
    - g is the offset that accounts for geometry and risk
    """
    def __init__(self, h, g_tilde):
        """
        Initialize a safe halfspace.
        
        Args:
            h: Normal vector of the halfspace
            g_tilde: Adjusted offset of the halfspace
        """
        self.h = h
        self.g_tilde = g_tilde
        self.info = None
    
    def is_point_safe(self, point):
        """
        Check if a point is in the safe halfspace.
        
        Args:
            point: Point to check
        
        Returns:
            True if the point is in the safe halfspace
        """
        return np.dot(self.h, point) + self.g_tilde <= 0
    
    def distance_to_boundary(self, point):
        """
        Compute the signed distance from a point to the halfspace boundary.
        
        Args:
            point: Point to check
        
        Returns:
            Signed distance (negative if point is in the halfspace)
        """
        h_norm = self.h / np.linalg.norm(self.h)
        return np.dot(h_norm, point) + self.g_tilde / np.linalg.norm(self.h)
    
    def get_constraint_params(self):
        """
        Get the parameters for the halfspace constraint in the form h·y + g ≤ 0.
        
        Returns:
            h: Normal vector
            g: Offset
        """
        return self.h, self.g_tilde

class MeanSafeHalfspace(SafeHalfspace):
    """
    Safe halfspace based on the mean value of obstacle positions.
    """
    @staticmethod
    def create(samples, robot_radius, obstacle_radius):
        """
        Create a safe halfspace based on mean obstacle position.
        
        Args:
            samples: Array of obstacle position samples
            robot_radius: Radius of the ego robot
            obstacle_radius: Radius of the obstacle
        
        Returns:
            MeanSafeHalfspace object
        """
        # Compute the mean obstacle position
        mean_pos = np.mean(samples, axis=0)
        
        # Compute the separating vector (from ego to obstacle)
        # We use (0,0) as the reference ego position
        h = compute_separating_vector(np.zeros(2), mean_pos) 
        
        # Compute the combined radius term
        combined_radius = robot_radius + obstacle_radius
        
        # Compute g_tilde directly (no optimization needed)
        g_tilde = -(np.dot(h, mean_pos) - combined_radius * np.linalg.norm(h))
        
        # Create and return halfspace
        halfspace = MeanSafeHalfspace(h, g_tilde)
        
        # Add timing info for consistency (all zeros since this is analytical)
        halfspace.info = {
            'setup_time': 0,
            'solve_time': 0,
            'solve_call_time': 0
        }
        
        return halfspace

class CVaRSafeHalfspace(SafeHalfspace):
    """
    Safe halfspace based on the CVaR of obstacle positions.
    """
    @staticmethod
    @timeit
    def create(samples, ego_ref_pos, alpha, delta, robot_radius, obstacle_radius):
        """
        Create a safe halfspace based on CVaR.
        
        Args:
            samples: Array of obstacle position samples
            ego_ref_pos: Reference position of the ego robot
            alpha: CVaR confidence level
            delta: Risk bound
            robot_radius: Radius of the ego robot
            obstacle_radius: Radius of the obstacle
        
        Returns:
            CVaRSafeHalfspace object
        """
        # Compute the separating vector
        h = compute_separating_vector(ego_ref_pos, np.mean(samples, axis=0))
        
        # Use the optimized CVaR halfspace function
        with Timer("CVaR Optimization"):
            g_value = cvar_halfspace(
                samples, h, alpha, delta, robot_radius, obstacle_radius
            )
        
        # Create the halfspace
        halfspace = CVaRSafeHalfspace(h, g_value)
        
        # Get timing info from the file
        try:
            with open('tmp/timing_info_cvar.json', 'r') as f:
                timing_info = json.load(f)
                halfspace.info = timing_info
        except Exception:
            pass
            
        return halfspace

class DRCVaRSafeHalfspace(SafeHalfspace): 
    """
    Safe halfspace based on the Distributionally Robust CVaR.
    """
    @staticmethod
    @timeit
    def create(samples, ego_ref_pos, alpha, delta, epsilon, robot_radius, obstacle_radius):
        """
        Create a safe halfspace based on DR-CVaR.
        
        Args:
            samples: Array of obstacle position samples
            ego_ref_pos: Reference position of the ego robot
            alpha: CVaR confidence level
            delta: Risk bound
            epsilon: Wasserstein radius
            robot_radius: Radius of the ego robot
            obstacle_radius: Radius of the obstacle
        
        Returns:
            DRCVaRSafeHalfspace object
        """
        # Compute the separating vector
        h = compute_separating_vector(ego_ref_pos, np.mean(samples, axis=0))
        
        # Compute DR-CVaR halfspace using our optimized function
        with Timer("DR-CVaR Optimization"):
            g_star, g_tilde = dr_cvar_halfspace(
                samples, h, alpha, delta, epsilon, 
                robot_radius, obstacle_radius
            )
        
        # Create the halfspace
        halfspace = DRCVaRSafeHalfspace(h, g_tilde)
        
        # Get timing info from the file
        try:
            with open('tmp/timing_info_drcvar.json', 'r') as f:
                timing_info = json.load(f)
                halfspace.info = timing_info
        except Exception:
            pass
            
        return halfspace

def compute_safe_halfspaces(obstacle_samples, ego_ref_pos, robot_radius, obstacle_radius, alpha, delta, epsilon):
    """
    Compute safe halfspaces for a list of obstacle samples.
    
    Args:
        obstacle_samples: List of obstacle position samples [obstacle_idx][sample_idx]
        ego_ref_pos: Reference position of the ego robot
        robot_radius: Radius of the ego robot
        obstacle_radius: Radius of the obstacle
        alpha: CVaR confidence level
        delta: Risk bound
        epsilon: Wasserstein radius
    
    Returns:
        Dict of safe halfspaces for each obstacle:
        {
            'mean': [MeanSafeHalfspace objects],
            'cvar': [CVaRSafeHalfspace objects],
            'dr_cvar': [DRCVaRSafeHalfspace objects]
        }
    """
    n_obstacles = len(obstacle_samples)
    
    safe_halfspaces = {
        'mean': [],
        'cvar': [],
        'dr_cvar': []
    }
    
    for i in range(n_obstacles):
        samples = obstacle_samples[i]
        
        # Compute mean safe halfspace
        mean_halfspace = MeanSafeHalfspace.create(
            samples, robot_radius, obstacle_radius
        )
        safe_halfspaces['mean'].append(mean_halfspace)
        
        # Compute CVaR safe halfspace
        cvar_halfspace = CVaRSafeHalfspace.create(
            samples, ego_ref_pos, alpha, delta,
            robot_radius, obstacle_radius
        )
        safe_halfspaces['cvar'].append(cvar_halfspace)
        
        # Compute DR-CVaR safe halfspace
        dr_cvar_halfspace = DRCVaRSafeHalfspace.create(
            samples, ego_ref_pos, alpha, delta, epsilon,
            robot_radius, obstacle_radius
        )
        safe_halfspaces['dr_cvar'].append(dr_cvar_halfspace)
    
    return safe_halfspaces 