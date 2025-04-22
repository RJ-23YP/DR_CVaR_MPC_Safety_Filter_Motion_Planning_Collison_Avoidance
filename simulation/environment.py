"""
Simulation environment for testing the DR-CVaR safety filtering.
"""
import numpy as np
from core.dynamics import create_double_integrator_matrices
from core.halfspaces import compute_safe_halfspaces

class SafetyFilteringEnvironment:
    """
    Environment for safety filtering simulations.
    """
    def __init__(self, ROBOT_RADIUS, OBSTACLE_RADIUS, HORIZON, DT, ALPHA, DELTA, EPSILON):
        """
        Initialize the environment.
        
        Args:
            ROBOT_RADIUS: Radius of the ego robot
            OBSTACLE_RADIUS: Radius of obstacles
            HORIZON: MPC horizon
            DT: Time step
            ALPHA: CVaR confidence level
            DELTA: Risk bound
            EPSILON: Wasserstein radius
        """
        self.ROBOT_RADIUS = ROBOT_RADIUS
        self.OBSTACLE_RADIUS = OBSTACLE_RADIUS
        self.HORIZON = HORIZON
        self.DT = DT
        self.ALPHA = ALPHA
        self.DELTA = DELTA
        self.EPSILON = EPSILON 
        
        # Create system matrices
        self.A, self.B, self.C = create_double_integrator_matrices(DT) 
        
        # State and input dimensions
        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        self.n_outputs = self.C.shape[0] 
        
        # # Weight matrices for MPC
        # self.Q = params.Q_WEIGHT * np.eye(self.n_states)
        # self.R = params.R_WEIGHT * np.eye(self.n_inputs) 
        
        # Environment bounds
        self.state_bounds = None
        self.input_bounds = None
    
    def set_bounds(self, state_bounds=None, input_bounds=None):
        """
        Set bounds for the environment.
        
        Args:
            state_bounds: (min, max) tuple for states
            input_bounds: (min, max) tuple for inputs
        """
        self.state_bounds = state_bounds
        self.input_bounds = input_bounds
    
    def compute_safe_halfspaces_for_trajectory(self, obstacle_sample_trajectories, ego_ref_trajectory):
        """
        Compute safe halfspaces for a given reference trajectory.
        
        Args:
            obstacle_sample_trajectories: List of obstacle sample trajectories
            ego_ref_trajectory: Reference trajectory for the ego robot
        
        Returns:
            dict: Safe halfspaces for each time step and risk metric
        """
        n_obstacles = len(obstacle_sample_trajectories)
        n_steps = min(len(ego_ref_trajectory), self.HORIZON) 
        
        # Initialize safe halfspaces
        safe_halfspaces = {
            'mean': [[] for _ in range(n_steps)],
            'cvar': [[] for _ in range(n_steps)],
            'dr_cvar': [[] for _ in range(n_steps)]
        }
        
        # Compute safe halfspaces for each time step
        for t in range(n_steps):
            # Extract obstacle samples at time t
            obstacle_samples_t = []
            
            for i in range(n_obstacles):
                # Extract samples for obstacle i at time step t
                samples_i_t = obstacle_sample_trajectories[i][:, t, :]
                obstacle_samples_t.append(samples_i_t)
            
            # Extract ego reference position at time t
            ego_ref_pos_t = self.C @ ego_ref_trajectory[t]
            
            # Compute safe halfspaces
            halfspaces_t = compute_safe_halfspaces(
                obstacle_samples_t, ego_ref_pos_t, 
                self.ROBOT_RADIUS, self.OBSTACLE_RADIUS,
                self.ALPHA, self.DELTA, self.EPSILON
            )
            
            # Store the safe halfspaces
            safe_halfspaces['mean'][t] = halfspaces_t['mean']
            safe_halfspaces['cvar'][t] = halfspaces_t['cvar']
            safe_halfspaces['dr_cvar'][t] = halfspaces_t['dr_cvar']
        
        return safe_halfspaces
    
    def compute_distance_to_collision(self, ego_trajectory, obstacle_trajectories):
        """
        Compute the distance to collision between the ego robot and obstacles.
        
        Args:
            ego_trajectory: Ego robot trajectory
            obstacle_trajectories: Obstacle trajectories
        
        Returns:
            distances: Array of distances to collision over time
        """
        n_steps = min(len(ego_trajectory), len(obstacle_trajectories[0]))
        n_obstacles = len(obstacle_trajectories)
        
        # Initialize distances
        distances = np.inf * np.ones(n_steps)
        
        # Compute distances for each time step
        for t in range(n_steps):
            # Extract ego position at time t
            ego_pos_t = self.C @ ego_trajectory[t]
            
            # Compute distances to each obstacle
            for i in range(n_obstacles):
                obstacle_pos_t = obstacle_trajectories[i][t]
                
                # Compute Euclidean distance
                dist = np.linalg.norm(ego_pos_t - obstacle_pos_t) - self.ROBOT_RADIUS - self.OBSTACLE_RADIUS 
                
                # Update minimum distance
                distances[t] = min(distances[t], dist)
        
        return distances