"""
Obstacle trajectory generation for safety filtering simulations.
"""
import numpy as np
from core.dynamics import create_single_integrator_matrices, simulate_linear_system

def generate_nominal_trajectory(start_pos, direction, speed, n_steps, dt):
    """
    Generate a nominal obstacle trajectory.
    
    Args:
        start_pos: Initial position
        direction: Direction of motion (normalized)
        speed: Speed of the obstacle
        n_steps: Number of time steps
        dt: Time step 
    
    Returns:
        positions: Array of positions [n_steps+1, dim]
    """
    dim = len(start_pos)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-10:
        # Stationary obstacle
        return np.tile(start_pos, (n_steps+1, 1))
    
    # Normalize direction
    direction = direction / direction_norm
    
    # Create velocity control inputs
    velocity = speed * direction
    u_sequence = np.tile(velocity, (n_steps, 1))
    
    # Create single integrator dynamics
    A, B, C = create_single_integrator_matrices(dt, dim)
    
    # Simulate the system
    x_sequence, y_sequence = simulate_linear_system(start_pos, u_sequence, A, B, C)
    
    return y_sequence

def generate_obstacle_sample_trajectories(nominal_trajectory, n_samples, noise_cov, dt):
    """
    Generate sample trajectories around a nominal trajectory with Gaussian noise.
    
    Args:
        nominal_trajectory: Nominal obstacle trajectory [n_steps+1, dim]
        n_samples: Number of samples to generate
        noise_cov: Covariance matrix for the Gaussian noise
        dt: Time step
    
    Returns:
        sample_trajectories: Array of sample trajectories [n_samples, n_steps+1, dim]
    """
    n_steps = nominal_trajectory.shape[0] - 1
    dim = nominal_trajectory.shape[1]
    
    # Initialize sample trajectories
    sample_trajectories = np.zeros((n_samples, n_steps+1, dim))
    
    # Set initial positions (same for all samples)
    sample_trajectories[:, 0, :] = nominal_trajectory[0, :]
    
    # Generate noise for each time step and sample
    for t in range(1, n_steps+1):
        # Generate Gaussian noise
        noise = np.random.multivariate_normal(
            mean=np.zeros(dim),
            cov=noise_cov,
            size=n_samples
        )
        
        # Add noise to the nominal trajectory
        sample_trajectories[:, t, :] = nominal_trajectory[t, :] + noise
    
    return sample_trajectories

def generate_laplace_realization(nominal_trajectory, noise_cov, dt):
    """
    Generate a Laplace-distributed realization for the actual obstacle trajectory.
    
    Args:
        nominal_trajectory: Nominal obstacle trajectory [n_steps+1, dim]
        noise_cov: Covariance matrix for the underlying noise
        dt: Time step
    
    Returns:
        realization: Actual obstacle trajectory [n_steps+1, dim]
    """
    n_steps = nominal_trajectory.shape[0] - 1
    dim = nominal_trajectory.shape[1]
    
    # Initialize realization trajectory
    realization = np.zeros_like(nominal_trajectory)
    realization[0, :] = nominal_trajectory[0, :]
    
    # Compute Laplace scale parameter from covariance
    # For Laplace, var = 2*b^2 where b is the scale parameter
    scale = np.sqrt(np.diag(noise_cov) / 2)
    
    # Generate Laplace noise for each time step
    for t in range(1, n_steps+1):
        # Generate Laplace noise
        # Laplace can be generated from the difference of two exponentials
        u1 = np.random.exponential(scale=1.0, size=dim)
        u2 = np.random.exponential(scale=1.0, size=dim)
        noise = scale * (u1 - u2)
        
        # Add noise to the nominal trajectory
        realization[t, :] = nominal_trajectory[t, :] + noise
    
    return realization

def generate_obstacle_scenarios(scenario_config, horizon, dt, n_samples=100):
    """
    Generate obstacle trajectories for a given scenario.
    
    Args:
        scenario_config: Scenario configuration
        horizon: Time horizon
        dt: Time step
        n_samples: Number of samples to generate
    
    Returns:
        Dict containing:
        - nominal_trajectories: List of nominal obstacle trajectories
        - sample_trajectories: List of sample trajectories for each obstacle
        - realization_trajectories: List of realization trajectories for each obstacle
    """
    n_steps = int(horizon / dt)
    
    # Default noise covariance
    noise_cov = np.diag([0.01, 0.01])
    
    # Check if this is a multi-obstacle scenario
    if 'obstacles' in scenario_config:
        # Process multiple obstacles
        obstacles_config = scenario_config['obstacles']
        n_obstacles = len(obstacles_config)
        
        nominal_trajectories = []
        sample_trajectories = []
        realization_trajectories = []
        
        for i in range(n_obstacles):
            obstacle = obstacles_config[i]
            start_pos = obstacle['start']
            direction = obstacle['direction']
            speed = obstacle.get('speed', 1.0)
            
            # Generate nominal trajectory
            nominal_traj = generate_nominal_trajectory(
                start_pos, direction, speed, n_steps, dt
            )
            nominal_trajectories.append(nominal_traj)
            
            # Generate sample trajectories
            samples = generate_obstacle_sample_trajectories(
                nominal_traj, n_samples, noise_cov, dt
            )
            sample_trajectories.append(samples)
            
            # Generate realization
            realization = generate_laplace_realization(
                nominal_traj, noise_cov, dt
            )
            realization_trajectories.append(realization)
    else:
        # Process single obstacle
        start_pos = scenario_config['obstacle_start']
        direction = scenario_config['obstacle_direction']
        speed = scenario_config.get('obstacle_speed', 1.0)
        
        # Generate nominal trajectory
        nominal_traj = generate_nominal_trajectory(
            start_pos, direction, speed, n_steps, dt
        )
        nominal_trajectories = [nominal_traj]
        
        # Generate sample trajectories
        samples = generate_obstacle_sample_trajectories(
            nominal_traj, n_samples, noise_cov, dt
        )
        sample_trajectories = [samples]
        
        # Generate realization
        realization = generate_laplace_realization(
            nominal_traj, noise_cov, dt
        )
        realization_trajectories = [realization]
    
    return {
        'nominal_trajectories': nominal_trajectories,
        'sample_trajectories': sample_trajectories,
        'realization_trajectories': realization_trajectories
    }