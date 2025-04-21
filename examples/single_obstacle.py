"""
Example script demonstrating the DR-CVaR safety filtering for single obstacle scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import *
from config.scenarios import get_scenario_config
from core.dynamics import create_double_integrator_matrices
from simulation.environment import SafetyFilteringEnvironment
from simulation.obstacles import generate_obstacle_scenarios
from simulation.planner import ReferenceTrajectoryPlanner
from core.mpc_filter import MPCSafetyFilter
from simulation.visualization import plot_scenario, plot_distance_to_collision
from utils.timing import Timer, TimingStats

def run_single_obstacle_scenario(scenario_name, save_results=False, output_dir='results'):
    """
    Run a single obstacle safety filtering scenario.
    
    Args:
        scenario_name: Name of the scenario ('head_on', 'overtaking', 'intersection')
        save_results: Whether to save the results
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    # Validate scenario name
    if scenario_name not in ['head_on', 'overtaking', 'intersection']:
        raise ValueError(f"Unknown scenario: {scenario_name}. Choose from 'head_on', 'overtaking', 'intersection'")
    
    # Create output directory if needed
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize timing statistics
    timing_stats = TimingStats()
    
    # Get scenario configuration
    scenario_config = get_scenario_config(scenario_name)
    print(f"Running scenario: {scenario_config['description']}")
    
    # Initialize environment
    env = SafetyFilteringEnvironment(
        ROBOT_RADIUS=ROBOT_RADIUS,
        OBSTACLE_RADIUS=OBSTACLE_RADIUS,
        HORIZON=HORIZON, DT=DT,
        ALPHA=ALPHA, DELTA=DELTA,
        EPSILON=EPSILON
    )
    
    # Create system matrices
    A, B, C = create_double_integrator_matrices(DT)
    
    # Set up the weight matrices
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    Q = Q_WEIGHT * np.eye(n_states)
    R = R_WEIGHT * np.eye(n_inputs)
    
    # Set up bounds
    state_bounds = (np.array([-10, -10, -5, -5]), np.array([10, 10, 5, 5]))
    input_bounds = (np.array([-5, -5]), np.array([5, 5]))
    env.set_bounds(state_bounds, input_bounds)
    
    # Generate obstacle scenarios
    with Timer("Obstacle Generation") as timer:
        obstacle_data = generate_obstacle_scenarios(
            scenario_config, SIM_TIME, DT, NUM_SAMPLES
        )
    timing_stats.add("Obstacle Generation", timer.elapsed)
    
    # Extract obstacle data
    nominal_trajectories = obstacle_data['nominal_trajectories']
    sample_trajectories = obstacle_data['sample_trajectories']
    realization_trajectories = obstacle_data['realization_trajectories']
    
    # Setup reference trajectory planner
    planner = ReferenceTrajectoryPlanner(A, B, C, Q, R, HORIZON, DT)
    
    # Set up initial state and goal
    x0 = np.zeros(n_states)
    x0[:2] = scenario_config['ego_start']
    
    goal_state = np.zeros(n_states)
    goal_state[:2] = scenario_config['ego_goal']
    
    # Plan reference trajectory
    with Timer("Reference Planning") as timer:
        x_ref, u_ref, plan_info = planner.straight_line_trajectory(
            scenario_config['ego_start'], scenario_config['ego_goal']
        )
    timing_stats.add("Reference Planning", timer.elapsed)
    
    if x_ref is None:
        print("Failed to plan reference trajectory:")
        print(plan_info)
        return None
    
    # Compute safe halfspaces for the reference trajectory
    with Timer("Computing Safe Halfspaces") as timer:
        safe_halfspaces = env.compute_safe_halfspaces_for_trajectory(
            sample_trajectories, x_ref
        )
    timing_stats.add("Computing Safe Halfspaces", timer.elapsed)
    
    # Set up MPC safety filter
    mpc_filter = MPCSafetyFilter(A, B, C, Q, R, HORIZON, DT)
    
    # Filter the trajectory using different risk metrics
    filtered_trajectories = {}
    filtered_inputs = {}
    filter_infos = {}
    
    for risk_metric in ['mean', 'cvar', 'dr_cvar']:
        with Timer(f"MPC Filtering ({risk_metric})") as timer:
            x_filtered, u_filtered, filter_info = mpc_filter.filter_trajectory(
                x0, x_ref, u_ref, safe_halfspaces[risk_metric],
                input_bounds, state_bounds[:2]  # Only position bounds
            )
        timing_stats.add(f"MPC Filtering ({risk_metric})", timer.elapsed)
        
        filtered_trajectories[risk_metric] = x_filtered
        filtered_inputs[risk_metric] = u_filtered
        filter_infos[risk_metric] = filter_info
    
    # Evaluate safety by computing distance to collision with the realization
    distances = {}
    for risk_metric in ['mean', 'cvar', 'dr_cvar']:
        distances[risk_metric] = env.compute_distance_to_collision(
            filtered_trajectories[risk_metric], realization_trajectories
        )
    
    # Compute distance for reference trajectory
    distances['reference'] = env.compute_distance_to_collision(
        x_ref, realization_trajectories
    )
    
    # Print collision information
    for method, dist in distances.items():
        min_dist = np.min(dist)
        collision = min_dist < 0
        print(f"{method.ljust(10)}: Min distance = {min_dist:.3f} - {'COLLISION' if collision else 'Safe'}")
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot trajectories
    for i, obstacle_traj in enumerate(realization_trajectories):
        ax1.plot(obstacle_traj[:, 0], obstacle_traj[:, 1], 'k-', label=f'Obstacle' if i == 0 else "")
    ax1.plot(x_ref[:, 0], x_ref[:, 1], 'r--', linewidth=2, label='Reference')
    ax1.plot(filtered_trajectories['mean'][:, 0], filtered_trajectories['mean'][:, 1], 'g-', linewidth=2, label='Mean')
    ax1.plot(filtered_trajectories['cvar'][:, 0], filtered_trajectories['cvar'][:, 1], 'b-', linewidth=2, label='CVaR')
    ax1.plot(filtered_trajectories['dr_cvar'][:, 0], filtered_trajectories['dr_cvar'][:, 1], 'm-', linewidth=2, label='DR-CVaR')
    
    # Add markers for start and goal
    ax1.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_state[0], goal_state[1], 'r*', markersize=10, label='Goal')
    
    ax1.set_title(f'Trajectories - {scenario_config["description"]}')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot distance to collision
    for method, dist in distances.items():
        ax2.plot(dist, label=method, linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', label='Collision threshold')
    ax2.set_title('Distance to Collision')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance [m]')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save results if requested
    if save_results:
        plt.savefig(os.path.join(output_dir, f'{scenario_name}_scenario.png'))
    
    # Print timing statistics
    print("\nTiming Statistics:")
    timing_stats.print_stats()
    
    return {
        'filtered_trajectories': filtered_trajectories,
        'distances': distances,
        'timing_stats': timing_stats
    }

if __name__ == "__main__":
    # Run the example for all three scenarios
    scenarios = ['head_on', 'overtaking', 'intersection']
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*50}\n")
        
        results = run_single_obstacle_scenario(scenario, save_results=True)
        
        plt.show()