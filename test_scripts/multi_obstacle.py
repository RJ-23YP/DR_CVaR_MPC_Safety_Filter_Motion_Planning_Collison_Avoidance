"""
Test Script for the DR-CVaR safety filtering for multi-obstacle scenarios.
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
from utils.timing import Timer, TimingStats

def run_multi_obstacle_scenario(save_results=False, output_dir='results'):
    """
    Run a multi-obstacle safety filtering scenario.
    
    Args:
        save_results: Whether to save the results
        output_dir: Directory to save results
    
    Returns:
        Dictionary of results
    """
    # Create output directory if needed
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize timing statistics
    timing_stats = TimingStats()
    
    # Get scenario configuration
    scenario_config = get_scenario_config('multi_obstacle')
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
    
    # We'll focus on DR-CVaR for multi-obstacle scenario
    risk_metric = 'dr_cvar'
    
    with Timer(f"MPC Filtering ({risk_metric})") as timer:
        x_filtered, u_filtered, filter_info = mpc_filter.filter_trajectory(
            x0, x_ref, u_ref, safe_halfspaces[risk_metric],
            input_bounds, state_bounds[:2]  # Only position bounds
        )
    timing_stats.add(f"MPC Filtering ({risk_metric})", timer.elapsed)
    
    # Check if the filtering was successful
    if filter_info.get('used_fallback', False):
        print("Warning: MPC Safety Filter used fallback strategy")
    
    # Evaluate safety by computing distance to collision with the realization
    ref_distances = env.compute_distance_to_collision(x_ref, realization_trajectories)
    filtered_distances = env.compute_distance_to_collision(x_filtered, realization_trajectories)
    
    # Print collision information
    min_ref_dist = np.min(ref_distances)
    min_filtered_dist = np.min(filtered_distances)
    
    print(f"Reference: Min distance = {min_ref_dist:.3f} - {'COLLISION' if min_ref_dist < 0 else 'Safe'}")
    print(f"Filtered:  Min distance = {min_filtered_dist:.3f} - {'COLLISION' if min_filtered_dist < 0 else 'Safe'}")
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(realization_trajectories)))
    
    for i, obstacle_traj in enumerate(realization_trajectories):
        ax1.plot(obstacle_traj[:, 0], obstacle_traj[:, 1], color=colors[i], linestyle='-', 
                label=f'Obstacle {i+1}')
        
        # Add circles for initial and final positions
        ax1.add_patch(plt.Circle(obstacle_traj[0], OBSTACLE_RADIUS, color=colors[i], alpha=0.3))
        ax1.add_patch(plt.Circle(obstacle_traj[-1], OBSTACLE_RADIUS, color=colors[i], alpha=0.7))
    
    # Plot reference and filtered trajectories
    ax1.plot(x_ref[:, 0], x_ref[:, 1], 'r--', linewidth=2, label='Reference')
    ax1.plot(x_filtered[:, 0], x_filtered[:, 1], 'm-', linewidth=2, label='DR-CVaR Filtered')
    
    # Add circles for ego robot initial and final positions
    ax1.add_patch(plt.Circle(x0[:2], ROBOT_RADIUS, color='blue', alpha=0.3))
    ax1.add_patch(plt.Circle(x_filtered[-1, :2], ROBOT_RADIUS, color='blue', alpha=0.7))
    
    # Add markers for start and goal
    ax1.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_state[0], goal_state[1], 'r*', markersize=10, label='Goal')
    
    # Add DR-CVaR safe halfspaces for better visualization
    # We'll plot every few steps to avoid clutter
    step = 3
    for t in range(0, min(len(safe_halfspaces[risk_metric]), HORIZON), step):
        halfspaces_t = safe_halfspaces[risk_metric][t]
        
        for i, halfspace in enumerate(halfspaces_t):
            h, g = halfspace.get_constraint_params()
            
            # Create a line representing the halfspace boundary
            # First, find two points on the boundary
            if abs(h[1]) > 1e-6:  # Non-vertical line
                x_vals = np.array([-10, 10])
                y_vals = (-g - h[0] * x_vals) / h[1]
            else:  # Vertical line
                y_vals = np.array([-10, 10])
                x_vals = -g / h[0] * np.ones_like(y_vals)
            
            # Plot the boundary
            ax1.plot(x_vals, y_vals, color=colors[i], alpha=0.2)
    
    ax1.set_title('Multi-Obstacle Scenario with DR-CVaR Safety Filtering')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-4, 4])
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Plot distance to collision
    ax2.plot(ref_distances, 'r--', label='Reference', linewidth=2)
    ax2.plot(filtered_distances, 'm-', label='DR-CVaR Filtered', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', label='Collision threshold')
    ax2.set_title('Distance to Collision')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance [m]')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save results if requested
    if save_results:
        plt.savefig(os.path.join(output_dir, 'multi_obstacle_scenario.png'))
    
    # Plot control inputs
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(u_ref[:, 0], 'r--', label='Reference x-input')
    plt.plot(u_filtered[:, 0], 'm-', label='Filtered x-input')
    plt.title('Control Inputs - x direction')
    plt.xlabel('Time step')
    plt.ylabel('Input [m/s²]')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(u_ref[:, 1], 'r--', label='Reference y-input')
    plt.plot(u_filtered[:, 1], 'm-', label='Filtered y-input')
    plt.title('Control Inputs - y direction')
    plt.xlabel('Time step')
    plt.ylabel('Input [m/s²]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_results:
        plt.savefig(os.path.join(output_dir, 'multi_obstacle_inputs.png'))
    
    # Print timing statistics
    print("\nTiming Statistics:")
    timing_stats.print_stats()
    
    return {
        'x_ref': x_ref,
        'u_ref': u_ref,
        'x_filtered': x_filtered,
        'u_filtered': u_filtered,
        'ref_distances': ref_distances,
        'filtered_distances': filtered_distances,
        'timing_stats': timing_stats
    }

if __name__ == "__main__":
    results = run_multi_obstacle_scenario(save_results=True)
    plt.show()