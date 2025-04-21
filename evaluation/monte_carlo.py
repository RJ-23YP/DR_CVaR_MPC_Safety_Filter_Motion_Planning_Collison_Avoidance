"""
Monte Carlo simulation framework for evaluating safety filtering performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.timing import Timer, TimingStats
from simulation.visualization import compare_risk_metrics, plot_distance_to_collision

def run_monte_carlo_simulation(env, scenario_config, n_runs, params):
    """
    Run Monte Carlo simulations to evaluate different risk metrics.
    
    Args:
        env: Safety filtering environment
        scenario_config: Scenario configuration
        n_runs: Number of Monte Carlo runs
        params: Configuration parameters
    
    Returns:
        results: Dictionary of results
    """
    # Initialize timing statistics
    timing_stats = TimingStats()
    
    # Arrays to store the minimum distance to collision for each run
    min_distances = {
        'reference': np.zeros(n_runs),
        'mean': np.zeros(n_runs),
        'cvar': np.zeros(n_runs),
        'dr_cvar': np.zeros(n_runs)
    }
    
    # Collision counts for each method
    collision_counts = {
        'reference': 0,
        'mean': 0,
        'cvar': 0,
        'dr_cvar': 0
    }
    
    # Create a planner and MPC filter
    from simulation.planner import ReferenceTrajectoryPlanner
    from core.mpc_filter import MPCSafetyFilter
    
    # Set up initial state and goal
    x0 = np.zeros(env.n_states)
    x0[:2] = scenario_config['ego_start']
    
    goal_state = np.zeros(env.n_states)
    goal_state[:2] = scenario_config['ego_goal']
    
    # Set up weight matrices for MPC
    Q = params.Q_WEIGHT * np.eye(env.n_states)
    R = params.R_WEIGHT * np.eye(env.n_inputs)
    
    # Create planner and MPC filter
    planner = ReferenceTrajectoryPlanner(env.A, env.B, env.C, Q, R, params.HORIZON, params.DT)
    mpc_filter = MPCSafetyFilter(env.A, env.B, env.C, Q, R, params.HORIZON, params.DT)
    
    # Input and state bounds
    state_bounds = (np.array([-10, -10, -5, -5]), np.array([10, 10, 5, 5]))
    input_bounds = (np.array([-5, -5]), np.array([5, 5]))
    
    # Plan reference trajectory (same for all runs)
    with Timer("Reference Planning") as timer:
        x_ref, u_ref, plan_info = planner.straight_line_trajectory(
            scenario_config['ego_start'], scenario_config['ego_goal']
        )
    timing_stats.add("Reference Planning", timer.elapsed)
    
    if x_ref is None:
        print("Failed to plan reference trajectory:")
        print(plan_info)
        return None
    
    # Run Monte Carlo simulations
    print(f"Running {n_runs} Monte Carlo simulations...")
    
    for run in range(n_runs):
        if run % 10 == 0:
            print(f"Run {run}/{n_runs}")
        
        # Generate obstacle scenarios for this run
        from simulation.obstacles import generate_obstacle_scenarios
        
        with Timer("Obstacle Generation") as timer:
            obstacle_data = generate_obstacle_scenarios(
                scenario_config, params.SIM_TIME, params.DT, params.NUM_SAMPLES
            )
        timing_stats.add("Obstacle Generation", timer.elapsed)
        
        # Extract obstacle data
        sample_trajectories = obstacle_data['sample_trajectories']
        realization_trajectories = obstacle_data['realization_trajectories']
        
        # Compute safe halfspaces for the reference trajectory
        with Timer("Computing Safe Halfspaces") as timer:
            safe_halfspaces = env.compute_safe_halfspaces_for_trajectory(
                sample_trajectories, x_ref
            )
        timing_stats.add("Computing Safe Halfspaces", timer.elapsed)
        
        # Filter the trajectory using different risk metrics
        filtered_trajectories = {}
        
        for risk_metric in ['mean', 'cvar', 'dr_cvar']:
            with Timer(f"MPC Filtering ({risk_metric})") as timer:
                x_filtered, u_filtered, _ = mpc_filter.filter_trajectory(
                    x0, x_ref, u_ref, safe_halfspaces[risk_metric],
                    input_bounds, state_bounds[:2]  # Only position bounds
                )
            timing_stats.add(f"MPC Filtering ({risk_metric})", timer.elapsed)
            
            filtered_trajectories[risk_metric] = x_filtered
        
        # Evaluate safety for each method
        for method in ['reference', 'mean', 'cvar', 'dr_cvar']:
            if method == 'reference':
                trajectory = x_ref
            else:
                trajectory = filtered_trajectories[method]
            
            # Compute minimum distance to collision
            distances = env.compute_distance_to_collision(trajectory, realization_trajectories)
            min_distance = np.min(distances)
            min_distances[method][run] = min_distance
            
            # Count collisions (distance < 0)
            if min_distance < 0:
                collision_counts[method] += 1
    
    # Compute collision probability
    collision_probs = {
        method: count / n_runs for method, count in collision_counts.items()
    }
    
    # Print results
    print("\nMonte Carlo Simulation Results:")
    print(f"Total runs: {n_runs}")
    print("\nCollision Counts:")
    for method, count in collision_counts.items():
        print(f"  {method}: {count} ({collision_probs[method]*100:.2f}%)")
    
    print("\nMinimum Distance Statistics:")
    for method, distances in min_distances.items():
        print(f"  {method}:")
        print(f"    Mean: {np.mean(distances):.4f}")
        print(f"    Min:  {np.min(distances):.4f}")
        print(f"    Max:  {np.max(distances):.4f}")
        print(f"    Std:  {np.std(distances):.4f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    compare_risk_metrics(
        min_distances['mean'], 
        min_distances['cvar'], 
        min_distances['dr_cvar'],
        title=f"Minimum Distance to Collision Comparison ({n_runs} runs)"
    )
    
    # Return results
    return {
        'min_distances': min_distances,
        'collision_counts': collision_counts,
        'collision_probs': collision_probs,
        'timing_stats': timing_stats
    }