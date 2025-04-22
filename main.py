"""
Main entry point for testing the DR-CVaR safety filtering implementation.
"""
import numpy as np 
import matplotlib.pyplot as plt
import os 
from config.parameters import *
from config.scenarios import get_scenario_config
from core.dynamics import create_double_integrator_matrices
from simulation.environment import SafetyFilteringEnvironment
from simulation.obstacles import generate_obstacle_scenarios
from simulation.planner import ReferenceTrajectoryPlanner
from core.mpc_filter import MPCSafetyFilter 
from simulation.visualization import animate_scenario, visualize_trajectory_with_halfspaces
from utils.timing import Timer, TimingStats
import argparse 
from evaluation.timing_analysis import analyze_dr_cvar_computation_time 

def run_single_scenario(scenario_name, save_dir=None):
    """
    Run a single safety filtering scenario.
    
    Args:
        scenario_name: Name of the scenario
        save_dir: Directory to save results
    """
    # Create output directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize timing statistics
    timing_stats = TimingStats()
    
    # Get scenario configuration
    scenario_config = get_scenario_config(scenario_name)
    print(f"Running scenario: {scenario_config['description']}")
    
    # Initialize environment
    env = SafetyFilteringEnvironment(ROBOT_RADIUS=ROBOT_RADIUS, 
                                    OBSTACLE_RADIUS=OBSTACLE_RADIUS,
                                    HORIZON=HORIZON, DT=DT,
                                    ALPHA=ALPHA, DELTA=DELTA, 
                                    EPSILON=EPSILON)
    
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
        return
    
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

    # ADD DEBUG PRINTS HERE:
    print("\nMPC Feasibility Information:")
    for risk_metric in ['mean', 'cvar', 'dr_cvar']:
        print(f"{risk_metric} status: {filter_infos[risk_metric]['status']}")
        if 'used_fallback' in filter_infos[risk_metric]:
            print(f"{risk_metric} used fallback: {filter_infos[risk_metric]['used_fallback']}")
    print()  # Add an empty line for readability
    
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
    
    # Plot the results
    
    # Distance to collision
    # plt.subplot(2, 2, 2)

    # Plot the distance to collision
    plt.figure(figsize=(10, 6))
    for risk_metric, distance in distances.items():
        plt.plot(distance, label=risk_metric)
    plt.axhline(y=0, color='r', linestyle='--', label='Collision threshold')
    plt.title('Distance to collision')
    plt.xlabel('Time step')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{scenario_name}_results.png')) 

    # Create halfspace visualization only for DR-CVaR
    risk_metric = 'dr_cvar'
    print(f"\nCreating halfspace visualization for {risk_metric}...")
    visualize_trajectory_with_halfspaces(
        filtered_trajectories[risk_metric],
        realization_trajectories,
        safe_halfspaces[risk_metric],
        ROBOT_RADIUS, 
        OBSTACLE_RADIUS,
        xlim=(-6, 6), 
        ylim=(-4, 4),
        title=f'{scenario_name.capitalize()} Scenario with {risk_metric.upper()} Safe Halfspaces',
        save_path=os.path.join(save_dir, f'{scenario_name}_{risk_metric}_halfspaces.png')
    )
    print(f"Halfspace visualization saved to {os.path.join(save_dir, f'{scenario_name}_{risk_metric}_halfspaces.png')}")
    
    # plt.tight_layout() 
    
    # Print timing information
    timing_stats.print_stats()
    
    return {
        'filtered_trajectories': filtered_trajectories,
        'distances': distances,
        'timing_stats': timing_stats,
        'realization_trajectories': realization_trajectories,  # Add this
        'safe_halfspaces': safe_halfspaces  # Add this
    } 

if __name__ == "__main__": 
    # Argument parser setup
    # Set a fixed random seed for reproducibility
    np.random.seed(42)  # You can choose any integer value as the seed 

    parser = argparse.ArgumentParser(description="Run DR-CVaR Safety Filtering Scenarios")

    parser.add_argument(
        "--scenario", 
        choices=['head_on', 'overtaking', 'intersection', 'multi_obstacle'], 
        default='head_on',
        help="Scenario to run"
    )

    parser.add_argument(
        "--mode", 
        choices=['single', 'timing_analysis'], 
        default='single',
        help="Test mode to run"
    ) 

    parser.add_argument(
        "--animate", 
        action="store_true", 
        help="Create animation (only applies to 'single' mode)"
    )

    parser.add_argument(
        "--metric", 
        choices=['mean', 'cvar', 'dr_cvar'], 
        default='dr_cvar',
        help="Risk metric to animate"
    )

    parser.add_argument(
        "--sample_sizes", 
        type=str,
        default="10,50,100,500,1000,1500",
        help="Comma-separated list of sample sizes for timing analysis"
    )

    parser.add_argument(
        "--timing_runs", 
        type=int, 
        default=50,
        help="Number of runs per sample size for timing analysis"
    )

    args = parser.parse_args()

    # Create results directory
    save_dir = 'results' 
    os.makedirs(save_dir, exist_ok=True)

    # Run scenario
    if args.mode == 'single':
        results = run_single_scenario(args.scenario, save_dir)

        if args.animate and results:
            print("\nCreating animation...")

            anim, fig = animate_scenario(
                results['filtered_trajectories'][args.metric],
                results.get('realization_trajectories', []),
                ROBOT_RADIUS, 
                OBSTACLE_RADIUS,
                results.get('safe_halfspaces', {}).get(args.metric, None),
                xlim=(-6, 6), 
                ylim=(-4, 4),
                title=f'{args.scenario.capitalize()} Scenario with {args.metric.upper()} Safety Filtering',
                interval=100,
                save_path=os.path.join(save_dir, f'{args.scenario}_{args.metric}_animation.mp4')
            )
            print(f"Animation saved to {os.path.join(save_dir, f'{args.scenario}_{args.metric}_animation.mp4')}")
        
    elif args.mode == 'timing_analysis':
        print("\nRunning DR-CVaR computation time analysis...")
        
        # Parse sample sizes from the string argument
        sample_sizes = [int(n.strip()) for n in args.sample_sizes.split(',')]
        
        # Run timing analysis
        timing_data = analyze_dr_cvar_computation_time(
            sample_sizes=sample_sizes, 
            n_runs=args.timing_runs, 
            save_dir=save_dir
        )
        print(f"Timing analysis complete. Results saved to {save_dir}")

    plt.show() 

