"""
Timing analysis for DR-CVaR safe halfspace computation.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.timing import Timer
from core.halfspaces import CVaRSafeHalfspace, DRCVaRSafeHalfspace
from config.parameters import ALPHA, DELTA, EPSILON, ROBOT_RADIUS, OBSTACLE_RADIUS
import pandas as pd
import os 
import json 

def analyze_dr_cvar_computation_time(sample_sizes=[10, 50, 100, 500, 1000, 1500], n_runs=50, save_dir=None):
    """
    Analyze the computation time of DR-CVaR safe halfspace calculation for different sample sizes.
    
    Args:
        sample_sizes: List of sample sizes to test
        n_runs: Number of runs for each sample size
        save_dir: Directory to save results
    
    Returns:
        Timing statistics for each sample size
    """
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Parameters
    alpha = ALPHA
    epsilon = EPSILON
    delta = DELTA
    
    # Initialize data structures for timing stats
    timing_data = {
        'setup_times': {n: [] for n in sample_sizes},
        'solve_times': {n: [] for n in sample_sizes},
        'call_times': {n: [] for n in sample_sizes},
        'cvar_setup_times': {n: [] for n in sample_sizes},
        'cvar_solve_times': {n: [] for n in sample_sizes},
        'cvar_call_times': {n: [] for n in sample_sizes}
    }
    
    # Clean any old timing files
    os.makedirs('tmp', exist_ok=True)
    if os.path.exists('tmp/timing_info_drcvar.json'):
        os.remove('tmp/timing_info_drcvar.json')
    if os.path.exists('tmp/timing_info_cvar.json'):
        os.remove('tmp/timing_info_cvar.json')
    
    for n_samples in sample_sizes:
        print(f"Testing with {n_samples} samples...")
        
        for run in range(n_runs):
            if run % 10 == 0 and run > 0:
                print(f"  Run {run}/{n_runs}") 
            
            # Generate random samples and setup
            h = np.array([1., 1])
            h = h / np.linalg.norm(h)
            
            # Generate random obstacle samples
            mean_pos = np.array([0.5, 0.0])
            scale = np.array([0.1, 0.1])
            samples = np.zeros((n_samples, 2))
            for i in range(n_samples):
                samples[i, 0] = np.random.normal(mean_pos[0], scale[0])
                samples[i, 1] = np.random.normal(mean_pos[1], scale[1])
            
            ego_ref_pos = np.array([0.0, 0.0])
            
            # Time the operations for DR-CVaR using static create method
            with Timer() as timer:
                dr_halfspace = DRCVaRSafeHalfspace.create(
                    samples, ego_ref_pos, alpha, delta, epsilon, 
                    ROBOT_RADIUS, OBSTACLE_RADIUS
                )
            call_time = timer.elapsed * 1000  # Convert to ms 

            # Read timing info from file
            setup_time = 0
            solve_time = 0
                
            drcvar_timing_file = 'tmp/timing_info_drcvar.json' 
            if os.path.exists(drcvar_timing_file):
                try:
                    with open(drcvar_timing_file, 'r') as f:
                        timing_info = json.load(f)
                        setup_time = timing_info.get('setup_time', 0) * 1000
                        solve_time = timing_info.get('solve_time', 0) * 1000
                except Exception as e:
                    print(f"Error reading timing file: {e}")
            
            # Record DR-CVaR timings
            timing_data['setup_times'][n_samples].append(setup_time)
            timing_data['solve_times'][n_samples].append(solve_time)
            timing_data['call_times'][n_samples].append(call_time)
            
            # Time the operations for CVaR using static create method
            with Timer() as timer:
                cvar_halfspace = CVaRSafeHalfspace.create(
                    samples, ego_ref_pos, alpha, delta,
                    ROBOT_RADIUS, OBSTACLE_RADIUS
                )
            cvar_call_time = timer.elapsed * 1000  # Convert to ms
            
            # Read CVaR timing info from file
            cvar_setup_time = 0
            cvar_solve_time = 0
                
            cvar_timing_file = 'tmp/timing_info_cvar.json'
            if os.path.exists(cvar_timing_file):
                try:
                    with open(cvar_timing_file, 'r') as f:
                        timing_info = json.load(f)
                        cvar_setup_time = timing_info.get('setup_time', 0) * 1000
                        cvar_solve_time = timing_info.get('solve_time', 0) * 1000
                except Exception as e:
                    print(f"Error reading CVaR timing file: {e}")
            
            # Record CVaR timings
            timing_data['cvar_setup_times'][n_samples].append(cvar_setup_time)
            timing_data['cvar_solve_times'][n_samples].append(cvar_solve_time)
            timing_data['cvar_call_times'][n_samples].append(cvar_call_time)
    
    # Create and save box plots
    plot_timing_results(timing_data, sample_sizes, save_dir)
    
    # Create and save comparison table
    create_comparison_table(timing_data, sample_sizes, save_dir) 
    
    return timing_data 

def plot_timing_results(timing_data, sample_sizes, save_dir=None):
    """
    Create box plots of timing results with outlier filtering.
    
    Args:
        timing_data: Dictionary of timing data
        sample_sizes: List of sample sizes
        save_dir: Directory to save results
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Define outlier thresholds
    setup_threshold = 2  # ms
    solve_threshold = 100  # ms
    call_threshold = 400  # ms 
    
    # Filter data to remove extreme outliers
    filtered_setup_data = []
    filtered_solve_data = []
    filtered_call_data = []
    
    for n in sample_sizes:
        # Filter setup times
        setup_times = np.array(timing_data['setup_times'][n])
        filtered_setup = setup_times[setup_times < setup_threshold]
        filtered_setup_data.append(filtered_setup)
        
        # Filter solve times
        solve_times = np.array(timing_data['solve_times'][n])
        filtered_solve = solve_times[solve_times < solve_threshold]
        filtered_solve_data.append(filtered_solve)
        
        # Filter call times
        call_times = np.array(timing_data['call_times'][n])
        filtered_call = call_times[call_times < call_threshold]
        filtered_call_data.append(filtered_call)
        
        # Print filtering statistics
        setup_removed = len(setup_times) - len(filtered_setup)
        solve_removed = len(solve_times) - len(filtered_solve)
        call_removed = len(call_times) - len(filtered_call)
        
        print(f"Sample size {n}:")
        print(f"  Setup Time: Removed {setup_removed}/{len(setup_times)} outliers > {setup_threshold}ms")
        print(f"  Solve Time: Removed {solve_removed}/{len(solve_times)} outliers > {solve_threshold}ms")
        print(f"  Call Time: Removed {call_removed}/{len(call_times)} outliers > {call_threshold}ms")
    
    # Setup time
    axs[0].boxplot(filtered_setup_data, labels=sample_sizes)
    axs[0].set_title(f'Setup Time (outliers > {setup_threshold}ms removed)')
    axs[0].set_ylabel('Time (ms)')
    
    # Solve time
    axs[1].boxplot(filtered_solve_data, labels=sample_sizes)
    axs[1].set_title(f'Solve Time (outliers > {solve_threshold}ms removed)')
    axs[1].set_ylabel('Time (ms)')
    
    # Call time
    axs[2].boxplot(filtered_call_data, labels=sample_sizes)
    axs[2].set_title(f'Call Time (outliers > {call_threshold}ms removed)')
    axs[2].set_ylabel('Time (ms)')
    axs[2].set_xlabel('Number Samples')
    
    plt.tight_layout()
    
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, 'dr_cvar_computation_time.png'))
    
    # Also create a version with the original unfiltered data for comparison
    fig_unfiltered, axs_unfiltered = plt.subplots(3, 1, figsize=(10, 12))
    
    # Setup time (unfiltered)
    axs_unfiltered[0].boxplot([timing_data['setup_times'][n] for n in sample_sizes], labels=sample_sizes)
    axs_unfiltered[0].set_title('Setup Time (with outliers)')
    axs_unfiltered[0].set_ylabel('Time (ms)')
    
    # Solve time (unfiltered)
    axs_unfiltered[1].boxplot([timing_data['solve_times'][n] for n in sample_sizes], labels=sample_sizes)
    axs_unfiltered[1].set_title('Solve Time (with outliers)')
    axs_unfiltered[1].set_ylabel('Time (ms)')
    
    # Call time (unfiltered)
    axs_unfiltered[2].boxplot([timing_data['call_times'][n] for n in sample_sizes], labels=sample_sizes)
    axs_unfiltered[2].set_title('Call Time (with outliers)')
    axs_unfiltered[2].set_ylabel('Time (ms)')
    axs_unfiltered[2].set_xlabel('Number Samples')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'dr_cvar_computation_time_with_outliers.png'))


def create_comparison_table(timing_data, sample_sizes, save_dir=None):
    """
    Create a comparison table of mean timing results.
    
    Args:
        timing_data: Dictionary of timing data
        sample_sizes: List of sample sizes
        save_dir: Directory to save results
    """
    
    # Use all available sample sizes provided by user
    display_sizes = sample_sizes 
    
    # Create table data
    table_data = []
    for n in display_sizes:
        dr_setup = np.mean(timing_data['setup_times'][n])
        dr_solve = np.mean(timing_data['solve_times'][n])
        dr_call = np.mean(timing_data['call_times'][n])
        
        cvar_setup = np.mean(timing_data['cvar_setup_times'][n])
        cvar_solve = np.mean(timing_data['cvar_solve_times'][n])
        cvar_call = np.mean(timing_data['cvar_call_times'][n])
        
        table_data.append([
            n, 
            dr_setup, dr_solve, dr_call,
            cvar_setup, cvar_solve, cvar_call
        ])
    
    # Create dataframe
    df = pd.DataFrame(
        table_data, 
        columns=[
            'Samples', 
            'DR-CVaR Setup', 'DR-CVaR Solve', 'DR-CVaR Call',
            'CVaR Setup', 'CVaR Solve', 'CVaR Call'
        ]
    )
    
    # Print table
    print("\nTiming Comparison (times in ms):")
    print(df.to_string(index=False))
    
    # Save to file
    if save_dir:
        import os
        df.to_csv(os.path.join(save_dir, 'timing_comparison.csv'), index=False)