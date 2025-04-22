"""
Risk metrics for safety filtering, including CVaR and DR-CVaR.
"""
import numpy as np
import cvxpy as cp
from utils.timing import timeit
import time
import json 
import os

# For storing optimizer instances
drcvar_optimizer = None
cvar_optimizer = None

# Add a function to save timing info with a specific key
def save_timing_info(key, setup_time, solve_time):
    """Save timing information to a JSON file with the specified key."""
    # Create directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)
    
    # Create the file path
    file_path = f'tmp/timing_info_{key}.json'
    
    # Save the timing info
    timing_info = {
        'setup_time': setup_time,
        'solve_time': solve_time
    }
    
    with open(file_path, 'w') as f:
        json.dump(timing_info, f)
    
    print(f"DEBUG - Saved {key} timing: setup={setup_time*1000:.2f}ms, solve={solve_time*1000:.2f}ms")

def expected_value(samples):
    """
    Compute the expected value of samples.
    
    Args:
        samples: Array of samples
    
    Returns:
        Expected value
    """
    return np.mean(samples, axis=0)

def var_metric(samples, alpha):
    """
    Compute the Value-at-Risk (VaR) at level alpha.
    
    Args:
        samples: Array of samples
        alpha: Confidence level (e.g., 0.2)
    
    Returns:
        VaR at level alpha
    """
    sorted_samples = np.sort(samples)
    index = int(np.ceil(len(samples) * (1 - alpha)))
    return sorted_samples[index - 1]

def cvar_metric(samples, alpha):
    """
    Compute the Conditional Value-at-Risk (CVaR) at level alpha.
    
    Args:
        samples: Array of samples
        alpha: Confidence level (e.g., 0.2)
    
    Returns:
        CVaR at level alpha
    """
    # Compute VaR
    var = var_metric(samples, alpha)
    
    # Compute CVaR: mean of samples in the (1-alpha) tail
    tail_samples = samples[samples >= var]
    
    if len(tail_samples) == 0:
        return var
    
    return np.mean(tail_samples)

class DRCVaROptimizer:
    """Efficient optimizer for DR-CVaR safe halfspace computation."""
    
    def __init__(self, alpha, epsilon, delta, max_samples):
        """Initialize the optimizer with problem parameters."""
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.n_samples = max_samples
        
        # Create optimization variables once
        self.g_var = cp.Variable(1)
        self.tau_var = cp.Variable(1)
        self.lambda_var = cp.Variable(1, nonneg=True)
        self.eta_var = cp.Variable(self.n_samples)
        
        # Create parameters that will be updated for each run
        self.h_xi_param = cp.Parameter(self.n_samples)
        self.r_param = cp.Parameter(1)
        
        # Define coefficients based on paper formulation
        a_k = [-1.0 / alpha, 0.0]
        b_k = [-1.0 / alpha, 0.0]
        c_k = [1.0 - 1.0 / alpha, 1.0]
        
        # Create constraints
        constraints = [self.lambda_var * self.epsilon + (1.0 / self.n_samples) * cp.sum(self.eta_var) <= self.delta]
        
        # Add constraints for each sample 
        for i in range(self.n_samples):
            for k in range(len(a_k)):
                constraints.append(
                    a_k[k] * self.h_xi_param[i] + 
                    b_k[k] * (self.g_var - self.r_param) + 
                    c_k[k] * self.tau_var <= self.eta_var[i]
                )
        
        # Add lambda constraint from paper
        constraints.append(1.0 / alpha <= self.lambda_var)
        
        # Create problem once
        self.problem = cp.Problem(cp.Minimize(self.g_var), constraints)
    
    def solve(self, h, samples, combined_radius):
        """
        Solve the DR-CVaR optimization with current parameters.
        
        Args:
            h: Normal vector of halfspace
            samples: Obstacle samples [n_samples, 2]
            combined_radius: Combined robot and obstacle radius
            
        Returns:
            solved: Whether optimization succeeded
            g_star: Optimal value of g
            info: Dictionary with timing information
        """
        # Start setup timing
        setup_start = time.time()
        
        # Update parameter values (this is much faster than recreating the problem)
        self.h_xi_param.value = h @ samples.T  # Matrix multiplication
        self.r_param.value = [combined_radius]
        
        # End setup timing
        setup_end = time.time()
        setup_time = setup_end - setup_start
        
        # Start solve timing
        solve_start = time.time()
        
        # Solve problem with ECOS solver (as in the paper's code)
        self.problem.solve(solver='ECOS')
        
        # End solve timing
        solve_end = time.time()
        solve_time = solve_end - solve_start
        
        # Save timing info
        info = {
            'setup_time': setup_time,
            'solve_time': solve_time,
            'solve_call_time': setup_time + solve_time
        }
        
        # Save to file for analysis
        save_timing_info('drcvar', setup_time, solve_time)
        
        # Check if solved successfully
        if self.problem.status in ["optimal", "optimal_inaccurate"]:
            return True, float(self.g_var.value), info
        else:
            print(f"Warning: DR-CVaR optimization failed with status: {self.problem.status}")
            return False, 100.0, info

class CVaROptimizer:
    """Efficient optimizer for CVaR safe halfspace computation."""
    
    def __init__(self, alpha, delta, max_samples):
        """Initialize the optimizer with problem parameters."""
        self.alpha = alpha
        self.delta = delta
        self.n_samples = max_samples
        
        # Create optimization variables once
        self.g_var = cp.Variable(1)
        self.tau_var = cp.Variable(1)
        self.aux_vars = cp.Variable(self.n_samples)
        
        # Create parameters that will be updated for each run
        self.h_xi_param = cp.Parameter(self.n_samples)
        self.r_param = cp.Parameter(1)
        
        # Create constraints
        constraints = [self.aux_vars >= 0]
        
        # Loss function: -(h·p + g - r)
        # Since h and p are in parameters, we express this as a constraint
        for i in range(self.n_samples):
            constraints.append(
                self.aux_vars[i] >= -self.h_xi_param[i] - self.g_var + self.r_param - self.tau_var
            )
        
        # Add the CVaR constraint
        constraints.append(
            self.tau_var + (1.0 / (self.alpha * self.n_samples)) * cp.sum(self.aux_vars) <= self.delta
        )
        
        # Create problem once
        self.problem = cp.Problem(cp.Minimize(self.g_var), constraints)
    
    def solve(self, h, samples, combined_radius):
        """
        Solve the CVaR optimization with current parameters.
        
        Args:
            h: Normal vector of halfspace
            samples: Obstacle samples [n_samples, 2]
            combined_radius: Combined robot and obstacle radius
            
        Returns:
            solved: Whether optimization succeeded
            g_star: Optimal value of g
            info: Dictionary with timing information
        """
        # Start setup timing
        setup_start = time.time()
        
        # Update parameter values (this is much faster than recreating the problem)
        self.h_xi_param.value = h @ samples.T  # Matrix multiplication
        self.r_param.value = [combined_radius * np.linalg.norm(h)]
        
        # End setup timing
        setup_end = time.time()
        setup_time = setup_end - setup_start
        
        # Start solve timing
        solve_start = time.time()
        
        # Solve problem with ECOS solver (as in the paper's code)
        self.problem.solve(solver='ECOS')
        
        # End solve timing
        solve_end = time.time()
        solve_time = solve_end - solve_start
        
        # Save timing info
        info = {
            'setup_time': setup_time,
            'solve_time': solve_time,
            'solve_call_time': setup_time + solve_time
        }
        
        # Save to file for analysis
        save_timing_info('cvar', setup_time, solve_time)
        
        # Check if solved successfully
        if self.problem.status in ["optimal", "optimal_inaccurate"]:
            return True, float(self.g_var.value), info
        else:
            print(f"Warning: CVaR optimization failed with status: {self.problem.status}")
            return False, 100.0, info

@timeit
def dr_cvar_halfspace(samples, h, alpha, delta, epsilon, robot_radius, obstacle_radius):
    """
    Compute the DR-CVaR safe halfspace using the paper's formulation.
    Uses a persistent optimizer object for efficiency.
    
    Args:
        samples: Array of obstacle position samples [n_samples, dim]
        h: Normal vector of the halfspace
        alpha: CVaR confidence level
        delta: Risk bound
        epsilon: Wasserstein radius 
        robot_radius: Radius of the ego robot 
        obstacle_radius: Radius of the obstacle
    
    Returns:
        g_star: Optimal offset of the halfspace
        g_tilde: Adjusted offset accounting for robot and obstacle geometry
    """
    global drcvar_optimizer
    
    # Create optimizer if it doesn't exist (singleton pattern)
    if drcvar_optimizer is None or drcvar_optimizer.n_samples != len(samples):
        drcvar_optimizer = DRCVaROptimizer(alpha, epsilon, delta, len(samples))
    
    # Compute the combined radius term
    combined_radius = (robot_radius + obstacle_radius) * np.linalg.norm(h)
    
    # Solve using the optimizer
    solved, g_star, _ = drcvar_optimizer.solve(h, samples, combined_radius)
    
    if solved:
        g_tilde = g_star - combined_radius
        return g_star, g_tilde
    else:
        # Return conservative default
        return 100.0, 100.0 - combined_radius

@timeit
def cvar_halfspace(samples, h, alpha, delta, robot_radius, obstacle_radius):
    """
    Compute the CVaR safe halfspace using optimization.
    Uses a persistent optimizer object for efficiency.
    
    Args:
        samples: Array of obstacle position samples [n_samples, dim]
        h: Normal vector of the halfspace
        alpha: CVaR confidence level
        delta: Risk bound
        robot_radius: Radius of the ego robot 
        obstacle_radius: Radius of the obstacle
    
    Returns:
        g_value: Optimal value for the halfspace offset
    """
    global cvar_optimizer
    
    # Create optimizer if it doesn't exist (singleton pattern)
    if cvar_optimizer is None or cvar_optimizer.n_samples != len(samples):
        cvar_optimizer = CVaROptimizer(alpha, delta, len(samples))
    
    # Compute the combined radius term
    combined_radius = (robot_radius + obstacle_radius)
    
    # Solve using the optimizer
    solved, g_value, _ = cvar_optimizer.solve(h, samples, combined_radius)
    
    if solved:
        return g_value
    else:
        # Return conservative default
        return 100.0 