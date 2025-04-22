"""
MPC-based safety filter using DR-CVaR safe halfspaces.
"""
import numpy as np
import cvxpy as cp
from utils.timing import timeit, Timer
import time

class MPCSafetyFilter:
    """
    MPC-based safety filter for collision avoidance. 
    """
    def __init__(self, A, B, C, Q, R, horizon, dt):
        """
        Initialize the MPC safety filter.
        
        Args:
            A, B, C: System matrices
            Q, R: Cost matrices
            horizon: MPC horizon
            dt: Time step
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.horizon = horizon
        self.dt = dt
        
        # Get dimensions
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]
        
        # Store the last optimal control sequence
        self.last_optimal_u = None
    
    @timeit
    def filter_trajectory(self, x0, x_ref, u_ref, safe_halfspaces, input_constraints=None, position_constraints=None):
        """
        Filter a reference trajectory using MPC with safe halfspace constraints.
        
        Args:
            x0: Initial state
            x_ref: Reference state trajectory [horizon+1, n_states]
            u_ref: Reference input trajectory [horizon, n_inputs]
            safe_halfspaces: List of safe halfspaces for each time step
            input_constraints: Input constraints (min, max)
            position_constraints: Position constraints
        
        Returns:
            x_filtered: Filtered state trajectory
            u_filtered: Filtered input trajectory
            info: Additional information
        """
        # Start timer
        start_time = time.time()
        
        # Initialize variables
        x = cp.Variable((self.horizon+1, self.n_states))
        u = cp.Variable((self.horizon, self.n_inputs))
        
        # Initialize objective
        objective = 0
        
        # Add tracking costs
        for t in range(self.horizon):
            # State tracking cost
            state_error = x[t+1] - x_ref[t+1]
            objective += cp.quad_form(state_error, self.Q)
            
            # Input cost
            objective += cp.quad_form(u[t], self.R)
        
        # Initialize constraints
        constraints = []
        
        # Initial state constraint
        constraints.append(x[0] == x0)
        
        # Dynamics constraints
        for t in range(self.horizon):
            constraints.append(x[t+1] == self.A @ x[t] + self.B @ u[t])
        
        # Input constraints
        if input_constraints is not None:
            u_min, u_max = input_constraints
            for t in range(self.horizon):
                constraints.append(u[t] >= u_min)
                constraints.append(u[t] <= u_max)
        
        # Position constraints
        if position_constraints is not None:
            pos_min, pos_max = position_constraints
            for t in range(1, self.horizon+1):
                      
                # Extract only position part (first 2 elements for a 2D scenario)
                position = self.C @ x[t]  # This should give a 2D vector 
                
                # Check dimensions
                if len(pos_min) != position.shape[0]:
                    # Resize constraints if needed
                    pos_min_adjusted = pos_min[:position.shape[0]]
                    pos_max_adjusted = pos_max[:position.shape[0]]
                    
                    constraints.append(position >= pos_min_adjusted)
                    constraints.append(position <= pos_max_adjusted)
                else:
                    constraints.append(position >= pos_min)
                    constraints.append(position <= pos_max)
        
        
        # Add slack variables and safety costs for soft constraints
        safety_slacks = []
        for t in range(1, self.horizon+1):
            position = self.C @ x[t]
            
            if t-1 < len(safe_halfspaces):
                halfspaces_t = safe_halfspaces[t-1] 

                # DEBUG PRINTS
                print(f"Halfspace constraints at step {t}: {len(halfspaces_t)}")
                for i, halfspace in enumerate(halfspaces_t):
                    h, g = halfspace.get_constraint_params()
                    print(f"Halfspace {i}: h={h}, g={g}")
                # END DEBUG PRINTS 
                
                for j, halfspace in enumerate(halfspaces_t):
                    h, g = halfspace.get_constraint_params()
                    
                    # Add a slack variable for soft constraint
                    slack = cp.Variable(name=f"slack_t{t}_h{j}")
                    safety_slacks.append(slack)
                    
                    # Add constraint: h·position + g ≤ slack
                    constraints.append(cp.sum(cp.multiply(h, position)) + g <= slack)
                    
                    # Add non-negativity constraint
                    constraints.append(slack >= 0)
                    
                    # Add high-weight penalty for violating constraint
                    objective += 50.0 * slack  # Linear penalty
                    objective += 50.0 * cp.square(slack)  # Quadratic penalty 

        # Create and solve the optimization problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            with Timer("MPC Solve"):
                problem.solve()
            
            # Process the results
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                x_filtered = x.value
                u_filtered = u.value
                self.last_optimal_u = u_filtered
                
                solve_time = time.time() - start_time
                
                return x_filtered, u_filtered, {
                    'status': problem.status,
                    'solve_time': solve_time,
                    'objective': problem.value
                }
            else: 
                # Problem was not solved optimally, use fallback
                return self._fallback(x0, x_ref, u_ref, {
                    'status': problem.status,
                    'error': 'Problem could not be solved optimally' 
                })
                
        except Exception as e:
            # Handle any exceptions
            return self._fallback(x0, x_ref, u_ref, {
                'status': 'ERROR',
                'error': str(e)
            })
    
    def _fallback(self, x0, x_ref, u_ref, info):
        """
        Fallback strategy when the MPC problem is infeasible.
        
        Args:
            x0: Initial state
            x_ref: Reference state trajectory
            u_ref: Reference input trajectory
            info: Information about the failure
        
        Returns:
            x_filtered: Fallback state trajectory
            u_filtered: Fallback input trajectory
            info: Updated information
        """
        info['used_fallback'] = True
        
        if self.last_optimal_u is not None:
            # Use the remaining part of the last optimal solution
            u_filtered = np.zeros((self.horizon, self.n_inputs))
            
            # Shift the last optimal control sequence
            remaining_steps = min(self.horizon - 1, len(self.last_optimal_u) - 1)
            u_filtered[:remaining_steps] = self.last_optimal_u[1:remaining_steps+1]
            
            # Fill the rest with zeros or the reference
            if remaining_steps < self.horizon:
                u_filtered[remaining_steps:] = u_ref[remaining_steps:]
        else:
            # No previous solution available, use the reference
            u_filtered = u_ref
        
        # Simulate the system with the fallback inputs
        x_filtered = np.zeros((self.horizon+1, self.n_states))
        x_filtered[0] = x0
        
        for t in range(self.horizon):
            x_filtered[t+1] = self.A @ x_filtered[t] + self.B @ u_filtered[t]
        
        return x_filtered, u_filtered, info