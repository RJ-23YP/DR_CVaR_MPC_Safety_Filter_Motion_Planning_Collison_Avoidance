"""
Reference trajectory planning for the DR-CVaR safety filtering simulations.
"""
import numpy as np
import cvxpy as cp
from utils.timing import timeit

class ReferenceTrajectoryPlanner:
    """
    Simple MPC-based reference trajectory planner.
    """
    def __init__(self, A, B, C, Q, R, horizon, dt):
        """
        Initialize the planner.
        
        Args:
            A, B, C: System matrices
            Q, R: Cost matrices
            horizon: Planning horizon
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
    
    @timeit
    def plan_trajectory(self, x0, goal_state, input_constraints=None, state_constraints=None):
        """
        Plan a reference trajectory from x0 to goal_state.
        
        Args:
            x0: Initial state
            goal_state: Goal state
            input_constraints: Input constraints (min, max)
            state_constraints: State constraints (min, max)
        
        Returns:
            x_ref: Reference state trajectory
            u_ref: Reference input trajectory
            info: Additional information
        """
        # Initialize variables
        x = cp.Variable((self.horizon+1, self.n_states))
        u = cp.Variable((self.horizon, self.n_inputs))
        
        # Initialize objective
        objective = 0
        
        # Add tracking costs
        for t in range(self.horizon):
            # State tracking cost (for reaching the goal)
            state_error = x[t+1] - goal_state
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
        
        # State constraints
        if state_constraints is not None:
            state_min, state_max = state_constraints
            for t in range(1, self.horizon+1):
                constraints.append(x[t] >= state_min)
                constraints.append(x[t] <= state_max)
        
        # Create and solve the optimization problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve()
            
            # Process the results
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                x_ref = x.value
                u_ref = u.value
                
                return x_ref, u_ref, {
                    'status': problem.status,
                    'objective': problem.value
                }
            else:
                # Problem was not solved optimally
                return None, None, {
                    'status': problem.status,
                    'error': 'Problem could not be solved optimally'
                }
                
        except Exception as e:
            # Handle any exceptions
            return None, None, {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def straight_line_trajectory(self, start_pos, goal_pos, velocity=1.5):
        """
        Generate a straight-line reference trajectory. 
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position
            velocity: Desired velocity
        
        Returns:
            x_ref: Reference state trajectory
            u_ref: Reference input trajectory
            plan_info: Additional information about the plan
        """
        # Compute the direction and distance
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction) 

        # Calculate total simulation steps based on SIM_TIME
        from config.parameters import SIM_TIME, DT
        total_steps = int(SIM_TIME / DT)
        
        # Ensure horizon doesn't exceed total steps
        simulation_horizon = min(self.horizon, total_steps) 
        
        if distance < 1e-10:
            # Start and goal are the same, return a stationary trajectory
            x_ref = np.zeros((self.horizon+1, self.n_states))
            u_ref = np.zeros((self.horizon, self.n_inputs))
            
            # Set the initial position
            x_ref[:, :2] = start_pos
            
            plan_info = {'status': 'OPTIMAL', 'distance': 0.0}
            return x_ref, u_ref, plan_info
        
        # Normalize the direction
        direction = direction / distance
        
        # Compute the time needed to reach the goal
        time_to_goal = distance / velocity
        n_steps = int(time_to_goal / self.dt)
        
        # Initialize trajectories
        x_ref = np.zeros((self.horizon+1, self.n_states))
        u_ref = np.zeros((self.horizon, self.n_inputs))
        
        # Set the initial state (position and zero velocity)
        x_ref[0, :2] = start_pos
        x_ref[0, 2:] = np.zeros(2)
        
        # Generate positions along the straight line
        for t in range(1, self.horizon+1):
            # Compute the position at time step t
            if t <= n_steps:
                # Moving towards the goal
                progress = t / n_steps
                x_ref[t, :2] = start_pos + progress * (goal_pos - start_pos)
                x_ref[t, 2:] = velocity * direction
            else:
                # Reached the goal, stay there
                x_ref[t, :2] = goal_pos
                x_ref[t, 2:] = np.zeros(2)
        
        # Compute the control inputs using the dynamics
        for t in range(self.horizon):
            # u[t] = B^+ (x[t+1] - A*x[t])
            u_ref[t] = np.linalg.pinv(self.B) @ (x_ref[t+1] - self.A @ x_ref[t])
        
        # Add additional plan information
        plan_info = {
            'status': 'OPTIMAL',
            'distance': distance,
            'time_to_goal': time_to_goal,
            'n_steps': n_steps
        }
        
        return x_ref, u_ref, plan_info 