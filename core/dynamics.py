"""
Robot and obstacle dynamics for the DR-CVaR safety filtering.
"""
import numpy as np
from config.parameters import DT 

def create_double_integrator_matrices(dt=DT, dim=2):
    """
    Create state-space matrices for a double integrator system.
    
    Args:
        dt: Time step
        dim: Dimension of the state space (2 for 2D)
    
    Returns:
        A, B, C: State, input, and output matrices
    """
    # State matrix
    A = np.block([
        [np.eye(dim), dt * np.eye(dim)],
        [np.zeros((dim, dim)), np.eye(dim)]
    ])
    
    # Input matrix
    B = np.block([
        [0.5 * dt**2 * np.eye(dim)],
        [dt * np.eye(dim)]
    ])
    
    # Output matrix (extract position)
    C = np.block([np.eye(dim), np.zeros((dim, dim))])
    
    return A, B, C

def create_single_integrator_matrices(dt=DT, dim=2):
    """
    Create state-space matrices for a single integrator system.
    
    Args:
        dt: Time step
        dim: Dimension of the state space (2 for 2D)
    
    Returns:
        A, B, C: State, input, and output matrices
    """
    # State matrix
    A = np.eye(dim)
    
    # Input matrix
    B = dt * np.eye(dim)
    
    # Output matrix (identity since state = position)
    C = np.eye(dim)
    
    return A, B, C

def simulate_linear_system(x0, u_sequence, A, B, C):
    """
    Simulate a linear system given initial state and control inputs.
    
    Args:
        x0: Initial state
        u_sequence: Sequence of control inputs
        A, B, C: System matrices
    
    Returns:
        x_sequence: Sequence of states
        y_sequence: Sequence of outputs
    """
    n_steps = len(u_sequence)
    x_dim = A.shape[0]
    y_dim = C.shape[0]
    
    x_sequence = np.zeros((n_steps + 1, x_dim))
    y_sequence = np.zeros((n_steps + 1, y_dim))
    
    x_sequence[0] = x0
    y_sequence[0] = C @ x0
    
    for t in range(n_steps):
        x_sequence[t+1] = A @ x_sequence[t] + B @ u_sequence[t]
        y_sequence[t+1] = C @ x_sequence[t+1]
    
    return x_sequence, y_sequence