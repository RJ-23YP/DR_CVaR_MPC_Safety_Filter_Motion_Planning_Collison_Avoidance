"""
Configuration parameters for the DR-CVaR safety filtering implementation.
"""

# Risk parameters
ALPHA = 0.1          # CVaR confidence level (1-alpha quantile)
DELTA = 0.1          # Risk bound 
EPSILON = 0.15       # Wasserstein radius  

# Robot parameters
ROBOT_RADIUS = 0.3   # Radius of the ego robot (modeled as a circle)
DT = 0.2             # Time step (sec) 

# MPC parameters
HORIZON = 30         # MPC horizon 
Q_WEIGHT = 2.0       # Weight for state tracking error
R_WEIGHT = 1.0      # Weight for control input   

# Simulation parameters
SIM_TIME = 30.0      # Total simulation time (sec) 
NUM_SAMPLES = 20    # Number of obstacle trajectory samples  

# Obstacle parameters 
OBSTACLE_RADIUS = 0.3  # Radius of obstacles (modeled as circles)
OBSTACLE_SPEED = 1.0   # Nominal speed of obstacles 

# Monte Carlo parameters
NUM_MC_RUNS = 300    # Number of Monte Carlo simulation runs


# """
# Configuration parameters for the DR-CVaR safety filtering implementation.
# """

# # Risk parameters
# ALPHA = 0.2          # CVaR confidence level (1-alpha quantile)
# DELTA = 0.1          # Risk bound 
# EPSILON = 0.15       # Wasserstein radius  

# # Robot parameters
# ROBOT_RADIUS = 0.3   # Radius of the ego robot (modeled as a circle)
# DT = 0.2             # Time step (sec) 

# # MPC parameters
# HORIZON = 30         # MPC horizon 
# Q_WEIGHT = 2.0       # Weight for state tracking error
# R_WEIGHT = 1.0      # Weight for control input   

# # Simulation parameters
# SIM_TIME = 30.0      # Total simulation time (sec) 
# NUM_SAMPLES = 20    # Number of obstacle trajectory samples  

# # Obstacle parameters 
# OBSTACLE_RADIUS = 0.3  # Radius of obstacles (modeled as circles)
# OBSTACLE_SPEED = 1.0   # Nominal speed of obstacles 

# # Monte Carlo parameters
# NUM_MC_RUNS = 300    # Number of Monte Carlo simulation runs


