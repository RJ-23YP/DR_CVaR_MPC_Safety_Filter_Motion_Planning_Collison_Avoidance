"""
Scenario definitions for testing the DR-CVaR safety filtering.
"""
import numpy as np

def get_scenario_config(scenario_name):
    """
    Get configuration for a specific scenario.
    
    Args:
        scenario_name: Name of the scenario ('head_on', 'overtaking', 'intersection')
    
    Returns:
        dict: Configuration for the scenario
    """
    if scenario_name == 'head_on':
        return {
            'ego_start': np.array([-4.0, 0.0]),
            'ego_goal': np.array([4.0, 0.0]), 
            'obstacle_start': np.array([4.0, 0.0]),
            'obstacle_direction': np.array([-1.0, 0.0]),
            'description': 'Head-on collision scenario' 
        }
    
    elif scenario_name == 'overtaking':
        return {
            'ego_start': np.array([-4.0, 0.0]),
            'ego_goal': np.array([4.0, 0.0]),
            'obstacle_start': np.array([-2.0, 0.0]), 
            'obstacle_direction': np.array([1.0, 0.0]),
            'obstacle_speed': 0.7,  # Slower than ego
            'description': 'Overtaking scenario'
        }
        
    elif scenario_name == 'intersection':
        return {
            'ego_start': np.array([-4.0, 0.0]),
            'ego_goal': np.array([4.0, 0.0]),
            'obstacle_start': np.array([0.0, 4.0]),
            'obstacle_direction': np.array([0.0, -1.0]), 
            'obstacle_speed': 1.5,
            'description': 'Intersection crossing scenario'
        } 
        
    elif scenario_name == 'multi_obstacle':
        return {
            'ego_start': np.array([-2.0, -1.0]),
            'ego_goal': np.array([4.0, 0.0]), 
            'obstacles': [
                # Obstacle 1: Moving downward from top
                {'start': np.array([0.0, 2.0]), 'direction': np.array([0.0, -0.5]), 'speed': 0.8},
                
                # Obstacle 2: Moving from left to right, crossing the ego's path
                {'start': np.array([-3.0, 0.5]), 'direction': np.array([0.7, 0.0]), 'speed': 0.6},
                
                # Obstacle 3: Moving diagonally from bottom right
                {'start': np.array([1.5, -2.0]), 'direction': np.array([-0.2, 0.5]), 'speed': 0.7}, 
            ],
            'description': 'Multiple obstacle scenario' 
        }
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}") 


# """
# Scenario definitions for testing the DR-CVaR safety filtering.
# """
# import numpy as np

# # Environment limits
# ENV_LIM = 5.0

# def get_scenario_config(scenario_name):
#     """
#     Get configuration for a specific scenario.
    
#     Args: 
#         scenario_name: Name of the scenario ('head_on', 'overtaking', 'intersection', 'multi_obstacle')
    
#     Returns:
#         dict: Configuration for the scenario
#     """
#     if scenario_name == 'head_on':
#         return {
#             'ego_start': np.array([-ENV_LIM + 0.3, 0.0]),
#             'ego_goal': np.array([ENV_LIM - 0.3, 0.0]),
#             'ego_velocity': np.array([1.5, 0.0]),
#             'obstacle_start': np.array([2.0, -0.01]),
#             'obstacle_direction': np.array([-1.0, 0.0]),
#             'obstacle_speed': 1.0,
#             'sim_time': 3.0,  # 3 seconds as in the paper
#             'description': 'Head-on collision scenario' 
#         }
    
#     elif scenario_name == 'overtaking':
#         return {
#             'ego_start': np.array([-ENV_LIM + 0.3, 0.0]),
#             'ego_goal': np.array([ENV_LIM - 0.3, 0.0]),
#             'ego_velocity': np.array([1.5, 0.0]),
#             'obstacle_start': np.array([-2.0, -0.05]),
#             'obstacle_direction': np.array([1.0, 0.0]),
#             'obstacle_speed': 1.0,
#             'sim_time': 3.0,  # 3 seconds as in the paper
#             'description': 'Overtaking scenario'
#         }
        
#     elif scenario_name == 'intersection':
#         return {
#             'ego_start': np.array([-3.5, 1.0]),
#             'ego_goal': np.array([1.0, -3.0]),
#             'ego_velocity': np.array([1.5, 0.0]),
#             'obstacle_start': np.array([-3.5, -1.0]),
#             'obstacle_direction': np.array([1.5, 0.0]),
#             'obstacle_speed': 1.5,
#             'sim_time': 3.0,  # 3 seconds as in the paper
#             'description': 'Intersection crossing scenario'
#         }
        
#     elif scenario_name == 'multi_obstacle':
#         return {
#             'ego_start': np.array([-ENV_LIM + 0.3, -1.0]),
#             'ego_goal': np.array([ENV_LIM - 0.3, 0.0]),
#             'ego_velocity': np.array([1.5, 0.0]),
#             'sim_time': 5.0,  # 5 seconds as in the paper
#             'obstacles': [
#                 {'start': np.array([-1.1, 1.01]), 'direction': np.array([0.7, 0.0]), 'speed': 0.7},
#                 {'start': np.array([-2.0, -1.01]), 'direction': np.array([1.0, 0.0]), 'speed': 1.0},
#                 {'start': np.array([-1.0, -2.01]), 'direction': np.array([0.7, 0.0]), 'speed': 0.7}
#             ],
#             'description': 'Multiple obstacle scenario with three dynamic obstacles'
#         }
        
#     else:
#         raise ValueError(f"Unknown scenario: {scenario_name}") 