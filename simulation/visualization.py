"""
Visualization utilities for the DR-CVaR safety filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

def plot_trajectory(ax, trajectory, color='b', marker='o', linestyle='-', markersize=4, label=None):
    """
    Plot a trajectory.
    
    Args:
        ax: Matplotlib axis
        trajectory: Trajectory to plot [n_steps, 2]
        color: Color of the trajectory
        marker: Marker style
        linestyle: Line style
        markersize: Marker size
        label: Label for the legend
    """
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, marker=marker, 
            linestyle=linestyle, markersize=markersize, label=label)

def plot_robot(ax, position, radius, color='b', alpha=0.7, label=None):
    """
    Plot a robot as a circle.
    
    Args:
        ax: Matplotlib axis
        position: Position of the robot [x, y]
        radius: Radius of the robot
        color: Color of the robot
        alpha: Transparency
        label: Label for the legend
    """
    circle = Circle(position, radius, color=color, alpha=alpha, label=label)
    ax.add_patch(circle)

def plot_halfspace(ax, halfspace, xlim, ylim, color='g', alpha=0.2, label=None):
    """
    Plot a halfspace.
    
    Args:
        ax: Matplotlib axis
        halfspace: Halfspace to plot
        xlim, ylim: Limits of the plot
        color: Color of the halfspace
        alpha: Transparency
        label: Label for the legend
    """
    h, g = halfspace.get_constraint_params()
    
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the halfspace function
    Z = h[0] * X + h[1] * Y + g
    
    # Plot the contour at 0
    ax.contour(X, Y, Z, levels=[0], colors=color)
    
    # Fill the safe region
    ax.contourf(X, Y, Z, levels=[-np.inf, 0], colors=color, alpha=alpha, label=label)

def plot_scenario(ego_trajectory, obstacle_trajectories, robot_radius, obstacle_radius, 
                  safe_halfspaces=None, xlim=(-5, 5), ylim=(-5, 5), title=None):
    """
    Plot a complete scenario.
    
    Args:
        ego_trajectory: Ego robot trajectory
        obstacle_trajectories: Obstacle trajectories
        robot_radius: Radius of the ego robot
        obstacle_radius: Radius of the obstacle
        safe_halfspaces: List of safe halfspaces
        xlim, ylim: Limits of the plot
        title: Title of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ego trajectory
    plot_trajectory(ax, ego_trajectory, color='b', label='Ego')
    
    # Plot ego start and end
    plot_robot(ax, ego_trajectory[0], robot_radius, color='b', alpha=0.3)
    plot_robot(ax, ego_trajectory[-1], robot_radius, color='b', alpha=0.7)
    
    # Plot obstacles
    colors = cm.tab10(np.linspace(0, 1, len(obstacle_trajectories)))
    
    for i, obstacle_traj in enumerate(obstacle_trajectories):
        color = colors[i]
        plot_trajectory(ax, obstacle_traj, color=color, label=f'Obstacle {i}')
        
        # Plot obstacle start and end
        plot_robot(ax, obstacle_traj[0], obstacle_radius, color=color, alpha=0.3)
        plot_robot(ax, obstacle_traj[-1], obstacle_radius, color=color, alpha=0.7)
    
    # Plot safe halfspaces if provided
    if safe_halfspaces is not None:
        for t, halfspaces_t in enumerate(safe_halfspaces):
            if t % 3 == 0:  # Plot every third to avoid clutter
                for i, halfspace in enumerate(halfspaces_t):
                    plot_halfspace(ax, halfspace, xlim, ylim, 
                                  color=colors[i], alpha=0.1)
    
    # Set plot limits and title
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Safety Filtering Scenario')
    
    ax.legend()
    
    return fig, ax

def plot_distance_to_collision(ax, distances, threshold=0, 
                             color='b', label=None, show_boxplot=False):
    """
    Plot the distance to collision over time.
    
    Args:
        ax: Matplotlib axis
        distances: Array of distances to collision over time
        threshold: Collision threshold (typically 0)
        color: Color of the plot
        label: Label for the legend
        show_boxplot: Whether to show a boxplot instead of individual lines
    """
    if show_boxplot:
        # For Monte Carlo results
        ax.boxplot(distances, patch_artist=True,
                  boxprops=dict(facecolor=color, color='black'),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'),
                  medianprops=dict(color='white'))
        
        # Add a red line at the threshold
        ax.axhline(y=threshold, color='r', linestyle='--', label='Collision threshold')
        
        # Set labels
        ax.set_xlabel('Time step')
        ax.set_ylabel('Distance to collision')
        
    else:
        # For a single run
        time_steps = np.arange(len(distances))
        ax.plot(time_steps, distances, color=color, label=label)
        
        # Add a red line at the threshold
        ax.axhline(y=threshold, color='r', linestyle='--', label='Collision threshold')
        
        # Set labels
        ax.set_xlabel('Time step')
        ax.set_ylabel('Distance to collision')
        
        # Highlight collisions
        collision_mask = distances < threshold
        if np.any(collision_mask):
            ax.scatter(time_steps[collision_mask], distances[collision_mask],
                      color='r', s=50, label='Collision')

def compare_risk_metrics(mean_distances, cvar_distances, dr_cvar_distances, title=None):
    """
    Compare different risk metrics with boxplots.
    
    Args:
        mean_distances: Distances with mean risk metric
        cvar_distances: Distances with CVaR risk metric
        dr_cvar_distances: Distances with DR-CVaR risk metric
        title: Title of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for boxplot
    data = [mean_distances, cvar_distances, dr_cvar_distances]
    labels = ['Mean', 'CVaR', 'DR-CVaR']
    colors = ['red', 'blue', 'green']
    
    # Create boxplot
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    
    # Color the boxes
    for i, color in enumerate(colors):
        box['boxes'][i].set_facecolor(color)
    
    # Add a black line at the collision threshold
    ax.axhline(y=0, color='k', linestyle='--', label='Collision threshold')
    
    # Set labels and title
    ax.set_xlabel('Risk metric')
    ax.set_ylabel('Distance to collision')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Comparison of risk metrics')
    
    return fig, ax

def animate_scenario(ego_trajectory, obstacle_trajectories, robot_radius, obstacle_radius, 
                     safe_halfspaces=None, xlim=(-5, 5), ylim=(-5, 5), title=None, 
                     interval=100, save_path=None): 
    """
    Create an animation of the safety filtering scenario.
    
    Args:
        ego_trajectory: Ego robot trajectory [n_steps, n_states]
        obstacle_trajectories: List of obstacle trajectories [n_obstacles][n_steps, 2]
        robot_radius: Radius of the ego robot
        obstacle_radius: Radius of the obstacle
        safe_halfspaces: List of safe halfspaces for each time step
        xlim, ylim: Limits of the plot
        title: Title of the animation
        interval: Interval between frames in milliseconds
        save_path: Path to save the animation (if None, animation is not saved)
    
    Returns:
        animation: Matplotlib animation object
    """
    import matplotlib.animation as animation
    
    # Get the number of time steps
    n_steps = min(len(ego_trajectory), 
                 min([len(traj) for traj in obstacle_trajectories]))
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set plot limits and title
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Safety Filtering Animation')
    
    # Initialize robot and obstacle circles
    ego_circle = Circle((0, 0), robot_radius, color='blue', alpha=0.7, label='Ego Robot')
    ax.add_patch(ego_circle)
    
    obstacle_circles = []
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan']  # Predefined color list
    
    for i in range(len(obstacle_trajectories)):
        color = colors[i % len(colors)]  # Use modulo to cycle through colors
        circle = Circle((0, 0), obstacle_radius, color=color, alpha=0.7, 
                      label=f'Obstacle {i+1}')
        ax.add_patch(circle)
        obstacle_circles.append(circle)
    
    # Plot the full trajectories as dotted lines for reference
    ax.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 'b--', alpha=0.3, label='Ego Path')
    
    for i, obstacle_traj in enumerate(obstacle_trajectories):
        color = colors[i % len(colors)]
        ax.plot(obstacle_traj[:, 0], obstacle_traj[:, 1], '--', color=color, alpha=0.3, 
               label=f'Obstacle {i+1} Path')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Text annotation for time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Store halfspace artists to remove them between frames
    halfspace_artists = []
    
    def init():
        """Initialize the animation."""
        ego_circle.center = (ego_trajectory[0, 0], ego_trajectory[0, 1])
        
        for i, circle in enumerate(obstacle_circles):
            circle.center = (obstacle_trajectories[i][0, 0], obstacle_trajectories[i][0, 1])
        
        time_text.set_text('')
        
        return [ego_circle] + obstacle_circles + [time_text]
    
    def update(frame):
        """Update the animation for a given frame."""
        # Clear previous halfspace artists
        for artist in halfspace_artists:
            artist.remove()
        halfspace_artists.clear()
        
        # Update ego position
        ego_circle.center = (ego_trajectory[frame, 0], ego_trajectory[frame, 1])
        
        # Update obstacle positions
        for i, circle in enumerate(obstacle_circles):
            circle.center = (obstacle_trajectories[i][frame, 0], obstacle_trajectories[i][frame, 1])
        
        # Update time text
        time_text.set_text(f'Time: {frame * interval/1000:.1f}s')
        
        # Update safe halfspaces if provided
        if safe_halfspaces is not None and frame < len(safe_halfspaces):
            halfspaces_t = safe_halfspaces[frame]
            
            for i, halfspace in enumerate(halfspaces_t):
                h, g = halfspace.get_constraint_params()
                
                # Create a line showing the halfspace boundary
                if abs(h[1]) > 1e-6:  # Non-vertical line
                    x_vals = np.array([xlim[0], xlim[1]])
                    y_vals = (-g - h[0] * x_vals) / h[1]
                else:  # Vertical line
                    y_vals = np.array([ylim[0], ylim[1]])
                    x_vals = -g / h[0] * np.ones_like(y_vals)
                
                # Only draw the line if it's within the plot limits
                mask = (y_vals >= ylim[0]) & (y_vals <= ylim[1])
                if np.any(mask):
                    color = colors[i % len(colors)]
                    line = ax.plot(x_vals[mask], y_vals[mask], color=color, linestyle='-', 
                                  alpha=0.4, linewidth=2)[0]
                    halfspace_artists.append(line)
                    
                    # Calculate a point in the safe direction
                    mid_x = (xlim[0] + xlim[1]) / 2
                    mid_y = (ylim[0] + ylim[1]) / 2
                    mid_point = np.array([mid_x, mid_y])
                    
                    # If h·p + g <= 0 defines the safe region, 
                    # then the safe direction is opposite to h
                    safe_dir = -h / np.linalg.norm(h)
                    arrow_start = ego_circle.center
                    
                    # Calculate the arrow's endpoints
                    arrow_length = robot_radius * 2
                    arrow_dx = safe_dir[0] * arrow_length
                    arrow_dy = safe_dir[1] * arrow_length
                    
                    # Draw a small arrow in the safe direction
                    arrow = ax.arrow(arrow_start[0], arrow_start[1], 
                                    arrow_dx, arrow_dy, 
                                    color=color, alpha=0.4, width=0.05)
                    halfspace_artists.append(arrow)
        
        return [ego_circle] + obstacle_circles + [time_text] + halfspace_artists
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=n_steps, init_func=init, 
                                  interval=interval, blit=False)
    
    # Save the animation if requested
    if save_path:
        try:
            # First try with ffmpeg
            anim.save(save_path, writer='ffmpeg', fps=1000/interval, dpi=100)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save with ffmpeg: {e}")
            
            try:
                # Try with pillow
                print("Trying to save with Pillow...")
                anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=10)
                print(f"Animation saved as GIF to {save_path.replace('.mp4', '.gif')}")
            except Exception as e2:
                print(f"Could not save with Pillow either: {e2}")
                print("Animation will be displayed but not saved.")
    
    return anim, fig


# Create a function to evaluate a halfspace at all grid points
def evaluate_halfspace(h, g, positions):
    return np.dot(positions, h) + g 

def visualize_trajectory_with_halfspaces(ego_trajectory, obstacle_trajectories, safe_halfspaces, 
                                         robot_radius, obstacle_radius, xlim=(-5, 5), ylim=(-5, 5),
                                         title=None, save_path=None):
    """
    Visualize the ego trajectory, obstacles and safe halfspaces at each step.
    """
    # Create figure with specific figsize with equal dimensions
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Set plot limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')  # This ensures the plot scale is equal in both dimensions
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    
    # Add title if provided
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Trajectory with Safe Halfspaces')
    
    # Create a grid of points to visualize the halfspace intersections
    x = np.linspace(xlim[0], xlim[1], 300)
    y = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.flatten(), Y.flatten())) 
    
    # Sample time steps to reduce clutter
    sampled_steps = range(0, len(safe_halfspaces), 3)
    
    # For each time step, determine the safe region formed by the intersection of all halfspaces
    for t in sampled_steps:
        if t >= len(ego_trajectory):
            continue
            
        # Create a safety mask (1 = safe, 0 = unsafe)
        safety_mask = np.ones(len(grid_points), dtype=bool)
        
        # Apply each halfspace constraint
        for halfspace in safe_halfspaces[t]:
            h, g = halfspace.get_constraint_params()
            constraint_values = np.dot(grid_points, h) + g
            # Points where constraint_values <= 0 are inside the halfspace (safe)
            safety_mask = safety_mask & (constraint_values <= 0)
        
        # Reshape to grid for visualization
        safety_grid = safety_mask.reshape(X.shape)
        
        # Plot the safe region with a low alpha to show overlapping regions
        ax.contourf(X, Y, safety_grid, levels=[0.5, 1.5], colors=['green'], alpha=0.1)
        
        # Plot the boundary of the safe region 
        try:
            ax.contour(X, Y, safety_grid, levels=[0.5], colors=['green'], linewidths=0.5, alpha=0.4)
        except:
            # Skip if contour fails
            pass
    
    # Plot the trajectories and robots
    # Plot ego trajectory
    ego_pos = ego_trajectory[:, :2]  # Extract position from state
    ax.plot(ego_pos[:, 0], ego_pos[:, 1], 'b-', linewidth=2, label='ego')
    
    # Plot ego robot at selected positions
    for t in range(0, len(ego_trajectory), 2):  # Every 2nd position
        circle = Circle(ego_pos[t], robot_radius, color='blue', alpha=0.7)
        ax.add_patch(circle)
        # Add timestep label
        ax.text(ego_pos[t, 0], ego_pos[t, 1], str(t), color='white', 
               horizontalalignment='center', verticalalignment='center')
    
    # Plot obstacle trajectories
    colors = ['red', 'orange', 'magenta', 'green']
    for i, obstacle_traj in enumerate(obstacle_trajectories):
        color = colors[i % len(colors)]
        label = f'ob{i}'
        ax.plot(obstacle_traj[:, 0], obstacle_traj[:, 1], '-', color=color, linewidth=2, label=label)
        
        # Plot obstacle at selected positions
        for t in range(0, len(obstacle_traj), 4):  # Every 4th position
            circle = Circle(obstacle_traj[t], obstacle_radius, color=color, alpha=0.7)
            ax.add_patch(circle)
            # Add timestep label
            ax.text(obstacle_traj[t, 0], obstacle_traj[t, 1], str(t), color='white', 
                   horizontalalignment='center', verticalalignment='center')
    
    ax.legend()
    ax.grid(True)
    
    # Adjust layout to ensure proper display
    fig.tight_layout()
    
    if save_path:
        # Make sure the figure is rendered with correct dimensions
        fig.savefig(save_path, dpi=300, bbox_inches=None, pad_inches=0.1, format='png')
        # plt.close(fig)  # Close the figure to free memory
    
    return fig, ax