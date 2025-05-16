import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Set the style to be clean and academic
plt.style.use('seaborn-v0_8-whitegrid')

# Create the figure with a grid layout
fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 3, figure=fig)

# Set up color scheme
colors = {
    'fk': '#1f77b4',     # Blue
    'ik': '#ff7f0e',     # Orange
    'error': '#d62728',  # Red
    'raw': '#7f7f7f',    # Gray
    'filtered': '#2ca02c', # Green
    'expected': '#9467bd', # Purple
    'actual': '#8c564b',   # Brown
}

# ----- LEFT PANEL: FK vs IK Joint Position Accuracy -----
ax1 = fig.add_subplot(gs[0, 0])

# Time data
time = np.linspace(0, 10, 200)

# Sample joint angle data for shoulder_pitch
fk_angles = 0.5 * np.sin(time) + 0.2 * np.cos(2*time) + 0.3
ik_angles = fk_angles + 0.02 * np.sin(5*time)  # Small oscillating error

# Error band calculation (convert radians to mm for visualization)
error_band = 0.05 * np.sin(5*time) * 20  # 20mm per radian (example conversion)
error_threshold = 5  # 5mm threshold

# Plot FK and IK angles
ax1.plot(time, fk_angles, color=colors['fk'], label='FK (Ground Truth)')
ax1.plot(time, ik_angles, color=colors['ik'], label='IK Solution')

# Highlight areas with error > threshold
error_mask = np.abs(error_band) > error_threshold
if np.any(error_mask):
    error_regions = np.where(error_mask)[0]
    for start_idx in error_regions:
        if start_idx > 0 and not error_mask[start_idx-1]:  # Start of a region
            end_idx = start_idx
            while end_idx < len(error_mask) and error_mask[end_idx]:
                end_idx += 1
            ax1.axvspan(time[start_idx], time[end_idx-1], alpha=0.3, color=colors['error'])

# Add labels and legend
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Joint Angle (radians)')
ax1.set_title('FK vs IK Joint Position Accuracy\n(shoulder_pitch)')
ax1.legend()

# Add annotation about error band
ax1.text(0.98, 0.02, 'Red areas: position error > 5 mm', 
         transform=ax1.transAxes, ha='right', fontsize=8, 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# ----- CENTER PANEL: Raw vs Filtered Joint Velocity -----
ax2 = fig.add_subplot(gs[0, 1])

# Generate some noisy velocity data
raw_velocity = np.diff(ik_angles) * 20  # Convert to velocity and scale
time_vel = time[:-1]  # Adjusted time array for velocity (one shorter)

# Add noise to raw velocity
np.random.seed(42)
raw_velocity += 0.1 * np.random.randn(len(raw_velocity))

# Create a simple filtered version (moving average)
window_size = 10
filtered_velocity = np.convolve(raw_velocity, np.ones(window_size)/window_size, mode='same')

# Add velocity limiting
velocity_limit = 0.2
limited_velocity = np.clip(filtered_velocity, -velocity_limit, velocity_limit)

# Plot raw and processed velocities
ax2.plot(time_vel, raw_velocity, color=colors['raw'], linestyle='-', alpha=0.5, label='Raw Velocity')
ax2.plot(time_vel, filtered_velocity, color=colors['filtered'], linestyle='-', linewidth=1.5, label='Filtered')
ax2.plot(time_vel, limited_velocity, color=colors['filtered'], linestyle='--', linewidth=2, label='Limited')

# Add horizontal lines at the limits
ax2.axhline(y=velocity_limit, color=colors['error'], linestyle=':', alpha=0.7)
ax2.axhline(y=-velocity_limit, color=colors['error'], linestyle=':', alpha=0.7)

# Add labels and legend
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Joint Velocity (rad/s)')
ax2.set_title('Velocity Filtering and Limiting\n(elbow joint)')
ax2.legend()

# Annotate the velocity limit
ax2.annotate('Velocity limit', xy=(5, velocity_limit), xytext=(5, velocity_limit + 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=8)

# ----- RIGHT PANEL: Expected vs Actual Trajectory -----
ax3 = fig.add_subplot(gs[0, 2], projection='3d')

# Create a sample trajectory (e.g., circular path)
t = np.linspace(0, 2*np.pi, 100)
x_expected = 150 * np.cos(t)
y_expected = 150 * np.sin(t)
z_expected = 30 * np.sin(2*t)

# Create actual trajectory with some deviation
deviation_magnitude = 8
x_actual = x_expected + deviation_magnitude * np.sin(3*t)
y_actual = y_expected + deviation_magnitude * np.cos(5*t)
z_actual = z_expected + deviation_magnitude * np.sin(7*t)

# Plot the trajectories
ax3.plot(x_expected, y_expected, z_expected, color=colors['expected'], linestyle=':', linewidth=2, label='Expected')
ax3.plot(x_actual, y_actual, z_actual, color=colors['actual'], linewidth=2, label='Actual')

# Plot deviation arrows at intervals
arrow_indices = np.arange(0, len(t), 10)
for i in arrow_indices:
    ax3.quiver(x_expected[i], y_expected[i], z_expected[i],
               x_actual[i] - x_expected[i], y_actual[i] - y_expected[i], z_actual[i] - z_expected[i],
               color=colors['error'], arrow_length_ratio=0.2, alpha=0.7)

# Set labels and title
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_zlabel('Z (mm)')
ax3.set_title('End-Effector Trajectory\nExpected vs Actual')
ax3.legend()

# Set equal aspect ratio
ax3.set_box_aspect([1, 1, 0.4])  # Adjust for better visualization

# Add a text annotation about the deviation
ax3.text2D(0.05, 0.95, 'Red arrows show deviation\nbetween expected and actual paths', 
          transform=ax3.transAxes, fontsize=8,
          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Final adjustments
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.suptitle('7-DOF Robot Arm Inverse Kinematics Analysis', fontsize=16, y=0.98)

# Save the figure
plt.savefig('ik_analysis_diagram.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show() 