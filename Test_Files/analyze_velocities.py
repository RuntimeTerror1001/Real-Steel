import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Joint limits in radians for the humanoid robot
JOINT_LIMITS = {
    'right_shoulder_pitch': (-2.0, 2.0),  # approximately ±115 degrees
    'right_shoulder_roll': (-1.5, 1.5),   # approximately ±85 degrees
    'right_elbow': (0, 2.27),             # approximately 0 to 130 degrees
    'left_shoulder_pitch': (-2.0, 2.0),
    'left_shoulder_roll': (-1.5, 1.5),
    'left_elbow': (0, 2.27)
}

def calculate_velocities(data):
    """Calculate velocities from joint angle data"""
    velocities = {}
    timestamps = data['timestamp'].values
    dt = 0.1  # 10 Hz sampling rate
    
    for column in data.columns:
        if column != 'timestamp':
            angles = data[column].values
            # Calculate velocity as difference between consecutive angles divided by time step
            velocity = np.diff(angles) / dt
            velocities[column] = velocity
            
    return velocities, timestamps[1:]  # Return velocities and corresponding timestamps

def plot_velocities(velocities, timestamps):
    """Plot joint velocities over time with peak detection"""
    plt.figure(figsize=(15, 12))
    
    # Plot left arm velocities
    plt.subplot(2, 1, 1)
    for joint in velocities:
        if 'left' in joint:
            vel = velocities[joint]
            plt.plot(timestamps, vel, label=joint, linewidth=2)
            
            # Find peaks (local maxima) in absolute velocity
            peaks = np.where((np.abs(vel[1:-1]) > np.abs(vel[:-2])) & 
                           (np.abs(vel[1:-1]) > np.abs(vel[2:])))[0] + 1
            if len(peaks) > 0:
                plt.plot(timestamps[peaks], vel[peaks], 'ro', label=f'{joint} peaks')
                
                # Annotate significant peaks (>1 rad/s)
                for peak_idx in peaks:
                    if abs(vel[peak_idx]) > 1.0:
                        plt.annotate(f'{vel[peak_idx]:.2f} rad/s', 
                                   (timestamps[peak_idx], vel[peak_idx]),
                                   xytext=(10, 10), textcoords='offset points')
    
    plt.title('Left Arm Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot right arm velocities
    plt.subplot(2, 1, 2)
    for joint in velocities:
        if 'right' in joint:
            vel = velocities[joint]
            plt.plot(timestamps, vel, label=joint, linewidth=2)
            
            # Find peaks in absolute velocity
            peaks = np.where((np.abs(vel[1:-1]) > np.abs(vel[:-2])) & 
                           (np.abs(vel[1:-1]) > np.abs(vel[2:])))[0] + 1
            if len(peaks) > 0:
                plt.plot(timestamps[peaks], vel[peaks], 'ro', label=f'{joint} peaks')
                
                # Annotate significant peaks (>1 rad/s)
                for peak_idx in peaks:
                    if abs(vel[peak_idx]) > 1.0:
                        plt.annotate(f'{vel[peak_idx]:.2f} rad/s', 
                                   (timestamps[peak_idx], vel[peak_idx]),
                                   xytext=(10, 10), textcoords='offset points')
    
    plt.title('Right Arm Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Add a text box with motion analysis
    analysis_text = "Motion Analysis:\n"
    for joint in velocities:
        max_vel = np.max(np.abs(velocities[joint]))
        if max_vel > 0.1:  # Only show joints with significant motion
            analysis_text += f"{joint}:\n"
            analysis_text += f"  Max velocity: {max_vel:.2f} rad/s\n"
            
            # Calculate acceleration
            accel = np.diff(velocities[joint]) / 0.1
            max_accel = np.max(np.abs(accel))
            analysis_text += f"  Max acceleration: {max_accel:.2f} rad/s²\n"
    
    plt.figtext(1.15, 0.5, analysis_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for the legend
    
    # Save the plot with high DPI for better quality
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/velocity_analysis.png', dpi=300, bbox_inches='tight')
    print("Velocity analysis plot saved as 'analysis/velocity_analysis.png'")
    plt.close()

def check_joint_limits(data):
    """Check if joint angles are within their limits"""
    limit_violations = {}
    
    for joint, (min_limit, max_limit) in JOINT_LIMITS.items():
        if joint in data.columns:
            angles = data[joint].values
            violations = np.where((angles < min_limit) | (angles > max_limit))[0]
            if len(violations) > 0:
                limit_violations[joint] = {
                    'timestamps': violations * 0.1,  # Convert frame indices to timestamps
                    'values': angles[violations],
                    'limits': (min_limit, max_limit)
                }
    
    return limit_violations

def analyze_ik_quality(data):
    """Analyze the quality of IK solutions"""
    analysis = {}
    
    # Calculate joint angle changes between consecutive frames
    for joint in JOINT_LIMITS.keys():
        if joint in data.columns:
            angles = data[joint].values
            changes = np.abs(np.diff(angles))
            
            analysis[joint] = {
                'max_change': np.max(changes),
                'mean_change': np.mean(changes),
                'std_change': np.std(changes)
            }
    
    return analysis

def plot_joint_errors(data, limit_violations, ik_analysis):
    """Plot joint errors and IK quality metrics"""
    plt.figure(figsize=(15, 12))
    
    # Plot joint limit violations
    plt.subplot(2, 1, 1)
    for joint in JOINT_LIMITS.keys():
        if joint in data.columns:
            angles = data[joint].values
            timestamps = np.arange(len(angles)) * 0.1
            plt.plot(timestamps, angles, label=joint)
            
            # Plot joint limits as horizontal lines
            min_limit, max_limit = JOINT_LIMITS[joint]
            plt.axhline(y=min_limit, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=max_limit, color='r', linestyle='--', alpha=0.3)
            
            # Highlight violations
            if joint in limit_violations:
                violation_data = limit_violations[joint]
                plt.scatter(violation_data['timestamps'], violation_data['values'], 
                          color='red', marker='x', s=100, label=f'{joint} violations')
    
    plt.title('Joint Angles and Limit Violations')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot IK solution quality metrics
    plt.subplot(2, 1, 2)
    joints = list(ik_analysis.keys())
    max_changes = [ik_analysis[j]['max_change'] for j in joints]
    mean_changes = [ik_analysis[j]['mean_change'] for j in joints]
    
    x = np.arange(len(joints))
    width = 0.35
    
    plt.bar(x - width/2, max_changes, width, label='Max Change per Frame')
    plt.bar(x + width/2, mean_changes, width, label='Mean Change per Frame')
    plt.xticks(x, joints, rotation=45)
    plt.title('IK Solution Quality Metrics')
    plt.ylabel('Joint Angle Change (rad/frame)')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/ik_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read the CSV file
    data = pd.read_csv('/recordings/robot_motion_20250424-101531.csv')
    
    # Check joint limits
    limit_violations = check_joint_limits(data)
    
    # Analyze IK quality
    ik_analysis = analyze_ik_quality(data)
    
    # Plot validation results
    plot_joint_errors(data, limit_violations, ik_analysis)
    
    # Print analysis results
    print("\nJoint Limit Violations:")
    if limit_violations:
        for joint, violations in limit_violations.items():
            print(f"\n{joint}:")
            print(f"  Number of violations: {len(violations['timestamps'])}")
            print(f"  Allowed range: [{violations['limits'][0]:.2f}, {violations['limits'][1]:.2f}] rad")
            print("  Violations at timestamps (s):", 
                  ", ".join(f"{t:.1f}" for t in violations['timestamps'][:5]))
            if len(violations['timestamps']) > 5:
                print(f"  ... and {len(violations['timestamps'])-5} more")
    else:
        print("  No joint limit violations found")
    
    print("\nIK Solution Quality:")
    for joint, metrics in ik_analysis.items():
        print(f"\n{joint}:")
        print(f"  Maximum change between frames: {metrics['max_change']:.3f} rad")
        print(f"  Average change between frames: {metrics['mean_change']:.3f} rad")
        print(f"  Standard deviation of changes: {metrics['std_change']:.3f} rad")
        
        # Add warnings for potentially problematic changes
        if metrics['max_change'] > 0.5:  # More than ~30 degrees per frame
            print("  WARNING: Large sudden movement detected")
        if metrics['mean_change'] > 0.2:  # More than ~11 degrees average change
            print("  WARNING: Motion might be too fast")
    
    # Calculate velocities and plot them as before
    velocities, timestamps = calculate_velocities(data)
    plot_velocities(velocities, timestamps)

if __name__ == "__main__":
    main() 