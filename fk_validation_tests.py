import numpy as np
from motion_validator import MotionValidator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_fk(joint_angles, link_lengths):
    """
    Compute forward kinematics for a simple arm (shoulder pitch, roll, yaw, elbow)
    Returns end-effector position and orientation using right-hand coordinate system:
    - X-axis points forward
    - Y-axis points left
    - Z-axis points up
    """
    shoulder_pitch = joint_angles.get('right_shoulder_pitch', 0)
    shoulder_roll = joint_angles.get('right_shoulder_roll', 0)
    shoulder_yaw = joint_angles.get('right_shoulder_yaw', 0)
    elbow = joint_angles.get('right_elbow', 0)
    
    # Link lengths
    l1 = link_lengths['upper_arm']  # Upper arm length
    l2 = link_lengths['forearm']    # Forearm length
    
    # Transformation matrices (right-hand coordinate system)
    def Rx(theta):
        # Rotation about X-axis (roll)
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def Ry(theta):
        # Rotation about Y-axis (pitch)
        # Positive pitch rotates up
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def Rz(theta):
        # Rotation about Z-axis (yaw)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    # Initial position (shoulder)
    pos = np.zeros(3)
    
    # Upper arm transformation
    # For right arm:
    # - Positive pitch rotates up (about Y)
    # - Positive roll rotates outward (about X)
    # - Positive yaw rotates forward (about Z)
    R_shoulder = Rz(-shoulder_yaw) @ Ry(-shoulder_pitch) @ Rx(shoulder_roll)
    upper_arm_vec = np.array([l1, 0, 0])
    pos += R_shoulder @ upper_arm_vec
    
    # Forearm transformation
    # Positive elbow angle flexes inward (about Y)
    # For right arm, positive elbow flexion moves the forearm to the left (+Y)
    R_elbow = Rx(-np.pi/2) @ Ry(-elbow) @ Rx(np.pi/2)
    R_total = R_shoulder @ R_elbow
    forearm_vec = np.array([l2, 0, 0])
    pos += R_total @ forearm_vec
    
    return pos, R_total

def test_fk_basic_poses():
    """Test FK validation for basic poses"""
    print("\nTesting FK for basic poses...")
    
    # Define link lengths (in meters)
    link_lengths = {
        'upper_arm': 0.3,  # 30cm upper arm
        'forearm': 0.25    # 25cm forearm
    }
    
    # Test cases with expected results
    test_cases = [
        {
            'name': 'Rest position',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            },
            'expected_pos': np.array([0.55, 0, 0])  # Full extension along x-axis
        },
        {
            'name': 'Raised arm',
            'angles': {
                'right_shoulder_pitch': np.pi/2,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            },
            'expected_pos': np.array([0, 0, 0.55])  # Full extension along z-axis
        },
        {
            'name': '90-degree elbow',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': np.pi/2
            },
            'expected_pos': np.array([0.3, 0.25, 0])  # L-shape in x-y plane
        }
    ]
    
    # Run tests
    for test in test_cases:
        pos, rot = compute_fk(test['angles'], link_lengths)
        error = np.linalg.norm(pos - test['expected_pos'])
        print(f"\n{test['name']}:")
        print(f"Expected position: {test['expected_pos']}")
        print(f"Computed position: {pos}")
        print(f"Position error: {error:.6f} meters")
        
        # Check if error is within tolerance
        if error > 0.01:  # 1cm tolerance
            print("WARNING: Position error exceeds tolerance!")
            print("Rotation matrix:")
            print(rot)

def visualize_arm_motion(timestamps, positions):
    """Visualize the arm motion with intermediate positions"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='End-effector trajectory')
    
    # Plot some intermediate positions
    num_poses = len(positions)
    for i in range(0, num_poses, num_poses//5):  # Plot 5 intermediate poses
        ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], 
                  c='r', marker='o', s=100)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set axis limits
    max_val = np.max(np.abs(positions)) * 1.2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Arm End-Effector Trajectory')
    
    # Add timestamp annotations
    for i in range(0, num_poses, num_poses//5):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                f't={timestamps[i]:.1f}s', fontsize=8)
    
    plt.savefig('fk_trajectory.png')
    plt.close()

def test_complex_trajectory():
    """Test a more complex arm trajectory"""
    print("\nTesting complex trajectory...")
    
    link_lengths = {
        'upper_arm': 0.3,
        'forearm': 0.25
    }
    
    # Generate a figure-8 trajectory
    t = np.linspace(0, 4*np.pi, 100)
    shoulder_pitch = 0.5 * np.sin(t/2)  # Vertical motion
    shoulder_yaw = 0.3 * np.sin(t)      # Horizontal motion
    elbow = 0.5 + 0.3 * np.cos(t)       # Elbow flexion/extension
    
    # Compute trajectory points
    positions = []
    for i in range(len(t)):
        angles = {
            'right_shoulder_pitch': shoulder_pitch[i],
            'right_shoulder_roll': 0,
            'right_shoulder_yaw': shoulder_yaw[i],
            'right_elbow': elbow[i]
        }
        pos, _ = compute_fk(angles, link_lengths)
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Calculate trajectory statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    max_reach = np.max(np.linalg.norm(positions, axis=1))
    
    print(f"Trajectory statistics:")
    print(f"Total distance traveled: {total_distance:.3f} meters")
    print(f"Maximum reach: {max_reach:.3f} meters")
    
    # Visualize trajectory
    visualize_arm_motion(t, positions)
    print("Generated figure-8 trajectory visualization")

def test_workspace_limits():
    """Test if arm positions respect workspace limits"""
    print("\nTesting workspace limits...")
    
    link_lengths = {
        'upper_arm': 0.3,
        'forearm': 0.25
    }
    
    # Maximum reach
    max_reach = link_lengths['upper_arm'] + link_lengths['forearm']
    
    # Test extreme positions
    extreme_angles = [
        {
            'name': 'Full extension',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            }
        },
        {
            'name': 'Full flexion',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': np.pi
            }
        },
        {
            'name': 'Overhead reach',
            'angles': {
                'right_shoulder_pitch': np.pi/2,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            }
        }
    ]
    
    for test in extreme_angles:
        pos, _ = compute_fk(test['angles'], link_lengths)
        distance = np.linalg.norm(pos)
        print(f"\n{test['name']}:")
        print(f"End-effector position: {pos}")
        print(f"Distance from origin: {distance:.3f} meters")
        print(f"Maximum reach: {max_reach:.3f} meters")
        
        if distance > max_reach:
            print("WARNING: Position exceeds maximum reach!")

def main():
    print("Running FK validation tests...")
    
    test_fk_basic_poses()
    test_complex_trajectory()
    test_workspace_limits()
    
    print("\nFK validation tests completed!")

if __name__ == "__main__":
    main() 