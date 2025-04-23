import numpy as np
import pandas as pd
import os
from ik_analytical3d import IKAnalytical3D

def generate_boxing_motion(num_frames=30, dt=0.1):
    """Generate a sample boxing motion (jab) with proper joint angles."""
    
    # Initialize time array
    timestamps = np.arange(0, num_frames * dt, dt)
    
    # Define joint limits (in radians)
    joint_limits = {
        'shoulder_pitch': (-2.0, 2.0),    # Forward/backward motion
        'shoulder_yaw': (-2.0, 2.0),      # Side-to-side motion
        'shoulder_roll': (-1.5, 1.5),     # Rotation
        'elbow': (0.0, 2.5),              # Elbow flexion
        'wrist_pitch': (-1.0, 1.0),       # Wrist up/down
        'wrist_yaw': (-1.0, 1.0),         # Wrist side-to-side
        'wrist_roll': (-1.0, 1.0)         # Wrist rotation
    }
    
    # Generate smooth motion using cosine interpolation
    def cosine_interpolate(start, end, fraction):
        cos_fraction = (1 - np.cos(fraction * np.pi)) / 2
        return start + (end - start) * cos_fraction
    
    # Initialize arrays for each joint
    motion_data = {
        'timestamp': timestamps,
        'left_shoulder_pitch_joint': np.zeros(num_frames),
        'left_shoulder_yaw_joint': np.zeros(num_frames),
        'left_shoulder_roll_joint': np.zeros(num_frames),
        'left_elbow_joint': np.zeros(num_frames),
        'left_wrist_pitch_joint': np.zeros(num_frames),
        'left_wrist_yaw_joint': np.zeros(num_frames),
        'left_wrist_roll_joint': np.zeros(num_frames),
        'right_shoulder_pitch_joint': np.zeros(num_frames),
        'right_shoulder_yaw_joint': np.zeros(num_frames),
        'right_shoulder_roll_joint': np.zeros(num_frames),
        'right_elbow_joint': np.zeros(num_frames),
        'right_wrist_pitch_joint': np.zeros(num_frames),
        'right_wrist_yaw_joint': np.zeros(num_frames),
        'right_wrist_roll_joint': np.zeros(num_frames)
    }
    
    # Instantiate IK solver and define a constant hand orientation (identity)
    solver = IKAnalytical3D()
    target_orientation = np.eye(3)
    
    # Generate a jab motion for the right arm
    for i, t in enumerate(timestamps):
        # Normalize time to [0, 1] for the full motion
        phase = min(t / (timestamps[-1]), 1.0)
        
        # Right arm jab motion (example values)
        if phase < 0.3:  # Wind up
            motion_data['right_shoulder_pitch_joint'][i] = cosine_interpolate(0, -0.5, phase/0.3)
            motion_data['right_shoulder_roll_joint'][i] = cosine_interpolate(0, -0.2, phase/0.3)
            motion_data['right_elbow_joint'][i] = cosine_interpolate(1.2, 1.5, phase/0.3)
        elif phase < 0.6:  # Extend
            local_phase = (phase - 0.3) / 0.3
            motion_data['right_shoulder_pitch_joint'][i] = cosine_interpolate(-0.5, -1.2, local_phase)
            motion_data['right_shoulder_roll_joint'][i] = cosine_interpolate(-0.2, -0.1, local_phase)
            motion_data['right_elbow_joint'][i] = cosine_interpolate(1.5, 0.3, local_phase)
        else:  # Retract
            local_phase = (phase - 0.6) / 0.4
            motion_data['right_shoulder_pitch_joint'][i] = cosine_interpolate(-1.2, 0, local_phase)
            motion_data['right_shoulder_roll_joint'][i] = cosine_interpolate(-0.1, 0, local_phase)
            motion_data['right_elbow_joint'][i] = cosine_interpolate(0.3, 1.2, local_phase)
        
        # Left arm stays in guard position
        motion_data['left_shoulder_pitch_joint'][i] = -0.3
        motion_data['left_shoulder_roll_joint'][i] = 0.2
        motion_data['left_elbow_joint'][i] = 1.2
        
        # Compute wrist angles using IK solver to satisfy target orientation
        # Right arm
        wp, wy, wr = solver._wrist_angles_with_orientation(
            None, None, None, target_orientation,
            motion_data['right_shoulder_pitch_joint'][i],
            motion_data['right_shoulder_yaw_joint'][i],
            motion_data['right_shoulder_roll_joint'][i],
            motion_data['right_elbow_joint'][i]
        )
        motion_data['right_wrist_pitch_joint'][i] = wp
        motion_data['right_wrist_yaw_joint'][i] = wy
        motion_data['right_wrist_roll_joint'][i] = wr
        
        # Left arm
        lp, ly, lr = solver._wrist_angles_with_orientation(
            None, None, None, target_orientation,
            motion_data['left_shoulder_pitch_joint'][i],
            motion_data['left_shoulder_yaw_joint'][i],
            motion_data['left_shoulder_roll_joint'][i],
            motion_data['left_elbow_joint'][i]
        )
        motion_data['left_wrist_pitch_joint'][i] = lp
        motion_data['left_wrist_yaw_joint'][i] = ly
        motion_data['left_wrist_roll_joint'][i] = lr
    
    # Clip all angles to their limits
    for joint_name, (min_limit, max_limit) in joint_limits.items():
        for side in ['left', 'right']:
            full_name = f'{side}_{joint_name}_joint'
            if full_name in motion_data:
                motion_data[full_name] = np.clip(motion_data[full_name], min_limit, max_limit)
    
    # Create DataFrame
    df = pd.DataFrame(motion_data)
    
    # Ensure the recordings directory exists
    os.makedirs('recordings', exist_ok=True)
    
    # Save to CSV
    output_file = 'recordings/retargeted_motion.csv'
    df.to_csv(output_file, index=False)
    print(f"Motion data saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Generate a 3-second motion at 10Hz (30 frames)
    csv_file = generate_boxing_motion(num_frames=30, dt=0.1)
    print("You can now run the simulation using:")
    print(f"mjpython run_simulation.py") 