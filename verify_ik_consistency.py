import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ik_validation.log'),
        logging.StreamHandler()
    ]
)

# Constants for validation
MAX_VELOCITY = 2.0  # rad/s
MAX_ACCELERATION = 1.0  # rad/s²
DT = 0.1  # seconds between frames

# Joint limits in radians
JOINT_LIMITS = {
    'right_shoulder_pitch_joint': (-2.0, 2.0),
    'right_shoulder_yaw_joint': (-2.0, 2.0),
    'right_shoulder_roll_joint': (-1.5, 1.5),
    'right_elbow_joint': (0, 2.27),
    'right_wrist_pitch_joint': (-1.5, 1.5),
    'right_wrist_yaw_joint': (-1.5, 1.5),
    'right_wrist_roll_joint': (-1.5, 1.5),
    'left_shoulder_pitch_joint': (-2.0, 2.0),
    'left_shoulder_yaw_joint': (-2.0, 2.0),
    'left_shoulder_roll_joint': (-1.5, 1.5),
    'left_elbow_joint': (0, 2.27),
    'left_wrist_pitch_joint': (-1.5, 1.5),
    'left_wrist_yaw_joint': (-1.5, 1.5),
    'left_wrist_roll_joint': (-1.5, 1.5)
}

def check_joint_limits(joint_angles: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Check if joint angles are within their limits
    
    Args:
        joint_angles: Dictionary of joint angles
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of violation messages)
    """
    violations = []
    is_valid = True
    
    for joint, (min_limit, max_limit) in JOINT_LIMITS.items():
        if joint in joint_angles:
            angle = joint_angles[joint]
            if not (min_limit <= angle <= max_limit):
                violations.append(
                    f"{joint}: {angle:.3f} rad exceeds limits [{min_limit:.3f}, {max_limit:.3f}]"
                )
                is_valid = False
    
    return is_valid, violations

def check_motion_smoothness(joint_trajectories: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
    """
    Check if joint trajectories satisfy velocity and acceleration constraints
    
    Args:
        joint_trajectories: Dictionary of joint angle trajectories
        
    Returns:
        Tuple[bool, List[str]]: (is_smooth, list of violation messages)
    """
    violations = []
    is_smooth = True
    
    for joint_name, angles in joint_trajectories.items():
        # Calculate velocities and accelerations
        velocities = np.diff(angles) / DT
        accelerations = np.diff(velocities) / DT
        
        # Check velocity constraints
        max_vel = np.max(np.abs(velocities))
        if max_vel > MAX_VELOCITY:
            violations.append(
                f"{joint_name}: Max velocity {max_vel:.3f} rad/s exceeds limit {MAX_VELOCITY:.1f}"
            )
            is_smooth = False
            
        # Check acceleration constraints
        max_acc = np.max(np.abs(accelerations))
        if max_acc > MAX_ACCELERATION:
            violations.append(
                f"{joint_name}: Max acceleration {max_acc:.3f} rad/s² exceeds limit {MAX_ACCELERATION:.1f}"
            )
            is_smooth = False
    
    return is_smooth, violations

def validate_motion_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate joint angles and motion smoothness
    
    Args:
        data: DataFrame containing joint trajectories
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (validated data, list of all violations)
    """
    all_violations = []
    valid_frames = []
    
    # Process each frame
    for idx in range(len(data)):
        frame_data = data.iloc[idx].to_dict()
        joint_angles = {k: v for k, v in frame_data.items() if k != 'timestamp'}
        
        # Check joint limits
        is_valid, limit_violations = check_joint_limits(joint_angles)
        if not is_valid:
            all_violations.extend([f"Frame {idx}: {v}" for v in limit_violations])
            continue
        
        valid_frames.append(frame_data)
    
    if not valid_frames:
        logging.error("No valid frames found in the motion data")
        return pd.DataFrame(), all_violations
    
    # Create DataFrame from valid frames
    validated_data = pd.DataFrame(valid_frames)
    
    # Check motion smoothness
    joint_trajectories = {col: validated_data[col].values 
                         for col in validated_data.columns if col != 'timestamp'}
    is_smooth, smoothness_violations = check_motion_smoothness(joint_trajectories)
    
    if not is_smooth:
        all_violations.extend(smoothness_violations)
    
    return validated_data, all_violations

def plot_validation_results(motion_data: pd.DataFrame, violations: List[str]):
    """Generate plots for validation results"""
    plt.figure(figsize=(15, 15))
    
    # Plot joint angles
    plt.subplot(3, 1, 1)
    for col in motion_data.columns:
        if col != 'timestamp':
            plt.plot(motion_data['timestamp'], motion_data[col], label=col)
    plt.title('Joint Angles Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot velocities
    plt.subplot(3, 1, 2)
    for col in motion_data.columns:
        if col != 'timestamp':
            velocities = np.diff(motion_data[col]) / DT
            plt.plot(motion_data['timestamp'][1:], velocities, label=f'{col} velocity')
    plt.axhline(y=MAX_VELOCITY, color='r', linestyle='--', label='Max velocity limit')
    plt.axhline(y=-MAX_VELOCITY, color='r', linestyle='--')
    plt.title('Joint Velocities Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot accelerations
    plt.subplot(3, 1, 3)
    for col in motion_data.columns:
        if col != 'timestamp':
            velocities = np.diff(motion_data[col]) / DT
            accelerations = np.diff(velocities) / DT
            plt.plot(motion_data['timestamp'][2:], accelerations, 
                    label=f'{col} acceleration')
    plt.axhline(y=MAX_ACCELERATION, color='r', linestyle='--', 
                label='Max acceleration limit')
    plt.axhline(y=-MAX_ACCELERATION, color='r', linestyle='--')
    plt.title('Joint Accelerations Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (rad/s²)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/validation_results.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save violations to file
    if violations:
        with open('analysis/validation_violations.txt', 'w') as f:
            f.write('\n'.join(violations))

def main():
    input_csv = "recordings/retargeted_motion.csv"
    output_csv = "recordings/validated_motion.csv"
    
    logging.info("Starting motion validation process...")
    
    try:
        # Read input data
        data = pd.read_csv(input_csv)
        
        # Validate motion
        validated_data, violations = validate_motion_data(data)
        
        if len(validated_data) > 0:
            # Save validated motion
            validated_data.to_csv(output_csv, index=False)
            logging.info(f"Saved validated motion to {output_csv}")
            
            # Generate validation plots
            plot_validation_results(validated_data, violations)
            
            if violations:
                logging.warning("Motion validation completed with violations:")
                for violation in violations:
                    logging.warning(violation)
            else:
                logging.info("Motion validation completed successfully with no violations")
                
            # Print summary statistics
            logging.info("\nMotion Statistics:")
            for col in validated_data.columns:
                if col != 'timestamp':
                    velocities = np.diff(validated_data[col]) / DT
                    accelerations = np.diff(velocities) / DT
                    logging.info(f"\n{col}:")
                    logging.info(f"  Max velocity: {np.max(np.abs(velocities)):.3f} rad/s")
                    logging.info(f"  Max acceleration: {np.max(np.abs(accelerations)):.3f} rad/s²")
        else:
            logging.error("No valid frames remained after validation")
            
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        return

if __name__ == "__main__":
    main() 