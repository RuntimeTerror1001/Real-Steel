import numpy as np
import csv
import math

# Define the column headers as specified
JOINT_HEADERS = [
    "timestamp",
    "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
    "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint",
    "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint"
]

def generate_joint_angle_trajectory(start_angles, target_angles, max_increment=0.25, time_step=0.1):
    """
    Generate a trajectory of joint angles from start to target position with maximum increments.
    
    Args:
        start_angles (dict): Starting joint angles with joint names as keys
        target_angles (dict): Target joint angles with joint names as keys
        max_increment (float): Maximum change in angle per step
        time_step (float): Time step between samples in seconds
        
    Returns:
        list: List of dictionaries containing timestamp and joint angles
    """
    # Ensure all joints are present in both start and target angles
    joint_names = [header for header in JOINT_HEADERS if header != "timestamp"]
    for joint in joint_names:
        if joint not in start_angles or joint not in target_angles:
            raise ValueError(f"Joint '{joint}' missing in start or target angles")
    
    # Initialize current angles with start angles
    current_angles = {k: v for k, v in start_angles.items()}
    
    # Calculate the differences between target and start angles
    angle_diffs = {joint: target_angles[joint] - start_angles[joint] for joint in joint_names}
    
    # Calculate the number of steps needed for each joint
    steps_needed = {joint: math.ceil(abs(diff) / max_increment) for joint, diff in angle_diffs.items()}
    
    # Find the maximum number of steps needed
    max_steps = max(steps_needed.values()) if steps_needed else 0
    
    # Calculate the actual increment for each joint per step
    actual_increments = {
        joint: diff / max_steps if max_steps > 0 else 0 
        for joint, diff in angle_diffs.items()
    }
    
    # Initialize the trajectory list
    trajectory = []
    
    # Generate the trajectory
    for step in range(max_steps + 1):  # +1 to include the final position
        # Calculate the current time
        current_time = step * time_step
        
        # Create a dictionary for the current state
        state = {
            "timestamp": round(current_time, 2)  # Round to 2 decimal places
        }
        
        # Add joint angles to the state
        for joint in joint_names:
            state[joint] = round(current_angles[joint], 4)  # Round to 4 decimal places
        
        # Add the state to the trajectory
        trajectory.append(state)
        
        # Update current angles (except for the last step)
        if step < max_steps:
            for joint in joint_names:
                current_angles[joint] += actual_increments[joint]
    
    return trajectory

def save_trajectory_to_csv(trajectory, filename='joint_angle_trajectory.csv'):
    """
    Save the trajectory to a CSV file with specific column ordering.
    
    Args:
        trajectory (list): List of dictionaries containing timestamp and joint angles
        filename (str): Output CSV filename
    """
    if not trajectory:
        raise ValueError("Trajectory cannot be empty")
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=JOINT_HEADERS)
        writer.writeheader()
        writer.writerows(trajectory)
    
    print(f"Trajectory saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Example start and target joint angles (in radians)
    # Create dictionaries with the specific joint names
    start_angles = {
        "left_shoulder_pitch_joint": -1.25,
        "left_shoulder_yaw_joint": 0.0,
        "left_shoulder_roll_joint": 0.2,
        "left_elbow_joint": -0.2,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": -1.85,
        "right_shoulder_yaw_joint": 0.0,
        "right_shoulder_roll_joint": -0.25,
        "right_elbow_joint": 0.75,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_roll_joint": 1.5
    }
    
    target_angles = {
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_elbow_joint": 1.28,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0,
        "right_shoulder_yaw_joint": 0.0,
        "right_shoulder_roll_joint": 0,
        "right_elbow_joint": 1.28,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_roll_joint": 0.0
    }
    
    # Generate the trajectory
    trajectory = generate_joint_angle_trajectory(
        start_angles=start_angles,
        target_angles=target_angles,
        max_increment=0.15,  # Maximum angle change per step
        time_step=0.1        # Time increment in seconds
    )
    
    # Print the first few entries
    print("Sample of trajectory:")
    for i, state in enumerate(trajectory[:3]):
        print(state)
    print("...")
    print(trajectory[-1])  # Print the last entry
    
    # Save to CSV
    save_trajectory_to_csv(trajectory, 'joint_angle_trajectory.csv')