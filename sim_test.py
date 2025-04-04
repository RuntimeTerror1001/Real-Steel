import csv
import time

def create_test_file(filename, joint_values):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with joint names
        header = ["timestamp"]
        joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        header.extend(joint_names)
        writer.writerow(header)
        
        # Single row with specified values
        row = [0.0]  # timestamp
        row.extend(joint_values)
        writer.writerow(row)

# Create all zeros test file
create_test_file("test_all_zeros.csv", [0.0] * 14)

# Create test file with only right shoulder pitch = 0.5
zero_values = [0.0] * 14
zero_values[7] = 0.5  # right_shoulder_pitch_joint
create_test_file("test_right_shoulder_pitch_pos.csv", zero_values)

# Create test file with only right shoulder pitch = -0.5
zero_values = [0.0] * 14
zero_values[7] = -0.5  # right_shoulder_pitch_joint
create_test_file("test_right_shoulder_pitch_neg.csv", zero_values)

# Create test file with only right elbow = 0.5
zero_values = [0.0] * 14
zero_values[10] = 0.5  # right_elbow_joint
create_test_file("test_right_elbow_pos.csv", zero_values)