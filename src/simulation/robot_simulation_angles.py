#!/usr/bin/env python3
"""
Generate a robot simulation-compatible CSV file with the updated DH parameters
This includes columns for both left and right arm joints
"""

import numpy as np
import csv
from ik_analytical3d import IKAnalytical3DRefined

def create_robot_simulation_file():
    # Create solver instance
    solver = IKAnalytical3DRefined()
    
    # Define test angles for the left arm
    test_angles = [
        # Zero position
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        # Each joint individually
        {'shoulder_yaw': np.radians(20.0), 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': np.radians(20.0), 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': np.radians(20.0), 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': np.radians(45.0), 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': np.radians(20.0), 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': np.radians(20.0), 'wrist_roll': 0.0},
        
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': np.radians(20.0)},
        
        # Combination movements
        {'shoulder_yaw': np.radians(15.0), 'shoulder_pitch': np.radians(15.0), 'shoulder_roll': np.radians(15.0), 
         'elbow': np.radians(30.0), 'wrist_pitch': np.radians(15.0), 'wrist_yaw': np.radians(15.0), 'wrist_roll': np.radians(15.0)},
        
        # Return to zero
        {'shoulder_yaw': 0.0, 'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 
         'elbow': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0, 'wrist_roll': 0.0},
    ]
    
    # Calculate end-effector positions for each set of angles
    results = []
    timestamp = 0.0
    
    for i, angles in enumerate(test_angles):
        # Calculate forward kinematics
        pos, ori = solver.forward_kinematics(angles)
        
        # Create result row with both left and right arm joints
        # Left arm uses the test angles, right arm is kept at zero
        row = {
            'timestamp': round(timestamp, 2),
            'left_shoulder_yaw_joint': round(np.degrees(angles['shoulder_yaw']), 2),
            'left_shoulder_pitch_joint': round(np.degrees(angles['shoulder_pitch']), 2),
            'left_shoulder_roll_joint': round(np.degrees(angles['shoulder_roll']), 2),
            'left_elbow_joint': round(np.degrees(angles['elbow']), 2),
            'left_wrist_pitch_joint': round(np.degrees(angles['wrist_pitch']), 2),
            'left_wrist_yaw_joint': round(np.degrees(angles['wrist_yaw']), 2),
            'left_wrist_roll_joint': round(np.degrees(angles['wrist_roll']), 2),
            'right_shoulder_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_elbow_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0,
            'left_pos_x': round(pos[0], 4),
            'left_pos_y': round(pos[1], 4),
            'left_pos_z': round(pos[2], 4)
        }
        
        results.append(row)
        timestamp += 0.5  # Half-second intervals for easier visualization
    
    # Write to CSV
    csv_path = 'robot_simulation_angles.csv'
    fieldnames = [
        'timestamp',
        'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint',
        'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint',
        'left_pos_x', 'left_pos_y', 'left_pos_z'
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Robot simulation CSV file created: {csv_path}")

if __name__ == "__main__":
    create_robot_simulation_file() 