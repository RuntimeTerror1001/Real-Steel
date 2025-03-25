import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time

class RobotRetargeter:
    """Class to handle retargeting human motion to robot figure and recording data."""
    def __init__(self, robot_type="unitree_g1", recording_freq=20):
        # Initialize matplotlib for 3D visualization of robot figure
        self.fig_robot = plt.figure(figsize=(6, 6))
        self.ax_robot = self.fig_robot.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Set initial view angle
        self.ax_robot.view_init(elev=0, azim=90)
        
        # Robot specifications - simplified Unitree G1 humanoid model
        # Focusing on upper body for boxing
        self.robot_type = robot_type
        
        # Robot joint limits (in radians) based on Unitree G1 specs
        self.joint_limits = {
            "shoulder_pitch": (-2.0, 2.0),  # Forward/backward motion
            "shoulder_roll": (-1.5, 1.5),   # Side motion
            "shoulder_yaw": (-1.5, 1.5),    # Rotation
            "elbow": (0.0, 2.5),            # Elbow bending
            "wrist": (-1.0, 1.0),           # Wrist movement
        }
        
        # Robot dimensions (in meters, approximate for Unitree G1)
        self.dimensions = {
            "shoulder_width": 0.4,
            "upper_arm_length": 0.25,
            "lower_arm_length": 0.25,
            "torso_height": 0.4,
        }
        
        # Current robot joint positions
        self.robot_joints = {
            "torso": np.array([0, 0, 0]),
            "right_shoulder": np.array([0, 0, 0]),
            "right_elbow": np.array([0, 0, 0]),
            "right_wrist": np.array([0, 0, 0]),
            "left_shoulder": np.array([0, 0, 0]),
            "left_elbow": np.array([0, 0, 0]),
            "left_wrist": np.array([0, 0, 0]),
        }
        
        # Recording settings
        self.is_recording = False
        self.recording_freq = recording_freq  # Target frequency in Hz
        self.csv_file = None
        self.csv_writer = None
        self.last_record_time = 0
        self.record_interval = 1.0 / recording_freq
        
    def start_recording(self, filename="robot_motion.csv"):
        """Start recording joint positions to CSV file."""
        if self.is_recording:
            print("Already recording")
            return
            
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header
        header = ["timestamp", 
                 "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
                 "right_elbow_x", "right_elbow_y", "right_elbow_z",
                 "right_wrist_x", "right_wrist_y", "right_wrist_z",
                 "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
                 "left_elbow_x", "left_elbow_y", "left_elbow_z",
                 "left_wrist_x", "left_wrist_y", "left_wrist_z"]
        self.csv_writer.writerow(header)
        
        self.is_recording = True
        self.start_time = time.time()
        self.last_record_time = self.start_time
        print(f"Recording started to {filename} at {self.recording_freq}Hz")
        
    def stop_recording(self):
        """Stop recording and close the CSV file."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("Recording stopped")
        
    def record_frame(self):
        """Record current joint positions to CSV if it's time for a new frame."""
        if not self.is_recording:
            return
            
        current_time = time.time()
        # Check if it's time to record based on desired frequency
        if current_time - self.last_record_time >= self.record_interval:
            # Calculate timestamp relative to start time
            timestamp = current_time - self.start_time
            
            # Create row with timestamp and joint positions
            row = [timestamp]
            
            # Add right arm joint positions
            for joint in ["right_shoulder", "right_elbow", "right_wrist"]:
                row.extend(self.robot_joints[joint])
                
            # Add left arm joint positions
            for joint in ["left_shoulder", "left_elbow", "left_wrist"]:
                row.extend(self.robot_joints[joint])
                
            # Write to CSV
            self.csv_writer.writerow(row)
            self.last_record_time = current_time
            
    def retarget_pose(self, human_landmarks, rotation_angle=0):
        """Retarget human pose to robot figure."""
        if not human_landmarks:
            return
            
        # Extract required landmarks
        landmarks = human_landmarks.landmark
        
        # Get torso landmarks for reference
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate torso center (between shoulders)
        torso_x = (left_shoulder.x + right_shoulder.x) / 2
        torso_y = (left_shoulder.y + right_shoulder.y) / 2
        torso_z = (left_shoulder.z + right_shoulder.z) / 2
        
        # Set torso as origin
        self.robot_joints["torso"] = np.array([0, 0, 0])
        
        # Scale and apply shoulder width from robot specs
        shoulder_width_half = self.dimensions["shoulder_width"] / 2
        
        # Right arm joints (apply rotation compensation)
        angle_rad = math.radians(rotation_angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        
        # Right shoulder position relative to torso
        r_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_shoulder_pos = np.array([-r_shoulder.z, r_shoulder.x])
        r_shoulder_rotated = np.dot(rotation_matrix, r_shoulder_pos)
        self.robot_joints["right_shoulder"] = np.array([
            r_shoulder_rotated[0],
            shoulder_width_half,
            -r_shoulder.y
        ])
        
        # Right elbow
        r_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
        r_elbow_pos = np.array([-r_elbow.z, r_elbow.x])
        r_elbow_rotated = np.dot(rotation_matrix, r_elbow_pos)
        self.robot_joints["right_elbow"] = np.array([
            r_elbow_rotated[0],
            shoulder_width_half,  
            -r_elbow.y
        ])
        
        # Right wrist
        r_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        r_wrist_pos = np.array([-r_wrist.z, r_wrist.x])
        r_wrist_rotated = np.dot(rotation_matrix, r_wrist_pos)
        self.robot_joints["right_wrist"] = np.array([
            r_wrist_rotated[0],
            shoulder_width_half,
            -r_wrist.y
        ])
        
        # Left arm joints
        # Left shoulder
        l_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        l_shoulder_pos = np.array([-l_shoulder.z, l_shoulder.x])
        l_shoulder_rotated = np.dot(rotation_matrix, l_shoulder_pos)
        self.robot_joints["left_shoulder"] = np.array([
            l_shoulder_rotated[0],
            -shoulder_width_half,
            -l_shoulder.y
        ])
        
        # Left elbow
        l_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
        l_elbow_pos = np.array([-l_elbow.z, l_elbow.x])
        l_elbow_rotated = np.dot(rotation_matrix, l_elbow_pos)
        self.robot_joints["left_elbow"] = np.array([
            l_elbow_rotated[0],
            -shoulder_width_half,
            -l_elbow.y
        ])
        
        # Left wrist
        l_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        l_wrist_pos = np.array([-l_wrist.z, l_wrist.x])
        l_wrist_rotated = np.dot(rotation_matrix, l_wrist_pos)
        self.robot_joints["left_wrist"] = np.array([
            l_wrist_rotated[0],
            -shoulder_width_half,
            -l_wrist.y
        ])
        
        # Scale all joints to robot dimensions
        self.scale_to_robot_dimensions()
        
        # Apply joint limits to ensure realistic robot motion
        self.apply_joint_limits()
        
    def scale_to_robot_dimensions(self):
        """Scale the retargeted joints to match robot dimensions."""
        # Calculate human upper arm and forearm lengths
        human_r_upper_arm = np.linalg.norm(
            self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        )
        human_r_forearm = np.linalg.norm(
            self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        )
        
        # Scale factors
        upper_arm_scale = self.dimensions["upper_arm_length"] / max(human_r_upper_arm, 0.001)
        forearm_scale = self.dimensions["lower_arm_length"] / max(human_r_forearm, 0.001)
        
        # Apply scaling to right arm
        r_shoulder_to_elbow = self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        self.robot_joints["right_elbow"] = self.robot_joints["right_shoulder"] + r_shoulder_to_elbow * upper_arm_scale
        
        r_elbow_to_wrist = self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        self.robot_joints["right_wrist"] = self.robot_joints["right_elbow"] + r_elbow_to_wrist * forearm_scale
        
        # Apply scaling to left arm
        l_shoulder_to_elbow = self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"]
        self.robot_joints["left_elbow"] = self.robot_joints["left_shoulder"] + l_shoulder_to_elbow * upper_arm_scale
        
        l_elbow_to_wrist = self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]
        self.robot_joints["left_wrist"] = self.robot_joints["left_elbow"] + l_elbow_to_wrist * forearm_scale
        
    def apply_joint_limits(self):
        """Apply joint limits to ensure robot stays within physical constraints."""
        # This is a simplified version - in a full implementation you would
        # calculate actual joint angles and apply limits
        
        # For this demo, we'll just ensure the arms don't go behind the body
        # by limiting x coordinates
        min_x = -0.3  # Don't let arms go too far back
        
        # Apply to right arm
        if self.robot_joints["right_elbow"][0] < min_x:
            self.robot_joints["right_elbow"][0] = min_x
        if self.robot_joints["right_wrist"][0] < min_x:
            self.robot_joints["right_wrist"][0] = min_x
            
        # Apply to left arm
        if self.robot_joints["left_elbow"][0] < min_x:
            self.robot_joints["left_elbow"][0] = min_x
        if self.robot_joints["left_wrist"][0] < min_x:
            self.robot_joints["left_wrist"][0] = min_x
            
    def update_robot_plot(self):
        """Update the 3D robot visualization."""
        self.ax_robot.clear()
        
        # Set fixed axis limits
        self.ax_robot.set_xlim3d(-0.5, 0.5)  # Forward/backward
        self.ax_robot.set_ylim3d(-0.5, 0.5)  # Left/right
        self.ax_robot.set_zlim3d(-0.8, 0.2)  # Up/down
        
        # Set labels
        self.ax_robot.set_xlabel('X (Forward)')
        self.ax_robot.set_ylabel('Y (Side)')
        self.ax_robot.set_zlabel('Z (Up)')
        self.ax_robot.set_title('Robot Figure')
        
        # Draw the robot stick figure
        # Torso
        torso_height = self.dimensions["torso_height"]
        torso_top = self.robot_joints["torso"] + np.array([0, 0, torso_height/2])
        torso_bottom = self.robot_joints["torso"] - np.array([0, 0, torso_height/2])
        
        # Draw torso
        self.ax_robot.plot([torso_top[0], torso_bottom[0]], 
                          [torso_top[1], torso_bottom[1]], 
                          [torso_top[2], torso_bottom[2]], 'k-', linewidth=3)
        
        # Draw shoulders line
        self.ax_robot.plot([self.robot_joints["right_shoulder"][0], self.robot_joints["left_shoulder"][0]],
                          [self.robot_joints["right_shoulder"][1], self.robot_joints["left_shoulder"][1]],
                          [self.robot_joints["right_shoulder"][2], self.robot_joints["left_shoulder"][2]], 
                          'k-', linewidth=3)
        
        # Draw right arm
        self.ax_robot.plot([self.robot_joints["right_shoulder"][0], self.robot_joints["right_elbow"][0]],
                          [self.robot_joints["right_shoulder"][1], self.robot_joints["right_elbow"][1]],
                          [self.robot_joints["right_shoulder"][2], self.robot_joints["right_elbow"][2]], 
                          'r-', linewidth=2)
        
        self.ax_robot.plot([self.robot_joints["right_elbow"][0], self.robot_joints["right_wrist"][0]],
                          [self.robot_joints["right_elbow"][1], self.robot_joints["right_wrist"][1]],
                          [self.robot_joints["right_elbow"][2], self.robot_joints["right_wrist"][2]], 
                          'r-', linewidth=2)
        
        # Draw left arm
        self.ax_robot.plot([self.robot_joints["left_shoulder"][0], self.robot_joints["left_elbow"][0]],
                          [self.robot_joints["left_shoulder"][1], self.robot_joints["left_elbow"][1]],
                          [self.robot_joints["left_shoulder"][2], self.robot_joints["left_elbow"][2]], 
                          'b-', linewidth=2)
        
        self.ax_robot.plot([self.robot_joints["left_elbow"][0], self.robot_joints["left_wrist"][0]],
                          [self.robot_joints["left_elbow"][1], self.robot_joints["left_wrist"][1]],
                          [self.robot_joints["left_elbow"][2], self.robot_joints["left_wrist"][2]], 
                          'b-', linewidth=2)
        
        # Draw joints as spheres
        for joint, color in [
            ("right_shoulder", "r"), 
            ("right_elbow", "r"), 
            ("right_wrist", "r"),
            ("left_shoulder", "b"), 
            ("left_elbow", "b"), 
            ("left_wrist", "b")
        ]:
            self.ax_robot.scatter(
                self.robot_joints[joint][0], 
                self.robot_joints[joint][1], 
                self.robot_joints[joint][2], 
                color=color, s=50
            )
        
        # Update the plot
        self.fig_robot.canvas.draw()
        self.fig_robot.canvas.flush_events()
