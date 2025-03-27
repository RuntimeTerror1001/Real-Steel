import mediapipe as mp
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
        
        # Set initial view angle - fixed to match human orientation
        self.ax_robot.view_init(elev=0, azim=0)
        
        # Robot specifications - Unitree G1 humanoid model
        self.robot_type = robot_type
        
        # Robot joint limits (in radians)
        self.joint_limits = {
            "shoulder_pitch": (-2.0, 2.0),  # Forward/backward motion
            "shoulder_roll": (-1.5, 1.5),   # Side motion
            "shoulder_yaw": (-1.5, 1.5),    # Rotation
            "elbow": (0.0, 2.5),            # Elbow bending
            "wrist_pitch": (-1.0, 1.0),     # Wrist up/down
            "wrist_yaw": (-1.0, 1.0),       # Wrist rotation
            "wrist_roll": (-1.0, 1.0),      # Wrist side to side
        }
        
        # Robot dimensions (in meters, approximate for Unitree G1)
        self.dimensions = {
            "shoulder_width": 0.4,
            "upper_arm_length": 0.25,
            "lower_arm_length": 0.25,
            "torso_height": 0.4,
        }
        
        # Current robot joint positions (3D coordinates)
        self.robot_joints = {
            "torso": np.array([0, 0, 0]),
            "right_shoulder": np.array([0, 0, 0]),
            "right_elbow": np.array([0, 0, 0]),
            "right_wrist": np.array([0, 0, 0]),
            "left_shoulder": np.array([0, 0, 0]),
            "left_elbow": np.array([0, 0, 0]),
            "left_wrist": np.array([0, 0, 0]),
        }
        
        # Joint angles (for robot control)
        self.joint_angles = {
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
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
        
        # Write CSV header with all joint names but combined coordinates
        header = ["timestamp"]
        
        # Add all joint names (without splitting x,y,z)
        for joint in [
            "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
            "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint",
            "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint"
        ]:
            header.append(joint)
            
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
            
            # Create row with timestamp
            row = [timestamp]
            
            # Calculate positions for each joint based on our skeleton
            joint_positions = self.calculate_joint_positions()
            
            # Add all joint positions to the row in {x,y,z} format
            for joint in [
                "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
                "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
                "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint",
                "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint"
            ]:
                if joint in joint_positions:
                    pos = joint_positions[joint]
                    # Format as {x,y,z}
                    formatted_pos = f"{{{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}}}"
                    row.append(formatted_pos)
                else:
                    # Default for missing positions
                    row.append("{0.0000,0.0000,0.0000}")
                
            # Write to CSV
            self.csv_writer.writerow(row)
            self.last_record_time = current_time
    
    def calculate_joint_positions(self):
        """Calculate 3D positions for all robot joints based on current skeleton."""
        positions = {}
        
        # Calculate positions for all joints based on our skeleton model
        # Left shoulder group
        positions["left_shoulder_pitch_joint"] = self.robot_joints["left_shoulder"]
        positions["left_shoulder_yaw_joint"] = self.robot_joints["left_shoulder"] + np.array([0.02, 0, 0])
        positions["left_shoulder_roll_joint"] = self.robot_joints["left_shoulder"] + np.array([0, 0, 0.02])
        
        # Left elbow and wrist
        positions["left_elbow_joint"] = self.robot_joints["left_elbow"]
        positions["left_wrist_pitch_joint"] = self.robot_joints["left_wrist"] + np.array([0, 0, 0.01])
        positions["left_wrist_yaw_joint"] = self.robot_joints["left_wrist"] + np.array([0.01, 0, 0])
        positions["left_wrist_roll_joint"] = self.robot_joints["left_wrist"] + np.array([0, 0.01, 0])
        
        # Right shoulder group
        positions["right_shoulder_pitch_joint"] = self.robot_joints["right_shoulder"]
        positions["right_shoulder_yaw_joint"] = self.robot_joints["right_shoulder"] + np.array([0.02, 0, 0])
        positions["right_shoulder_roll_joint"] = self.robot_joints["right_shoulder"] + np.array([0, 0, 0.02])
        
        # Right elbow and wrist
        positions["right_elbow_joint"] = self.robot_joints["right_elbow"]
        positions["right_wrist_pitch_joint"] = self.robot_joints["right_wrist"] + np.array([0, 0, 0.01])
        positions["right_wrist_yaw_joint"] = self.robot_joints["right_wrist"] + np.array([0.01, 0, 0])
        positions["right_wrist_roll_joint"] = self.robot_joints["right_wrist"] + np.array([0, 0.01, 0])
        
        return positions
            
    def retarget_pose(self, human_landmarks, rotation_angle=0):
        """
        Retarget pose ensuring the robot coordinates follow standard convention:
        - X-axis: positive right (horizontal)
        - Y-axis: positive up (vertical)
        - Z-axis: positive forward (depth/away from camera)
        """
        if not human_landmarks:
            return
            
        # Extract landmarks
        landmarks = human_landmarks.landmark
        
        # Set torso as origin
        self.robot_joints["torso"] = np.array([0, 0, 0])
        
        # Scale factor
        scale = 0.3
        
        # Apply rotation matrix for body orientation if needed
        angle_rad = math.radians(rotation_angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        
        # Map MediaPipe coordinates to standard coordinates:
        # - MediaPipe X (positive left) → -X (standard, positive right)
        # - MediaPipe Z (negative up) → Y (standard, positive up)
        # - MediaPipe Y (negative forward) → Z (standard, positive forward)
        
        # Get key joints with correct mapping
        for side in ["left", "right"]:
            # Get landmarks
            if side == "left":
                shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            else:
                shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
                
            # Apply rotation in X-Z plane of standard coordinates
            # In MediaPipe, this corresponds to X and Y axes
            
            # Map shoulder
            standard_shoulder_x = -shoulder.x  # Flip X to match standard
            standard_shoulder_y = -shoulder.z  # Map Z to Y (up)
            standard_shoulder_z = -shoulder.y  # Map Y to Z (forward)
            
            # For rotation, we need the X-Z plane in standard coordinates
            shoulder_xz = np.array([standard_shoulder_x, standard_shoulder_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, shoulder_xz)
                standard_shoulder_x, standard_shoulder_z = rotated
                
            self.robot_joints[f"{side}_shoulder"] = np.array([
                standard_shoulder_x * scale,  # Standard X (right)
                standard_shoulder_y * scale,  # Standard Y (up)
                standard_shoulder_z * scale   # Standard Z (forward)
            ])
            
            # Map elbow
            standard_elbow_x = -elbow.x  # Flip X to match standard
            standard_elbow_y = -elbow.z  # Map Z to Y (up)
            standard_elbow_z = -elbow.y  # Map Y to Z (forward)
            
            # Apply rotation
            elbow_xz = np.array([standard_elbow_x, standard_elbow_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, elbow_xz)
                standard_elbow_x, standard_elbow_z = rotated
                
            self.robot_joints[f"{side}_elbow"] = np.array([
                standard_elbow_x * scale,  # Standard X (right)
                standard_elbow_y * scale,  # Standard Y (up)
                standard_elbow_z * scale   # Standard Z (forward)
            ])
            
            # Map wrist
            standard_wrist_x = -wrist.x  # Flip X to match standard
            standard_wrist_y = -wrist.z  # Map Z to Y (up)
            standard_wrist_z = -wrist.y  # Map Y to Z (forward)
            
            # Apply rotation
            wrist_xz = np.array([standard_wrist_x, standard_wrist_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, wrist_xz)
                standard_wrist_x, standard_wrist_z = rotated
                
            self.robot_joints[f"{side}_wrist"] = np.array([
                standard_wrist_x * scale,  # Standard X (right)
                standard_wrist_y * scale,  # Standard Y (up)
                standard_wrist_z * scale   # Standard Z (forward)
            ])
        
        # Calculate joint angles from standard coordinates
        self.calculate_joint_angles()
        
    def calculate_joint_angles(self):
        """
        Calculate joint angles from positions using standard coordinate system:
        - X-axis: positive right (horizontal)
        - Y-axis: positive up (vertical)
        - Z-axis: positive forward (depth)
        """
        # Left arm joint calculations
        # Calculate vectors
        l_shoulder_to_elbow = self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"]
        l_elbow_to_wrist = self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]
        
        # Normalize vectors
        l_upper_arm = l_shoulder_to_elbow / np.linalg.norm(l_shoulder_to_elbow)
        l_forearm = l_elbow_to_wrist / np.linalg.norm(l_elbow_to_wrist)
        
        # Calculate shoulder angles
        # Pitch (forward/backward in Y-Z plane)
        l_shoulder_pitch = math.atan2(l_upper_arm[1], l_upper_arm[2])
        # Yaw (side-to-side in X-Z plane)
        l_shoulder_yaw = math.atan2(l_upper_arm[0], l_upper_arm[2])
        # Roll (rotation in X-Y plane)
        l_shoulder_roll = math.atan2(l_upper_arm[0], l_upper_arm[1])
        
        # Calculate elbow angle (cosine of dot product of normalized vectors)
        l_elbow_angle = math.acos(np.clip(np.dot(l_upper_arm, l_forearm), -1.0, 1.0))
        
        # Calculate wrist angles
        # Pitch (up/down in Y-Z plane)
        l_wrist_pitch = math.atan2(l_forearm[1], l_forearm[2])
        # Yaw (side-to-side in X-Z plane)
        l_wrist_yaw = math.atan2(l_forearm[0], l_forearm[2])
        # Roll (rotation in X-Y plane)
        l_wrist_roll = math.atan2(l_forearm[0], l_forearm[1])
        
        # Right arm joint calculations
        # Calculate vectors
        r_shoulder_to_elbow = self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        r_elbow_to_wrist = self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        
        # Normalize vectors
        r_upper_arm = r_shoulder_to_elbow / np.linalg.norm(r_shoulder_to_elbow)
        r_forearm = r_elbow_to_wrist / np.linalg.norm(r_elbow_to_wrist)
        
        # Calculate shoulder angles
        # Pitch (forward/backward in Y-Z plane)
        r_shoulder_pitch = math.atan2(r_upper_arm[1], r_upper_arm[2])
        # Yaw (side-to-side in X-Z plane)
        r_shoulder_yaw = math.atan2(r_upper_arm[0], r_upper_arm[2])
        # Roll (rotation in X-Y plane)
        r_shoulder_roll = math.atan2(r_upper_arm[0], r_upper_arm[1])
        
        # Calculate elbow angle (cosine of dot product)
        r_elbow_angle = math.acos(np.clip(np.dot(r_upper_arm, r_forearm), -1.0, 1.0))
        
        # Calculate wrist angles
        # Pitch (up/down in Y-Z plane)
        r_wrist_pitch = math.atan2(r_forearm[1], r_forearm[2])
        # Yaw (side-to-side in X-Z plane)
        r_wrist_yaw = math.atan2(r_forearm[0], r_forearm[2])
        # Roll (rotation in X-Y plane)
        r_wrist_roll = math.atan2(r_forearm[0], r_forearm[1])
        
        # Update joint angles dictionary
        self.joint_angles["left_shoulder_pitch_joint"] = l_shoulder_pitch
        self.joint_angles["left_shoulder_yaw_joint"] = l_shoulder_yaw
        self.joint_angles["left_shoulder_roll_joint"] = l_shoulder_roll
        self.joint_angles["left_elbow_joint"] = l_elbow_angle
        self.joint_angles["left_wrist_pitch_joint"] = l_wrist_pitch
        self.joint_angles["left_wrist_yaw_joint"] = l_wrist_yaw
        self.joint_angles["left_wrist_roll_joint"] = l_wrist_roll
        
        self.joint_angles["right_shoulder_pitch_joint"] = r_shoulder_pitch
        self.joint_angles["right_shoulder_yaw_joint"] = r_shoulder_yaw
        self.joint_angles["right_shoulder_roll_joint"] = r_shoulder_roll
        self.joint_angles["right_elbow_joint"] = r_elbow_angle
        self.joint_angles["right_wrist_pitch_joint"] = r_wrist_pitch
        self.joint_angles["right_wrist_yaw_joint"] = r_wrist_yaw
        self.joint_angles["right_wrist_roll_joint"] = r_wrist_roll

    def scale_to_robot_dimensions(self):
        """Scale the joint positions to match the robot's dimensions."""
        # Calculate current dimensions from joint positions
        current_upper_arm_length_right = np.linalg.norm(
            self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        )
        current_lower_arm_length_right = np.linalg.norm(
            self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        )
        
        current_upper_arm_length_left = np.linalg.norm(
            self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"]
        )
        current_lower_arm_length_left = np.linalg.norm(
            self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]
        )
        
        # Calculate scaling factors
        if current_upper_arm_length_right > 0:
            scale_upper_right = self.dimensions["upper_arm_length"] / current_upper_arm_length_right
        else:
            scale_upper_right = 1.0
            
        if current_lower_arm_length_right > 0:
            scale_lower_right = self.dimensions["lower_arm_length"] / current_lower_arm_length_right
        else:
            scale_lower_right = 1.0
            
        if current_upper_arm_length_left > 0:
            scale_upper_left = self.dimensions["upper_arm_length"] / current_upper_arm_length_left
        else:
            scale_upper_left = 1.0
            
        if current_lower_arm_length_left > 0:
            scale_lower_left = self.dimensions["lower_arm_length"] / current_lower_arm_length_left
        else:
            scale_lower_left = 1.0
        
        # Apply scaling to upper arm (shoulder to elbow)
        vector_right_upper = self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        self.robot_joints["right_elbow"] = self.robot_joints["right_shoulder"] + vector_right_upper * scale_upper_right
        
        vector_left_upper = self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"]
        self.robot_joints["left_elbow"] = self.robot_joints["left_shoulder"] + vector_left_upper * scale_upper_left
        
        # Apply scaling to lower arm (elbow to wrist)
        vector_right_lower = self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        self.robot_joints["right_wrist"] = self.robot_joints["right_elbow"] + vector_right_lower * scale_lower_right
        
        vector_left_lower = self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]
        self.robot_joints["left_wrist"] = self.robot_joints["left_elbow"] + vector_left_lower * scale_lower_left

    def apply_joint_limits(self):
        """Apply joint limits to ensure realistic robot motion."""
        # This will constrain joint angles to keep them within the robot's physical limits
        # We'll implement this based on the joint_angles dictionary and joint_limits
        
        # Calculate joint angles first if they haven't been calculated already
        # (they should be calculated in retarget_pose, but just in case)
        self.calculate_joint_angles()
        
        # Now apply limits to joint angles
        for joint_name, angle in self.joint_angles.items():
            # Extract the joint type from the name (e.g., "shoulder_pitch" from "left_shoulder_pitch_joint")
            parts = joint_name.split('_')
            if len(parts) >= 3:
                joint_type = parts[1] + "_" + parts[2]
                
                # Apply limits if defined
                if joint_type in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[joint_type]
                    self.joint_angles[joint_name] = max(min_limit, min(max_limit, angle))
        
        # Note: A complete implementation would update joint positions from the constrained angles
        # using forward kinematics. For now, we'll use the already calculated positions.

    def update_robot_plot(self):
        """
        Update robot visualization using standard coordinates:
        - X-axis: positive right (horizontal)
        - Y-axis: positive up (vertical)
        - Z-axis: positive forward (depth/away from camera)
        """
        self.ax_robot.clear()
        
        # Set up axes labels for standard coordinates
        self.ax_robot.set_xlabel('X (Right →)')
        self.ax_robot.set_ylabel('Y (Up ↑)')
        self.ax_robot.set_zlabel('Z (Forward ↗)')
        
        # Set limits
        limit = 0.4
        self.ax_robot.set_xlim3d(-limit, limit)
        self.ax_robot.set_ylim3d(-limit, limit)
        self.ax_robot.set_zlim3d(-limit, limit)
        
        # Grid
        self.ax_robot.grid(True)
        
        # Coordinate axes indicators
        origin = [0, 0, 0]
        self.ax_robot.quiver(*origin, 0.1, 0, 0, color='r', label='X (Right)')
        self.ax_robot.quiver(*origin, 0, 0.1, 0, color='g', label='Y (Up)')
        self.ax_robot.quiver(*origin, 0, 0, 0.1, color='b', label='Z (Forward)')
        
        # Plot torso
        self.ax_robot.scatter(*self.robot_joints["torso"], c='black', marker='o', s=50)
        
        # Connect shoulders
        self.ax_robot.plot(
            [self.robot_joints["left_shoulder"][0], self.robot_joints["right_shoulder"][0]],
            [self.robot_joints["left_shoulder"][1], self.robot_joints["right_shoulder"][1]],
            [self.robot_joints["left_shoulder"][2], self.robot_joints["right_shoulder"][2]],
            'k-', linewidth=3
        )
        
        # Plot arms
        for side, color in [("right", "blue"), ("left", "green")]:
            shoulder = self.robot_joints[f"{side}_shoulder"]
            elbow = self.robot_joints[f"{side}_elbow"]
            wrist = self.robot_joints[f"{side}_wrist"]
            
            # Plot segments
            self.ax_robot.plot(
                [shoulder[0], elbow[0]],
                [shoulder[1], elbow[1]],
                [shoulder[2], elbow[2]],
                color=color, linewidth=3
            )
            
            self.ax_robot.plot(
                [elbow[0], wrist[0]],
                [elbow[1], wrist[1]],
                [elbow[2], wrist[2]],
                color=color, linewidth=3
            )
            
            # Plot joints
            self.ax_robot.scatter(shoulder[0], shoulder[1], shoulder[2], 
                            color=color, s=80, marker='o')
            self.ax_robot.scatter(elbow[0], elbow[1], elbow[2], 
                            color=color, s=60, marker='o')
            self.ax_robot.scatter(wrist[0], wrist[1], wrist[2], 
                            color=color, s=60, marker='o')
        
        # Set view to show person from front (Y up, X right, Z toward viewer)
        # For standard coordinates, looking toward negative Z axis shows the front view
        self.ax_robot.view_init(elev=15, azim=270)
        
        # Add legend
        self.ax_robot.legend(loc='upper right', fontsize='small')
        
        # Update
        self.fig_robot.canvas.draw()
        self.fig_robot.canvas.flush_events()