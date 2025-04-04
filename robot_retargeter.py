import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time

class RobotRetargeter:
    """Class to handle retargeting human motion to robot figure and recording data."""
    def __init__(self, robot_type="unitree_g1", recording_freq=10, smoothing_window=15, smoothing_factor=0.85):
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
            "shoulder_pitch": (-3.089, 2.670),
            "shoulder_roll": (-1.588, 2.252),  # Left arm values
            "shoulder_yaw": (-2.618, 2.618),
            "elbow": (-1.047, 2.094),
            "wrist_pitch": (-1.614, 1.614),
            "wrist_yaw": (-1.614, 1.614),
            "wrist_roll": (-1.972, 1.972),
        }

        # Updated robot dimensions (in meters)
        self.dimensions = {
            "shoulder_width": 0.200,
            "upper_arm_length": 0.082,
            "lower_arm_length": 0.185,
            "torso_height": 0.4,  # Keep as is if no better data
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
        
        # ENHANCED: Joint angle history and smoothing
        self.joint_angle_history = {joint: [] for joint in self.joint_angles.keys()}
        self.smoothing_window = smoothing_window  # Increased from 5 to 15
        self.smoothing_factor = smoothing_factor  # Increased from 0.7 to 0.85
        
        # ENHANCED: Velocity limiting to prevent sudden changes
        self.previous_joint_angles = self.joint_angles.copy()
        self.max_velocity = 0.1  # Maximum change in angle per frame (radians)
        
        # ENHANCED: Joint-specific smoothing factors
        self.joint_type_smoothing = {
            "shoulder": 0.9,    # More smoothing for shoulders
            "elbow": 0.85,      # Medium smoothing for elbows
            "wrist": 0.8        # Less smoothing for wrists (more responsive)
        }
        
        # ENHANCED: Outlier rejection threshold (in radians)
        self.outlier_threshold = 0.5  # ~30 degrees sudden change
        
        # Recording settings
        self.is_recording = False
        self.recording_freq = recording_freq  # Target frequency in Hz
        self.csv_file = None
        self.csv_writer = None
        self.last_record_time = 0
        self.record_interval = 1.0 / recording_freq
        
        # Frame counter for constant-interval timestamps
        self.frame_counter = 0
        
    def start_recording(self, filename="robot_motion.csv"):
        """Start recording joint angles to CSV file."""
        if self.is_recording:
            print("Already recording")
            return
            
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header with joint angle names
        header = ["timestamp"]
        
        # Add all joint angle names
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
        
        # Reset frame counter to generate sequential timestamps
        self.frame_counter = 0
        
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
        """Record current joint angles to CSV if it's time for a new frame."""
        if not self.is_recording:
            return
            
        current_time = time.time()
        # Check if it's time to record based on desired frequency
        if current_time - self.last_record_time >= self.record_interval:
            # Calculate timestamp using fixed increments matching the expected frequency
            # For 10Hz, this gives timestamps of 0, 0.1, 0.2, 0.3, etc.
            timestamp = self.frame_counter * 0.1  # Hardcoded 0.1s interval (10Hz)
            
            # Create row with timestamp
            row = [f"{timestamp:.1f}"]
            
            # Add all joint angles to the row (scalar values)
            for joint in [
                "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
                "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
                "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint",
                "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint"
            ]:
                # Format the angle in radians to 4 decimal places
                angle = self.joint_angles.get(joint, 0.0)
                row.append(f"{angle:.4f}")
                
            # Write to CSV
            self.csv_writer.writerow(row)
            self.last_record_time = current_time
            
            # Increment frame counter for next timestamp
            self.frame_counter += 1
    
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

            for joint_name in self.robot_joints:
                self.robot_joints[joint_name] = self.robot_joints[joint_name].astype(float)

            # Level the shoulders to fix slanting issue
            shoulder_height_diff = self.robot_joints["left_shoulder"][1] - self.robot_joints["right_shoulder"][1]
            # Adjust both shoulders to be at the same height
            self.robot_joints["left_shoulder"][1] -= shoulder_height_diff/2
            self.robot_joints["right_shoulder"][1] += shoulder_height_diff/2
            
            # This fix also requires updating elbow and wrist positions to maintain arm structure
            # Adjust left arm joints
            left_elbow_offset = np.array([0, -shoulder_height_diff/2, 0])
            left_wrist_offset = np.array([0, -shoulder_height_diff/2, 0])
            self.robot_joints["left_elbow"] += left_elbow_offset
            self.robot_joints["left_wrist"] += left_wrist_offset
            
            # Adjust right arm joints
            right_elbow_offset = np.array([0, shoulder_height_diff/2, 0])
            right_wrist_offset = np.array([0, shoulder_height_diff/2, 0])
            self.robot_joints["right_elbow"] += right_elbow_offset
            self.robot_joints["right_wrist"] += right_wrist_offset
        
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

        l_shoulder_pitch -= math.pi/2
        r_shoulder_pitch -= math.pi/2
        
        # Apply joint limits before updating angles
        l_shoulder_pitch = self.apply_limit(l_shoulder_pitch, "shoulder_pitch")
        l_shoulder_yaw = self.apply_limit(l_shoulder_yaw, "shoulder_yaw")
        l_shoulder_roll = self.apply_limit(l_shoulder_roll, "shoulder_roll")
        l_elbow_angle = self.apply_limit(l_elbow_angle, "elbow")
        l_wrist_pitch = self.apply_limit(l_wrist_pitch, "wrist_pitch")
        l_wrist_yaw = self.apply_limit(l_wrist_yaw, "wrist_yaw")
        l_wrist_roll = self.apply_limit(l_wrist_roll, "wrist_roll")
        
        r_shoulder_pitch = self.apply_limit(r_shoulder_pitch, "shoulder_pitch")
        r_shoulder_yaw = self.apply_limit(r_shoulder_yaw, "shoulder_yaw")
        r_shoulder_roll = self.apply_limit(r_shoulder_roll, "shoulder_roll")
        r_elbow_angle = self.apply_limit(r_elbow_angle, "elbow")
        r_wrist_pitch = self.apply_limit(r_wrist_pitch, "wrist_pitch")
        r_wrist_yaw = self.apply_limit(r_wrist_yaw, "wrist_yaw")
        r_wrist_roll = self.apply_limit(r_wrist_roll, "wrist_roll")
        
        # Create raw angles dictionary
        raw_angles = {
            "left_shoulder_pitch_joint": l_shoulder_pitch,
            "left_shoulder_yaw_joint": l_shoulder_yaw,
            "left_shoulder_roll_joint": l_shoulder_roll,
            "left_elbow_joint": l_elbow_angle,
            "left_wrist_pitch_joint": l_wrist_pitch,
            "left_wrist_yaw_joint": l_wrist_yaw,
            "left_wrist_roll_joint": l_wrist_roll,
            "right_shoulder_pitch_joint": r_shoulder_pitch,
            "right_shoulder_yaw_joint": r_shoulder_yaw,
            "right_shoulder_roll_joint": r_shoulder_roll,
            "right_elbow_joint": r_elbow_angle,
            "right_wrist_pitch_joint": r_wrist_pitch,
            "right_wrist_yaw_joint": r_wrist_yaw,
            "right_wrist_roll_joint": r_wrist_roll
        }
        
        # ENHANCED: Update history with outlier detection
        self.update_joint_history_with_outlier_detection(raw_angles)
        
        # ENHANCED: Apply improved smoothing to all joints
        self.apply_enhanced_smoothing(raw_angles)

    def update_joint_history_with_outlier_detection(self, raw_angles):
        """
        Update joint angle history with outlier detection.
        
        Args:
            raw_angles: Dictionary of joint names to raw angle values
        """
        for joint, raw_angle in raw_angles.items():
            # Check if this is an outlier (if we have history)
            is_outlier = False
            if len(self.joint_angle_history[joint]) > 0:
                last_angle = self.joint_angle_history[joint][-1]
                if abs(raw_angle - last_angle) > self.outlier_threshold:
                    # This is likely an outlier - don't add it to history
                    is_outlier = True
            
            # Add to history if not an outlier
            if not is_outlier:
                self.joint_angle_history[joint].append(raw_angle)
                
                # Keep history within window size
                if len(self.joint_angle_history[joint]) > self.smoothing_window:
                    self.joint_angle_history[joint].pop(0)

    def apply_enhanced_smoothing(self, raw_angles):
        """
        Apply enhanced smoothing with weighted history, velocity limiting, and joint-specific parameters.
        
        Args:
            raw_angles: Dictionary of joint names to raw angle values
        """
        for joint, raw_angle in raw_angles.items():
            history = self.joint_angle_history[joint]
            
            if len(history) < 3:  # Need at least 3 frames for good smoothing
                self.joint_angles[joint] = raw_angle
                continue
                
            # Determine smoothing factor based on joint type
            smoothing_factor = self.smoothing_factor
            for joint_type, factor in self.joint_type_smoothing.items():
                if joint_type in joint:
                    smoothing_factor = factor
                    break
            
            # Use weighted average - newer values have higher weight
            weights = [i+1 for i in range(len(history))]  # [1, 2, 3, ...]
            total_weight = sum(weights)
            weighted_sum = sum(h * w for h, w in zip(history, weights))
            weighted_average = weighted_sum / total_weight
            
            # Mix with exponential moving average for stability
            if joint in self.joint_angles:
                current_value = self.joint_angles[joint]
                smoothed_value = current_value * smoothing_factor + weighted_average * (1 - smoothing_factor)
            else:
                smoothed_value = weighted_average
            
            # Apply velocity limiting to prevent sudden changes
            if joint in self.previous_joint_angles:
                prev_value = self.previous_joint_angles[joint]
                change = smoothed_value - prev_value
                
                # Limit change to maximum velocity
                if abs(change) > self.max_velocity:
                    smoothed_value = prev_value + self.max_velocity * np.sign(change)
            
            # Update the joint angle with smoothed value
            self.joint_angles[joint] = smoothed_value
        
        # Store current joint angles for next frame
        self.previous_joint_angles = self.joint_angles.copy()

    def apply_limit(self, angle, joint_type):
        """Apply joint limits to a given angle."""
        if joint_type in self.joint_limits:
            min_limit, max_limit = self.joint_limits[joint_type]
            return max(min_limit, min(max_limit, angle))
        return angle

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
        self.ax_robot.set_ylabel('Z (Forward ↗)')
        self.ax_robot.set_zlabel('Y (Up ↑)')
        
        # Set limits
        limit = 0.4
        self.ax_robot.set_xlim3d(-limit, limit)
        self.ax_robot.set_ylim3d(-limit, limit)
        self.ax_robot.set_zlim3d(-limit, limit)
        
        # Grid
        self.ax_robot.grid(True)
        
        # Calculate torso points
        shoulder_width = np.linalg.norm(self.robot_joints["right_shoulder"] - self.robot_joints["left_shoulder"])
        torso_height = self.dimensions["torso_height"] * 0.8  # 80% of defined torso height for better proportions
        
        # Calculate midpoint between shoulders
        shoulder_midpoint = (self.robot_joints["left_shoulder"] + self.robot_joints["right_shoulder"]) / 2
        
        # Define waist position (down from shoulders)
        waist_midpoint = shoulder_midpoint - np.array([0, torso_height, 0])
        
        # Define waist width (typically narrower than shoulders)
        waist_width = shoulder_width * 0.8  # 80% of shoulder width
        
        # Calculate waist left and right points
        waist_left = waist_midpoint + np.array([waist_width/2, 0, 0])
        waist_right = waist_midpoint - np.array([waist_width/2, 0, 0])
        
        # Draw torso rectangle (connecting shoulders to waist)
        torso_points = [
            self.robot_joints["left_shoulder"],
            self.robot_joints["right_shoulder"],
            waist_right,
            waist_left,
            self.robot_joints["left_shoulder"]
        ]
        
        # Extract x, y, z coordinates for plotting
        torso_x = [point[0] for point in torso_points]
        torso_y = [point[1] for point in torso_points]
        torso_z = [point[2] for point in torso_points]
        
        # Plot torso outline
        self.ax_robot.plot(torso_x, torso_y, torso_z, 'k-', linewidth=2)
        
        # Add vertical lines for torso sides (optional, for better 3D appearance)
        # Left side
        self.ax_robot.plot(
            [self.robot_joints["left_shoulder"][0], waist_left[0]],
            [self.robot_joints["left_shoulder"][1], waist_left[1]],
            [self.robot_joints["left_shoulder"][2], waist_left[2]],
            'k-', linewidth=2
        )
        
        # Right side
        self.ax_robot.plot(
            [self.robot_joints["right_shoulder"][0], waist_right[0]],
            [self.robot_joints["right_shoulder"][1], waist_right[1]],
            [self.robot_joints["right_shoulder"][2], waist_right[2]],
            'k-', linewidth=2
        )
        
        # Plot torso as a center marker
        self.ax_robot.scatter(*self.robot_joints["torso"], c='black', marker='o', s=50)
        
        # Connect shoulders
        self.ax_robot.plot(
            [self.robot_joints["left_shoulder"][0], self.robot_joints["right_shoulder"][0]],
            [self.robot_joints["left_shoulder"][1], self.robot_joints["right_shoulder"][1]],
            [self.robot_joints["left_shoulder"][2], self.robot_joints["right_shoulder"][2]],
            'k-', linewidth=3
        )
        
        # NEW: Calculate joint positions for visualization
        joint_positions = self.calculate_joint_positions()
        
        # Plot all joint positions
        for side, color in [("right", "blue"), ("left", "green")]:
            # Main arm segments
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
            
            # Plot all joint positions with different markers to distinguish them
            # Shoulder joint group
            shoulder_pitch = joint_positions[f"{side}_shoulder_pitch_joint"]
            shoulder_yaw = joint_positions[f"{side}_shoulder_yaw_joint"]
            shoulder_roll = joint_positions[f"{side}_shoulder_roll_joint"]
            
            # Elbow joint
            elbow_joint = joint_positions[f"{side}_elbow_joint"]
            
            # Wrist joint group
            wrist_pitch = joint_positions[f"{side}_wrist_pitch_joint"]
            wrist_yaw = joint_positions[f"{side}_wrist_yaw_joint"]
            wrist_roll = joint_positions[f"{side}_wrist_roll_joint"]
            
            # Plot each joint with a different marker
            # Shoulder group
            self.ax_robot.scatter(shoulder_pitch[0], shoulder_pitch[1], shoulder_pitch[2], 
                           color=color, s=80, marker='o', label=f"{side.capitalize()} Shoulder Pitch" if side == "right" else "")
            self.ax_robot.scatter(shoulder_yaw[0], shoulder_yaw[1], shoulder_yaw[2], 
                           color=color, s=50, marker='s', label=f"{side.capitalize()} Shoulder Yaw" if side == "right" else "")
            self.ax_robot.scatter(shoulder_roll[0], shoulder_roll[1], shoulder_roll[2], 
                           color=color, s=50, marker='^', label=f"{side.capitalize()} Shoulder Roll" if side == "right" else "")
            
            # Elbow joint (already plotted as main joint, but with specific label)
            self.ax_robot.scatter(elbow_joint[0], elbow_joint[1], elbow_joint[2], 
                           color=color, s=60, marker='o', label=f"{side.capitalize()} Elbow" if side == "right" else "")
            
            # Wrist group
            self.ax_robot.scatter(wrist_pitch[0], wrist_pitch[1], wrist_pitch[2], 
                           color=color, s=40, marker='d', label=f"{side.capitalize()} Wrist Pitch" if side == "right" else "")
            self.ax_robot.scatter(wrist_yaw[0], wrist_yaw[1], wrist_yaw[2], 
                           color=color, s=40, marker='*', label=f"{side.capitalize()} Wrist Yaw" if side == "right" else "")
            self.ax_robot.scatter(wrist_roll[0], wrist_roll[1], wrist_roll[2], 
                           color=color, s=40, marker='p', label=f"{side.capitalize()} Wrist Roll" if side == "right" else "")
            
            # Connect dots to visualize the joint structure
            # Connect shoulder group
            self.ax_robot.plot(
                [shoulder_pitch[0], shoulder_yaw[0]],
                [shoulder_pitch[1], shoulder_yaw[1]],
                [shoulder_pitch[2], shoulder_yaw[2]],
                color=color, linewidth=1, linestyle=':'
            )
            self.ax_robot.plot(
                [shoulder_pitch[0], shoulder_roll[0]],
                [shoulder_pitch[1], shoulder_roll[1]],
                [shoulder_pitch[2], shoulder_roll[2]],
                color=color, linewidth=1, linestyle=':'
            )
            
            # Connect wrist group
            self.ax_robot.plot(
                [wrist_pitch[0], wrist_yaw[0]],
                [wrist_pitch[1], wrist_yaw[1]],
                [wrist_pitch[2], wrist_yaw[2]],
                color=color, linewidth=1, linestyle=':'
            )
            self.ax_robot.plot(
                [wrist_pitch[0], wrist_roll[0]],
                [wrist_pitch[1], wrist_roll[1]],
                [wrist_pitch[2], wrist_roll[2]],
                color=color, linewidth=1, linestyle=':'
            )
        
        # Set view to show person from front (Y up, X right, Z toward viewer)
        # For standard coordinates, looking toward negative Z axis shows the front view
        self.ax_robot.view_init(elev=15, azim=270)
        
        # Add legend to show all the different joint types
        # Only show for right arm to avoid duplicate entries
        self.ax_robot.legend(loc='upper right', fontsize='x-small', ncol=2)
        
        # Update
        self.fig_robot.canvas.draw()
        self.fig_robot.canvas.flush_events()