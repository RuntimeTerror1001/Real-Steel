import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time
from Arm_Kinematics import ArmKinematics
import xml.etree.ElementTree as ET

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

        # Arm kinematics module
        self.kinematics_solver = ArmKinematics()
        
        # Extract Joint Limits from XML file
        xml_path = "unitree_g1/g1.xml"
        self.joint_limits = self.load_joint_limits_from_xml(xml_path)

        # Updated robot dimensions (in meters)
        self.dimensions = {
            "shoulder_width": 0.200,
            "upper_arm_length": 0.082,
            "lower_arm_length": 0.185,
            "torso_height": 0.4,  # Use measured value if available
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
        
        # Enhanced: Joint angle history and smoothing
        self.joint_angle_history = {joint: [] for joint in self.joint_angles.keys()}
        self.smoothing_window = smoothing_window
        self.smoothing_factor = smoothing_factor
        
        # Enhanced: Velocity limiting to prevent sudden changes
        self.previous_joint_angles = self.joint_angles.copy()
        self.max_velocity = 0.1  # Maximum change in angle per frame (radians)
        
        # Enhanced: Joint-specific smoothing factors
        self.joint_type_smoothing = {
            "shoulder": 0.9,    # More smoothing for shoulders
            "elbow": 0.85,      # Medium smoothing for elbows
            "wrist": 0.8        # Less smoothing for wrists (more responsive)
        }
        
        # Enhanced: Outlier rejection threshold (in radians)
        self.outlier_threshold = 0.5  # ~30 degrees
        
        # Recording settings
        self.is_recording = False
        self.recording_freq = recording_freq  # Target frequency in Hz
        self.csv_file = None
        self.csv_writer = None
        self.last_record_time = 0
        self.record_interval = 1.0 / recording_freq
        self.frame_counter = 0
        
    def start_recording(self, filename="robot_motion.csv"):
        """Start recording joint angles to CSV file."""
        if self.is_recording:
            print("Already recording")
            return
            
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        header = ["timestamp"]
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
        """Record current joint angles to CSV in MuJoCo-compatible format."""
        if not self.is_recording:
            return
        current_time = time.time()
        if current_time - self.last_record_time >= self.record_interval:
            timestamp = self.frame_counter * (1.0 / self.recording_freq)
            mujoco_angles = self.convert_to_mujoco_precise(self.joint_angles, apply_offset=True)
            row = [f"{timestamp:.1f}"]
            for joint in [
                     'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
                     'left_elbow_joint', 
                     'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                     'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
                     'right_elbow_joint', 
                     'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
            ]:
                angle = mujoco_angles.get(joint, 0.0)
                row.append(f"{angle:.4f}")
            self.csv_writer.writerow(row)
            self.last_record_time = current_time
            self.frame_counter += 1
    
    def calculate_joint_positions(self):
        """Calculate 3D positions for all robot joints based on current skeleton."""
        positions = {}
        offset_small = 0.02  # Shoulder visualization offset
        offset_tiny = 0.01   # Wrist visualization offset

        # LEFT ARM
        positions["left_shoulder_pitch_joint"] = self.robot_joints["left_shoulder"]
        positions["left_shoulder_yaw_joint"] = self.robot_joints["left_shoulder"] + np.array([offset_small, 0, 0])
        positions["left_shoulder_roll_joint"] = self.robot_joints["left_shoulder"] + np.array([0, 0, offset_small])
        positions["left_elbow_joint"] = self.robot_joints["left_elbow"]
        positions["left_wrist_pitch_joint"] = self.robot_joints["left_wrist"] + np.array([0, 0, offset_tiny])
        positions["left_wrist_yaw_joint"] = self.robot_joints["left_wrist"] + np.array([offset_tiny, 0, 0])
        positions["left_wrist_roll_joint"] = self.robot_joints["left_wrist"] + np.array([0, offset_tiny, 0])

        # RIGHT ARM
        positions["right_shoulder_pitch_joint"] = self.robot_joints["right_shoulder"]
        positions["right_shoulder_yaw_joint"] = self.robot_joints["right_shoulder"] + np.array([offset_small, 0, 0])
        positions["right_shoulder_roll_joint"] = self.robot_joints["right_shoulder"] + np.array([0, 0, offset_small])
        positions["right_elbow_joint"] = self.robot_joints["right_elbow"]
        positions["right_wrist_pitch_joint"] = self.robot_joints["right_wrist"] + np.array([0, 0, offset_tiny])
        positions["right_wrist_yaw_joint"] = self.robot_joints["right_wrist"] + np.array([offset_tiny, 0, 0])
        positions["right_wrist_roll_joint"] = self.robot_joints["right_wrist"] + np.array([0, offset_tiny, 0])

        return positions

    def retarget_pose(self, human_landmarks, rotation_angle=0):
        print("[INFO] Retargeting new frame")
        if not human_landmarks:
            print("[WARN] No human landmarks detected")
            return
            
        # Use the torso as the origin.
        landmarks = human_landmarks.landmark
        self.robot_joints["torso"] = np.array([0, 0, 0])
        scale = 0.3  # Scale factor to adjust for robot size
        angle_rad = math.radians(rotation_angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        
        # Process both sides: left and right.
        for side in ["left", "right"]:
            if side == "left":
                shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
                wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            else:
                shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
                wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
                
            # Map shoulder coordinates from MediaPipe to standard coordinates:
            # X: flip sign; Y: use -shoulder.z; Z: use -shoulder.y
            standard_shoulder_x = -shoulder.x      
            standard_shoulder_y = -shoulder.z      
            standard_shoulder_z = -shoulder.y      
            shoulder_xz = np.array([standard_shoulder_x, standard_shoulder_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, shoulder_xz)
                standard_shoulder_x, standard_shoulder_z = rotated
            self.robot_joints[f"{side}_shoulder"] = np.array([
                standard_shoulder_x * scale,
                standard_shoulder_y * scale,
                standard_shoulder_z * scale
            ])
            
            # Map elbow coordinates
            standard_elbow_x = -elbow.x
            standard_elbow_y = -elbow.z
            standard_elbow_z = -elbow.y
            elbow_xz = np.array([standard_elbow_x, standard_elbow_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, elbow_xz)
                standard_elbow_x, standard_elbow_z = rotated
            self.robot_joints[f"{side}_elbow"] = np.array([
                standard_elbow_x * scale,
                standard_elbow_y * scale,
                standard_elbow_z * scale
            ])
            
            # Map wrist coordinates
            standard_wrist_x = -wrist.x
            standard_wrist_y = -wrist.z
            standard_wrist_z = -wrist.y
            wrist_xz = np.array([standard_wrist_x, standard_wrist_z])
            if rotation_angle != 0:
                rotated = np.dot(rotation_matrix, wrist_xz)
                standard_wrist_x, standard_wrist_z = rotated
            self.robot_joints[f"{side}_wrist"] = np.array([
                standard_wrist_x * scale,
                standard_wrist_y * scale,
                standard_wrist_z * scale
            ])
            
            # Ensure all joint positions are floats
            for joint_name in self.robot_joints:
                self.robot_joints[joint_name] = self.robot_joints[joint_name].astype(float)
        
        # After both sides have been processed, perform shoulder leveling.
        shoulder_height_diff = self.robot_joints["left_shoulder"][1] - self.robot_joints["right_shoulder"][1]
        self.robot_joints["left_shoulder"][1] -= shoulder_height_diff / 2
        self.robot_joints["right_shoulder"][1] += shoulder_height_diff / 2
        
        # Adjust elbows and wrists to maintain proper arm structure after leveling.
        left_elbow_offset = np.array([0, -shoulder_height_diff / 2, 0])
        left_wrist_offset = np.array([0, -shoulder_height_diff / 2, 0])
        self.robot_joints["left_elbow"] += left_elbow_offset
        self.robot_joints["left_wrist"] += left_wrist_offset
        
        right_elbow_offset = np.array([0, shoulder_height_diff / 2, 0])
        right_wrist_offset = np.array([0, shoulder_height_diff / 2, 0])
        self.robot_joints["right_elbow"] += right_elbow_offset
        self.robot_joints["right_wrist"] += right_wrist_offset
        
        # Finally, compute the robot joint angles from the updated positions.
        self.calculate_joint_angles()
        
    def calculate_joint_angles(self):
        # Compute right arm joint angles.
        angles_3d_r = self.kinematics_solver.solve_3d_joint_angles(
            self.robot_joints["right_shoulder"],
            self.robot_joints["right_elbow"],
            self.robot_joints["right_wrist"]
        )
        self.joint_angles["right_shoulder_pitch_joint"] = self.apply_limit(angles_3d_r["shoulder_pitch"], "shoulder_pitch")
        self.joint_angles["right_shoulder_yaw_joint"] = self.apply_limit(angles_3d_r["shoulder_yaw"], "shoulder_yaw")
        self.joint_angles["right_shoulder_roll_joint"] = self.apply_limit(angles_3d_r["shoulder_roll"], "shoulder_roll", "right")
        self.joint_angles["right_elbow_joint"] = self.apply_limit(angles_3d_r["elbow"], "elbow")
        self.joint_angles["right_wrist_pitch_joint"] = self.apply_limit(angles_3d_r["wrist_pitch"], "wrist_pitch")
        self.joint_angles["right_wrist_yaw_joint"] = self.apply_limit(angles_3d_r["wrist_yaw"], "wrist_yaw")
        self.joint_angles["right_wrist_roll_joint"] = self.apply_limit(angles_3d_r["wrist_roll"], "wrist_roll")

        # Compute left arm joint angles.
        angles_3d_l = self.kinematics_solver.solve_3d_joint_angles(
            self.robot_joints["left_shoulder"],
            self.robot_joints["left_elbow"],
            self.robot_joints["left_wrist"]
        )
        self.joint_angles["left_shoulder_pitch_joint"] = self.apply_limit(angles_3d_l["shoulder_pitch"], "shoulder_pitch")
        self.joint_angles["left_shoulder_yaw_joint"] = self.apply_limit(angles_3d_l["shoulder_yaw"], "shoulder_yaw")
        self.joint_angles["left_shoulder_roll_joint"] = self.apply_limit(angles_3d_l["shoulder_roll"], "shoulder_roll", "left")
        self.joint_angles["left_elbow_joint"] = self.apply_limit(angles_3d_l["elbow"], "elbow")
        self.joint_angles["left_wrist_pitch_joint"] = self.apply_limit(angles_3d_l["wrist_pitch"], "wrist_pitch")
        self.joint_angles["left_wrist_yaw_joint"] = self.apply_limit(angles_3d_l["wrist_yaw"], "wrist_yaw")
        self.joint_angles["left_wrist_roll_joint"] = self.apply_limit(angles_3d_l["wrist_roll"], "wrist_roll")

        self.update_joint_history_with_outlier_detection(self.joint_angles)
        self.apply_enhanced_smoothing(self.joint_angles)

    def update_joint_history_with_outlier_detection(self, raw_angles):
        for joint, raw_angle in raw_angles.items():
            is_outlier = False
            if len(self.joint_angle_history[joint]) > 0:
                last_angle = self.joint_angle_history[joint][-1]
                if abs(raw_angle - last_angle) > self.outlier_threshold:
                    is_outlier = True
            if not is_outlier:
                self.joint_angle_history[joint].append(raw_angle)
                if len(self.joint_angle_history[joint]) > self.smoothing_window:
                    self.joint_angle_history[joint].pop(0)

    def apply_enhanced_smoothing(self, raw_angles):
        for joint, raw_angle in raw_angles.items():
            history = self.joint_angle_history[joint]
            if len(history) < 3:
                self.joint_angles[joint] = raw_angle
                continue
                
            smoothing_factor = self.smoothing_factor
            for joint_type, factor in self.joint_type_smoothing.items():
                if joint_type in joint:
                    smoothing_factor = factor
                    break
                    
            weights = [i+1 for i in range(len(history))]
            total_weight = sum(weights)
            weighted_sum = sum(h * w for h, w in zip(history, weights))
            weighted_average = weighted_sum / total_weight
            
            current_value = self.joint_angles.get(joint, raw_angle)
            smoothed_value = current_value * smoothing_factor + weighted_average * (1 - smoothing_factor)
            
            if joint in self.previous_joint_angles:
                prev_value = self.previous_joint_angles[joint]
                change = smoothed_value - prev_value
                if abs(change) > self.max_velocity:
                    smoothed_value = prev_value + self.max_velocity * np.sign(change)
            
            self.joint_angles[joint] = smoothed_value
        
        self.previous_joint_angles = self.joint_angles.copy()

    def apply_limit(self, angle, joint_type, side=None):
        limits = self.joint_limits.get(joint_type)
        if isinstance(limits, dict):
            if side and side in limits:
                min_limit, max_limit = limits[side]
                return max(min_limit, min(max_limit, angle))
        elif isinstance(limits, tuple):
            min_limit, max_limit = limits
            return max(min_limit, min(max_limit, angle))
        return angle
    
    def load_joint_limits_from_xml(self, xml_path):
        joint_limits = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if limit is not None and "range" in limit.attrib:
                min_val, max_val = map(float, limit.get("range").split())
                joint_limits[name] = (min_val, max_val)
        return joint_limits

    def scale_to_robot_dimensions(self):
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
        
        scale_upper_right = self.dimensions["upper_arm_length"] / current_upper_arm_length_right if current_upper_arm_length_right > 0 else 1.0
        scale_lower_right = self.dimensions["lower_arm_length"] / current_lower_arm_length_right if current_lower_arm_length_right > 0 else 1.0
        scale_upper_left = self.dimensions["upper_arm_length"] / current_upper_arm_length_left if current_upper_arm_length_left > 0 else 1.0
        scale_lower_left = self.dimensions["lower_arm_length"] / current_lower_arm_length_left if current_lower_arm_length_left > 0 else 1.0
        
        vector_right_upper = self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"]
        self.robot_joints["right_elbow"] = self.robot_joints["right_shoulder"] + vector_right_upper * scale_upper_right
        
        vector_left_upper = self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"]
        self.robot_joints["left_elbow"] = self.robot_joints["left_shoulder"] + vector_left_upper * scale_upper_left
        
        vector_right_lower = self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]
        self.robot_joints["right_wrist"] = self.robot_joints["right_elbow"] + vector_right_lower * scale_lower_right
        
        vector_left_lower = self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]
        self.robot_joints["left_wrist"] = self.robot_joints["left_elbow"] + vector_left_lower * scale_lower_left

    def convert_to_mujoco_precise(self, angles, apply_offset=True):
     """
     Convert joint angles from the retargeter convention to MuJoCo convention.
     In this updated version, excessive fixed offsets have been removed.
    
     Parameters:
      angles : dict
          Dictionary of joint names to angle values.
          apply_offset : bool
          If True, apply calibration offsets (here set to zero or negligible values).
    
     Returns:
      dict
          Dictionary of joint names to MuJoCo-compatible angle values.
    """
     mujoco_angles = {}
     if not apply_offset:
         return angles.copy()
    
    # Instead of using large fixed offsets, we now either use zero offsets or calibrated ones.
    # For now, we assume zero offsets. Adjust these values if you obtain calibration data.
     offset_left_shoulder_pitch = 0.0
     offset_left_shoulder_roll  = 0.0
     offset_left_shoulder_yaw   = 0.0
     offset_left_elbow          = 0.0
     offset_left_wrist_pitch    = 0.0
     offset_left_wrist_roll     = 0.0
     offset_left_wrist_yaw      = 0.0

     offset_right_shoulder_pitch = 0.0
     offset_right_shoulder_roll  = 0.0
     offset_right_shoulder_yaw   = 0.0
     offset_right_elbow          = 0.0
     offset_right_wrist_pitch    = 0.0
     offset_right_wrist_roll     = 0.0
     offset_right_wrist_yaw      = 0.0
    
    # Apply offsets as needed. Notice for right arm joints, the raw angle is negated.
     mujoco_angles["left_shoulder_pitch_joint"] = angles["left_shoulder_pitch_joint"] + offset_left_shoulder_pitch
     mujoco_angles["left_shoulder_roll_joint"]  = angles["left_shoulder_roll_joint"]  + offset_left_shoulder_roll
     mujoco_angles["left_shoulder_yaw_joint"]   = angles["left_shoulder_yaw_joint"]   + offset_left_shoulder_yaw
     mujoco_angles["left_elbow_joint"]          = angles["left_elbow_joint"]          + offset_left_elbow
     mujoco_angles["left_wrist_pitch_joint"]    = angles["left_wrist_pitch_joint"]    + offset_left_wrist_pitch
     mujoco_angles["left_wrist_roll_joint"]     = angles["left_wrist_roll_joint"]     + offset_left_wrist_roll
     mujoco_angles["left_wrist_yaw_joint"]      = angles["left_wrist_yaw_joint"]      + offset_left_wrist_yaw

     mujoco_angles["right_shoulder_pitch_joint"] = -angles["right_shoulder_pitch_joint"] + offset_right_shoulder_pitch
     mujoco_angles["right_shoulder_roll_joint"]  = -angles["right_shoulder_roll_joint"]  + offset_right_shoulder_roll
     mujoco_angles["right_shoulder_yaw_joint"]   = -angles["right_shoulder_yaw_joint"]   + offset_right_shoulder_yaw
     mujoco_angles["right_elbow_joint"]          = angles["right_elbow_joint"]           + offset_right_elbow
     mujoco_angles["right_wrist_pitch_joint"]    = -angles["right_wrist_pitch_joint"]    + offset_right_wrist_pitch
     mujoco_angles["right_wrist_roll_joint"]     = -angles["right_wrist_roll_joint"]     + offset_right_wrist_roll
     mujoco_angles["right_wrist_yaw_joint"]      = -angles["right_wrist_yaw_joint"]      + offset_right_wrist_yaw
     
     return mujoco_angles


    def update_robot_plot(self):
        self.ax_robot.clear()
        self.ax_robot.set_xlabel('X (Right →)')
        self.ax_robot.set_ylabel('Z (Forward ↗)')
        self.ax_robot.set_zlabel('Y (Up ↑)')
        limit = 0.4
        self.ax_robot.set_xlim3d(-limit, limit)
        self.ax_robot.set_ylim3d(-limit, limit)
        self.ax_robot.set_zlim3d(-limit, limit)
        self.ax_robot.grid(True)
        
        shoulder_width = np.linalg.norm(self.robot_joints["right_shoulder"] - self.robot_joints["left_shoulder"])
        torso_height = self.dimensions["torso_height"] * 0.8
        shoulder_midpoint = (self.robot_joints["left_shoulder"] + self.robot_joints["right_shoulder"]) / 2
        waist_midpoint = shoulder_midpoint - np.array([0, torso_height, 0])
        waist_width = shoulder_width * 0.8
        waist_left = waist_midpoint + np.array([waist_width/2, 0, 0])
        waist_right = waist_midpoint - np.array([waist_width/2, 0, 0])
        torso_points = [
            self.robot_joints["left_shoulder"],
            self.robot_joints["right_shoulder"],
            waist_right,
            waist_left,
            self.robot_joints["left_shoulder"]
        ]
        torso_x = [point[0] for point in torso_points]
        torso_y = [point[1] for point in torso_points]
        torso_z = [point[2] for point in torso_points]
        self.ax_robot.plot(torso_x, torso_y, torso_z, 'k-', linewidth=2)
        
        self.ax_robot.plot(
            [self.robot_joints["left_shoulder"][0], waist_left[0]],
            [self.robot_joints["left_shoulder"][1], waist_left[1]],
            [self.robot_joints["left_shoulder"][2], waist_left[2]],
            'k-', linewidth=2
        )
        self.ax_robot.plot(
            [self.robot_joints["right_shoulder"][0], waist_right[0]],
            [self.robot_joints["right_shoulder"][1], waist_right[1]],
            [self.robot_joints["right_shoulder"][2], waist_right[2]],
            'k-', linewidth=2
        )
        self.ax_robot.scatter(*self.robot_joints["torso"], c='black', marker='o', s=50)
        self.ax_robot.plot(
            [self.robot_joints["left_shoulder"][0], self.robot_joints["right_shoulder"][0]],
            [self.robot_joints["left_shoulder"][1], self.robot_joints["right_shoulder"][1]],
            [self.robot_joints["left_shoulder"][2], self.robot_joints["right_shoulder"][2]],
            'k-', linewidth=3
        )
        
        joint_positions = self.calculate_joint_positions()
        
        for side, color in [("right", "blue"), ("left", "green")]:
            shoulder = self.robot_joints[f"{side}_shoulder"]
            elbow = self.robot_joints[f"{side}_elbow"]
            wrist = self.robot_joints[f"{side}_wrist"]
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
            shoulder_pitch = joint_positions[f"{side}_shoulder_pitch_joint"]
            shoulder_yaw = joint_positions[f"{side}_shoulder_yaw_joint"]
            shoulder_roll = joint_positions[f"{side}_shoulder_roll_joint"]
            elbow_joint = joint_positions[f"{side}_elbow_joint"]
            wrist_pitch = joint_positions[f"{side}_wrist_pitch_joint"]
            wrist_yaw = joint_positions[f"{side}_wrist_yaw_joint"]
            wrist_roll = joint_positions[f"{side}_wrist_roll_joint"]
            
            self.ax_robot.scatter(shoulder_pitch[0], shoulder_pitch[1], shoulder_pitch[2], 
                           color=color, s=80, marker='o', label=f"{side.capitalize()} Shoulder Pitch" if side=="right" else "")
            self.ax_robot.scatter(shoulder_yaw[0], shoulder_yaw[1], shoulder_yaw[2], 
                           color=color, s=50, marker='s', label=f"{side.capitalize()} Shoulder Yaw" if side=="right" else "")
            self.ax_robot.scatter(shoulder_roll[0], shoulder_roll[1], shoulder_roll[2], 
                           color=color, s=50, marker='^', label=f"{side.capitalize()} Shoulder Roll" if side=="right" else "")
            self.ax_robot.scatter(elbow_joint[0], elbow_joint[1], elbow_joint[2], 
                           color=color, s=60, marker='o', label=f"{side.capitalize()} Elbow" if side=="right" else "")
            self.ax_robot.scatter(wrist_pitch[0], wrist_pitch[1], wrist_pitch[2], 
                           color=color, s=40, marker='d', label=f"{side.capitalize()} Wrist Pitch" if side=="right" else "")
            self.ax_robot.scatter(wrist_yaw[0], wrist_yaw[1], wrist_yaw[2], 
                           color=color, s=40, marker='*', label=f"{side.capitalize()} Wrist Yaw" if side=="right" else "")
            self.ax_robot.scatter(wrist_roll[0], wrist_roll[1], wrist_roll[2], 
                           color=color, s=40, marker='p', label=f"{side.capitalize()} Wrist Roll" if side=="right" else "")
            
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
            
        self.ax_robot.view_init(elev=15, azim=270)
        self.ax_robot.legend(loc='upper right', fontsize='x-small', ncol=2)
        self.fig_robot.canvas.draw()
        self.fig_robot.canvas.flush_events()
