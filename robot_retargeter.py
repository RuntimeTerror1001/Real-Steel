import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time
from ik_analytical3d import IKAnalytical3D
import xml.etree.ElementTree as ET

class RobotRetargeter:
    """Class to handle retargeting human motion to robot figure and recording data."""
    def __init__(self, robot_type="unitree_g1", recording_freq=10):
        """Initialize the RobotRetargeter."""
        # Robot specifications - Unitree G1 humanoid model
        self.robot_type = robot_type
        self.recording = False
        self.recording_freq = recording_freq
        self.last_record_time = 0
        self.recording_file = None

        # Robot dimensions (in meters)
        self.dimensions = {
            "shoulder_width": 0.200,
            "upper_arm_length": 0.300,
            "lower_arm_length": 0.300,
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
        self.recording_freq = recording_freq
        self.csv_file = None
        self.csv_writer = None
        self.frame_counter = 0
        
        # Last valid angles for error recovery
        self.last_valid_angles = self.safe_copy(self.joint_angles)
        
        # Arm kinematics module - analytical solver
        self.analytical_solver = IKAnalytical3D(
            upper_arm_length=self.dimensions["upper_arm_length"],
            lower_arm_length=self.dimensions["lower_arm_length"]
        )
        
        # Extract Joint Limits from XML file
        xml_path = "unitree_g1/g1.xml"
        self.joint_limits = self.load_joint_limits_from_xml(xml_path)
        
        # Reference to the visualization axis
        self.ax_robot = None

        # Visualization joint positions (for plotting)
        self.left_shoulder_pos = None
        self.left_elbow_pos = None
        self.left_wrist_pos = None
        self.right_shoulder_pos = None
        self.right_elbow_pos = None
        self.right_wrist_pos = None

        # MediaPipe setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def start_recording(self, filename="robot_motion.csv"):
        """Start recording robot motion to CSV file."""
        print(f"[TRACE] Started recording to {filename}")

        if self.recording:
            print("Already recording")
            return
            
        self.recording = True
        self.start_time = time.time()
        self.recording_file = filename
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        header = ['timestamp']
        for joint in self.joint_angles.keys():
            header.append(joint)
        self.csv_writer.writerow(header)
        
        self.last_record_time = time.time()
        print(f"Recording started to {filename} at {self.recording_freq}Hz")

    def stop_recording(self):
        """Stop recording and close the CSV file."""
        if not self.recording:
            print("Not currently recording")
            return
            
        self.recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print(f"Recording stopped: {self.recording_file}")
        self.recording_file = None

    def record_frame(self):
        """Record current joint angles to CSV with validation."""
        print("[TRACE] record_frame() called")

        if not self.recording:
            print("[TRACE] Not Recording - skipping frame")
            return
        
        if self.frame_counter == 0 or (time.time() - self.start_time) >= self.frame_counter * (1.0 / self.recording_freq):

            # Get current target position and orientation
            target_position = self.robot_joints["right_wrist"]
            target_orientation = None  # Can be added if needed
            
            try:
                # Validate IK solution but don't block recording
                is_valid, position_error = self.analytical_solver.validate_ik_fk(
                    self.joint_angles, target_position, target_orientation
                )
                
                if not is_valid:
                    print(f"[WARNING] IK validation error {position_error:.6f}")

                # Compute timestamp from frame index and frequency
                timestamp = round(self.frame_counter * (1.0 / self.recording_freq), 3)

                print(f"[RECORDING] Frame {self.frame_counter}, angles: {list(self.joint_angles.values())[:3]}...")

                timestamp = self.frame_counter * (1.0 / self.recording_freq)
                # Create row with timestamp and joint angles
                row = [f"{timestamp:.3f}"]
                for joint in self.joint_angles.keys():
                    row.append(f"{self.joint_angles[joint]:.6f}")
                
                # Write row and flush to ensure data is saved
                self.csv_writer.writerow(row)
                self.csv_file.flush()
                
                # Update timing and frame counter
                self.frame_counter += 1
                
                if self.frame_counter % 100 == 0:  # Reduced frequency of progress updates
                    print(f"Recorded {self.frame_counter} frames")
                    
            except Exception as e:
                print(f"[ERROR] Recording failed: {str(e)}")

    def validate_recorded_motion(self, csv_file):
        """Validate the entire recorded motion sequence."""
        joint_angles_sequence = []
        
        # Read the CSV file
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Convert row to joint angles dictionary
                angles = {}
                for i, joint in enumerate(header[1:], 1):  # Skip timestamp
                    angles[joint] = float(row[i])
                joint_angles_sequence.append(angles)
        
        # Check motion continuity
        is_continuous, problematic_joints = self.analytical_solver.check_motion_continuity(
            joint_angles_sequence
        )
        
        if not is_continuous:
            print("Warning: Motion contains sudden changes:")
            for joint, frame, velocity in problematic_joints:
                print(f"  - {joint} at frame {frame} with velocity {velocity:.3f} rad/s")
        
        return is_continuous, problematic_joints

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
        """Retarget human pose to robot."""
        if not human_landmarks:
            print("[ERROR] No human landmarks provided to retarget_pose.")
            return

        # Get landmarks
        landmarks = human_landmarks.landmark
        self.robot_joints["torso"] = np.array([0, 0, 0])
        scale = 0.3  # Scale factor to adjust for robot size
        for side in ["left", "right"]:
            if side == "left":
                try:
                    shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                    elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
                    wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
                except Exception as e:
                    print(f"[ERROR] Missing left arm landmarks: {e}")
                    continue
            else:
                try:
                    shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                    elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
                    wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
                except Exception as e:
                    print(f"[ERROR] Missing right arm landmarks: {e}")
                    continue
            sign = -1 if side == "left" else 1
            self.robot_joints[f"{side}_shoulder"] = np.array([
                sign * shoulder.x * scale,
                -shoulder.z * scale,
                -shoulder.y * scale
            ])
            self.robot_joints[f"{side}_elbow"] = np.array([
                sign * elbow.x * scale,
                -elbow.z * scale,
                -elbow.y * scale
            ])
            self.robot_joints[f"{side}_wrist"] = np.array([
                sign * wrist.x * scale,
                -wrist.z * scale,
                -wrist.y * scale
            ])
        self.scale_to_robot_dimensions()
        try:
            left_angles = self.calculate_joint_angles("left")
            right_angles = self.calculate_joint_angles("right")
            # Merge both arms' angles into self.joint_angles (standardized names)
            self.joint_angles.update({
                "left_shoulder_pitch_joint": left_angles.get("left_shoulder_pitch_joint", 0.0),
                "left_shoulder_roll_joint": left_angles.get("left_shoulder_roll_joint", 0.0),
                "left_shoulder_yaw_joint": left_angles.get("left_shoulder_yaw_joint", 0.0),
                "left_elbow_joint": left_angles.get("left_elbow_joint", 0.0),
                "right_shoulder_pitch_joint": right_angles.get("right_shoulder_pitch_joint", 0.0),
                "right_shoulder_roll_joint": right_angles.get("right_shoulder_roll_joint", 0.0),
                "right_shoulder_yaw_joint": right_angles.get("right_shoulder_yaw_joint", 0.0),
                "right_elbow_joint": right_angles.get("right_elbow_joint", 0.0),
                # Add wrist joints if your IK returns them
                "left_wrist_pitch_joint": left_angles.get("left_wrist_pitch_joint", 0.0),
                "left_wrist_yaw_joint": left_angles.get("left_wrist_yaw_joint", 0.0),
                "left_wrist_roll_joint": left_angles.get("left_wrist_roll_joint", 0.0),
                "right_wrist_pitch_joint": right_angles.get("right_wrist_pitch_joint", 0.0),
                "right_wrist_yaw_joint": right_angles.get("right_wrist_yaw_joint", 0.0),
                "right_wrist_roll_joint": right_angles.get("right_wrist_roll_joint", 0.0),
            })
        except Exception as e:
            print(f"[ERROR] IK calculation failed: {str(e)}")
            self.joint_angles = self.safe_copy(self.last_valid_angles)

        print("[DEBUG] Computed joint angles:", self.joint_angles)

        self.update_visualization_positions()

    def calculate_joint_angles(self, side="right"):
        """Calculate joint angles for the specified arm using analytical IK."""
        try:
            shoulder = self.robot_joints[f"{side}_shoulder"]
            elbow = self.robot_joints[f"{side}_elbow"]
            wrist = self.robot_joints[f"{side}_wrist"]
            
            angles = self.analytical_solver.solve(shoulder, elbow, wrist)
            
            # Standardize output keys to *_joint
            angles_3d = {
                f"{side}_shoulder_pitch_joint": angles.get("shoulder_pitch", 0.0),
                f"{side}_shoulder_roll_joint": angles.get("shoulder_roll", 0.0),
                f"{side}_shoulder_yaw_joint": angles.get("shoulder_yaw", 0.0),
                f"{side}_elbow_joint": angles.get("elbow", 0.0),
                f"{side}_wrist_pitch_joint": angles.get("wrist_pitch", 0.0),
                f"{side}_wrist_yaw_joint": angles.get("wrist_yaw", 0.0),
                f"{side}_wrist_roll_joint": angles.get("wrist_roll", 0.0),
            }
            
            # Check joint limits silently and clip values
            for joint, angle in angles_3d.items():
                if joint in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[joint]
                    if not (min_limit <= angle <= max_limit):
                        angles_3d[joint] = np.clip(angle, min_limit, max_limit)
            
            self.last_valid_angles.update(angles_3d)
            return angles_3d
        except Exception as e:
            print(f"[ERROR] IK failed for {side} arm: {str(e)}")
            return self.safe_copy(self.last_valid_angles)

    def apply_smooth_limit(self, angle, joint_type, side=None):
        """Apply joint limits with smooth transitions."""
        limits = self.joint_limits.get(joint_type)
        if not limits:
            return angle
            
        # Get appropriate limits
        if isinstance(limits, dict) and side in limits:
            min_limit, max_limit = limits[side]
        elif isinstance(limits, tuple):
            min_limit, max_limit = limits
        else:
            return angle
            
        # Calculate current velocity
        current_velocity = 0
        if len(self.transition_history[f"{side}_{joint_type}"]) > 0:
            current_velocity = abs(angle - self.transition_history[f"{side}_{joint_type}"][-1])
            
        # Update transition history
        self.transition_history[f"{side}_{joint_type}"].append(angle)
        if len(self.transition_history[f"{side}_{joint_type}"]) > self.transition_window:
            self.transition_history[f"{side}_{joint_type}"].pop(0)
            
        # Apply smooth transition if approaching limits
        if angle < min_limit or angle > max_limit:
            # Calculate transition factor based on velocity
            transition_factor = min(1.0, current_velocity / self.motion_speed_threshold)
            
            # Smoothly transition to limit
            if angle < min_limit:
                return min_limit + (angle - min_limit) * transition_factor
            else:
                return max_limit + (angle - max_limit) * transition_factor
                
        return angle

    def apply_adaptive_smoothing(self, raw_angles):
        """Apply adaptive smoothing based on motion speed."""
        for joint, raw_angle in raw_angles.items():
            # Calculate motion speed
            if len(self.joint_angle_history[joint]) > 0:
                last_angle = self.joint_angle_history[joint][-1]
                motion_speed = abs(raw_angle - last_angle)
                
                # Determine smoothing factor based on motion speed
                if motion_speed > self.motion_speed_threshold:
                    smoothing_factor = self.fast_motion_factor
                else:
                    smoothing_factor = self.slow_motion_factor
                    
                # Apply joint-specific velocity limit
                joint_type = joint.split('_')[1]  # Extract joint type (shoulder, elbow, wrist)
                if joint_type in self.joint_velocity_limits:
                    max_velocity = self.joint_velocity_limits[joint_type]
                    if motion_speed > max_velocity:
                        raw_angle = last_angle + max_velocity * np.sign(raw_angle - last_angle)
                
                # Apply smoothing
                if len(self.joint_angle_history[joint]) >= self.smoothing_window:
                    smoothed_value = np.mean(self.joint_angle_history[joint])
                    self.joint_angles[joint] = (smoothed_value * smoothing_factor + 
                                              raw_angle * (1 - smoothing_factor))
                else:
                    self.joint_angles[joint] = raw_angle
                    
                # Update history
                self.joint_angle_history[joint].append(raw_angle)
                if len(self.joint_angle_history[joint]) > self.smoothing_window:
                    self.joint_angle_history[joint].pop(0)
            else:
                self.joint_angles[joint] = raw_angle
                self.joint_angle_history[joint].append(raw_angle)
        
            self.previous_joint_angles = self.safe_copy(self.joint_angles)

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
            if limit is not None:
                # Try both 'range' attribute and 'lower'/'upper'
                if "range" in limit.attrib:
                    min_val, max_val = map(float, limit.get("range").split())
                else:
                    min_val = float(limit.get("lower", "-3.14"))
                    max_val = float(limit.get("upper", "3.14"))
                joint_limits[name] = (min_val, max_val)
            elif "range" in joint.attrib:
                min_val, max_val = map(float, joint.get("range").split())
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
        """Convert joint angles from the retargeter convention to MuJoCo convention."""
        mujoco_angles = {}
        if not apply_offset:
            return self.safe_copy(angles)
        
        # Apply simple mirroring for right arm joints
        for joint, angle in angles.items():
            if joint.startswith('right_'):
                mujoco_angles[joint] = -angle
            else:
                mujoco_angles[joint] = angle
                
        return mujoco_angles

    def update_robot_plot(self, ax=None):
        """Update the robot visualization plot."""
        if ax is None:
            ax = self.ax_robot

        # Clear previous plot
        ax.clear()

        # Plot robot skeleton
        self.plot_robot_skeleton(ax)

        # Set axis properties
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-0.2, 1.4])
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.view_init(elev=0, azim=270)  # or azim=90, depending on your coordinate system

    def plot_robot_skeleton(self, ax):
        """Plot the robot skeleton using current joint positions."""
        # Plot base
        ax.scatter3D(0, 0, 0, c='r', marker='s')

        # Plot left arm
        if self.left_shoulder_pos is not None:
            ax.scatter3D(self.left_shoulder_pos[0], self.left_shoulder_pos[2], self.left_shoulder_pos[1], c='b')
            if self.left_elbow_pos is not None:
                ax.plot3D([self.left_shoulder_pos[0], self.left_elbow_pos[0]],
                         [self.left_shoulder_pos[2], self.left_elbow_pos[2]],
                         [self.left_shoulder_pos[1], self.left_elbow_pos[1]], 'b-')
                ax.scatter3D(self.left_elbow_pos[0], self.left_elbow_pos[2], self.left_elbow_pos[1], c='g')
                if self.left_wrist_pos is not None:
                    ax.plot3D([self.left_elbow_pos[0], self.left_wrist_pos[0]],
                             [self.left_elbow_pos[2], self.left_wrist_pos[2]],
                             [self.left_elbow_pos[1], self.left_wrist_pos[1]], 'g-')
                    ax.scatter3D(self.left_wrist_pos[0], self.left_wrist_pos[2], self.left_wrist_pos[1], c='r')

        # Plot right arm
        if self.right_shoulder_pos is not None:
            ax.scatter3D(self.right_shoulder_pos[0], self.right_shoulder_pos[2], self.right_shoulder_pos[1], c='b')
            if self.right_elbow_pos is not None:
                ax.plot3D([self.right_shoulder_pos[0], self.right_elbow_pos[0]],
                         [self.right_shoulder_pos[2], self.right_elbow_pos[2]],
                         [self.right_shoulder_pos[1], self.right_elbow_pos[1]], 'b-')
                ax.scatter3D(self.right_elbow_pos[0], self.right_elbow_pos[2], self.right_elbow_pos[1], c='g')
                if self.right_wrist_pos is not None:
                    ax.plot3D([self.right_elbow_pos[0], self.right_wrist_pos[0]],
                             [self.right_elbow_pos[2], self.right_wrist_pos[2]],
                             [self.right_elbow_pos[1], self.right_wrist_pos[1]], 'g-')
                    ax.scatter3D(self.right_wrist_pos[0], self.right_wrist_pos[2], self.right_wrist_pos[1], c='r')

    def calculate_forward_kinematics(self):
        """Calculate joint positions using forward kinematics."""
        positions = {}
        
        # Base positions (shoulders)
        shoulder_width = 0.3  # meters
        positions['left_shoulder'] = np.array([-shoulder_width/2, 0, 0])
        positions['right_shoulder'] = np.array([shoulder_width/2, 0, 0])
        
        for side in ['left', 'right']:
            # Get current angles
            shoulder_roll = np.radians(self.joint_angles[f'{side}_shoulder_roll_joint'])
            shoulder_pitch = np.radians(self.joint_angles[f'{side}_shoulder_pitch_joint'])
            shoulder_yaw = np.radians(self.joint_angles[f'{side}_shoulder_yaw_joint'])
            elbow = np.radians(self.joint_angles[f'{side}_elbow_joint'])
            
            # Calculate elbow position
            upper_arm = 0.3  # meters
            elbow_offset = np.array([0, 0, -upper_arm])
            
            # Apply shoulder transformations
            R_roll = np.array([[1, 0, 0],
                             [0, np.cos(shoulder_roll), -np.sin(shoulder_roll)],
                             [0, np.sin(shoulder_roll), np.cos(shoulder_roll)]])
            
            R_pitch = np.array([[np.cos(shoulder_pitch), 0, np.sin(shoulder_pitch)],
                              [0, 1, 0],
                              [-np.sin(shoulder_pitch), 0, np.cos(shoulder_pitch)]])
            
            R_yaw = np.array([[np.cos(shoulder_yaw), -np.sin(shoulder_yaw), 0],
                            [np.sin(shoulder_yaw), np.cos(shoulder_yaw), 0],
                            [0, 0, 1]])
            
            # Combined rotation
            R = R_yaw @ R_pitch @ R_roll
            elbow_pos = positions[f'{side}_shoulder'] + R @ elbow_offset
            positions[f'{side}_elbow'] = elbow_pos
            
            # Calculate wrist position
            forearm = 0.3  # meters
            wrist_offset = np.array([0, 0, -forearm])
            
            # Apply elbow rotation
            R_elbow = np.array([[np.cos(elbow), -np.sin(elbow), 0],
                              [np.sin(elbow), np.cos(elbow), 0],
                              [0, 0, 1]])
            
            wrist_pos = elbow_pos + R @ R_elbow @ wrist_offset
            positions[f'{side}_wrist'] = wrist_pos
        
        return positions

    def update_visualization_positions(self):
        positions = self.calculate_joint_positions()
        self.left_shoulder_pos = positions["left_shoulder_pitch_joint"]
        self.left_elbow_pos = positions["left_elbow_joint"]
        self.left_wrist_pos = positions["left_wrist_pitch_joint"]
        self.right_shoulder_pos = positions["right_shoulder_pitch_joint"]
        self.right_elbow_pos = positions["right_elbow_joint"]
        self.right_wrist_pos = positions["right_wrist_pitch_joint"]

    def results_from_image(self, image):
        results = self.pose.process(image)
        if not results.pose_landmarks:
            print("[WARN] No pose landmarks detected.")
        else:
            print("[INFO] Pose landmarks detected.")
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    
    @staticmethod
    def safe_copy(d):
        return d.copy() if d is not None else {}
