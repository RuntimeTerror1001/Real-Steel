import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import sys
from ik_analytical3d import IKAnalytical3DRefined
import xml.etree.ElementTree as ET
import ikpy.chain
import ikpy.utils.plot as ikpy_plot

# Ensure CSV support is available
try:
    import csv
except ImportError:
    print("CSV module not available, implementing basic CSV functionality")
    # Basic CSV writer implementation
    class CsvWriter:
        def __init__(self, file):
            self.file = file
            
        def writerow(self, row):
            line = ','.join(str(item) for item in row) + '\n'
            self.file.write(line)
    
    class CsvModule:
        @staticmethod
        def writer(file):
            return CsvWriter(file)
    
    csv = CsvModule()

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

        # Robot dimensions (in meters) — match IKAnalytical3D defaults for consistency
        self.dimensions = {
            "shoulder_width": 0.200,
            "upper_arm_length": 0.1032,  # use same L1 as IKAnalytical3D
            "lower_arm_length": 0.1000,  # use same L2 as IKAnalytical3D
            "torso_height": 0.4,
        }
        
        # Scale factor for converting human dimensions to robot dimensions
        self.scale = 0.5  # Adjust this value based on your specific needs
        
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
        self.recording_freq = recording_freq
        self.csv_file = None
        self.csv_writer = None
        self.frame_counter = 0
        
        # Last valid angles for error recovery
        self.last_valid_angles = self.safe_copy(self.joint_angles)
        
        # Arm kinematics module - analytical solver with refinement
        self.analytical_solver = IKAnalytical3DRefined(
            upper_arm_length=self.dimensions["upper_arm_length"],
            lower_arm_length=self.dimensions["lower_arm_length"],
            position_tolerance=1e-5,  # Higher tolerance for more stable solutions
            refinement_gain=0.3       # Conservative gain for refinement
        )
        
        # IK solver error tracking
        self.ik_error_tracking = {
            'analytical_errors': 0,
            'ikpy_errors': 0,
            'joint_limit_clips': 0,
            'total_frames': 0,
            'last_error_message': '',
        }
        
        # IK solver state
        self.pause_on_error = False
        
        # Try to load the IKPy chain - will be None if URDF isn't found
        try:
            # Check if URDF file exists, if not convert from XML
            urdf_path = "unitree_g1/g1.urdf"
            if not os.path.exists(urdf_path):
                print("URDF file not found, attempting to convert from XML...")
                try:
                    from mjcf_to_urdf import mjcf_to_urdf
                    xml_path = "unitree_g1/g1.xml"
                    mjcf_to_urdf(xml_path, urdf_path)
                    print(f"Successfully converted {xml_path} to {urdf_path}")
                except Exception as e:
                    print(f"Failed to convert XML to URDF: {e}")
            
            # Initialize IKPy chains for left and right arms
            self.ikpy_chain_right = ikpy.chain.Chain.from_urdf_file(
                urdf_path,                                          # Path to the URDF file
                base_elements=["base_link"],                        # Base element of the chain 
                active_links_mask=[False, False, False, True, True, True, True, True, True],
                name="right_arm"
            )
            self.ikpy_chain_left = ikpy.chain.Chain.from_urdf_file(
                urdf_path,                                          # Path to the URDF file
                base_elements=["base_link"],                        # Base element of the chain
                active_links_mask=[False, False, False, True, True, True, True, True, True],
                name="left_arm"
            )
            print("IKPy chains initialized for fallback IK solving")
            
            # Print chain structure for debugging
            print("Right arm chain links:")
            for i, link in enumerate(self.ikpy_chain_right.links):
                print(f"  Link {i}: {link.name} (active: {link.isactive})")
                
        except Exception as e:
            self.ikpy_chain_right = None
            self.ikpy_chain_left = None
            print(f"IKPy chain initialization failed: {e}")
            print("Fallback solver will not be available - consider generating a URDF")
        
        # Extract Joint Limits from XML file
        xml_path = "unitree_g1/g1.xml"
        self.joint_limits = self.load_joint_limits_from_xml(xml_path)
        
        # Reference to the visualization axis
        self.ax_robot = None
        self.fig_robot = None

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

        # Enhanced motion smoothing parameters
        self.smoothing_window = 5  # Number of frames for moving average
        self.velocity_window = 3   # Window for velocity smoothing
        self.max_joint_velocity = 1.0  # rad/s
        self.max_acceleration = 0.5    # rad/s^2
        
        # Smoothing factors
        self.position_alpha = 0.7  # For position-based smoothing (higher = more previous value)
        
        # Initialize motion history buffers
        self.position_history = {joint: [] for joint in self.joint_angles.keys()}
        self.velocity_history = {joint: [] for joint in self.joint_angles.keys()}
        self.timestamp_history = []
        self.last_timestamp = 0
        
        # Initialize robot visualization
        self.robot_plot = None

    def clip_angle(self, joint_name, angle):
        """Clip angle to joint limits and track statistics"""
        if joint_name in self.joint_limits:
            min_limit, max_limit = self.joint_limits[joint_name]
            
            if angle < min_limit:
                self.ik_error_tracking['joint_limit_clips'] += 1
                return min_limit
            elif angle > max_limit:
                self.ik_error_tracking['joint_limit_clips'] += 1
                return max_limit
                
        return angle

    def start_recording(self, filename=None):
        """Start recording robot motion to CSV file."""
        try:
            if filename is None:
                os.makedirs("recordings", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/robot_motion_{timestamp}.csv"

            print(f"Starting recording to {filename}")

            if self.recording:
                print("Already recording")
                return
                
            # Simple approach - just write the header
            with open(filename, 'w') as f:
                # Write header
                header = "timestamp"
                for joint in self.joint_angles.keys():
                    header += f",{joint}"
                f.write(header + "\n")
            
            # Now open for appending
            self.csv_file = open(filename, 'a')
            self.recording = True
            self.start_time = time.time()
            self.last_record_time = self.start_time  # Initialize last record time
            self.recording_file = filename
            self.frame_counter = 0
            print(f"Recording started to {filename}")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
            self.csv_file = None

    def stop_recording(self):
        """Stop recording and close the file."""
        self.recording = False
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"Recording stopped: {self.recording_file}")
                print(f"Recorded {self.frame_counter} frames")
            except Exception as e:
                print(f"Error stopping recording: {e}")
            self.csv_file = None
        self.recording_file = None

    def record_frame(self):
        """Record current joint angles to CSV with validation."""
        if not self.recording or not self.csv_file:
            return
        
        try:
            current_time = time.time()
            time_since_last_record = current_time - self.last_record_time
            
            # Enforce frequency: skip frames that are too fast
            if time_since_last_record < (1.0 / self.recording_freq):
                return
            
            # Update last record time
            self.last_record_time = current_time
            
            # Fixed timestamp based on frame count
            elapsed = self.frame_counter * (1.0 / self.recording_freq)
            
            line = f"{elapsed:.3f}"
            for joint in self.joint_angles.keys():
                line += f",{self.joint_angles[joint]:.6f}"
                
            self.csv_file.write(line + "\n")
            self.csv_file.flush()
            
            self.frame_counter += 1
            
            if self.frame_counter % 100 == 0:
                print(f"Recorded {self.frame_counter} frames")
                
        except Exception as e:
            print(f"Error recording frame: {e}")

    def load_joint_limits_from_xml(self, xml_path):
        """Load joint limits from MuJoCo XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find all joint elements
            joint_limits = {}
            for joint in root.findall(".//joint"):
                name = joint.get("name")
                
                # Check if this joint has range defined
                if "range" in joint.attrib:
                    range_str = joint.get("range")
                    range_values = [float(x) for x in range_str.split()]
                    
                    if len(range_values) >= 2:
                        joint_limits[name] = (range_values[0], range_values[1])
            
            print(f"Loaded {len(joint_limits)} joint limits from {xml_path}")
            return joint_limits
            
        except Exception as e:
            print(f"Error loading joint limits from XML: {e}")
            return {}

    def calculate_joint_positions(self):
        """Calculate robot joint positions from human pose landmarks."""
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

    def calculate_joint_angles(self, side="right"):
        """Calculate joint angles using Inverse Kinematics."""
        # Get updated joint positions based on human pose
        joints = self.robot_joints
        
        if side == "right":
            shoulder = joints["right_shoulder"] 
            elbow = joints["right_elbow"]
            wrist = joints["right_wrist"]
            prefix = "right"
        else:
            shoulder = joints["left_shoulder"]
            elbow = joints["left_elbow"]
            wrist = joints["left_wrist"]
            prefix = "left"
        
        # Track total frames for stats
        self.ik_error_tracking['total_frames'] += 1
        
        # First try analytical IK
        try:
            # Compute orientation from joint positions
            # This creates a local coordinate system for the wrist
            # X axis: from elbow to wrist (forward)
            x_axis = wrist - elbow
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            
            # Y axis: perpendicular to the plane formed by shoulder, elbow and wrist
            # (using the cross product of elbow-shoulder and wrist-elbow)
            v1 = elbow - shoulder
            v2 = wrist - elbow
            y_axis = np.cross(v1, v2)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
            
            # Z axis: perpendicular to X and Y (cross product)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
            # Construct rotation matrix from these three orthogonal axes
            target_orientation = np.column_stack((x_axis, y_axis, z_axis))
            
            # Use refined analytical IK solver with Newton refinement and orientation
            ik_solution = self.analytical_solver.solve(shoulder, elbow, wrist, target_orientation)
            
            # Store the result with proper joint naming
            new_angles = {
                f"{prefix}_shoulder_pitch_joint": ik_solution["shoulder_pitch"],
                f"{prefix}_shoulder_yaw_joint": ik_solution["shoulder_yaw"],
                f"{prefix}_shoulder_roll_joint": ik_solution["shoulder_roll"],
                f"{prefix}_elbow_joint": ik_solution["elbow"],
                f"{prefix}_wrist_pitch_joint": ik_solution["wrist_pitch"],
                f"{prefix}_wrist_yaw_joint": ik_solution["wrist_yaw"],
                f"{prefix}_wrist_roll_joint": ik_solution["wrist_roll"]
            }
            
            # Forward kinematics validation for extra security
            fk_pos, _ = self.analytical_solver.forward_kinematics(ik_solution)
            target_pos = wrist - shoulder
            error = np.linalg.norm(fk_pos - target_pos)
            
            # Check for error tolerance
            if error > 1e-2:  # Larger tolerance for practical use
                print(f"[WARNING] IK-FK validation error: {error:.6f} m")
                self.ik_error_tracking['analytical_errors'] += 1
            
            # Apply the solution to the joint angles
            for joint, angle in new_angles.items():
                self.joint_angles[joint] = self.clip_angle(joint, angle)
                
            # Store last valid angles
            self.last_valid_angles = self.safe_copy(self.joint_angles)
            return True
            
        except Exception as e:
            self.ik_error_tracking['analytical_errors'] += 1
            self.ik_error_tracking['last_error_message'] = str(e)
            print(f"[ERROR] Analytical IK failed: {str(e)}")
            
            # Fall back to IKPy
            if self.try_ikpy_fallback(shoulder, elbow, wrist, prefix):
                return True
                
            # If both fail, use last valid angles
            print(f"[WARN] Using last valid angles for {prefix} arm")
            return False

    def try_ikpy_fallback(self, shoulder, elbow, wrist, prefix):
        """Try to use IKPy as a fallback solver."""
        if prefix == "right" and self.ikpy_chain_right is not None:
            chain = self.ikpy_chain_right
        elif prefix == "left" and self.ikpy_chain_left is not None:
            chain = self.ikpy_chain_left
        else:
            return False
            
        try:
            # Target position for end effector in chain frame
            target_position = wrist - shoulder
            
            # Prepare for IK solving
            initial_position = []
            for joint_name in [
                f"{prefix}_shoulder_pitch_joint",
                f"{prefix}_shoulder_yaw_joint", 
                f"{prefix}_shoulder_roll_joint",
                f"{prefix}_elbow_joint",
                f"{prefix}_wrist_pitch_joint",
                f"{prefix}_wrist_yaw_joint",
                f"{prefix}_wrist_roll_joint"
            ]:
                initial_position.append(self.joint_angles[joint_name])
                
            # Run inverse kinematics with IKPy
            ik_solution = chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=None,
                initial_position=initial_position
            )
            
            # Active joints only, extract angles in order
            angles = []
            for i, link in enumerate(chain.links):
                if link.isactive:
                    angles.append(ik_solution[i])
            
            # Apply IKPy solution to joint angles
            joint_names = [
                f"{prefix}_shoulder_pitch_joint",
                f"{prefix}_shoulder_yaw_joint",
                f"{prefix}_shoulder_roll_joint",
                f"{prefix}_elbow_joint",
                f"{prefix}_wrist_pitch_joint",
                f"{prefix}_wrist_yaw_joint",
                f"{prefix}_wrist_roll_joint"
            ]
            
            for i, joint_name in enumerate(joint_names):
                if i < len(angles):
                    self.joint_angles[joint_name] = self.clip_angle(joint_name, angles[i])
            
            # Store last valid angles
            self.last_valid_angles = self.safe_copy(self.joint_angles)
            return True
            
        except Exception as e:
            self.ik_error_tracking['ikpy_errors'] += 1
            print(f"[ERROR] IKPy fallback also failed: {str(e)}")
            return False

    def reset_calibration(self):
        """Reset calibration and tracking variables."""
        # Reset scale to default
        self.scale = 0.5  # Or prompt for new value
        # Reset angle offset
        self.angle_offset = 0.0 if hasattr(self, 'angle_offset') else 0.0
        # Reset error state
        self.pause_on_error = False
        print("[INFO] Calibration reset. New scale: 0.5")
        
    def draw_error_overlay(self, ax=None):
        """Draw error information overlay."""
        if self.pause_on_error and ax is not None:
            # Save current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            # Create a red rectangle overlay
            error_box = plt.Rectangle(
                (xlim[0], ylim[0]), 
                width=(xlim[1]-xlim[0]), 
                height=(ylim[1]-ylim[0]),
                color='red', 
                alpha=0.2
            )
            ax.add_patch(error_box)
            
            # Add error text
            text_x = xlim[0] + (xlim[1]-xlim[0])*0.1
            text_y = ylim[0] + (ylim[1]-ylim[0])*0.5
            text_z = zlim[0] + (zlim[1]-zlim[0])*0.8
            
            ax.text(text_x, text_y, text_z, 
                   self.ik_error_tracking['last_error_message'],
                   color='white', 
                   backgroundcolor='red',
                   fontsize=10)
            
            # Add statistics
            total = self.ik_error_tracking['total_frames']
            analytical = self.ik_error_tracking['analytical_errors']
            ikpy = self.ik_error_tracking['ikpy_errors']
            
            stats_text = (
                f"Total frames: {total}\n"
                f"Analytical errors: {analytical} ({analytical/total*100:.1f}%)\n"
                f"IKPy errors: {ikpy} ({ikpy/total*100:.1f}%)"
            )
            
            ax.text(text_x, text_y, text_z-0.1, 
                   stats_text,
                   color='white', 
                   backgroundcolor='blue',
                   fontsize=8)
            
            # Restore axis limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    def update_robot_plot(self, ax=None):
        """Update the robot visualization plot."""
        if ax is not None:
            self.ax_robot = ax
            
        # Scale joint positions and update visualization
        self.scale_to_robot_dimensions()
        self.plot_robot_skeleton(self.ax_robot)
        
        # Draw error overlay if in error state
        if self.pause_on_error:
            self.draw_error_overlay(self.ax_robot)
            
        return self.ax_robot

    def scale_to_robot_dimensions(self):
        """Scale human landmarks to robot dimensions."""
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
        """Convert to MuJoCo compatible angle format."""
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

    def plot_robot_skeleton(self, ax):
        """Plot current robot skeleton state."""
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
        """Calculate end effector position using forward kinematics."""
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
        """Update positions for visualization."""
        positions = self.calculate_joint_positions()
        self.left_shoulder_pos = positions["left_shoulder_pitch_joint"]
        self.left_elbow_pos = positions["left_elbow_joint"]
        self.left_wrist_pos = positions["left_wrist_pitch_joint"]
        self.right_shoulder_pos = positions["right_shoulder_pitch_joint"]
        self.right_elbow_pos = positions["right_elbow_joint"]
        self.right_wrist_pos = positions["right_wrist_pitch_joint"]

    def results_from_image(self, image):
        """Process image with MediaPipe and return pose results."""
        results = self.pose.process(image)
        if not results.pose_landmarks:
            print("[WARN] No pose landmarks detected.")
        else:
            print("[INFO] Pose landmarks detected.")
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    
    @staticmethod
    def safe_copy(d):
        """Create a safe copy of a dictionary."""
        return {k: v for k, v in d.items()}

    def smooth_motion(self, raw_angles):
        """Apply motion smoothing to joint angles using position-based and velocity-based methods."""
        current_time = time.time()
        
        if not self.timestamp_history:
            # First frame - store initial values
            for joint in raw_angles:
                self.position_history[joint].append(raw_angles[joint])
            self.timestamp_history.append(current_time)
            self.last_timestamp = current_time
            return raw_angles
        
        # Calculate time delta
        dt = current_time - self.last_timestamp
        dt = max(dt, 0.001)  # Prevent division by zero
        self.last_timestamp = current_time
        self.timestamp_history.append(current_time)
        
        # Limit history size
        if len(self.timestamp_history) > self.smoothing_window:
            self.timestamp_history.pop(0)
        
        smoothed_angles = {}
        
        for joint, raw_angle in raw_angles.items():
            # Store current raw angle
            self.position_history[joint].append(raw_angle)
            
            # Limit history size
            if len(self.position_history[joint]) > self.smoothing_window:
                self.position_history[joint].pop(0)
            
            # Apply position-based smoothing (moving average with exponential weighting)
            if len(self.position_history[joint]) > 1:
                # Calculate previous smoothed value
                prev_smooth = self.position_history[joint][-2]
                # Apply exponential smoothing
                smooth_pos = self.position_alpha * prev_smooth + (1 - self.position_alpha) * raw_angle
            else:
                smooth_pos = raw_angle
            
            # Calculate current velocity
            if len(self.position_history[joint]) > 1:
                prev_pos = self.position_history[joint][-2]
                velocity = (raw_angle - prev_pos) / dt
                
                # Store velocity
                self.velocity_history[joint].append(velocity)
                
                # Limit velocity history size
                if len(self.velocity_history[joint]) > self.velocity_window:
                    self.velocity_history[joint].pop(0)
                
                # Apply velocity limiting if needed
                if abs(velocity) > self.max_joint_velocity:
                    max_change = self.max_joint_velocity * dt
                    if velocity > 0:
                        smooth_pos = prev_pos + max_change
                    else:
                        smooth_pos = prev_pos - max_change
            
            # Clip to joint limits
            smooth_pos = self.clip_angle(joint, smooth_pos)
            
            # Store final smoothed value
            smoothed_angles[joint] = smooth_pos
        
        return smoothed_angles

    def initialize_from_world_landmarks(self, world_landmarks):
        """Initialize robot joint positions from MediaPipe world landmarks."""
        landmarks = world_landmarks.landmark
        
        try:
            # Extract key landmarks for joint tracking
            # RIGHT ARM
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # LEFT ARM
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            # Convert to robot coordinate system and apply scaling
            # In MediaPipe world coordinates:
            # - X: left (negative) to right (positive)
            # - Y: down (negative) to up (positive)
            # - Z: backward (negative) to forward (positive)
            
            # For our robot coordinate system:
            # - X: right to left
            # - Y: up to down (gravity direction)
            # - Z: forward to backward
            
            # Coordinate transformation with normalization
            self.robot_joints["right_shoulder"] = np.array([-right_shoulder.x, -right_shoulder.z, -right_shoulder.y]) * self.scale
            self.robot_joints["right_elbow"] = np.array([-right_elbow.x, -right_elbow.z, -right_elbow.y]) * self.scale
            self.robot_joints["right_wrist"] = np.array([-right_wrist.x, -right_wrist.z, -right_wrist.y]) * self.scale
            
            self.robot_joints["left_shoulder"] = np.array([-left_shoulder.x, -left_shoulder.z, -left_shoulder.y]) * self.scale
            self.robot_joints["left_elbow"] = np.array([-left_elbow.x, -left_elbow.z, -left_elbow.y]) * self.scale
            self.robot_joints["left_wrist"] = np.array([-left_wrist.x, -left_wrist.z, -left_wrist.y]) * self.scale
            
            # Verify that the joint positions make physical sense
            # Check for NaN values or extreme values that indicate tracking errors
            for joint, pos in self.robot_joints.items():
                if np.isnan(pos).any() or np.max(np.abs(pos)) > 10.0:
                    # Use last valid positions instead
                    print(f"[WARNING] Invalid joint position detected for {joint}. Using previous valid positions.")
                    return False
                    
            # Calculate distances between joints to verify reasonableness
            r_upper_arm_length = np.linalg.norm(self.robot_joints["right_elbow"] - self.robot_joints["right_shoulder"])
            r_lower_arm_length = np.linalg.norm(self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"])
            l_upper_arm_length = np.linalg.norm(self.robot_joints["left_elbow"] - self.robot_joints["left_shoulder"])
            l_lower_arm_length = np.linalg.norm(self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"])
            
            # Normalize and scale the positions to robot dimensions
            self.scale_to_robot_dimensions()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to process landmarks: {e}")
            return False

    def update_robot_state(self, results):
        """Update robot state based on human pose results."""
        if not results:
            return False
            
        success = False
        if results.pose_world_landmarks:
            # Initialize joint positions from world landmarks (preferred method)
            landmarks_success = self.initialize_from_world_landmarks(results.pose_world_landmarks)
            if not landmarks_success:
                return False
                
            # Calculate and apply joint angles for both arms
            right_success = self.calculate_joint_angles(side="right")
            left_success = self.calculate_joint_angles(side="left")
            
            # Smooth the motion after IK solving
            self.joint_angles = self.smooth_motion(self.joint_angles)
            
            # Enforce body yaw to prevent spinning (only if the joint exists in limits)
            if "waist_yaw_joint" in self.joint_limits and "waist_yaw_joint" in self.joint_angles:
                self.joint_angles["waist_yaw_joint"] = 0.0
            
            # Update visualization positions for display
            self.update_visualization_positions()
            
            # Record frame if we're recording
            if self.recording:
                self.record_frame()
            
            success = right_success and left_success
        elif results.pose_landmarks:
            # Fallback to 2D landmarks - kept for backward compatibility
            # Calculate joint positions from human pose
            self.robot_joints = self.calculate_joint_positions()
            
            # Calculate and apply joint angles for both arms
            right_success = self.calculate_joint_angles(side="right")
            left_success = self.calculate_joint_angles(side="left")
            
            # Smooth the motion after IK solving
            self.joint_angles = self.smooth_motion(self.joint_angles)
            
            # Enforce body yaw to prevent spinning (only if the joint exists in limits)
            if "waist_yaw_joint" in self.joint_limits and "waist_yaw_joint" in self.joint_angles:
                self.joint_angles["waist_yaw_joint"] = 0.0
            
            # Update visualization positions for display
            self.update_visualization_positions()
            
            # Record frame if we're recording
            if self.recording:
                self.record_frame()
            
            success = right_success and left_success
                
        return success
