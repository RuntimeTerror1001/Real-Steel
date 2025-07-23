import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import csv
from datetime import datetime
import pandas as pd
import os
import sys
import matplotlib.patches as mpatches
try:
    from .brpso_ik_solver import BRPSO_IK_Solver  # Try relative import first
except ImportError:
    from brpso_ik_solver import BRPSO_IK_Solver  # Fallback to direct import

# Create a simplified Robot Retargeter for visualization
class VisualRobotRetargeter:
    def __init__(self, ik_solver_backend='analytical'):
        print('[VisualRobotRetargeter] -> __init__()')
        self.mp_pose = mp.solutions.pose
        
        # Define joint limits based on actual robot hardware constraints
        self.joint_limits = {
            "right_shoulder_pitch_joint": (-0.9, 1.57),  # Forward/backward motion
            "right_shoulder_roll_joint": (-0.6, 1.0),    # Lateral abduction/adduction
            "right_shoulder_yaw_joint": (-1.57, 1.57),   # Rotation at shoulder
            "right_elbow_joint": (0.2, 2.0),             # Elbow bending - can't fully straighten or bend beyond ~115°
            "right_wrist_pitch_joint": (-0.8, 0.8),      # Wrist up/down
            "right_wrist_roll_joint": (-0.6, 0.6),       # Wrist rotation
            "right_wrist_yaw_joint": (-0.8, 0.8),        # Wrist side to side
            "left_shoulder_pitch_joint": (-0.9, 1.57),
            "left_shoulder_roll_joint": (-1.0, 0.6),     # Mirrored for left side
            "left_shoulder_yaw_joint": (-1.57, 1.57),
            "left_elbow_joint": (0.2, 2.0),
            "left_wrist_pitch_joint": (-0.8, 0.8),
            "left_wrist_roll_joint": (-0.6, 0.6),
            "left_wrist_yaw_joint": (-0.8, 0.8),
        }
        
        # Initial joint angles based on natural rest pose
        self.joint_angles = {
            "right_shoulder_pitch_joint": 0.1,   # Very slight forward
            "right_shoulder_roll_joint": 0.1,    # Slightly away from body
            "right_shoulder_yaw_joint": 0.0,     # Neutral rotation
            "right_elbow_joint": 0.6,            # Slightly bent
            "right_wrist_pitch_joint": 0.0,      # Neutral
            "right_wrist_roll_joint": 0.0,       # Neutral
            "right_wrist_yaw_joint": 0.0,        # Neutral
            "left_shoulder_pitch_joint": 0.1,
            "left_shoulder_roll_joint": -0.1,    # Mirrored
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.6,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
        }
        
        # Robot dimensions - scaled to match human proportions
        self.dimensions = {
            "shoulder_width": 0.36,        # Distance between shoulders 
            "upper_arm_length": 0.28,      # Shoulder to elbow
            "lower_arm_length": 0.26,      # Elbow to wrist
            "head_height": 0.15,           # Height of head from shoulder line
            "torso_length": 0.40,          # Length of torso from shoulder to hip
        }
        
        # Initialize robot joint positions
        shoulder_half_width = self.dimensions["shoulder_width"] / 2
        self.robot_joints = {
            "right_shoulder": np.array([shoulder_half_width, 0.0, 0.0]),
            "right_elbow": np.array([shoulder_half_width, 0.0, 0.0]),  # Will be calculated
            "right_wrist": np.array([shoulder_half_width, 0.0, 0.0]),  # Will be calculated
            "left_shoulder": np.array([-shoulder_half_width, 0.0, 0.0]),
            "left_elbow": np.array([-shoulder_half_width, 0.0, 0.0]),  # Will be calculated
            "left_wrist": np.array([-shoulder_half_width, 0.0, 0.0]),  # Will be calculated
            "head": np.array([0.0, self.dimensions["head_height"], 0.0]),
            "torso_bottom": np.array([0.0, -self.dimensions["torso_length"], 0.0]),
        }
        
        # Reference to visualization
        self.ax_robot = None
        self.fig_robot = None
        self.timer = 0
        self.paused = False
        
        # Target poses for animation
        # self.pose_sequence = self.generate_boxing_sequence()
        # self.current_pose_index = 0
        # self.transition_time = 0
        # self.pose_duration = 0.8  # Time to hold each pose
        # self.transition_duration = 0.4  # Time to transition between poses
        
        self.ik_solver_backend = ik_solver_backend
        if ik_solver_backend == 'brpso':
            self.brpso_solver = BRPSO_IK_Solver(
                upper_arm_length=self.dimensions["upper_arm_length"],
                lower_arm_length=self.dimensions["lower_arm_length"]
            )
        else:
            self.brpso_solver = None
        
        # Motion status tracking
        self.current_motion_status = "Demo Motion"
        self.last_pose_detected = False
        
        self.recording = False
        self.recording_file = None
    
    def update_robot_state(self, results):
        """Update robot state using real pose data and retargeted joint angles."""
        # Only update when not paused
        if not self.paused:
            # Check if we have pose data
            if results and results.pose_world_landmarks:
                # Extract human pose landmarks
                landmarks = results.pose_world_landmarks.landmark
                
                # Get key joint positions (shoulder, elbow, wrist for both arms)
                try:
                    # Right arm landmarks (MediaPipe indices)
                    right_shoulder = landmarks[12]  # RIGHT_SHOULDER
                    right_elbow = landmarks[14]     # RIGHT_ELBOW
                    right_wrist = landmarks[16]     # RIGHT_WRIST
                    
                    # Left arm landmarks
                    left_shoulder = landmarks[11]   # LEFT_SHOULDER
                    left_elbow = landmarks[13]      # LEFT_ELBOW
                    left_wrist = landmarks[15]      # LEFT_WRIST
                    
                    # Convert MediaPipe coordinates to robot coordinates
                    # MediaPipe: X=left-right, Y=up-down, Z=forward-back
                    # Robot: X=right-left, Y=forward-back, Z=up-down
                    scale = 0.5  # Scale factor
                    
                    # Update robot joint positions with coordinate transformation
                    self.robot_joints["right_shoulder"] = np.array([-right_shoulder.x, -right_shoulder.z, -right_shoulder.y]) * scale
                    self.robot_joints["right_elbow"] = np.array([-right_elbow.x, -right_elbow.z, -right_elbow.y]) * scale
                    self.robot_joints["right_wrist"] = np.array([-right_wrist.x, -right_wrist.z, -right_wrist.y]) * scale
                    
                    self.robot_joints["left_shoulder"] = np.array([-left_shoulder.x, -left_shoulder.z, -left_shoulder.y]) * scale
                    self.robot_joints["left_elbow"] = np.array([-left_elbow.x, -left_elbow.z, -left_elbow.y]) * scale
                    self.robot_joints["left_wrist"] = np.array([-left_wrist.x, -left_wrist.z, -left_wrist.y]) * scale
                    
                    # Calculate joint angles using inverse kinematics
                    self._retarget_to_robot_joints()
                    
                    # Update motion status
                    self.current_motion_status = "Live Human Pose"
                    self.last_pose_detected = True
                    
                except (IndexError, AttributeError) as e:
                    print(f"[WARNING] Could not extract pose landmarks: {e}")
                    # Fallback to demo motion
                    self._animate_demo_motion()
                    self.current_motion_status = "Demo Motion (Pose Lost)"
                    self.last_pose_detected = False
            else:
                # No pose detected, use demo motion
                self._animate_demo_motion()
                self.current_motion_status = "Demo Motion"
                self.last_pose_detected = False
        
        # Update forward kinematics to calculate joint positions
        self.calculate_forward_kinematics()
        return True
        
    def _retarget_to_robot_joints(self):
        """Convert human pose to robot joint angles using inverse kinematics."""
        # Calculate joint angles for both arms
        self._calculate_arm_joint_angles("right")
        self._calculate_arm_joint_angles("left")
        
    def _calculate_arm_joint_angles(self, side):
        """Calculate joint angles for one arm using simplified IK."""
        try:
            # Get joint positions
            shoulder = self.robot_joints[f"{side}_shoulder"]
            elbow = self.robot_joints[f"{side}_elbow"]  
            wrist = self.robot_joints[f"{side}_wrist"]
            
            # Calculate vectors
            upper_arm_vec = elbow - shoulder
            forearm_vec = wrist - elbow
            
            # Calculate lengths
            upper_arm_length = np.linalg.norm(upper_arm_vec)
            forearm_length = np.linalg.norm(forearm_vec)
            
            # Avoid division by zero
            if upper_arm_length < 0.01 or forearm_length < 0.01:
                return
                
            # Normalize vectors
            upper_arm_norm = upper_arm_vec / upper_arm_length
            forearm_norm = forearm_vec / forearm_length
            
            # Calculate elbow angle (angle between upper arm and forearm)
            dot_product = np.clip(np.dot(upper_arm_norm, forearm_norm), -1.0, 1.0)
            elbow_angle = np.pi - np.arccos(dot_product)  # External angle
            
            # Calculate shoulder angles
            # Shoulder pitch (up/down motion)
            shoulder_pitch = np.arctan2(upper_arm_vec[2], np.sqrt(upper_arm_vec[0]**2 + upper_arm_vec[1]**2))
            
            # Shoulder yaw (left/right rotation)  
            shoulder_yaw = np.arctan2(upper_arm_vec[1], upper_arm_vec[0])
            
            # Shoulder roll (arm rotation) - simplified calculation
            shoulder_roll = 0.0  # Can be enhanced later
            
            # Wrist angles - simplified (can be enhanced with actual orientation)
            wrist_pitch = 0.0
            wrist_yaw = 0.0
            wrist_roll = 0.0
            
            # Update joint angles with clipping
            prefix = side + "_"
            self.joint_angles[prefix + "shoulder_pitch_joint"] = self._clip_angle(prefix + "shoulder_pitch_joint", shoulder_pitch)
            self.joint_angles[prefix + "shoulder_yaw_joint"] = self._clip_angle(prefix + "shoulder_yaw_joint", shoulder_yaw)
            self.joint_angles[prefix + "shoulder_roll_joint"] = self._clip_angle(prefix + "shoulder_roll_joint", shoulder_roll)
            self.joint_angles[prefix + "elbow_joint"] = self._clip_angle(prefix + "elbow_joint", elbow_angle)
            self.joint_angles[prefix + "wrist_pitch_joint"] = self._clip_angle(prefix + "wrist_pitch_joint", wrist_pitch)
            self.joint_angles[prefix + "wrist_yaw_joint"] = self._clip_angle(prefix + "wrist_yaw_joint", wrist_yaw)
            self.joint_angles[prefix + "wrist_roll_joint"] = self._clip_angle(prefix + "wrist_roll_joint", wrist_roll)
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate {side} arm joint angles: {e}")
    
    def _clip_angle(self, joint_name, angle):
        """Clip angle to joint limits."""
        if joint_name in self.joint_limits:
            min_angle, max_angle = self.joint_limits[joint_name]
            return np.clip(angle, min_angle, max_angle)
        return angle
        
    def _animate_demo_motion(self):
        """Animate demo boxing motion when no pose is detected."""
        self.timer += 0.033  # ~30 FPS
        
        # Simple boxing motion animation
        t = self.timer * 0.5  # Slow down animation
        
        # Right arm punching motion
        right_punch = 0.5 * (1 + np.sin(t))
        self.joint_angles["right_shoulder_pitch_joint"] = 0.3 + 0.8 * right_punch
        self.joint_angles["right_elbow_joint"] = 1.5 - 1.0 * right_punch
        
        # Left arm guard motion  
        left_guard = 0.5 * (1 + np.sin(t + np.pi))
        self.joint_angles["left_shoulder_pitch_joint"] = 0.5 + 0.3 * left_guard
        self.joint_angles["left_elbow_joint"] = 1.2 + 0.3 * left_guard
    
    def ease_in_out(self, t):
        print('[VisualRobotRetargeter] -> ease_in_out()')
        """Apply cubic easing function for smoother motion"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def calculate_forward_kinematics(self):
        """Calculate joint positions using forward kinematics based on joint angles"""
        # Calculate right arm positions
        self.calculate_arm_fk("right")
        # Calculate left arm positions
        self.calculate_arm_fk("left")
        
        # Run validation if verification is enabled
        if hasattr(self, 'validation_enabled') and self.validation_enabled:
            self.validate_ik_fk_consistency()
    
    def validate_ik_fk_consistency(self):
        print('[VisualRobotRetargeter] -> validate_ik_fk_consistency()')
        """Validate consistency between FK and IK by performing a round trip calculation"""
        # Store original joint angles
        original_angles = self.joint_angles.copy()
        original_positions = {
            "right_elbow": self.robot_joints["right_elbow"].copy(),
            "right_wrist": self.robot_joints["right_wrist"].copy(),
            "left_elbow": self.robot_joints["left_elbow"].copy(),
            "left_wrist": self.robot_joints["left_wrist"].copy()
        }
        
        # Calculate IK from current FK positions
        ik_angles = self.calculate_inverse_kinematics()
        
        # Now use IK angles to calculate FK positions
        # First, save current angles and set to IK angles
        temp_angles = self.joint_angles.copy()
        self.joint_angles = ik_angles
        
        # Calculate FK with IK angles
        fk_positions = {}
        self.calculate_forward_kinematics()
        
        # Store resulting positions
        fk_positions = {
            "right_elbow": self.robot_joints["right_elbow"].copy(),
            "right_wrist": self.robot_joints["right_wrist"].copy(),
            "left_elbow": self.robot_joints["left_elbow"].copy(),
            "left_wrist": self.robot_joints["left_wrist"].copy()
        }
        
        # Restore original joint angles and positions
        self.joint_angles = original_angles
        self.robot_joints.update({
            "right_elbow": original_positions["right_elbow"],
            "right_wrist": original_positions["right_wrist"],
            "left_elbow": original_positions["left_elbow"],
            "left_wrist": original_positions["left_wrist"]
        })
        
        # Calculate errors
        position_error = 0
        angle_error = 0
        
        # Position errors (euclidean distance)
        for joint in original_positions:
            err = np.linalg.norm(original_positions[joint] - fk_positions[joint])
            position_error += err
        position_error /= len(original_positions)  # Average error
        
        # Angle errors
        for joint in original_angles:
            if joint in ik_angles:
                err = abs(original_angles[joint] - ik_angles[joint])
                angle_error += err
        angle_error /= len(original_angles)  # Average error
        
        # Store validation results
        self.validation_results = {
            "position_error": position_error,
            "angle_error": angle_error,
            "timestamp": time.time(),
            "details": {
                "original_angles": original_angles,
                "ik_angles": ik_angles,
                "original_positions": original_positions,
                "fk_positions": fk_positions
            }
        }
        
        # Get current motion name for logging
        # current_time = self.timer % (self.pose_duration + self.transition_duration) * len(self.pose_sequence)
        # pose_index = int(current_time / (self.pose_duration + self.transition_duration))
        # pose_names = ["Guard", "Left Jab", "Guard", "Right Cross", "Guard", "Left Hook", "Guard", "Right Uppercut", "Guard"]
        # current_motion = pose_names[pose_index % len(pose_names)]
        
        # Log the validation result
        self._log_validation_result(self.validation_results, "Unknown")
        
        # Print validation info if significant error
        if position_error > 0.01 or angle_error > 0.01:
            print(f"[VALIDATION] Position error: {position_error:.4f}m, Angle error: {angle_error:.4f}rad in Unknown")
        
        return self.validation_results
    
    def calculate_inverse_kinematics(self):
        print('[VisualRobotRetargeter] -> calculate_inverse_kinematics()')
        """Calculate joint angles using inverse kinematics from current end effector positions"""
        # Initialize angles dict
        ik_angles = {}
        
        # Calculate IK for each arm
        right_angles = self.calculate_arm_ik("right")
        left_angles = self.calculate_arm_ik("left")
        
        # Combine results
        ik_angles.update(right_angles)
        ik_angles.update(left_angles)
        
        return ik_angles
    
    def calculate_arm_ik(self, side):
        print('[VisualRobotRetargeter] -> calculate_arm_ik()')
        """Calculate inverse kinematics for one arm using selected approach"""
        prefix = side + "_"
        angles = {}
        # Get joint positions
        shoulder = self.robot_joints[side + "_shoulder"]
        elbow = self.robot_joints[side + "_elbow"]
        wrist = self.robot_joints[side + "_wrist"]
        L1 = self.dimensions["upper_arm_length"]
        L2 = self.dimensions["lower_arm_length"]
        if self.ik_solver_backend == 'brpso' and self.brpso_solver is not None:
            # Use BRPSO solver
            target = wrist - shoulder
            solution = self.brpso_solver.solve(target_position=target)
            # Map solution to our joint naming
            brpso_angles = solution['joint_angles']
            # Map BRPSO joint order to our joint names
            angles[prefix + "shoulder_yaw_joint"] = brpso_angles['shoulder_yaw']
            angles[prefix + "shoulder_pitch_joint"] = brpso_angles['shoulder_pitch']
            angles[prefix + "shoulder_roll_joint"] = brpso_angles['shoulder_roll']
            angles[prefix + "elbow_joint"] = brpso_angles['elbow']
            angles[prefix + "wrist_pitch_joint"] = brpso_angles['wrist_pitch']
            angles[prefix + "wrist_yaw_joint"] = brpso_angles['wrist_yaw']
            angles[prefix + "wrist_roll_joint"] = brpso_angles['wrist_roll']
            return angles
        # ... existing analytical IK code ...
        # (leave the rest of the function unchanged for analytical)
        # Link lengths from DH parameters (scaled to our model dimensions)
        L1 = self.dimensions["upper_arm_length"]
        L2 = self.dimensions["lower_arm_length"]
        
        # Convert positions to local coordinate system
        local_elbow = elbow - shoulder
        local_wrist = wrist - shoulder
        
        # Calculate vectors and lengths
        upper_arm_vec = local_elbow
        forearm_vec = wrist - elbow
        
        upper_arm_length = np.linalg.norm(upper_arm_vec)
        forearm_length = np.linalg.norm(forearm_vec)
        
        # If vectors are too small, use default values
        if upper_arm_length < 0.01:
            upper_arm_vec = np.array([0, -1, 0]) * L1
            upper_arm_length = L1
            
        if forearm_length < 0.01:
            forearm_vec = np.array([0, -1, 0]) * L2
            forearm_length = L2
        
        # Normalize vectors
        upper_arm_vec_norm = upper_arm_vec / upper_arm_length
        forearm_vec_norm = forearm_vec / forearm_length
        
        # Calculate wrist position relative to shoulder in DH coordinate frame
        wrist_local = local_wrist
        wrist_distance = np.linalg.norm(wrist_local)
        
        # ---- Analytical IK solution using DH chain ----
        # We're solving for: θ_sy, θ_sp, θ_sr, θ_el, θ_wp, θ_wy, θ_wr
        
        # 1. First, calculate elbow angle from cosine law
        # The angle between upper arm and forearm
        cos_elbow = np.clip((upper_arm_length**2 + forearm_length**2 - wrist_distance**2) / 
                           (2 * upper_arm_length * forearm_length), -1.0, 1.0)
        elbow_angle = np.arccos(cos_elbow)
        
        # Ensure the elbow only bends in one direction (usually positive for humans)
        elbow_angle = abs(elbow_angle)
        
        # 2. Calculate the shoulder orientation to reach the wrist position
        # We need to find the shoulder orientation that places the arm in the right direction
        
        # Project the wrist position onto different planes to find shoulder angles
        # XY plane projection for shoulder yaw
        xy_proj = np.array([wrist_local[0], wrist_local[1], 0])
        xy_proj_len = np.linalg.norm(xy_proj)
        
        if xy_proj_len > 0.01:
            # Shoulder yaw is the angle in the XY plane
            shoulder_yaw = np.arctan2(wrist_local[1], wrist_local[0])
        else:
            shoulder_yaw = 0.0
            
        # After yaw rotation, calculate pitch in the rotated plane
        # Create rotation matrix for yaw
        yaw_mat = np.array([
            [np.cos(shoulder_yaw), -np.sin(shoulder_yaw), 0],
            [np.sin(shoulder_yaw), np.cos(shoulder_yaw), 0],
            [0, 0, 1]
        ])
        
        # Rotate wrist position by negative yaw to get it in the YZ plane
        rotated_wrist = np.dot(yaw_mat.T, wrist_local)
        
        # Calculate shoulder pitch in the rotated YZ plane
        yz_proj = np.array([0, rotated_wrist[1], rotated_wrist[2]])
        yz_proj_len = np.linalg.norm(yz_proj)
        
        if yz_proj_len > 0.01:
            # Shoulder pitch is the angle in the YZ plane
            shoulder_pitch = np.arctan2(-rotated_wrist[2], rotated_wrist[1])
        else:
            shoulder_pitch = 0.0
            
        # After pitch rotation, calculate roll
        # Create rotation matrix for pitch
        pitch_mat = np.array([
            [1, 0, 0],
            [0, np.cos(shoulder_pitch), -np.sin(shoulder_pitch)],
            [0, np.sin(shoulder_pitch), np.cos(shoulder_pitch)]
        ])
        
        # Apply both yaw and pitch rotations
        yaw_pitch_rotated = np.dot(pitch_mat.T, np.dot(yaw_mat.T, wrist_local))
        
        # The remaining rotation is the shoulder roll
        # In a perfect DH chain, this would align with x axis
        # We can use the upper arm and rotated elbow position to determine this
        
        # First, calculate where the elbow would be with just yaw and pitch
        expected_elbow_dir = np.array([0, L1, 0])  # Default direction along y after yaw & pitch
        expected_elbow = np.dot(yaw_mat, np.dot(pitch_mat, expected_elbow_dir))
        
        # Compare this with the actual elbow position to find roll
        if np.linalg.norm(local_elbow) > 0.01 and np.linalg.norm(expected_elbow) > 0.01:
            # Project both onto a plane perpendicular to the arm direction
            # This is the plane defined by the cross product of the two directions
            cross_vec = np.cross(expected_elbow, local_elbow)
            cross_vec_len = np.linalg.norm(cross_vec)
            
            if cross_vec_len > 0.01:
                cross_vec = cross_vec / cross_vec_len
                
                # Calculate the angle between the expected and actual elbow positions
                dot_product = np.dot(expected_elbow, local_elbow) / (np.linalg.norm(expected_elbow) * np.linalg.norm(local_elbow))
                shoulder_roll = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # Determine the sign using the cross product
                if np.dot(np.cross(expected_elbow, local_elbow), np.array([1, 0, 0])) < 0:
                    shoulder_roll = -shoulder_roll
            else:
                shoulder_roll = 0.0
        else:
            shoulder_roll = 0.0
            
        # 3. For wrist orientation, we would typically use the end effector orientation
        # Since we're not tracking full orientation, we'll use default values
        wrist_pitch = 0.0
        wrist_yaw = 0.0
        wrist_roll = 0.0
        
        # 4. Apply side-specific adjustments
        if side == "left":
            shoulder_yaw = -shoulder_yaw
            shoulder_roll = -shoulder_roll
            
        # 5. Apply joint limits
        if prefix + "shoulder_pitch_joint" in self.joint_limits:
            min_val, max_val = self.joint_limits[prefix + "shoulder_pitch_joint"]
            shoulder_pitch = max(min_val, min(max_val, shoulder_pitch))
            
        if prefix + "shoulder_roll_joint" in self.joint_limits:
            min_val, max_val = self.joint_limits[prefix + "shoulder_roll_joint"]
            shoulder_roll = max(min_val, min(max_val, shoulder_roll))
            
        if prefix + "shoulder_yaw_joint" in self.joint_limits:
            min_val, max_val = self.joint_limits[prefix + "shoulder_yaw_joint"]
            shoulder_yaw = max(min_val, min(max_val, shoulder_yaw))
            
        if prefix + "elbow_joint" in self.joint_limits:
            min_val, max_val = self.joint_limits[prefix + "elbow_joint"]
            elbow_angle = max(min_val, min(max_val, elbow_angle))
        
        # 6. Store the calculated angles
        angles[prefix + "shoulder_yaw_joint"] = shoulder_yaw
        angles[prefix + "shoulder_pitch_joint"] = shoulder_pitch
        angles[prefix + "shoulder_roll_joint"] = shoulder_roll
        angles[prefix + "elbow_joint"] = elbow_angle
        angles[prefix + "wrist_pitch_joint"] = wrist_pitch
        angles[prefix + "wrist_yaw_joint"] = wrist_yaw
        angles[prefix + "wrist_roll_joint"] = wrist_roll
        
        return angles
    
    def enable_validation(self, enabled=True):
        print('[VisualRobotRetargeter] -> enable_validation()')
        """Enable or disable round-trip IK-FK validation"""
        self.validation_enabled = enabled
        if enabled:
            print("Round-trip IK-FK validation enabled")
            # Initialize validation log file
            self._initialize_validation_log()
        else:
            print("Round-trip IK-FK validation disabled")
        return enabled
    
    def _initialize_validation_log(self):
        print('[VisualRobotRetargeter] -> _initialize_validation_log()')
        """Initialize the validation log file"""
        # Create logs directory if it doesn't exist
        log_dir = "validation_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_log_path = os.path.join(log_dir, f"ik_fk_validation_{timestamp}.csv")
        
        # Create and initialize the log file with headers
        with open(self.validation_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Position_Error', 'Angle_Error', 
                             'Current_Motion', 'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 
                             'Right_Shoulder_Yaw', 'Right_Elbow', 'Left_Shoulder_Pitch', 
                             'Left_Shoulder_Roll', 'Left_Shoulder_Yaw', 'Left_Elbow'])
        
        print(f"Validation log initialized at {self.validation_log_path}")
    
    def _log_validation_result(self, validation_result, current_motion="Unknown"):
        print('[VisualRobotRetargeter] -> _log_validation_result()')
        """Log validation result to CSV file"""
        if not hasattr(self, 'validation_log_path') or not os.path.exists(self.validation_log_path):
            self._initialize_validation_log()
        
        # Extract current joint angles
        angles = [
            self.joint_angles.get("right_shoulder_pitch_joint", 0),
            self.joint_angles.get("right_shoulder_roll_joint", 0),
            self.joint_angles.get("right_shoulder_yaw_joint", 0),
            self.joint_angles.get("right_elbow_joint", 0),
            self.joint_angles.get("left_shoulder_pitch_joint", 0),
            self.joint_angles.get("left_shoulder_roll_joint", 0),
            self.joint_angles.get("left_shoulder_yaw_joint", 0),
            self.joint_angles.get("left_elbow_joint", 0)
        ]
        
        # Prepare log data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        position_error = validation_result["position_error"]
        angle_error = validation_result["angle_error"]
        
        # Write to the log file
        with open(self.validation_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, position_error, angle_error, current_motion] + angles)
    
    def get_last_validation_results(self):
        print('[VisualRobotRetargeter] -> get_last_validation_results()')
        """Get the results from the last validation run"""
        if hasattr(self, 'validation_results'):
            return self.validation_results
        return {"position_error": 0, "angle_error": 0, "timestamp": 0}
    
    def generate_validation_report(self, show_plot=True):
        print('[VisualRobotRetargeter] -> generate_validation_report()')
        """Generate a comprehensive validation report from logged data"""
        if not hasattr(self, 'validation_log_path') or not os.path.exists(self.validation_log_path):
            print("No validation log found. Enable validation first with 'V' key.")
            return None
        
        try:
            # Read the log file
            df = pd.read_csv(self.validation_log_path)
            
            # Basic statistics
            stats = {
                "total_samples": len(df),
                "avg_position_error": df['Position_Error'].mean(),
                "max_position_error": df['Position_Error'].max(),
                "min_position_error": df['Position_Error'].min(),
                "std_position_error": df['Position_Error'].std(),
                "avg_angle_error": df['Angle_Error'].mean(),
                "max_angle_error": df['Angle_Error'].max(),
                "min_angle_error": df['Angle_Error'].min(),
                "std_angle_error": df['Angle_Error'].std(),
            }
            
            # Group by motion type
            motion_stats = df.groupby('Current_Motion').agg({
                'Position_Error': ['mean', 'max', 'min', 'std'],
                'Angle_Error': ['mean', 'max', 'min', 'std']
            }).reset_index()
            
            # Print report
            print("\n===== IK-FK VALIDATION REPORT =====")
            print(f"Validation log: {self.validation_log_path}")
            print(f"Total samples: {stats['total_samples']}")
            print("\nOVERALL STATISTICS:")
            print(f"Position Error (m): Avg={stats['avg_position_error']:.4f}, Max={stats['max_position_error']:.4f}, Min={stats['min_position_error']:.4f}, StdDev={stats['std_position_error']:.4f}")
            print(f"Angle Error (rad): Avg={stats['avg_angle_error']:.4f}, Max={stats['max_angle_error']:.4f}, Min={stats['min_angle_error']:.4f}, StdDev={stats['std_angle_error']:.4f}")
            
            print("\nERROR BY MOTION TYPE:")
            for _, row in motion_stats.iterrows():
                motion = row['Current_Motion']
                pos_mean = row[('Position_Error', 'mean')]
                pos_max = row[('Position_Error', 'max')]
                ang_mean = row[('Angle_Error', 'mean')]
                ang_max = row[('Angle_Error', 'max')]
                print(f"{motion}: Pos Err Avg={pos_mean:.4f}m (Max={pos_max:.4f}m), Ang Err Avg={ang_mean:.4f}rad (Max={ang_max:.4f}rad)")
            
            # Visualize the data if requested
            if show_plot:
                self.visualize_validation_data(df)
            
            return {
                "stats": stats,
                "motion_stats": motion_stats,
                "data": df
            }
        
        except Exception as e:
            print(f"Error generating validation report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_validation_data(self, df=None):
        print('[VisualRobotRetargeter] -> visualize_validation_data()')
        """Visualize validation data from the log file"""
        if df is None:
            try:
                df = pd.read_csv(self.validation_log_path)
            except Exception as e:
                print(f"No validation log found or unable to read the log file: {str(e)}")
                return
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Add a time index for x-axis
        df['TimeIndex'] = range(len(df))
        
        # Plot position error over time
        ax1.plot(df['TimeIndex'], df['Position_Error'], 'b-', linewidth=2)
        ax1.set_title('Position Error Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Position Error (m)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold lines
        ax1.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (0.01m)')
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Error Threshold (0.05m)')
        ax1.legend()
        
        # Plot angle error over time
        ax2.plot(df['TimeIndex'], df['Angle_Error'], 'g-', linewidth=2)
        ax2.set_title('Angle Error Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('Angle Error (rad)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add threshold lines
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold (0.05rad)')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Error Threshold (0.1rad)')
        ax2.legend()
        
        # Color regions by motion type
        if 'Current_Motion' in df.columns:
            # Get unique motions
            motions = df['Current_Motion'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(motions)))
            
            # Create a color map
            motion_color_map = {motion: color for motion, color in zip(motions, colors)}
            
            # Mark motion regions
            current_motion = None
            start_idx = 0
            
            for idx, motion in enumerate(df['Current_Motion']):
                if motion != current_motion:
                    if current_motion is not None:
                        # Add colored background for this region
                        ax1.axvspan(start_idx, idx-1, alpha=0.2, color=motion_color_map[current_motion])
                        ax2.axvspan(start_idx, idx-1, alpha=0.2, color=motion_color_map[current_motion])
                    
                    current_motion = motion
                    start_idx = idx
            
            # Add the last motion region
            if current_motion is not None:
                ax1.axvspan(start_idx, len(df)-1, alpha=0.2, color=motion_color_map[current_motion])
                ax2.axvspan(start_idx, len(df)-1, alpha=0.2, color=motion_color_map[current_motion])
            
            # Add legend for motions
            legend_patches = [mpatches.Patch(color=color, alpha=0.5, label=motion) 
                             for motion, color in motion_color_map.items()]
            fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.97),
                      ncol=len(motions), fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle('IK-FK Validation Error Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Save the figure
        report_dir = "validation_reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"validation_report_{timestamp}.png")
        # plt.savefig(report_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        print(f"Visualization saved to {report_path}")
        
        return fig
    
    def calculate_arm_fk(self, side):
        """Calculate forward kinematics for one arm using DH parameters"""
        prefix = side + "_"
        shoulder_pos = self.robot_joints[side + "_shoulder"]
        
        # Get joint angles
        shoulder_yaw = self.joint_angles[prefix + "shoulder_yaw_joint"]
        shoulder_pitch = self.joint_angles[prefix + "shoulder_pitch_joint"]
        shoulder_roll = self.joint_angles[prefix + "shoulder_roll_joint"]
        elbow_angle = self.joint_angles[prefix + "elbow_joint"]
        wrist_pitch = self.joint_angles[prefix + "wrist_pitch_joint"]
        wrist_yaw = self.joint_angles[prefix + "wrist_yaw_joint"]
        wrist_roll = self.joint_angles[prefix + "wrist_roll_joint"]
        
        # Apply sign correction for left arm
        if side == "left":
            shoulder_roll = -shoulder_roll
            shoulder_yaw = -shoulder_yaw
        
        # Link lengths from DH parameters
        L1 = 0.1032  # Upper arm length in meters
        L2 = 0.1000  # Forearm length in meters
        
        # Scale lengths according to our model dimensions
        L1 = self.dimensions["upper_arm_length"]
        L2 = self.dimensions["lower_arm_length"]
        
        # DH chain transformation using homogeneous transformation matrices
        # Using the DH parameters: transform_matrix(theta, d, a, alpha)
        T = self.transform_matrix(shoulder_yaw, 0, 0, np.pi/2)      # Shoulder Yaw
        T = T @ self.transform_matrix(shoulder_pitch, 0, 0, -np.pi/2)  # Shoulder Pitch
        T = T @ self.transform_matrix(shoulder_roll, 0, 0, np.pi/2)    # Shoulder Roll
        T = T @ self.transform_matrix(elbow_angle, 0, L1, 0)           # Elbow Flex
        T = T @ self.transform_matrix(wrist_pitch, 0, L2, np.pi/2)     # Wrist Pitch
        T = T @ self.transform_matrix(wrist_yaw, 0, 0, -np.pi/2)       # Wrist Yaw
        T = T @ self.transform_matrix(wrist_roll, 0, 0, 0)             # Wrist Roll
        
        # Extract position and orientation from transformation matrix
        pos = T[:3, 3]
        orientation = T[:3, :3]
        
        # Transform to world coordinates
        # Adjust based on shoulder position
        world_pos = shoulder_pos + pos
        
        # Calculate elbow position (up to 4th transformation)
        T_elbow = self.transform_matrix(shoulder_yaw, 0, 0, np.pi/2)
        T_elbow = T_elbow @ self.transform_matrix(shoulder_pitch, 0, 0, -np.pi/2)
        T_elbow = T_elbow @ self.transform_matrix(shoulder_roll, 0, 0, np.pi/2)
        T_elbow = T_elbow @ self.transform_matrix(elbow_angle, 0, L1, 0)
        elbow_pos = T_elbow[:3, 3]
        elbow_world_pos = shoulder_pos + elbow_pos
        
        # Update joint positions
        self.robot_joints[side + "_elbow"] = elbow_world_pos
        self.robot_joints[side + "_wrist"] = world_pos
        
        return True
    
    def transform_matrix(self, theta, d, a, alpha):
        """
        Create a DH transformation matrix from DH parameters
        
        Parameters:
        - theta: joint angle (rotation around z)
        - d: link offset (translation along z)
        - a: link length (translation along x)
        - alpha: link twist (rotation around x)
        
        Returns:
        - 4x4 homogeneous transformation matrix
        """
        c_t = np.cos(theta)
        s_t = np.sin(theta)
        c_a = np.cos(alpha)
        s_a = np.sin(alpha)
        
        return np.array([
            [c_t, -s_t*c_a, s_t*s_a, a*c_t],
            [s_t, c_t*c_a, -c_t*s_a, a*s_t],
            [0, s_a, c_a, d],
            [0, 0, 0, 1]
        ])
    
    def rotation_matrix_x(self, angle):
        print('[VisualRobotRetargeter] -> rotation_matrix_x()')
        """Create rotation matrix around X axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def rotation_matrix_y(self, angle):
        print('[VisualRobotRetargeter] -> rotation_matrix_y()')
        """Create rotation matrix around Y axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def rotation_matrix_z(self, angle):
        print('[VisualRobotRetargeter] -> rotation_matrix_z()')
        """Create rotation matrix around Z axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def axis_angle_rotation(self, axis, angle):
        print('[VisualRobotRetargeter] -> axis_angle_rotation()')
        """Create rotation matrix around arbitrary axis"""
        # Normalize axis
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        
        return np.array([
            [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
            [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
            [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c]
        ])
    
    def toggle_pause(self):
        print('[VisualRobotRetargeter] -> toggle_pause()')
        """Toggle the pause state for the animation"""
        self.paused = not self.paused
        return self.paused
    
    def update_robot_plot(self, ax=None):
        """Update robot plot for visualization"""
        if ax is None:
            ax = self.ax_robot
        
        self.ax_robot = ax
        ax.clear()
        
        # Draw robot skeleton with thicker, more visible lines
        # Right arm
        ax.plot([self.robot_joints["right_shoulder"][0], self.robot_joints["right_elbow"][0]],
                [self.robot_joints["right_shoulder"][1], self.robot_joints["right_elbow"][1]],
                [self.robot_joints["right_shoulder"][2], self.robot_joints["right_elbow"][2]], 'r-', linewidth=4)
        
        ax.plot([self.robot_joints["right_elbow"][0], self.robot_joints["right_wrist"][0]],
                [self.robot_joints["right_elbow"][1], self.robot_joints["right_wrist"][1]],
                [self.robot_joints["right_elbow"][2], self.robot_joints["right_wrist"][2]], 'r-', linewidth=4)
        
        # Left arm
        ax.plot([self.robot_joints["left_shoulder"][0], self.robot_joints["left_elbow"][0]],
                [self.robot_joints["left_shoulder"][1], self.robot_joints["left_elbow"][1]],
                [self.robot_joints["left_shoulder"][2], self.robot_joints["left_elbow"][2]], 'b-', linewidth=4)
        
        ax.plot([self.robot_joints["left_elbow"][0], self.robot_joints["left_wrist"][0]],
                [self.robot_joints["left_elbow"][1], self.robot_joints["left_wrist"][1]],
                [self.robot_joints["left_elbow"][2], self.robot_joints["left_wrist"][2]], 'b-', linewidth=4)
        
        # Draw joints as larger spheres
        ax.scatter(self.robot_joints["right_shoulder"][0], self.robot_joints["right_shoulder"][1], 
                  self.robot_joints["right_shoulder"][2], color='r', s=150)
        ax.scatter(self.robot_joints["right_elbow"][0], self.robot_joints["right_elbow"][1], 
                  self.robot_joints["right_elbow"][2], color='r', s=120)
        ax.scatter(self.robot_joints["right_wrist"][0], self.robot_joints["right_wrist"][1], 
                  self.robot_joints["right_wrist"][2], color='r', s=100)
        
        ax.scatter(self.robot_joints["left_shoulder"][0], self.robot_joints["left_shoulder"][1], 
                  self.robot_joints["left_shoulder"][2], color='b', s=150)
        ax.scatter(self.robot_joints["left_elbow"][0], self.robot_joints["left_elbow"][1], 
                  self.robot_joints["left_elbow"][2], color='b', s=120)
        ax.scatter(self.robot_joints["left_wrist"][0], self.robot_joints["left_wrist"][1], 
                  self.robot_joints["left_wrist"][2], color='b', s=100)
        
        # Draw body - wider shoulders
        ax.plot([self.robot_joints["right_shoulder"][0], self.robot_joints["left_shoulder"][0]],
                [self.robot_joints["right_shoulder"][1], self.robot_joints["left_shoulder"][1]],
                [self.robot_joints["right_shoulder"][2], self.robot_joints["left_shoulder"][2]], 'g-', linewidth=5)
        
        # Add a head representation
        ax.scatter(self.robot_joints["head"][0], self.robot_joints["head"][1], 
                  self.robot_joints["head"][2], color='g', s=200)
        
        # Add neck
        ax.plot([0, self.robot_joints["head"][0]], 
                [0, self.robot_joints["head"][1]], 
                [0, self.robot_joints["head"][2]], 'g-', linewidth=4)
        
        # Add simple torso
        ax.plot([0, self.robot_joints["torso_bottom"][0]], 
                [0, self.robot_joints["torso_bottom"][1]], 
                [0, self.robot_joints["torso_bottom"][2]], 'g-', linewidth=5)
        
        # Add boxing gloves at the wrists
        # Right glove
        r_punch_extension = np.linalg.norm(self.robot_joints["right_wrist"] - self.robot_joints["right_elbow"]) / self.dimensions["lower_arm_length"]
        right_glove_size = 150 if r_punch_extension > 0.9 else 120
        
        # Left glove
        l_punch_extension = np.linalg.norm(self.robot_joints["left_wrist"] - self.robot_joints["left_elbow"]) / self.dimensions["lower_arm_length"]
        left_glove_size = 150 if l_punch_extension > 0.9 else 120
        
        # Draw gloves
        ax.scatter(self.robot_joints["right_wrist"][0], self.robot_joints["right_wrist"][1], 
                  self.robot_joints["right_wrist"][2], color='#990000', s=right_glove_size)
        ax.scatter(self.robot_joints["left_wrist"][0], self.robot_joints["left_wrist"][1], 
                  self.robot_joints["left_wrist"][2], color='#000099', s=left_glove_size)
        
        # Add labels showing the current motion
        # current_time = self.timer % (self.pose_duration + self.transition_duration) * len(self.pose_sequence)
        # pose_index = int(current_time / (self.pose_duration + self.transition_duration))
        # pose_names = ["Guard", "Left Jab", "Guard", "Right Cross", "Guard", "Left Hook", "Guard", "Right Uppercut", "Guard"]
        # current_motion = pose_names[pose_index % len(pose_names)]
        
        # Add label showing current motion
        if not self.paused:
            ax.text(0, 0.6, 0, f"Unknown", fontsize=14, color='black', 
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax.text(0, 0.6, 0, f"Unknown (PAUSED)", fontsize=14, color='red', 
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        # Display IK-FK validation results if enabled
        if hasattr(self, 'validation_enabled') and self.validation_enabled:
            validation_results = self.get_last_validation_results()
            pos_err = validation_results["position_error"]
            ang_err = validation_results["angle_error"]
            
            # Use color to indicate error level
            error_color = 'green'
            if pos_err > 0.05 or ang_err > 0.1:
                error_color = 'red'
            elif pos_err > 0.01 or ang_err > 0.05:
                error_color = 'orange'
                
            # Add text showing validation errors
            ax.text(0, -0.4, 0, 
                   f"IK-FK Validation:\nPosition Error: {pos_err:.4f}m\nAngle Error: {ang_err:.4f}rad", 
                   fontsize=10, color=error_color, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Add coordinate system helper
        axis_len = 0.15
        # X axis
        ax.plot([-axis_len, axis_len], [0, 0], [0, 0], 'r-', linewidth=1)
        ax.text(axis_len, 0, 0, "X", color='r')
        # Y axis
        ax.plot([0, 0], [-axis_len, axis_len], [0, 0], 'g-', linewidth=1)
        ax.text(0, axis_len, 0, "Y", color='g')
        # Z axis
        ax.plot([0, 0], [0, 0], [-axis_len, axis_len], 'b-', linewidth=1)
        ax.text(0, 0, axis_len, "Z", color='b')
        
        # Set axis properties
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.6)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X - Lateral')
        ax.set_ylabel('Y - Vertical')
        ax.set_zlabel('Z - Forward/Back')
        ax.set_title('Robot Motion Retargeting', fontsize=14, fontweight='bold')
        ax.view_init(elev=20, azim=-60)  # Better viewing angle
        
        # Add grid for better depth perception
        ax.grid(True, linestyle='--', alpha=0.6)

    def start_recording(self, filename=None):
        self.recording = True
        self.recording_file = filename or f"src/core/recordings/robot_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(self.recording_file), exist_ok=True)
        print(f"[Visual] Started recording to {self.recording_file}")

    def stop_recording(self):
        self.recording = False
        print(f"[Visual] Stopped recording: {self.recording_file}")
        self.recording_file = None

    def record_frame(self, timestamp, robot_joint_angles):
        if not self.recording:
            return
        row = [
            round(timestamp, 2),
            round(robot_joint_angles.get('left_shoulder_pitch_joint', 0.0), 2),
            round(robot_joint_angles.get('left_shoulder_yaw_joint', 0.0), 2),
            round(robot_joint_angles.get('left_shoulder_roll_joint', 0.0), 2),
            round(robot_joint_angles.get('left_elbow_joint', 0.0), 2),
            round(robot_joint_angles.get('left_wrist_pitch_joint', 0.0), 2),
            round(robot_joint_angles.get('left_wrist_yaw_joint', 0.0), 2),
            round(robot_joint_angles.get('left_wrist_roll_joint', 0.0), 2),
            round(robot_joint_angles.get('right_shoulder_pitch_joint', 0.0), 2),
            round(robot_joint_angles.get('right_shoulder_yaw_joint', 0.0), 2),
            round(robot_joint_angles.get('right_shoulder_roll_joint', 0.0), 2),
            round(robot_joint_angles.get('right_elbow_joint', 0.0), 2),
            round(robot_joint_angles.get('right_wrist_pitch_joint', 0.0), 2),
            round(robot_joint_angles.get('right_wrist_yaw_joint', 0.0), 2),
            round(robot_joint_angles.get('right_wrist_roll_joint', 0.0), 2),
        ]
        with open(self.recording_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

class PoseMirror3DWithRetargeting:
    def __init__(self, window_size=(1280, 720), ik_solver_backend='analytical', camera_index=0, 
                 debug=False, auto_record=False, record_frequency=30, dual_mode=False, 
                 execution_mode='live', input_file=None, enable_visualizations=True):
        print('[PoseMirror3DWithRetargeting] -> __init__()')
        """Initialize the PoseMirror3D system with robot retargeting."""
        self.window_size = window_size
        self.screen_width, self.screen_height = window_size
        self.camera_index = camera_index
        self.debug = debug
        self.auto_record = auto_record
        self.record_frequency = record_frequency
        self.dual_mode = dual_mode
        self.execution_mode = execution_mode
        self.input_file = input_file
        self.enable_visualizations = enable_visualizations
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
            enable_segmentation=True,
            smooth_landmarks=False
        )
        
        # Initialize visualization
        self._setup_visualization()
        
        # Initialize robot retargeter
        self.robot_retargeter = VisualRobotRetargeter(ik_solver_backend=ik_solver_backend)
        self.robot_retargeter.ax_robot = self.ax_robot
        self.robot_retargeter.fig_robot = self.fig
        
        # Initialize pose tracking variables
        self.pose_history = []
        self.smoothing_window = 5
        self.current_rotation_angle = 0
        self.initial_angle_set = False
        self.angle_offset = 0
        self.smoothing_factor = 0.8
        self.recent_chest_vectors = []
        self.max_history = 5
        self.joint_angles = self.robot_retargeter.joint_angles
        
        # Enhanced recording system with visual indicators
        self.recording_status = {
            'active': False,
            'start_time': None,
            'frame_count': 0,
            'filename': None,
            'show_indicator': False,
            'indicator_timer': 0,
            'blink_state': True
        }
        
        # Legacy recording variables for compatibility
        self.recording = False
        self.paused = False
        self.csv_file = None
        self.csv_writer = None
        self.recording_file = None
        self.start_time = 0
        self.frame_counter = 0
        
        # IK-FK validation flag
        self.validation_enabled = False
        
        # Auto-start recording if requested
        if self.auto_record:
            self.start_recording()

    def _setup_visualization(self):
        print('[PoseMirror3DWithRetargeting] -> _setup_visualization()')
        """Set up all visualization components."""
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption('Motion Retargeting')
        self.screen = pygame.display.set_mode(self.window_size)
        
        # Initialize matplotlib with larger figure size
        plt.ion()
        self.fig = plt.figure(figsize=(22, 8))
        self.fig.canvas.manager.set_window_title('Motion Retargeting Visualization')
        
        # Add a super title
        self.fig.suptitle('Human Motion Retargeting to Humanoid Robot', fontsize=18, fontweight='bold', y=0.98)
        
        # Create subplots for different views with better spacing
        self.ax_camera = self.fig.add_subplot(141)  # Camera feed
        self.ax_3d = self.fig.add_subplot(142, projection='3d')
        self.ax_robot = self.fig.add_subplot(143, projection='3d')
        self.ax_angles = self.fig.add_subplot(144)
        
        # Set up camera view
        self.ax_camera.set_title('Camera Feed', fontsize=14, fontweight='bold')
        self.ax_camera.axis('off')
        
        # Set up 3D view limits with better proportions
        self.ax_3d.set_xlim(-0.6, 0.6)
        self.ax_3d.set_ylim(-0.6, 0.6)
        self.ax_3d.set_zlim(-0.6, 0.6)
        
        # Set up robot view limits
        self.ax_robot.set_xlim(-0.5, 0.5)
        self.ax_robot.set_ylim(-0.5, 0.5)
        self.ax_robot.set_zlim(-0.5, 0.5)
        
        # Set labels
        self.ax_3d.set_title('Human Pose', fontsize=14, fontweight='bold')
        self.ax_robot.set_title('Robot Pose', fontsize=14, fontweight='bold')
        self.ax_angles.set_title('Joint Angles', fontsize=14, fontweight='bold')
        
        # Add better labels for 3D plots
        self.ax_3d.set_xlabel('X', fontsize=12)
        self.ax_3d.set_ylabel('Y', fontsize=12)
        self.ax_3d.set_zlabel('Z', fontsize=12)
        
        self.ax_robot.set_xlabel('X', fontsize=12)
        self.ax_robot.set_ylabel('Y', fontsize=12)
        self.ax_robot.set_zlabel('Z', fontsize=12)
        
        # Adjust subplot spacing for better layout
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.9, wspace=0.2)
        
        # Initialize empty plots
        self.human_plot = None
        self.robot_plot = None
        self.angle_plot = None
        self.camera_plot = None
        
    def calculate_body_plane_angle(self, landmarks):
        print('[PoseMirror3DWithRetargeting] -> calculate_body_plane_angle()')
        """Calculate the angle between the body plane and camera plane."""
        if not landmarks:
            return 0
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Calculate chest midpoint
        chest_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Calculate chest-to-nose vector
        chest_to_nose_x = -(nose.x - chest_mid_x)
        chest_to_nose_z = -(nose.y - chest_mid_y)
        
        # Return fixed angle for visualization
        return 0

    def update_visualization(self, results, image=None):
        """Update all real-time visualizations with latest pose data."""
        # Clear all plots except camera feed
        self.ax_3d.clear()
        self.ax_robot.clear()
        self.ax_angles.clear()
        
        # Update camera feed (Window 1)
        if image is not None:
            if not hasattr(self, 'camera_image'):
                self.camera_image = self.ax_camera.imshow(image)
                self.ax_camera.axis('off')
            else:
                self.camera_image.set_data(image)
            self.ax_camera.set_title('🎥 Live Camera Feed', fontweight='bold')

        if results and results.pose_world_landmarks:
            # Update human pose visualization (Window 2)
            self._update_human_pose(results.pose_world_landmarks)
            
            # Update robot visualization (Window 3)
            self.robot_retargeter.update_robot_plot(self.ax_robot)
            
            # Update joint angles visualization (Window 4)
            self._update_joint_angles()

        # Add status information overlay
        self._add_status_info()

        # Refresh the display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_human_pose(self, world_landmarks):
        """Update the human pose 3D plot using the provided world landmarks from MediaPipe."""
        """Update the human pose visualization."""
        landmarks = world_landmarks.landmark
        
        # Extract coordinates with correct mapping
        x_coords = [-lm.x for lm in landmarks]  # Flip x for right-positive
        y_coords = [-lm.z for lm in landmarks]  # Use -z for up
        z_coords = [-lm.y for lm in landmarks]  # Use -y for forward

        # Plot landmarks
        self.ax_3d.scatter3D(x_coords, z_coords, y_coords, c='b', marker='o')

        # Plot connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            self.ax_3d.plot3D([x_coords[start_idx], x_coords[end_idx]],
                               [z_coords[start_idx], z_coords[end_idx]],
                               [y_coords[start_idx], y_coords[end_idx]], 'b-')

        # Set plot properties
        self.ax_3d.set_xlabel('X (Left-Right)')
        self.ax_3d.set_ylabel('Z (Forward-Back)')
        self.ax_3d.set_zlabel('Y (Up-Down)')
        self.ax_3d.view_init(elev=0, azim=270)
        self.ax_3d.set_box_aspect([1,1,1])

    def _update_joint_angles(self):
        """Update the joint angles bar plot for the robot arms using the current joint angles."""
        """Update the joint angles visualization with realistic data."""
        joint_limits = self.robot_retargeter.joint_limits
        self.joint_angles = self.robot_retargeter.joint_angles
        
        # Create structured joint groups for better visualization
        joint_groups = {
            "Right Arm": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint", 
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ],
            "Left Arm": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ]
        }
        
        # Get current pose name from robot retargeter
        current_time = self.robot_retargeter.timer
        # pose_duration = self.robot_retargeter.pose_duration
        # transition_duration = self.robot_retargeter.transition_duration
        # total_pose_time = pose_duration + transition_duration
        # pose_sequence_length = len(self.robot_retargeter.pose_sequence)
        
        # cycle_time = current_time % (total_pose_time * pose_sequence_length)
        # pose_index = int(cycle_time / total_pose_time)
        # pose_names = ["Guard", "Left Jab", "Guard", "Right Cross", "Guard", "Left Hook", "Guard", "Right Uppercut", "Guard"]
        # current_motion = pose_names[pose_index % len(pose_names)]
        
        # Clear the plot
        self.ax_angles.clear()
        
        # Track positions for labels
        current_y_pos = 0
        max_val = 0
        min_val = 0
        
        # Colors for joint groups
        colors = {
            "Right Arm": "#ff6666",  # Light red
            "Left Arm": "#6666ff"    # Light blue
        }
        
        # Joint descriptions
        joint_descriptions = {
            "right_shoulder_pitch_joint": "Shoulder Forward/Back",
            "right_shoulder_roll_joint": "Shoulder In/Out",
            "right_shoulder_yaw_joint": "Shoulder Rotation",
            "right_elbow_joint": "Elbow Bend",
            "left_shoulder_pitch_joint": "Shoulder Forward/Back",
            "left_shoulder_roll_joint": "Shoulder In/Out",
            "left_shoulder_yaw_joint": "Shoulder Rotation",
            "left_elbow_joint": "Elbow Bend"
        }
        
        # Draw each group
        for group, joints in joint_groups.items():
            # Add group title
            self.ax_angles.text(-90, current_y_pos + 0.5, group, 
                              fontsize=12, fontweight='bold', ha='center',
                              bbox=dict(facecolor=colors[group], alpha=0.3, pad=5))
            current_y_pos += 1.5
            
            # Add joints in this group
            group_y_positions = {}
            group_values = {}
            
            for joint in joints:
                angle_deg = math.degrees(self.joint_angles[joint])
                group_y_positions[joint] = current_y_pos
                group_values[joint] = angle_deg
                current_y_pos += 1
                
                # Track min/max for axis scaling
                max_val = max(max_val, angle_deg)
                min_val = min(min_val, angle_deg)
            
            # Draw bars for this group
            for joint, y_pos in group_y_positions.items():
                value = group_values[joint]
                color = colors[group]
                
                # Check joint limits
                limit_exceeded = False
                if joint in joint_limits:
                    min_limit_deg = math.degrees(joint_limits[joint][0])
                    max_limit_deg = math.degrees(joint_limits[joint][1])
                    
                    if value < min_limit_deg or value > max_limit_deg:
                        color = '#ff0000'  # Red for out of limits
                        limit_exceeded = True
                
                # Draw the bar
                bar = self.ax_angles.barh(y_pos, value, color=color, height=0.7)
                
                # Add angle value at the end of the bar
                text_x = value + 2 * np.sign(value)
                self.ax_angles.text(text_x, y_pos, f"{value:.1f}°", 
                                  va='center', ha='left' if value >= 0 else 'right',
                                  fontsize=9)
                
                # Add limit indicators
                if joint in joint_limits:
                    min_limit_deg = math.degrees(joint_limits[joint][0])
                    max_limit_deg = math.degrees(joint_limits[joint][1])
                    
                    # Draw limit lines
                    self.ax_angles.axvline(x=min_limit_deg, ymin=(y_pos-0.4)/current_y_pos, 
                                        ymax=(y_pos+0.4)/current_y_pos,
                                        color='#ff0000', linestyle='--', alpha=0.7, linewidth=1.5)
                    self.ax_angles.axvline(x=max_limit_deg, ymin=(y_pos-0.4)/current_y_pos, 
                                        ymax=(y_pos+0.4)/current_y_pos,
                                        color='#ff0000', linestyle='--', alpha=0.7, linewidth=1.5)
                    
                    # Add warning symbol if needed
                    if limit_exceeded:
                        if value < min_limit_deg:
                            self.ax_angles.text(min_limit_deg - 5, y_pos, "⚠", 
                                             va='center', ha='right', fontsize=14, color='#ff0000')
                        else:
                            self.ax_angles.text(max_limit_deg + 5, y_pos, "⚠", 
                                             va='center', ha='left', fontsize=14, color='#ff0000')
                
                # Add joint description
                self.ax_angles.text(-95, y_pos, joint_descriptions.get(joint, joint), 
                                  va='center', ha='right', fontsize=9)
            
            # Add separator after group
            current_y_pos += 0.5
            self.ax_angles.axhline(y=current_y_pos, color='gray', linestyle='-', alpha=0.3)
            current_y_pos += 0.5
        
        # Set plot properties
        margin = 10  # Add margin to make sure bars and labels are fully visible
        max_abs_val = max(abs(min_val), abs(max_val))
        limit_val = max(max_abs_val * 1.2, 100)  # At least show ±100 degrees
        
        self.ax_angles.set_xlim(-limit_val - margin, limit_val + margin)
        self.ax_angles.set_yticks([])  # Hide y-ticks as we have custom labels
        self.ax_angles.set_ylim(0, current_y_pos)
        
        # Add a vertical line at 0 for reference
        self.ax_angles.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add legend for warning symbol
        self.ax_angles.text(0, current_y_pos + 0.5, "⚠ = Joint angle exceeds mechanical limits", 
                         ha='center', fontsize=10, color='#ff0000')
        
        # Add grid for horizontal lines only
        self.ax_angles.grid(axis='x', linestyle=':', alpha=0.4)
        
        # Add title with current motion status
        if hasattr(self.robot_retargeter, 'current_motion_status'):
            motion_status = self.robot_retargeter.current_motion_status
        else:
            motion_status = "Real-time Retargeting"
            
        title = f'Joint Angles - {motion_status}'
        if self.robot_retargeter.paused:
            title += " (PAUSED)"
        self.ax_angles.set_title(title, fontsize=14, fontweight='bold')
        
        # Add x-axis label
        self.ax_angles.set_xlabel('Angle (degrees)', fontsize=11)
    
    def _add_status_info(self):
        """Add status information overlay to the visualization"""
        # Create status text with current system info
        status_text = f"IK Solver: {self.robot_retargeter.ik_solver_backend.upper()}"
        
        if hasattr(self, 'dual_mode') and self.dual_mode:
            status_text += " | 🔄 DUAL MODE"
        
        # Recording status with visual indicators
        bg_color = 'yellow'
        text_color = 'black'
        
        if hasattr(self, 'recording_status') and self.recording_status:
            current_time = time.time()
            if self.recording_status['active']:
                # Blinking recording indicator
                if current_time - self.recording_status.get('indicator_timer', 0) > 0.5:
                    self.recording_status['blink_state'] = not self.recording_status['blink_state']
                    self.recording_status['indicator_timer'] = current_time
                
                if self.recording_status['blink_state']:
                    record_indicator = "🔴 RECORDING"
                    bg_color = 'red'
                    text_color = 'white'
                else:
                    record_indicator = "⭕ RECORDING"
                    bg_color = 'darkred'
                    text_color = 'lightgray'
                
                duration = current_time - self.recording_status.get('start_time', current_time)
                frame_count = self.recording_status.get('frame_count', 0)
                status_text += f" | {record_indicator} ({frame_count} frames, {duration:.1f}s)"
                
            elif self.recording_status.get('show_indicator', False):
                # Show "stopped recording" briefly
                if current_time - self.recording_status.get('indicator_timer', 0) < 2.0:
                    status_text += " | ⏹️ STOPPED RECORDING"
                    bg_color = 'gray'
                    text_color = 'white'
                else:
                    self.recording_status['show_indicator'] = False
        
        # Add pause status
        if hasattr(self, 'paused') and self.paused:
            status_text += " | ⏸️ PAUSED"
            bg_color = 'orange'
            text_color = 'black'
        
        # Add the status text as an overlay
        self.fig.text(0.5, 0.02, status_text, ha='center', fontsize=12, color=text_color,
                     bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor='black', linewidth=1))
    
    def start_recording(self):
        """Start recording with enhanced visual feedback"""
        import os
        from datetime import datetime
        
        # Create recordings directory if it doesn't exist
        os.makedirs('recordings', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"recordings/robot_motion_{timestamp}.csv"
        
        try:
            # Open CSV file for writing
            self.csv_file = open(filename, 'w', newline='')
            
            # Write CSV header (14 joint angles)
            joint_names = [
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint',
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint'
            ]
            header = 'timestamp,' + ','.join(joint_names)
            self.csv_file.write(header + '\n')
            
            # Update recording status
            current_time = time.time()
            self.recording_status.update({
                'active': True,
                'start_time': current_time,
                'frame_count': 0,
                'filename': filename,
                'show_indicator': True,
                'indicator_timer': current_time,
                'blink_state': True
            })
            
            # Legacy compatibility
            self.recording = True
            self.recording_file = filename
            self.start_time = current_time
            self.frame_counter = 0
            
            print(f"✅ Recording started: {filename}")
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            self.recording_status['active'] = False
            if hasattr(self, 'csv_file') and self.csv_file:
                self.csv_file.close()
                self.csv_file = None
    
    def stop_recording(self):
        """Stop recording with enhanced visual feedback"""
        if self.recording_status['active']:
            try:
                if self.csv_file:
                    self.csv_file.close()
                
                filename = self.recording_status.get('filename', 'Unknown')
                frame_count = self.recording_status.get('frame_count', 0)
                duration = time.time() - self.recording_status.get('start_time', time.time())
                
                print(f"⏹️ Recording stopped: {filename}")
                print(f"📊 Recorded {frame_count} frames in {duration:.1f}s")
                
                # Update recording status
                self.recording_status.update({
                    'active': False,
                    'show_indicator': True,
                    'indicator_timer': time.time()
                })
                
                # Legacy compatibility
                self.recording = False
                self.csv_file = None
                self.recording_file = None
                
            except Exception as e:
                print(f"❌ Error stopping recording: {e}")
        else:
            print("⚠️ No active recording to stop")
    
    def record_frame_to_csv(self):
        """Record current frame to CSV with enhanced tracking"""
        if self.recording_status['active'] and self.csv_file:
            try:
                # Calculate timestamp
                current_time = time.time()
                elapsed = current_time - self.recording_status['start_time']
                
                # Get joint angles
                joint_angles = self.robot_retargeter.joint_angles
                
                # Write CSV line
                line = f"{elapsed:.3f}"
                for joint_name in [
                    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                    'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint',
                    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                    'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint'
                ]:
                    angle = joint_angles.get(joint_name, 0.0)
                    line += f",{angle:.2f}"
                
                self.csv_file.write(line + '\n')
                self.csv_file.flush()
                
                # Update frame count
                self.recording_status['frame_count'] += 1
                
            except Exception as e:
                print(f"❌ Error recording frame: {e}")

    def run(self):
        """Main loop to run the PoseMirror3D system: captures camera, processes pose, updates visualizations, and handles user input."""
        print('[PoseMirror3DWithRetargeting] -> run()')
        """Run the PoseMirror3D system with robot retargeting."""
        # Initialize the camera capture
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        clock = pygame.time.Clock()
        running = True
        
        # Keep track of frame times
        prev_time = time.time()
        fps_values = []
        
        # Initialize animation timer for pausing
        self.animation_timer = time.time()
        self.paused = False
        
        while running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        # Enhanced recording toggle with visual feedback
                        if hasattr(self, 'recording_status'):
                            if self.recording_status['active']:
                                self.stop_recording()
                                print("🔴➡️⏹️ Recording stopped via 'S' key")
                            else:
                                self.start_recording()
                                print("⏹️➡️🔴 Recording started via 'S' key")
                        else:
                            # Fallback to old system
                            if not self.robot_retargeter.recording:
                                self.robot_retargeter.start_recording()
                                print("Started recording")
                            else:
                                self.robot_retargeter.stop_recording()
                                print("Stopped recording")
                    elif event.key == pygame.K_r:
                        # Reset calibration
                        self.initial_angle_set = False
                        self.angle_offset = 0
                        self.recent_chest_vectors = []
                        print("Calibration reset")
                    elif event.key == pygame.K_p:
                        # Toggle pause state
                        self.paused = self.robot_retargeter.toggle_pause()
                        if self.paused:
                            print("Motion paused - good for taking screenshots")
                        else:
                            print("Motion resumed")
                    elif event.key == pygame.K_i:
                        # Switch IK solver (if dual mode enabled)
                        if hasattr(self, 'dual_mode') and self.dual_mode:
                            if hasattr(self.robot_retargeter, 'switch_ik_solver'):
                                old_solver = self.robot_retargeter.ik_solver_backend
                                new_solver = self.robot_retargeter.switch_ik_solver()
                                print(f"🔄 IK Solver switched: {old_solver.upper()} → {new_solver.upper()}")
                            else:
                                print("⚠️ IK solver switching not available in this retargeter")
                        else:
                            print("⚠️ Dual mode not enabled. Use --dual-mode to enable IK switching.")
                    elif event.key == pygame.K_v:
                        # Toggle validation
                        self.validation_enabled = not self.validation_enabled
                        self.robot_retargeter.enable_validation(self.validation_enabled)
                        print(f"Validation {'enabled' if self.validation_enabled else 'disabled'}")
                        
            # Read frame from camera
            success, image = cap.read()
            if not success:
                print("Failed to capture video frame. Using a blank frame instead.")
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Process the image to detect human pose
            results = self.pose.process(image)
            
            # Draw the pose landmarks on the image
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
            # Update robot state
            self.robot_retargeter.update_robot_state(results)
            
            # Update all visualizations
            self.update_visualization(results, image)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            fps_values.append(fps)
            if len(fps_values) > 30:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)
            
            # Calculate frame rate and display as title
            pygame.display.set_caption(f'Motion Retargeting (FPS: {avg_fps:.1f})')
            
            # Display info text on pygame window
            self.screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 36)
            text = font.render("Please look at the matplotlib window", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.window_size[0]//2, self.window_size[1]//2))
            self.screen.blit(text, text_rect)
            
            # Add enhanced controls info
            controls_font = pygame.font.Font(None, 20)
            controls_lines = [
                "Controls: Q=Quit, S=Record Toggle, P=Pause, V=Validation",
            ]
            if hasattr(self, 'dual_mode') and self.dual_mode:
                controls_lines.append("I=Switch IK Solver (Analytical ↔ BRPSO)")
            
            for i, line in enumerate(controls_lines):
                controls_text = controls_font.render(line, True, (200, 200, 200))
                controls_rect = controls_text.get_rect(center=(self.window_size[0]//2, self.window_size[1]//2 + 50 + i*25))
                self.screen.blit(controls_text, controls_rect)
            
            pygame.display.flip()
            
            # Cap to 30 FPS max
            clock.tick(30)
            
            # Record frame if enhanced recording is active
            if hasattr(self, 'recording_status') and self.recording_status['active']:
                self.record_frame_to_csv()
            
            # Legacy recording support
            if self.robot_retargeter.recording:
                self.frame_counter += 1
                timestamp = time.time() - self.start_time if self.start_time > 0 else 0.0
                self.robot_retargeter.record_frame(timestamp, self.robot_retargeter.joint_angles)
        
        # Release resources
        cap.release()
        plt.ioff()
        plt.close(self.fig)
        pygame.quit()

# Main function to run directly
if __name__ == "__main__":
    print("Starting Motion Retargeting Visualization")
    pose_mirror = PoseMirror3DWithRetargeting()
    pose_mirror.run()