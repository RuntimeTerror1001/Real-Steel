# ik_analytical3d.py
"""
3D Analytical Inverse Kinematics Solver for 7-DOF Robotic Arm
Uses DH parameters and full geometric decomposition with orientation handling
"""

import numpy as np
from numpy import sin, cos, arctan2, arccos, sqrt
import math

class IKAnalytical3D:
    def __init__(self, upper_arm_length=0.1032, lower_arm_length=0.1):
        # DH parameters for Unitree G1 arm
        self.L1 = upper_arm_length  # Shoulder to elbow
        self.L2 = lower_arm_length  # Elbow to wrist
        
        # Joint limits from G1 specs
        self.joint_limits = {
            'shoulder_pitch': (-3.0892, 2.6704),    # -177° to 153°
            'shoulder_yaw': (-2.618, 2.618),        # -150° to 150°
            'shoulder_roll': (-1.5882, 2.2515),     # -91° to 129°
            'elbow': (-1.0472, 2.0944),            # -60° to 120°
            'wrist_pitch': (-1.61443, 1.61443),    # -92.5° to 92.5°
            'wrist_yaw': (-1.61443, 1.61443),      # -92.5° to 92.5°
            'wrist_roll': (-1.97222, 1.97222)      # -113° to 113°
        }

        # Validation parameters
        self.position_tolerance = 1e-6  # 1 µm tolerance
        self.orientation_tolerance = 1e-4  # ~0.0057 degrees

    def clip_to_limits(self, joint, angle):
        """Clip angle to joint limits"""
        if joint in self.joint_limits:
            min_limit, max_limit = self.joint_limits[joint]
            return np.clip(angle, min_limit, max_limit)
        return angle

    def transform_matrix(self, theta, d, a, alpha):
        """Calculate transformation matrix using DH parameters"""
        T = np.array([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0,          sin(alpha),             cos(alpha),            d],
            [0,          0,                      0,                     1]
        ])
        return T

    def solve(self, shoulder, elbow, wrist, target_orientation=None):
        """
        Solve full IK chain from shoulder to wrist with orientation
        :param shoulder: np.array(3,) Shoulder 3D point in world frame
        :param elbow:    np.array(3,) Elbow 3D point in world frame
        :param wrist:    np.array(3,) Wrist 3D point in world frame
        :param target_orientation: np.array(3,3) Optional target orientation matrix
        :return: Dictionary of 7 joint angles
        """
        # Convert points to shoulder-local coordinate system
        local_elbow = elbow - shoulder
        local_wrist = wrist - shoulder

        # Normalize vectors to unit length for stability
        local_elbow_norm = np.linalg.norm(local_elbow)
        local_wrist_norm = np.linalg.norm(local_wrist)
        
        if local_elbow_norm < 1e-8 or local_wrist_norm < 1e-8:
            raise ValueError("Points too close to shoulder")
            
        local_elbow = local_elbow / local_elbow_norm
        local_wrist = local_wrist / local_wrist_norm

        # 1. Solve shoulder angles
        sp, sy, sr = self._shoulder_angles(local_elbow, local_wrist)
        
        # Apply joint limits
        sp = self.clip_to_limits('shoulder_pitch', sp)
        sy = self.clip_to_limits('shoulder_yaw', sy)
        sr = self.clip_to_limits('shoulder_roll', sr)
        
        # 2. Solve elbow angle using cosine law with joint limits
        elb = self._elbow_angle(shoulder, elbow, wrist)
        elb = self.clip_to_limits('elbow', elb)
        
        # 3. Solve wrist angles considering orientation if provided
        if target_orientation is not None:
            wp, wy, wr = self._wrist_angles_with_orientation(
                shoulder, elbow, wrist, target_orientation, sp, sy, sr, elb
            )
        else:
            wp, wy, wr = self._wrist_angles(elbow, wrist)
        
        # Apply wrist joint limits
        wp = self.clip_to_limits('wrist_pitch', wp)
        wy = self.clip_to_limits('wrist_yaw', wy)
        wr = self.clip_to_limits('wrist_roll', wr)

        angles = {
            'shoulder_pitch': sp,
            'shoulder_yaw': sy,
            'shoulder_roll': sr,
            'elbow': elb,
            'wrist_pitch': wp,
            'wrist_yaw': wy,
            'wrist_roll': wr
        }

        # ——— one quick positional "nudge" ———
        fk_pos, _ = self.forward_kinematics(angles)
        target_local = wrist - shoulder
        err = target_local - fk_pos
        if np.linalg.norm(err) > self.position_tolerance:
            δ = 1e-6
            # approximate dP/dsp
            ang2 = angles.copy()
            ang2['shoulder_pitch'] += δ
            fk2, _ = self.forward_kinematics(ang2)
            jac_sp = (fk2 - fk_pos) / δ
            # project error onto that direction
            angles['shoulder_pitch'] += 0.1 * jac_sp.dot(err)
        # ————————————————————————————————

        return angles

    def _shoulder_angles(self, local_elbow, local_wrist):
        """Calculate shoulder angles using geometric approach"""
        # Project onto planes for better angle calculation
        xy_proj = sqrt(local_elbow[0]**2 + local_elbow[1]**2)
        
        # Shoulder pitch (elevation angle)
        pitch = arctan2(local_elbow[2], xy_proj)
        
        # Shoulder yaw (azimuth angle)
        yaw = arctan2(local_elbow[1], local_elbow[0])
        
        # Shoulder roll using arm plane normal
        elbow_dir = local_elbow / (np.linalg.norm(local_elbow) + 1e-8)
        wrist_dir = local_wrist / (np.linalg.norm(local_wrist) + 1e-8)
        plane_normal = np.cross(elbow_dir, wrist_dir)
        roll = arctan2(plane_normal[1], plane_normal[0])
        
        return pitch, yaw, roll

    def _elbow_angle(self, shoulder, elbow, wrist):
        """Calculate elbow angle using cosine law"""
        a = np.linalg.norm(elbow - shoulder)
        b = np.linalg.norm(wrist - elbow)
        c = np.linalg.norm(wrist - shoulder)
        
        # Apply cosine law with stability check
        cos_theta = (a**2 + b**2 - c**2) / (2 * a * b + 1e-8)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Compute elbow angle (pi - theta for interior angle)
        theta = np.pi - arccos(cos_theta)
        return theta

    def _wrist_angles(self, elbow, wrist):
        """Calculate wrist angles without orientation data"""
        vec = wrist - elbow
        vec_norm = np.linalg.norm(vec) + 1e-8
        dir_vec = vec / vec_norm

        # Calculate wrist angles relative to elbow
        pitch = arctan2(dir_vec[1], sqrt(dir_vec[0]**2 + dir_vec[2]**2))
        yaw = arctan2(-dir_vec[0], dir_vec[2])
        
        # Estimate roll based on natural arm pose
        roll = 0.0  # Can be enhanced with biomechanical constraints
        
        return pitch, yaw, roll

    def _wrist_angles_with_orientation(self, shoulder, elbow, wrist, target_orientation, sp, sy, sr, elb):
        """Calculate wrist angles considering target orientation"""
        # Build full upstream rotation (shoulder pitch, yaw, roll, then elbow)
        # Updated to match the fixed FK DH parameters with double initial rotation
        T_initial1 = self.transform_matrix(np.pi/2, 0, 0, 0)  # +90° around Y
        T_initial2 = self.transform_matrix(0, 0, 0, np.pi/2)  # +90° around X (alpha parameter)
        T_initial = T_initial2 @ T_initial1
        
        T_sp       = self.transform_matrix(sp,   0,       0,      0)
        T_sy       = self.transform_matrix(sy,   0,       0,      0)
        T_sr       = self.transform_matrix(sr,   0,       0,      0)
        T_elb      = self.transform_matrix(elb,  0,   self.L1,      0)
        R_partial  = (T_initial @ T_sp @ T_sy @ T_sr @ T_elb)[:3, :3]

        # Invert that to isolate the wrist orientation
        R_wrist    = np.linalg.inv(R_partial) @ target_orientation

        # Extract Euler angles for wrist pitch, yaw, roll
        pitch      = np.arctan2(R_wrist[2,1],                 R_wrist[2,2])
        yaw        = np.arctan2(-R_wrist[2,0], np.sqrt(R_wrist[2,1]**2 + R_wrist[2,2]**2))
        roll       = np.arctan2(R_wrist[1,0],                 R_wrist[0,0])

        # Clip to limits and return
        return (
            self.clip_to_limits('wrist_pitch', pitch),
            self.clip_to_limits('wrist_yaw',   yaw),
            self.clip_to_limits('wrist_roll',  roll)
        )

    def validate_ik_fk(self, joint_angles, target_position, target_orientation=None):
        """
        Validate IK solution using forward kinematics
        :param joint_angles: Dictionary of joint angles
        :param target_position: Target end-effector position in shoulder frame
        :param target_orientation: Optional target orientation matrix
        :return: (bool, float) - (is_valid, position_error)
        """
        # Check joint limits first
        if "right_shoulder_pitch_joint" in joint_angles:
            prefix = "right"
        elif "left_shoulder_pitch_joint" in joint_angles:
            prefix = "left"
        else:
            raise ValueError("No recognized joint prefix (left/right) in joint_angles")
        
        remapped = {
            'shoulder_pitch': joint_angles[f'{prefix}_shoulder_pitch_joint'],
            'shoulder_yaw': joint_angles[f'{prefix}_shoulder_yaw_joint'],
            'shoulder_roll': joint_angles[f'{prefix}_shoulder_roll_joint'],
            'elbow': joint_angles[f'{prefix}_elbow_joint'],
            'wrist_pitch': joint_angles[f'{prefix}_wrist_pitch_joint'],
            'wrist_yaw': joint_angles[f'{prefix}_wrist_yaw_joint'],
            'wrist_roll': joint_angles[f'{prefix}_wrist_roll_joint']
        }

        # Check joint limits
        for joint, angle in remapped.items():
            if joint in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint]
                if not (min_limit <= angle <= max_limit):
                    return False, float('inf')

        # Compute forward kinematics
        fk_pos, fk_ori = self.forward_kinematics(remapped)
        
        # Compute position error
        position_error = np.linalg.norm(fk_pos - target_position)
        
        # Check if error is within tolerance
        is_valid = position_error < self.position_tolerance
        
        # If orientation is provided, check orientation error
        if target_orientation is not None and fk_ori is not None:
            # Compute orientation error as the angle between rotation matrices
            R_error = fk_ori @ target_orientation.T
            orientation_error = np.arccos((np.trace(R_error) - 1) / 2)
            is_valid = is_valid and (orientation_error < self.orientation_tolerance)
        
        return is_valid, position_error

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector position and orientation from joint angles
        :param joint_angles: Dictionary of joint angles
        :return: (position, orientation_matrix)
        """
        # Extract joint angles
        sp = joint_angles['shoulder_pitch']
        sy = joint_angles['shoulder_yaw']
        sr = joint_angles['shoulder_roll']
        elb = joint_angles['elbow']
        wp = joint_angles['wrist_pitch']
        wy = joint_angles['wrist_yaw']
        wr = joint_angles['wrist_roll']
        
        # Calculate transformation matrices - FIXED to extend along Z-axis
        # Add initial rotations to align with Z-axis
        # First: rotate +90° around Y to point Y instead of X
        # Second: rotate +90° around X to point Z instead of Y (changed from -90° to +90°)
        T_initial1 = self.transform_matrix(np.pi/2, 0, 0, 0)  # +90° around Y
        T_initial2 = self.transform_matrix(0, 0, 0, np.pi/2)  # +90° around X (alpha parameter)
        T_initial = T_initial2 @ T_initial1
        
        # Regular DH chain
        T_shoulder = self.transform_matrix(sp, 0, 0, 0)
        T_yaw = self.transform_matrix(sy, 0, 0, 0)
        T_roll = self.transform_matrix(sr, 0, 0, 0)
        T_elbow = self.transform_matrix(elb, 0, self.L1, 0)
        T_wrist_pitch = self.transform_matrix(wp, 0, self.L2, 0)
        T_wrist_yaw = self.transform_matrix(wy, 0, 0, 0)
        T_wrist_roll = self.transform_matrix(wr, 0, 0, 0)
        
        # Calculate final transformation with initial rotation
        T_final = T_initial @ T_shoulder @ T_yaw @ T_roll @ T_elbow @ T_wrist_pitch @ T_wrist_yaw @ T_wrist_roll
        
        # Extract position and orientation
        position = T_final[:3, 3]
        orientation = T_final[:3, :3]
        
        return position, orientation

    def check_motion_continuity(self, joint_angles_sequence, max_velocity=1.0):
        """
        Check for sudden changes between consecutive joint angles
        :param joint_angles_sequence: List of joint angle dictionaries
        :param max_velocity: Maximum allowed joint velocity in rad/s
        :return: (bool, list) - (is_continuous, problematic_joints)
        """
        if len(joint_angles_sequence) < 2:
            return True, []
            
        problematic_joints = []
        is_continuous = True
        
        for i in range(1, len(joint_angles_sequence)):
            prev_angles = joint_angles_sequence[i-1]
            curr_angles = joint_angles_sequence[i]
            
            for joint in prev_angles.keys():
                velocity = abs(curr_angles[joint] - prev_angles[joint])
                if velocity > max_velocity:
                    is_continuous = False
                    problematic_joints.append((joint, i, velocity))
                    
        return is_continuous, problematic_joints

def validate_ik_fk(joint_angles, target_position):
    # Calculate FK from joint angles
    fk_position = forward_kinematics(joint_angles)
    # Compare with target position
    error = np.linalg.norm(fk_position - target_position)
    return error < threshold

def check_motion_continuity(joint_angles_sequence):
    # Check for sudden changes between consecutive positions
    max_velocity = calculate_joint_velocities(joint_angles_sequence)
    return max_velocity < velocity_threshold
