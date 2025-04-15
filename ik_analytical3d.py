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
        :param shoulder: np.array(3,) Shoulder 3D point
        :param elbow:    np.array(3,) Elbow 3D point
        :param wrist:    np.array(3,) Wrist 3D point
        :param target_orientation: np.array(3,3) Optional target orientation matrix
        :return: Dictionary of 7 joint angles
        """
        # Convert points to local coordinate system
        local_elbow = elbow - shoulder
        local_wrist = wrist - shoulder

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

        return {
            'shoulder_pitch': sp,
            'shoulder_yaw': sy,
            'shoulder_roll': sr,
            'elbow': elb,
            'wrist_pitch': wp,
            'wrist_yaw': wy,
            'wrist_roll': wr
        }

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
        # Calculate current orientation matrix
        R_shoulder = self.transform_matrix(sp, 0, 0, np.pi/2)[:3,:3]
        R_elbow = self.transform_matrix(elb, 0, self.L1, 0)[:3,:3]
        
        # Required wrist orientation
        R_wrist = np.linalg.inv(R_shoulder @ R_elbow) @ target_orientation
        
        # Extract Euler angles from rotation matrix
        pitch = arctan2(R_wrist[2,1], R_wrist[2,2])
        yaw = arctan2(-R_wrist[2,0], sqrt(R_wrist[2,1]**2 + R_wrist[2,2]**2))
        roll = arctan2(R_wrist[1,0], R_wrist[0,0])
        
        return pitch, yaw, roll
