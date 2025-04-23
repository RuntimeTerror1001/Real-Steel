import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

class MotionValidator:
    def __init__(self,
                 joint_limits: Dict[str, Tuple[float, float]],
                 max_velocity: float = 2.0,
                 max_acceleration: float = 1.0,
                 dt: float = 0.1,
                 link_lengths: Dict[str, float] = None):
        self.joint_limits = joint_limits
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.link_lengths = link_lengths or {
            'upper_arm': 0.3,  # 30cm upper arm
            'forearm': 0.25    # 25cm forearm
        }

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ik_validation.log'),
                logging.StreamHandler()
            ]
        )

    def compute_fk(self, joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for a simple arm (shoulder pitch, roll, yaw, elbow)
        Returns end-effector position and orientation using right-hand coordinate system:
        - X-axis points forward
        - Y-axis points left
        - Z-axis points up
        """
        shoulder_pitch = joint_angles.get('right_shoulder_pitch', 0)
        shoulder_roll = joint_angles.get('right_shoulder_roll', 0)
        shoulder_yaw = joint_angles.get('right_shoulder_yaw', 0)
        elbow = joint_angles.get('right_elbow', 0)
        
        # Link lengths
        l1 = self.link_lengths['upper_arm']
        l2 = self.link_lengths['forearm']
        
        # Transformation matrices (right-hand coordinate system)
        def Rx(theta):
            # Rotation about X-axis (roll)
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        
        def Ry(theta):
            # Rotation about Y-axis (pitch)
            # Positive pitch rotates up
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        
        def Rz(theta):
            # Rotation about Z-axis (yaw)
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        
        # Initial position (shoulder)
        pos = np.zeros(3)
        
        # Upper arm transformation
        # For right arm:
        # - Positive pitch rotates up (about Y)
        # - Positive roll rotates outward (about X)
        # - Positive yaw rotates forward (about Z)
        R_shoulder = Rz(-shoulder_yaw) @ Ry(-shoulder_pitch) @ Rx(shoulder_roll)
        upper_arm_vec = np.array([l1, 0, 0])
        pos += R_shoulder @ upper_arm_vec
        
        # Forearm transformation
        # Positive elbow angle flexes inward (about Y)
        # For right arm, positive elbow flexion moves the forearm to the left (+Y)
        R_elbow = Rx(-np.pi/2) @ Ry(-elbow) @ Rx(np.pi/2)
        R_total = R_shoulder @ R_elbow
        forearm_vec = np.array([l2, 0, 0])
        pos += R_total @ forearm_vec
        
        return pos, R_total

    def validate_fk_ik(self, joint_angles: Dict[str, float], target_position: np.ndarray,
                      position_tolerance: float = 0.01) -> Tuple[bool, float]:
        """
        Validate IK solution using forward kinematics
        Args:
            joint_angles: Dictionary of joint angles
            target_position: Target end-effector position
            position_tolerance: Maximum allowed position error (in meters)
        Returns:
            Tuple[bool, float]: (is_valid, position_error)
        """
        # Check joint limits first
        is_valid, violations = self.validate_joint_limits(joint_angles)
        if not is_valid:
            logging.warning("Joint limit violations detected:")
            for violation in violations:
                logging.warning(violation)
            return False, float('inf')
        
        # Calculate forward kinematics
        fk_position, _ = self.compute_fk(joint_angles)
        
        # Calculate position error
        position_error = np.linalg.norm(fk_position - target_position)
        is_valid = position_error < position_tolerance
        
        if not is_valid:
            logging.warning(f"Position error {position_error:.3f}m exceeds tolerance {position_tolerance:.3f}m")
            logging.warning(f"Target position: {target_position}")
            logging.warning(f"Achieved position: {fk_position}")
        
        return is_valid, position_error

    def validate_joint_limits(self, joint_angles: Dict[str, float]) -> Tuple[bool, List[str]]:
        violations = []
        is_valid = True
        for joint, angle in joint_angles.items():
            if joint in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint]
                if not (min_limit <= angle <= max_limit):
                    violations.append(
                        f"{joint}: {angle:.3f} rad out of bounds [{min_limit:.3f}, {max_limit:.3f}]"
                    )
                    is_valid = False
        return is_valid, violations

    def validate_motion_smoothness(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        violations = []
        is_smooth = True
        for joint in data.columns:
            if joint == 'timestamp':
                continue
            values = data[joint].values
            velocities = np.diff(values) / self.dt
            accelerations = np.diff(velocities) / self.dt
            max_vel = np.max(np.abs(velocities))
            max_acc = np.max(np.abs(accelerations))
            if max_vel > self.max_velocity:
                is_smooth = False
                violations.append(f"{joint}: max velocity {max_vel:.3f} rad/s > {self.max_velocity}")
            if max_acc > self.max_acceleration:
                is_smooth = False
                violations.append(f"{joint}: max acceleration {max_acc:.3f} rad/s^2 > {self.max_acceleration}")
        return is_smooth, violations

    def validate_from_csv(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        logging.info(f"Validating motion from CSV: {input_csv}")
        df = pd.read_csv(input_csv)
        valid_frames = []
        all_violations = []

        for idx, row in df.iterrows():
            joint_angles = row.drop('timestamp').to_dict()
            is_valid, violations = self.validate_joint_limits(joint_angles)
            if not is_valid:
                all_violations.extend([f"Frame {idx}: {v}" for v in violations])
                continue
            valid_frames.append(row)

        if not valid_frames:
            logging.error("No valid frames found.")
            return pd.DataFrame()

        validated_df = pd.DataFrame(valid_frames)
        is_smooth, smoothness_violations = self.validate_motion_smoothness(validated_df)
        all_violations.extend(smoothness_violations)

        if all_violations:
            logging.warning("Validation finished with warnings:")
            for v in all_violations:
                logging.warning(v)
        else:
            logging.info("Validation completed with no violations.")

        validated_df.to_csv(output_csv, index=False)
        logging.info(f"Validated CSV written to: {output_csv}")
        return validated_df

    def validate_from_stream(self, frames: List[Dict[str, Union[float, int]]]) -> pd.DataFrame:
        logging.info("Validating motion from live stream data...")
        df = pd.DataFrame(frames)
        return self.validate_from_csv_data(df)

    def validate_from_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_frames = []
        all_violations = []

        for idx, row in df.iterrows():
            joint_angles = row.drop('timestamp').to_dict()
            is_valid, violations = self.validate_joint_limits(joint_angles)
            if not is_valid:
                all_violations.extend([f"Frame {idx}: {v}" for v in violations])
                continue
            valid_frames.append(row)

        if not valid_frames:
            logging.error("No valid frames found.")
            return pd.DataFrame()

        validated_df = pd.DataFrame(valid_frames)
        is_smooth, smoothness_violations = self.validate_motion_smoothness(validated_df)
        all_violations.extend(smoothness_violations)

        if all_violations:
            logging.warning("Stream validation finished with warnings:")
            for v in all_violations:
                logging.warning(v)
        else:
            logging.info("Stream validation completed successfully.")

        return validated_df 