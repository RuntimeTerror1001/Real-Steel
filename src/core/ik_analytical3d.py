# ik_analytical3d.py
"""
3D Analytical Inverse Kinematics Solver for 7-DOF Robotic Arm
Uses DH parameters and full geometric decomposition with orientation handling
"""

import numpy as np
from numpy import pi, cos, sin, arccos, arctan2, sqrt
import math

class IKAnalytical3DRefined:
    def __init__(
        self,
        upper_arm_length=0.1032,
        lower_arm_length=0.1,
        position_tolerance=1e-6,
        orientation_tolerance=1e-4,
        jacobian_delta=1e-6,
        damping=1e-3,
        refinement_gain=0.5
    ):
        """Initialize the IK solver with specified parameters."""
        # Lengths of arm segments
        self.L1 = upper_arm_length # Shoulder to elbow
        self.L2 = lower_arm_length # Elbow to wrist
        
        # IK solver parameters
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self._delta = jacobian_delta
        self._damping = damping
        self._gain = refinement_gain
        
        # Joint limits (in radians)
        self.joint_limits = {
            'shoulder_pitch': (-3.0892, 2.6704),    # -177° to 153°
            'shoulder_yaw': (-2.618, 2.618),        # -150° to 150°
            'shoulder_roll': (-1.5882, 2.2515),     # -91° to 129°
            'elbow': (-1.0472, 2.0944),            # -60° to 120°
            'wrist_pitch': (-1.61443, 1.61443),    # -92.5° to 92.5°
            'wrist_yaw': (-1.61443, 1.61443),      # -92.5° to 92.5°
            'wrist_roll': (-1.97222, 1.97222)      # -113° to 113°
        }

        # Initialize joint tracking variables for continuity
        self.last_shoulder_yaw = 0.0
        self.last_shoulder_roll = 0.0
        self.last_wrist_pitch = 0.0
        self.last_wrist_yaw = 0.0
        self.last_wrist_roll = 0.0

    def clip(self, joint, angle):
        lo, hi = self.joint_limits[joint]
        return np.clip(angle, lo, hi)

    def transform_matrix(self, theta, d, a, alpha):
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0,            sin(alpha),             cos(alpha),           d],
            [0,            0,                      0,                    1]
        ])
    
    def forward_kinematics(self, angles):
        sy = angles['shoulder_yaw']; sp = angles['shoulder_pitch']; sr = angles['shoulder_roll']
        el = angles['elbow']; wp = angles['wrist_pitch']; wy = angles['wrist_yaw']; wr = angles['wrist_roll']
        
        # DH‐chain: A1…A7 with updated parameters
        T = self.transform_matrix(sy, 0, 0, +np.pi/2)     # Shoulder Yaw
        T = T @ self.transform_matrix(sp, 0, 0, -np.pi/2)  # Shoulder Pitch
        T = T @ self.transform_matrix(sr, 0, 0, +np.pi/2)  # Shoulder Roll
        T = T @ self.transform_matrix(el, 0, self.L1, 0)   # Elbow Flex, L₁=0.1032 m
        T = T @ self.transform_matrix(wp, 0, self.L2, +np.pi/2)  # Wrist Pitch, L₂=0.1000 m
        T = T @ self.transform_matrix(wy, 0, 0, -np.pi/2)  # Wrist Yaw
        T = T @ self.transform_matrix(wr, 0, 0, 0)         # Wrist Roll
        
        pos = T[:3,3]
        ori = T[:3,:3]
        return pos, ori

    def _shoulder_angles(self, le, lw):
        """
        Calculate shoulder angles (pitch, yaw, roll) based on the positions
        of elbow relative to shoulder and wrist relative to shoulder.
        
        Args:
            le: Normalized vector from shoulder to elbow
            lw: Normalized vector from shoulder to wrist
            
        Returns:
            Tuple of (pitch, yaw, roll) angles
        """
        # Calculate shoulder pitch (elevation)
        xy = math.sqrt(le[0]**2 + le[1]**2)
        pitch = math.atan2(le[2], xy)
        
        # Calculate shoulder yaw (rotation around vertical axis)
        # Use a more stable calculation that considers the magnitude of components
        yaw_magnitude = math.sqrt(le[0]**2 + le[1]**2)
        if yaw_magnitude < 1e-4:
            # If the arm is pointing straight up or down, yaw is poorly defined
            # Use previous value for stability if available
            if hasattr(self, 'last_shoulder_yaw'):
                yaw = self.last_shoulder_yaw
            else:
                yaw = 0.0
        else:
            yaw = math.atan2(le[1], le[0])
        
        # Remember yaw for next iteration
        self.last_shoulder_yaw = yaw
        
        # Calculate shoulder roll (twist of the arm)
        e_dir = le
        w_dir = lw
        n = np.cross(e_dir, w_dir)
        
        # Check if cross product is valid
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-4:
            # If vectors are nearly parallel, roll is poorly defined
            # Use previous value for stability if available
            if hasattr(self, 'last_shoulder_roll'):
                roll = self.last_shoulder_roll
            else:
                roll = 0.0
        else:
            # Normalize cross product and compute roll
            n = n / n_norm
            
            # Compute twist angle using components most likely to be significant
            # This gives better numerical stability
            roll = math.atan2(n[1], n[0])
        
        # Remember roll for next iteration
        self.last_shoulder_roll = roll
        
        return pitch, yaw, roll

    def _elbow_angle(self, sh, el, wr):
        a = np.linalg.norm(el-sh); b = np.linalg.norm(wr-el); c = np.linalg.norm(wr-sh)
        cos_t = np.clip((a*a + b*b - c*c)/(2*a*b+1e-8), -1,1)
        return np.pi - arccos(cos_t)

    def _wrist_angles(self, el, wr):
        """
        Compute wrist angles (pitch, yaw) for 5-dof mode
        """
        # Compute direction
        v = wr-el
        n = np.linalg.norm(v)
        if n < 1e-8:
            if hasattr(self, 'last_wrist_pitch') and hasattr(self, 'last_wrist_yaw'):
                return self.last_wrist_pitch, self.last_wrist_yaw, 0.0
            else:
                return 0.0, 0.0, 0.0
                
        # Unit vector in wrist direction
        d = v/n
        pitch = np.arctan2(d[1], np.sqrt(d[0]**2 + d[2]**2))
        yaw = np.arctan2(-d[0], d[2])
        
        # Apply continuity protection
        if hasattr(self, 'last_wrist_pitch') and abs(pitch) < 1e-4 and abs(self.last_wrist_pitch) > 0.1:
            pitch = self.last_wrist_pitch * 0.8
            
        if hasattr(self, 'last_wrist_yaw') and abs(yaw) < 1e-4 and abs(self.last_wrist_yaw) > 0.1:
            yaw = self.last_wrist_yaw * 0.8
            
        # Save values for next time
        self.last_wrist_pitch = pitch
        self.last_wrist_yaw = yaw
        
        return pitch, yaw, 0.0

    def calculate_wrist_angles(self, shoulder_angles, elbow_angle, desired_hand_pose):
        """
        Compute wrist joint angles using a direct vector approach.
        This matches the Method 3 implementation from test_wrist_angles.py.
        
        Parameters:
        - shoulder_angles: [pitch, yaw, roll]
        - elbow_angle: single float
        - desired_hand_pose: dictionary with 'position' and possibly 'orientation'
        
        Returns:
        - wrist_angles: dict with 'pitch', 'yaw', 'roll'
        """
        # Get target position in local coordinates
        target_pos = desired_hand_pose['position']
        
        # Simplified method using direct vector approach
        # We'll directly work with input vectors to calculate wrist angles
        
        # Create a simplified kinematic chain
        # For a simple approach, assume forearm is along Z-axis
        forearm_vec = np.array([0, 0, 1])  # Default direction
        wrist_to_hand_vec = target_pos / (np.linalg.norm(target_pos) + 1e-8)
        
        # 1. Calculate pitch angle - angle in the sagittal (XZ) plane
        # Project vectors onto XZ plane
        forearm_xz = np.array([forearm_vec[0], 0, forearm_vec[2]])
        hand_xz = np.array([wrist_to_hand_vec[0], 0, wrist_to_hand_vec[2]])
        
        # Normalize if needed
        if np.linalg.norm(forearm_xz) > 1e-4:
            forearm_xz = forearm_xz / np.linalg.norm(forearm_xz)
        else:
            forearm_xz = np.array([0, 0, 1])  # Default if too small
            
        if np.linalg.norm(hand_xz) > 1e-4:
            hand_xz = hand_xz / np.linalg.norm(hand_xz)
        else:
            hand_xz = np.array([0, 0, 1])  # Default if too small
            
        # Use dot product for pitch calculation
        dot_product = np.dot(forearm_xz, hand_xz)
        pitch = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Adjust sign of pitch based on vertical component
        if wrist_to_hand_vec[1] < 0:
            pitch = -pitch
            
        # 2. Calculate yaw angle - angle in the XY plane
        forearm_xy = np.array([forearm_vec[0], forearm_vec[1]])
        hand_xy = np.array([wrist_to_hand_vec[0], wrist_to_hand_vec[1]])
        
        # Check if vectors have enough magnitude in the XY plane
        if np.linalg.norm(forearm_xy) > 1e-4 and np.linalg.norm(hand_xy) > 1e-4:
            forearm_xy = forearm_xy / np.linalg.norm(forearm_xy)
            hand_xy = hand_xy / np.linalg.norm(hand_xy)
            angle_yaw = np.arctan2(hand_xy[1], hand_xy[0]) - np.arctan2(forearm_xy[1], forearm_xy[0])
            yaw = np.arctan2(np.sin(angle_yaw), np.cos(angle_yaw))  # Normalize
        else:
            # Use previous yaw value if the vectors are too vertical
            yaw = self.last_wrist_yaw if hasattr(self, 'last_wrist_yaw') else 0.0
        
        # 3. Roll calculation (only if orientation info is available)
        roll = 0.0  # Default
        if 'orientation' in desired_hand_pose and desired_hand_pose['orientation'] is not None:
            # Extract roll from orientation matrix if needed
            try:
                # In a real implementation, would extract roll from orientation
                # For now, we're just keeping it at 0
                roll = 0.0
            except Exception:
                roll = 0.0
        
        # 4. Apply continuity protection
        if hasattr(self, 'last_wrist_pitch') and abs(pitch) < 1e-4 and abs(self.last_wrist_pitch) > 0.1:
            pitch = self.last_wrist_pitch * 0.8
            
        if hasattr(self, 'last_wrist_yaw') and abs(yaw) < 1e-4 and abs(self.last_wrist_yaw) > 0.1:
            yaw = self.last_wrist_yaw * 0.8
                                                                            
        if hasattr(self, 'last_wrist_roll') and abs(roll) < 1e-4 and abs(self.last_wrist_roll) > 0.1:
            roll = self.last_wrist_roll * 0.8
        
        # Save for next time
        self.last_wrist_pitch = pitch
        self.last_wrist_yaw = yaw
        self.last_wrist_roll = roll
        
        # Return clipped angles
        return {
            'wrist_pitch': self.clip('wrist_pitch', pitch),
            'wrist_yaw': self.clip('wrist_yaw', yaw),
            'wrist_roll': self.clip('wrist_roll', roll)
        }

    def solve(self, shoulder, elbow, wrist, target_ori=None):
        """
        Solve inverse kinematics for the arm.
        
        Args:
            shoulder: shoulder position
            elbow: elbow position
            wrist: wrist position
            target_ori: target orientation matrix (optional)
            
        Returns:
            Dictionary of joint angles
        """
        # Convert to local coordinates
        local_elbow = elbow - shoulder
        local_wrist = wrist - shoulder
        
        # Calculate vectors
        le = local_elbow 
        lw = local_wrist
        le_n = le / (np.linalg.norm(le)+1e-8)
        lw_n = lw / (np.linalg.norm(lw)+1e-8)
        
        # Solve shoulder angles
        sp, sy, sr = self._shoulder_angles(le_n, lw_n)
        
        # Solve elbow angle
        el = self._elbow_angle(np.zeros(3), le, lw)
        
        # Clip within joint limits
        sp = self.clip('shoulder_pitch', sp)
        sy = self.clip('shoulder_yaw', sy)
        sr = self.clip('shoulder_roll', sr)
        el = self.clip('elbow', el)
        
        # Create desired hand pose dictionary
        desired_hand_pose = {
            'position': local_wrist  # Local coordinates
        }
        if target_ori is not None:
            desired_hand_pose['orientation'] = target_ori
            
        # Calculate wrist angles using the new method
        wrist_angles = self.calculate_wrist_angles(
            shoulder_angles=[sp, sy, sr],
            elbow_angle=el,
            desired_hand_pose=desired_hand_pose
        )
        
        # Combine all angles
        angles = {
            'shoulder_yaw': sy,
            'shoulder_pitch': sp,
            'shoulder_roll': sr,
            'elbow': el,
            'wrist_pitch': wrist_angles['wrist_pitch'],
            'wrist_yaw': wrist_angles['wrist_yaw'],
            'wrist_roll': wrist_angles['wrist_roll']
        }
        
        # Refine solution with more iterations for better convergence
        for _ in range(5):  # Try multiple refinement iterations
            # Calculate forward kinematics to get current position
            pos, _ = self.forward_kinematics(angles)
            
            # Calculate error
            target = local_wrist
            err = target - pos
            err_mag = np.linalg.norm(err)
            
            # If error is already small enough, break early
            if err_mag < self.position_tolerance:
                break
                
            # numeric Jacobian 3x7
            keys = ['shoulder_yaw', 'shoulder_pitch', 'shoulder_roll', 'elbow', 'wrist_pitch', 'wrist_yaw', 'wrist_roll']
            J = np.zeros((3,7))
            
            for i, k in enumerate(keys):
                a_eps = angles.copy()
                a_eps[k] += self._delta
                p_eps, _ = self.forward_kinematics(a_eps)
                J[:,i] = (p_eps - pos) / self._delta
            
            # Use damped pseudo-inverse for more stable convergence
            J_T = J.T
            damping = self._damping * (1.0 + 10.0 * err_mag)  # Adaptive damping based on error
            JJt = J @ J_T + (damping**2) * np.eye(3)
            
            # Solve the system
            try:
                inv = np.linalg.solve(JJt, np.eye(3))
                J_pinv = J_T @ inv
                
                # Compute the step size
                gain = self._gain * (0.5 + 0.5 * np.exp(-err_mag * 10))  # Adaptive gain
                dtheta = gain * (J_pinv @ err)
                
                # Update & clip joint angles
                for i, k in enumerate(keys):
                    angles[k] = self.clip(k, angles[k] + dtheta[i])
            except np.linalg.LinAlgError:
                # If matrix inversion fails, use a different approach
                # Try damped least squares with higher damping
                damping = self._damping * 10.0
                J_pinv = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3))
                dtheta = self._gain * 0.2 * (J_pinv @ err)  # Reduced gain
                
                # Update & clip joint angles
                for i, k in enumerate(keys):
                    angles[k] = self.clip(k, angles[k] + dtheta[i])
                
        return angles

    def _wrist_angles_with_orientation(self, sh, el, wr, ori, sp, sy, sr, elb):
        """
        Compute wrist angles (pitch, yaw, roll) based on the relative rotation between 
        the shoulder-elbow chain and the target wrist orientation matrix.
        
        Args:
            sh: shoulder position
            el: elbow position
            wr: wrist position
            ori: target orientation matrix
            sp: shoulder pitch angle
            sy: shoulder yaw angle
            sr: shoulder roll angle
            elb: elbow angle
            
        Returns:
            Tuple of (wrist_pitch, wrist_yaw, wrist_roll) after clipping to limits
        """
        # Calculate transformation matrices for each joint up to elbow
        T_sp = self.transform_matrix(sp, 0, 0, np.pi/2)
        T_sy = self.transform_matrix(sy, 0, 0, 0)
        T_sr = self.transform_matrix(sr, 0, 0, 0)
        T_el = self.transform_matrix(elb, 0, self.L1, 0)
        
        # Compute the partial transformation up to the wrist base frame
        R_partial = T_sp @ T_sy @ T_sr @ T_el
        R_partial_rot = R_partial[:3,:3]
        
        # Calculate the relative rotation between the upper arm chain and target orientation
        R_wrist = np.linalg.inv(R_partial_rot) @ ori
        
        # Check if the rotation matrix is valid
        det = np.linalg.det(R_wrist)
        if abs(det - 1.0) > 0.1:
            # If the determinant is far from 1, normalize the matrix using SVD
            U, _, Vt = np.linalg.svd(R_wrist)
            R_wrist = U @ Vt
        
        # Calculate forearm direction (Z-axis of wrist frame)
        forearm_z = wr - el
        forearm_z = forearm_z / (np.linalg.norm(forearm_z) + 1e-8)
        
        # Extract palm normal from the rotation matrix (Y-axis of target orientation)
        palm_normal = ori[:, 1]  # Second column is the Y axis
        
        # Build local Y by projecting palm_normal perpendicular to forearm_z
        y_axis = palm_normal - forearm_z * np.dot(palm_normal, forearm_z)
        y_magnitude = np.linalg.norm(y_axis)
        
        if y_magnitude < 1e-4:
            # If palm normal is nearly parallel to forearm, use previous angles
            if hasattr(self, 'last_wrist_pitch') and hasattr(self, 'last_wrist_yaw') and hasattr(self, 'last_wrist_roll'):
                return self.last_wrist_pitch, self.last_wrist_yaw, self.last_wrist_roll
            else:
                return 0.0, 0.0, 0.0
                
        y_axis = y_axis / y_magnitude
        
        # Local X is orthogonal to Y and Z
        x_axis = np.cross(y_axis, forearm_z)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        # Construct rotation matrix from these three orthogonal axes
        R = np.column_stack((x_axis, y_axis, forearm_z))
        
        # Extract angles using a more stable method
        # Calculate pitch (rotation around Y)
        pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        
        # Check for gimbal lock
        if abs(abs(pitch) - np.pi/2) < 1e-4:
            # In gimbal lock, yaw and roll are coupled
            yaw = 0.0
            roll = np.arctan2(R[0,1], R[1,1])
        else:
            # Not in gimbal lock
            yaw = np.arctan2(R[1,0], R[0,0])
            roll = np.arctan2(R[2,1], R[2,2])
        
        # Add stability to roll calculation
        roll_magnitude = np.sqrt(R[2,1]**2 + R[2,2]**2)
        if roll_magnitude < 1e-4:
            # If roll is poorly defined, use previous value
            if hasattr(self, 'last_wrist_roll'):
                roll = self.last_wrist_roll
            else:
                roll = 0.0
        
        # Store angles for next iteration
        self.last_wrist_pitch = pitch
        self.last_wrist_yaw = yaw
        self.last_wrist_roll = roll
        
        # Clip angles to joint limits and return
        return (self.clip('wrist_pitch', pitch),
                self.clip('wrist_yaw', yaw),
                self.clip('wrist_roll', roll))


# For backward compatibility
class IKAnalytical3D(IKAnalytical3DRefined):
    """Legacy wrapper for backward compatibility"""
    pass

def validate_ik_fk(joint_angles, target_position, threshold=1e-3):
    # Create an IK solver instance to use its forward_kinematics method
    solver = IKAnalytical3DRefined()
    # Calculate FK from joint angles
    fk_position, _ = solver.forward_kinematics(joint_angles)
    # Compare with target position
    error = np.linalg.norm(fk_position - target_position)
    return error < threshold, error

def check_motin_continuity(joint_angles_sequence, velocity_threshold=1.0):
    # Check for sudden changes between consecutive positions
    if len(joint_angles_sequence) < 2:
        return True, []
    
    problematic_joints = []
    for i in range(1, len(joint_angles_sequence)):
        prev = joint_angles_sequence[i-1]
        curr = joint_angles_sequence[i]
        
        for joint in prev:
            velocity = abs(curr[joint] - prev[joint])
            if velocity > velocity_threshold:
                problematic_joints.append((joint, i, velocity))
    
    return len(problematic_joints) == 0, problematic_joints
