import numpy as np
import mediapipe as mp

class RobotModel:
    def __init__(self):
        """Initialize the Unitree G1 robot model with correct specifications from MuJoCo XML."""
        
        # Physical dimensions (in meters)
        self.dimensions = {
            "shoulder_width": 0.2004,       # Distance between shoulder joints (from XML)
            "upper_arm_length": 0.2037,  # Shoulder to elbow (combined segments)
            "forearm_length": 0.1843,          # Elbow to wrist (measured from XML)
            "wrist_to_hand": 0.087  # Wrist to hand tip
        }
        
        # Joint axes (from XML axis attributes)
        self.joint_axes = {
            "shoulder_pitch": np.array([0, 1, 0]),  # Y-axis rotation
            "shoulder_roll": np.array([1, 0, 0]),   # X-axis rotation
            "shoulder_yaw": np.array([0, 0, 1]),    # Z-axis rotation
            "elbow": np.array([0, 1, 0]),           # Y-axis rotation 
            "wrist_roll": np.array([1, 0, 0]),      # X-axis rotation
            "wrist_pitch": np.array([0, 1, 0]),     # Y-axis rotation
            "wrist_yaw": np.array([0, 0, 1])        # Z-axis rotation
        }
        
        # Joint limits (in radians, from XML)
        self.joint_limits = {
            "left_shoulder_pitch_joint": (-3.0892, 2.6704),
            "left_shoulder_roll_joint": (-1.5882, 2.2515),
            "left_shoulder_yaw_joint": (-2.618, 2.618),
            "left_elbow_joint": (-1.0472, 2.0944),
            "left_wrist_roll_joint": (-1.97222, 1.97222),
            "left_wrist_pitch_joint": (-1.61443, 1.61443),
            "left_wrist_yaw_joint": (-1.61443, 1.61443),
            
            "right_shoulder_pitch_joint": (-3.0892, 2.6704),
            "right_shoulder_roll_joint": (-2.2515, 1.5882),  # Reversed from left
            "right_shoulder_yaw_joint": (-2.618, 2.618),
            "right_elbow_joint": (-1.0472, 2.0944),
            "right_wrist_roll_joint": (-1.97222, 1.97222),
            "right_wrist_pitch_joint": (-1.61443, 1.61443),
            "right_wrist_yaw_joint": (-1.61443, 1.61443)
        }
        
        # Reference pose from keyframe "stand" (from XML)
        self.reference_pose = {
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.28,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,  # Note: negative for right arm
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.28,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0
        }
        
        # Joint hierarchy from XML (parent-child relationships)
        self.joint_hierarchy = {
            "left_shoulder_pitch_joint": None,  # Root joint
            "left_shoulder_roll_joint": "left_shoulder_pitch_joint",
            "left_shoulder_yaw_joint": "left_shoulder_roll_joint",
            "left_elbow_joint": "left_shoulder_yaw_joint",
            "left_wrist_roll_joint": "left_elbow_joint",
            "left_wrist_pitch_joint": "left_wrist_roll_joint",
            "left_wrist_yaw_joint": "left_wrist_pitch_joint",
            
            # Same for right arm
            "right_shoulder_pitch_joint": None,
            "right_shoulder_roll_joint": "right_shoulder_pitch_joint",
            "right_shoulder_yaw_joint": "right_shoulder_roll_joint",
            "right_elbow_joint": "right_shoulder_yaw_joint",
            "right_wrist_roll_joint": "right_elbow_joint",
            "right_wrist_pitch_joint": "right_wrist_roll_joint",
            "right_wrist_yaw_joint": "right_wrist_pitch_joint"
        }
        
        # Calibration offsets (initialized to zero, set during calibration)
        self.calibration_offsets = {}
        for joint in self.reference_pose.keys():
            self.calibration_offsets[joint] = 0.0

        self.prev_palm_normal = {'left':None, 'right':None}
    
    def transform_to_robot_coords(self, landmark):
        """
        Transform MediaPipe landmark to robot coordinate system.
        
        Args:
            landmark: MediaPipe landmark object
            
        Returns:
            Dictionary with x, y, z in robot coordinates
        """
        # Convert from MediaPipe coordinate system to robot coordinate system
        # MediaPipe: X right, Y down, Z away from camera
        # Robot: X right, Y up, Z forward
        robot_coords = {
            'x': landmark.x,     # X stays the same (right is positive in both)
            'y': -landmark.y,    # Y is inverted (MediaPipe down → Robot up)
            'z': -landmark.z     # Z is inverted (MediaPipe away → Robot toward)
        }
        return robot_coords
    
    def rotation_matrix_x(self, angle):
        """Return 3x3 rotation matrix around X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rotation_matrix_y(self, angle):
        """Return 3x3 rotation matrix around Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rotation_matrix_z(self, angle):
        """Return 3x3 rotation matrix around Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def compute_wrist_angles(self, forearm_z, palm_normal):
        """
        Given:
          forearm_z   : (3,) unit vector = local Z-axis of the wrist (elbow→wrist)
          palm_normal : (3,) unit vector = normal to the “palm” plane

        Returns:
          roll  : rotation about local X
          pitch : rotation about local Y
          yaw   : rotation about local Z
        """

        # 1) Build local Y by projecting palm_normal ⟂ forearm_z
        y_axis = palm_normal - forearm_z * np.dot(palm_normal, forearm_z)
        y_axis /= np.linalg.norm(y_axis)

        # 2) Local X is orthogonal to Y and Z
        x_axis = np.cross(y_axis, forearm_z)

        # 3) Assemble rotation matrix R whose columns
        #    are [x_axis, y_axis, forearm_z], i.e. world↔local
        R = np.column_stack((x_axis, y_axis, forearm_z))

        # 4) Extract Euler angles (X–Y–Z, intrinsic = roll→pitch→yaw):
        #    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
        #
        #    From that convention:
        roll  = np.arctan2( R[2,1],  R[2,2] )
        pitch = np.arctan2(-R[2,0],  np.sqrt(R[2,1]**2 + R[2,2]**2) )
        yaw   = np.arctan2( R[1,0],  R[0,0] )

        return roll, pitch, yaw
    
    def compute_euler_yxz(self, R, eps=1e-6):
        """
        Given R = R_z(yaw) @ R_x(roll) @ R_y(pitch),
        extract intrinsic Y–X–Z angles (pitch, roll, yaw),
        with a gimbal-lock fallback.
        """

        # 1) roll = β = arcsin(R[2,1])
        sin_roll = np.clip(R[2, 1], -1.0, 1.0)
        roll     = np.arcsin(sin_roll)

        # 2) compute cos(β) to check for gimbal lock
        cos_roll = np.sqrt(1 - sin_roll*sin_roll)

        if cos_roll < eps:
            # --- GIMBAL-LOCK PATH ---
            # roll ≈ ±90°, so R[2,1]=±1 and rows/cols collapse.
            # We lose one DOF; pick yaw = 0 and compute pitch from R[1,0]/R[0,0].
            yaw   = 0.0
            pitch = np.arctan2(R[1, 0], R[0, 0])
        else:
            # --- NORMAL PATH ---
            # pitch = α from the X–Z submatrix
            pitch = np.arctan2(-R[2, 0], R[2, 2])
            # yaw   = γ from the top-left 2×2
            yaw   = np.arctan2(-R[0, 1], R[1, 1])

        return pitch, roll, yaw
    
    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector positions from joint angles using the correct joint hierarchy.
        
        Args:
            joint_angles: Dictionary of joint angles in radians
            
        Returns:
            Dictionary of joint positions in 3D space
        """
        positions = {}
        
        # Base positions (torso)
        torso_pos = np.array([0, 0, 0])
        
        # Set shoulder positions relative to torso
        left_shoulder_pos = torso_pos + np.array([-self.dimensions["shoulder_width"]/2, 0, 0])
        right_shoulder_pos = torso_pos + np.array([self.dimensions["shoulder_width"]/2, 0, 0])
        
        positions["torso"] = torso_pos
        positions["left_shoulder"] = left_shoulder_pos
        positions["right_shoulder"] = right_shoulder_pos
        
        # Left arm chain 
        # 1. Shoulder pitch (Y-axis)
        l_pitch = joint_angles.get("left_shoulder_pitch_joint", 0)
        R_l_pitch = self.rotation_matrix_y(l_pitch)
        
        # 2. Shoulder roll (X-axis) - applied after pitch
        l_roll = joint_angles.get("left_shoulder_roll_joint", 0)
        R_l_roll = self.rotation_matrix_x(l_roll)
        
        # 3. Shoulder yaw (Z-axis) - applied after pitch and roll
        l_yaw = joint_angles.get("left_shoulder_yaw_joint", 0)
        R_l_yaw = self.rotation_matrix_z(l_yaw)
        
        # Combined shoulder rotation (CORRECT ORDER: pitch → roll → yaw)
        R_l_shoulder = R_l_pitch @ R_l_roll @ R_l_yaw
        
        # Upper arm vector (shoulder to elbow) in neutral pose
        l_upper_arm_vec = np.array([0, 0, self.dimensions["upper_arm_length"]])
        
        # Apply shoulder rotation to upper arm
        l_upper_arm_rotated = R_l_shoulder @ l_upper_arm_vec
        
        # Calculate elbow position
        l_elbow_pos = left_shoulder_pos + l_upper_arm_rotated
        positions["left_elbow"] = l_elbow_pos
        
        # 4. Elbow (Y-axis) - applied after all shoulder rotations
        l_elbow = joint_angles.get("left_elbow_joint", 0)
        R_l_elbow = self.rotation_matrix_y(l_elbow)
        
        # Combined arm rotation (shoulder + elbow)
        R_l_arm = R_l_shoulder @ R_l_elbow
        
        # Forearm vector (elbow to wrist) in neutral pose
        l_forearm_vec = np.array([0, 0, self.dimensions["forearm_length"]])
        
        # Apply combined rotation to forearm
        l_forearm_rotated = R_l_arm @ l_forearm_vec
        
        # Calculate wrist position
        l_wrist_pos = l_elbow_pos + l_forearm_rotated
        positions["left_wrist"] = l_wrist_pos
        
        # Right arm chain 
        # 1. Shoulder pitch (Y-axis)
        r_pitch = joint_angles.get("right_shoulder_pitch_joint", 0)
        R_r_pitch = self.rotation_matrix_y(r_pitch)
        
        # 2. Shoulder roll (X-axis) - applied after pitch
        r_roll = joint_angles.get("right_shoulder_roll_joint", 0)
        R_r_roll = self.rotation_matrix_x(r_roll)
        
        # 3. Shoulder yaw (Z-axis) - applied after pitch and roll
        r_yaw = joint_angles.get("right_shoulder_yaw_joint", 0)
        R_r_yaw = self.rotation_matrix_z(r_yaw)
        
        # Combined shoulder rotation (CORRECT ORDER: pitch → roll → yaw)
        R_r_shoulder = R_r_pitch @ R_r_roll @ R_r_yaw
        
        # Upper arm vector (shoulder to elbow) in neutral pose
        r_upper_arm_vec = np.array([0, 0, self.dimensions["upper_arm_length"]])
        
        # Apply shoulder rotation to upper arm
        r_upper_arm_rotated = R_r_shoulder @ r_upper_arm_vec
        
        # Calculate elbow position
        r_elbow_pos = right_shoulder_pos + r_upper_arm_rotated
        positions["right_elbow"] = r_elbow_pos
        
        # 4. Elbow (Y-axis) - applied after all shoulder rotations
        r_elbow = joint_angles.get("right_elbow_joint", 0)
        R_r_elbow = self.rotation_matrix_y(r_elbow)
        
        # Combined arm rotation (shoulder + elbow)
        R_r_arm = R_r_shoulder @ R_r_elbow
        
        # Forearm vector (elbow to wrist) in neutral pose
        r_forearm_vec = np.array([0, 0, self.dimensions["forearm_length"]])
        
        # Apply combined rotation to forearm
        r_forearm_rotated = R_r_arm @ r_forearm_vec
        
        # Calculate wrist position
        r_wrist_pos = r_elbow_pos + r_forearm_rotated
        positions["right_wrist"] = r_wrist_pos
        
        return positions
    
    def inverse_kinematics(self, landmarks):
        """Complete revision of the inverse kinematics algorithm."""
        joint_angles = {}
        mp_pose = mp.solutions.pose.PoseLandmark
        
        # Transform landmarks with correct mirroring
        l_shoulder = self.transform_to_robot_coords(landmarks[mp_pose.LEFT_SHOULDER.value])
        l_elbow = self.transform_to_robot_coords(landmarks[mp_pose.LEFT_ELBOW.value])
        l_wrist = self.transform_to_robot_coords(landmarks[mp_pose.LEFT_WRIST.value])
        
        r_shoulder = self.transform_to_robot_coords(landmarks[mp_pose.RIGHT_SHOULDER.value])
        r_elbow = self.transform_to_robot_coords(landmarks[mp_pose.RIGHT_ELBOW.value])
        r_wrist = self.transform_to_robot_coords(landmarks[mp_pose.RIGHT_WRIST.value])
        
        # Convert to numpy arrays
        l_shoulder_pos = np.array([l_shoulder['x'], l_shoulder['y'], l_shoulder['z']])
        l_elbow_pos = np.array([l_elbow['x'], l_elbow['y'], l_elbow['z']])
        l_wrist_pos = np.array([l_wrist['x'], l_wrist['y'], l_wrist['z']])
        
        r_shoulder_pos = np.array([r_shoulder['x'], r_shoulder['y'], r_shoulder['z']])
        r_elbow_pos = np.array([r_elbow['x'], r_elbow['y'], r_elbow['z']])
        r_wrist_pos = np.array([r_wrist['x'], r_wrist['y'], r_wrist['z']])
        
        # Calculate vectors and normalize
        l_upper_arm_vec = l_elbow_pos - l_shoulder_pos
        l_forearm_vec = l_wrist_pos - l_elbow_pos
        
        r_upper_arm_vec = r_elbow_pos - r_shoulder_pos
        r_forearm_vec = r_wrist_pos - r_elbow_pos
        
        # Normalize vectors
        l_upper_arm_len = np.linalg.norm(l_upper_arm_vec)
        l_forearm_len = np.linalg.norm(l_forearm_vec)
        l_upper_arm_norm = l_upper_arm_vec / l_upper_arm_len if l_upper_arm_len > 0 else np.array([0, -1, 0])
        l_forearm_norm = l_forearm_vec / l_forearm_len if l_forearm_len > 0 else np.array([0, -1, 0])
        
        r_upper_arm_len = np.linalg.norm(r_upper_arm_vec)
        r_forearm_len = np.linalg.norm(r_forearm_vec)
        r_upper_arm_norm = r_upper_arm_vec / r_upper_arm_len if r_upper_arm_len > 0 else np.array([0, -1, 0])
        r_forearm_norm = r_forearm_vec / r_forearm_len if r_forearm_len > 0 else np.array([0, -1, 0])
        
        # ===== LEFT ARM =====
        
        # 1. Elbow angle - simple angle between arm segments
        l_elbow_angle = np.arccos(np.clip(np.dot(l_upper_arm_norm, l_forearm_norm), -1.0, 1.0))
        joint_angles["left_elbow_joint"] = -l_elbow_angle
        
        # 2. Shoulder pitch (Y-axis rotation) - project onto Y-Z plane
        # Angle from +Z axis (downward) in the Y-Z plane
        y_z_proj = np.array([0, l_upper_arm_norm[1], l_upper_arm_norm[2]])
        y_z_len = np.linalg.norm(y_z_proj)
        if y_z_len > 0.01:
            y_z_proj /= y_z_len
            # For shoulders, negative is arm up
            l_shoulder_pitch = - np.arctan2(y_z_proj[1], y_z_proj[2])
        else:
            # Handle the case where arm is pointing along X axis
            l_shoulder_pitch = 0.0
        
        # 3. Shoulder roll (X-axis rotation)
        # After applying pitch, calculate roll as deviation from Y-Z plane
        # Zero is arm in the Y-Z plane, positive roll moves arm outward
        x_proj = l_upper_arm_norm[0]
        l_shoulder_roll = np.arcsin(np.clip(x_proj, -1.0, 1.0))
        
        # 4. Shoulder yaw (Z-axis rotation)
        x_z_proj = np.array([l_upper_arm_norm[0], l_upper_arm_norm[2]])
        x_z_len = np.linalg.norm(x_z_proj)
        if x_z_len > 0.01:
            x_z_proj /= x_z_len
            #positive when swings rightward, negative leftward
            l_shoulder_yaw = np.arctan2(x_z_proj[0], x_z_proj[1])
        else:
            l_shoulder_yaw = 0.0

        R_ly = self.rotation_matrix_y(l_shoulder_pitch)
        R_lx = self.rotation_matrix_x(l_shoulder_roll)
        R_lz = self.rotation_matrix_z(l_shoulder_yaw)
        R_lshoulder = R_lz @ R_lx @ R_ly

        l_spitch, l_sroll, l_syaw = self.compute_euler_yxz(R_lshoulder)
        
        # Assign angles
        joint_angles["left_shoulder_pitch_joint"] = l_spitch
        joint_angles["left_shoulder_roll_joint"] = l_sroll
        joint_angles["left_shoulder_yaw_joint"] = l_syaw
        
        # ===== RIGHT ARM =====
        
        # 1. Elbow angle
        r_elbow_angle = np.arccos(np.clip(np.dot(r_upper_arm_norm, r_forearm_norm), -1.0, 1.0))
        joint_angles["right_elbow_joint"] = -r_elbow_angle
        
        # 2. Shoulder pitch
        y_z_proj = np.array([0, r_upper_arm_norm[1], r_upper_arm_norm[2]])
        y_z_len = np.linalg.norm(y_z_proj)
        if y_z_len > 0.01:
            y_z_proj /= y_z_len
            r_shoulder_pitch = - np.arctan2(y_z_proj[1], y_z_proj[2])
        else:
            r_shoulder_pitch = 0.0
        
        # 3. Shoulder roll - note sign change for right arm
        x_proj = r_upper_arm_norm[0]
        r_shoulder_roll = np.arcsin(np.clip(x_proj, -1.0, 1.0))  # Negative for right arm
        
        # 4. Shoulder yaw
        x_z_proj = np.array([r_upper_arm_norm[0], r_upper_arm_norm[2]])
        x_z_len = np.linalg.norm(x_z_proj)
        if x_z_len > 0.01:
            x_z_proj /= x_z_len
            r_shoulder_yaw = -np.arctan2(x_z_proj[0], x_z_proj[1])
        else:
            r_shoulder_yaw = 0.0

        R_ry = self.rotation_matrix_y(r_shoulder_pitch)
        R_rx = self.rotation_matrix_x(r_shoulder_roll)
        R_rz = self.rotation_matrix_z(r_shoulder_yaw)
        R_rshoulder = R_rz @ R_rx @ R_ry

        r_spitch, r_sroll, r_syaw = self.compute_euler_yxz(R_rshoulder)
        
        # Assign angles
        joint_angles["right_shoulder_pitch_joint"] = r_spitch
        joint_angles["right_shoulder_roll_joint"] = r_sroll
        joint_angles["right_shoulder_yaw_joint"] = r_syaw
        
        # Wrist angles 
        l_palm_norm = np.cross(l_upper_arm_norm, l_forearm_norm)
        l_palm_norm /= np.linalg.norm(l_palm_norm)
        prev = self.prev_palm_normal['left']
        if prev is not None and np.dot(prev, l_palm_norm) < 0:
            l_palm_norm = - l_palm_norm
        self.prev_palm_normal['left'] = l_palm_norm

        r_palm_norm = np.cross(r_upper_arm_norm, r_forearm_norm)
        r_palm_norm /= np.linalg.norm(r_palm_norm)
        prev = self.prev_palm_normal['right']
        if prev is not None and np.dot(prev, r_palm_norm) < 0:
            r_palm_norm = - r_palm_norm
        self.prev_palm_normal['right'] = r_palm_norm

        l_wr, l_wp, l_wy = self.compute_wrist_angles(l_forearm_norm, l_palm_norm)
        joint_angles["left_wrist_roll_joint"] = 0
        joint_angles["left_wrist_pitch_joint"] = 0
        joint_angles["left_wrist_yaw_joint"] = 0

        r_wr, r_wp, r_wy = self.compute_wrist_angles(r_forearm_norm, r_palm_norm)
        joint_angles["right_wrist_roll_joint"] = 0
        joint_angles["right_wrist_pitch_joint"] = 0
        joint_angles["right_wrist_yaw_joint"] = 0
        
        # Apply joint limits
        for joint, angle in joint_angles.items():
            if joint in self.joint_limits:
                min_limit, max_limit = self.joint_limits[joint]
                joint_angles[joint] = np.clip(angle, min_limit, max_limit)
        
        return joint_angles
    
    def calibrate(self, landmarks):
        """
        Calibrate the system using the current pose as reference.
        
        Args:
            landmarks: MediaPipe landmarks in calibration pose
        """
        # Calculate raw joint angles from the calibration pose
        raw_angles = self.inverse_kinematics(landmarks)
        
        # Calculate offsets to match the expected reference pose
        self.calibration_offsets = {}
        
        for joint, expected in self.reference_pose.items():
            if joint in raw_angles:
                # Offset = expected reference value - actual value
                self.calibration_offsets[joint] = expected - raw_angles[joint]
                print(f"Calibration offset for {joint}: {self.calibration_offsets[joint]:.4f}")
        
        print("Calibration complete!")
    
    def apply_calibration(self, joint_angles):
        """
        Apply calibration offsets to calculated joint angles.
        
        Args:
            joint_angles: Raw calculated joint angles
            
        Returns:
            Calibrated joint angles
        """
        calibrated = {}
        
        for joint, angle in joint_angles.items():
            if joint in self.calibration_offsets:
                calibrated[joint] = angle + self.calibration_offsets[joint]
            else:
                calibrated[joint] = angle
                
        return calibrated