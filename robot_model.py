import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

class RobotModel:
    _ref = np.array([0, -1, 0])  # “down” in robot coords
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
    
    def shoulder_angles(self, u, f, is_left=True):
        """
        u: unit upper-arm vector (shoulder→elbow)
        f: unit forearm vector  (elbow→wrist)
        is_left: True for left arm, False for right
        returns: (roll, pitch, yaw) with your sign conventions
        """
        # 1) swing: rotate “down” → u
        axis  = np.cross(self._ref, u)
        angle = np.arccos(np.clip(np.dot(self._ref, u), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            # nearly aligned; pick any perpendicular axis
            axis = np.array([1,0,0])
        else:
            axis = axis / np.linalg.norm(axis)
        q_swing = R.from_rotvec(axis * angle)

        # 2) twist: rotate around u by the dynamic yaw you already compute
        def compute_yaw(u, f):
            """
            u: unit vector along upper arm
            f: unit vector along forearm
            returns: signed twist angle around u (radians)
            """
            # 1) pick a stable “not‐parallel” axis
            arb = np.array([0, 0, 1])
            if abs(np.dot(u, arb)) > 0.9:
                arb = np.array([0, 1, 0])

            # 2) Gram–Schmidt to get two perpendicular basis vectors in plane ⟂ u
            v = arb - u * np.dot(arb, u)
            v /= np.linalg.norm(v)
            w = np.cross(u, v)

            # 3) project forearm vector onto that plane and measure angle
            x = np.dot(f, v)
            y = np.dot(f, w)
            return np.arctan2(y, x)
        
        raw_yaw = compute_yaw(u, f)
        # for right arm we'll flip sign later
        q_twist = R.from_rotvec(u * raw_yaw)

        # 3) combined orientation
        q_total = q_twist * q_swing
        # test = q_total.apply(self._ref)
        # print("swinged+twisted 'down' →", test, "should equal u:", u)

        # 4) intrinsic Y-X-Z decomposition
        #    (i.e. R = R_y(roll) · R_x(pitch) · R_z(yaw))
        pitch, roll, yaw = q_total.as_euler('xyz', degrees=False)
        # print("raw Euler X,Y,Z:", pitch, roll, yaw)

        # 5) apply your sign rules             # raising arm should decrease pitch
        if is_left:
            pitch = +pitch
            roll = +roll           # positive when moving left
            yaw  = -yaw            # positive for palm forward
        else:
            pitch = -pitch
            roll = -roll           # negative when moving right
            yaw  = +yaw            # negative for palm forward

        return roll, pitch, yaw
    
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
        """
        Inverse kinematics for both arms:
        - elbow via simple dot‐product angle
        - shoulder roll/pitch/yaw via quaternion→Euler (YXZ) decomposition
        - wrist via existing compute_wrist_angles
        """

        joint_angles = {}
        mp_pose = mp.solutions.pose.PoseLandmark

        # --- 1) get 3D positions in robot coords ---
        def to_np(lm_idx):
            lm = self.transform_to_robot_coords(landmarks[lm_idx.value])
            return np.array([lm['x'], lm['y'], lm['z']])

        l_sh = to_np(mp_pose.LEFT_SHOULDER)
        l_el = to_np(mp_pose.LEFT_ELBOW)
        l_wr = to_np(mp_pose.LEFT_WRIST)
        r_sh = to_np(mp_pose.RIGHT_SHOULDER)
        r_el = to_np(mp_pose.RIGHT_ELBOW)
        r_wr = to_np(mp_pose.RIGHT_WRIST)

        # --- 2) compute unit upper-arm & forearm vectors ---
        def unit(a, b):
            v = b - a
            n = np.linalg.norm(v)
            return (v / n) if n > 1e-6 else np.array([0, -1, 0])

        l_u = unit(l_sh, l_el)
        l_f = unit(l_el, l_wr)
        r_u = unit(r_sh, r_el)
        r_f = unit(r_el, r_wr)

        # --- 3) helper to compute dynamic twist (yaw) around u ---
        def compute_yaw(u, f):
            """
            u: unit vector along upper arm
            f: unit vector along forearm
            returns: signed twist angle around u (radians)
            """
            # 1) pick a stable “not‐parallel” axis
            arb = np.array([0, 0, 1])
            if abs(np.dot(u, arb)) > 0.9:
                arb = np.array([0, 1, 0])

            # 2) Gram–Schmidt to get two perpendicular basis vectors in plane ⟂ u
            v = arb - u * np.dot(arb, u)
            v /= np.linalg.norm(v)
            w = np.cross(u, v)

            # 3) project forearm vector onto that plane and measure angle
            x = np.dot(f, v)
            y = np.dot(f, w)
            return np.arctan2(y, x)

        # --- LEFT ARM ---
        # 1) elbow
        l_elbow_ang = np.arccos(np.clip(np.dot(l_u, l_f), -1.0, 1.0))
        joint_angles["left_elbow_joint"] = -l_elbow_ang

        l_s_roll, l_s_pitch, l_s_yaw = self.shoulder_angles(l_u, l_f)

        # 2) roll (swing left/right) about Y (vertical)
        l_roll = np.arctan2(l_u[0], -l_u[1])

        # 3) pitch (forward/back) about X (horizontal)
        l_pitch = -np.arctan2(l_u[2], -l_u[1])

        # 4) yaw (true twist around upper-arm axis)
        l_yaw = compute_yaw(l_u, l_f)

        joint_angles["left_shoulder_roll_joint"]  = l_s_roll
        joint_angles["left_shoulder_pitch_joint"] = l_s_pitch
        joint_angles["left_shoulder_yaw_joint"]   = l_s_yaw

        # --- RIGHT ARM ---
        r_elbow_ang = np.arccos(np.clip(np.dot(r_u, r_f), -1.0, 1.0))
        joint_angles["right_elbow_joint"] = -r_elbow_ang

        r_s_roll, r_s_pitch, r_s_yaw = self.shoulder_angles(r_u,r_f,is_left=False)

        r_roll = np.arctan2(r_u[0], -r_u[1])
        r_pitch = -np.arctan2(r_u[2], -r_u[1])
        r_yaw = compute_yaw(r_u, r_f)

        joint_angles["right_shoulder_roll_joint"]  = r_roll
        joint_angles["right_shoulder_pitch_joint"] = r_pitch
        joint_angles["right_shoulder_yaw_joint"]   = r_yaw

        # --- WRIST ANGLES (unchanged) ---
        def palm_norm(u, f, prev):
            pn = np.cross(u, f)
            norm = np.linalg.norm(pn)
            if norm > 1e-6:
                pn /= norm
            if prev is not None and np.dot(prev, pn) < 0:
                pn = -pn
            return pn

        l_pn = palm_norm(l_u, l_f, self.prev_palm_normal['left'])
        self.prev_palm_normal['left'] = l_pn
        r_pn = palm_norm(r_u, r_f, self.prev_palm_normal['right'])
        self.prev_palm_normal['right'] = r_pn

        l_wr, l_wp, l_wy = self.compute_wrist_angles(l_f, l_pn)
        r_wr, r_wp, r_wy = self.compute_wrist_angles(r_f, r_pn)

        joint_angles["left_wrist_roll_joint"]  = 0
        joint_angles["left_wrist_pitch_joint"] = 0
        joint_angles["left_wrist_yaw_joint"]   = 0
        joint_angles["right_wrist_roll_joint"]  = 0
        joint_angles["right_wrist_pitch_joint"] = 0
        joint_angles["right_wrist_yaw_joint"]   = 0

        # --- 4) apply joint limits ---
        for joint, ang in joint_angles.items():
            if joint in self.joint_limits:
                mn, mx = self.joint_limits[joint]
                joint_angles[joint] = np.clip(ang, mn, mx)

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
    

"""
    def inverse_kinematics(self, landmarks):
    FIRST METHOD - OFF PITCH:
    joint_angles = {}
        mp_pose = mp.solutions.pose.PoseLandmark

        # --- 1) get 3D positions in robot coords ---
        def to_np(lm_idx):
            lm = self.transform_to_robot_coords(landmarks[lm_idx.value])
            return np.array([lm['x'], lm['y'], lm['z']])

        l_sh = to_np(mp_pose.LEFT_SHOULDER)
        l_el = to_np(mp_pose.LEFT_ELBOW)
        l_wr = to_np(mp_pose.LEFT_WRIST)
        r_sh = to_np(mp_pose.RIGHT_SHOULDER)
        r_el = to_np(mp_pose.RIGHT_ELBOW)
        r_wr = to_np(mp_pose.RIGHT_WRIST)

        # --- 2) compute unit upper-arm & forearm vectors ---
        def unit(a, b):
            v = b - a
            n = np.linalg.norm(v)
            return (v / n) if n > 1e-6 else np.array([0, -1, 0])

        l_u = unit(l_sh, l_el)
        l_f = unit(l_el, l_wr)
        r_u = unit(r_sh, r_el)
        r_f = unit(r_el, r_wr)

        # --- 3) helper to compute dynamic twist (yaw) around u ---
        def compute_yaw(u, f):
            # pick a world axis not too parallel to u
            world = np.array([1, 0, 0]) if abs(u.dot([1,0,0])) < 0.9 else np.array([0, 0, 1])
            # project into plane ⟂ u
            ref = world - u * np.dot(u, world)
            ref /= np.linalg.norm(ref)
            proj = f - u * np.dot(f, u)
            proj /= np.linalg.norm(proj) if np.linalg.norm(proj) > 1e-6 else 1.0
            cross = np.cross(ref, proj)
            return -np.arctan2(np.dot(cross, u), np.dot(ref, proj))

        # --- LEFT ARM ---
        # 1) elbow
        l_elbow_ang = np.arccos(np.clip(np.dot(l_u, l_f), -1.0, 1.0))
        joint_angles["left_elbow_joint"] = -l_elbow_ang

        # 2) roll (swing left/right) about Y (vertical)
        l_roll = np.arctan2(l_u[0], -l_u[1])

        # 3) pitch (forward/back) about X (horizontal)
        l_pitch = -np.arctan2(l_u[2], -l_u[1])

        # 4) yaw (true twist around upper-arm axis)
        l_yaw = compute_yaw(l_u, l_f)

        joint_angles["left_shoulder_roll_joint"]  = l_roll
        joint_angles["left_shoulder_pitch_joint"] = l_pitch
        joint_angles["left_shoulder_yaw_joint"]   = l_yaw

        # --- RIGHT ARM ---
        r_elbow_ang = np.arccos(np.clip(np.dot(r_u, r_f), -1.0, 1.0))
        joint_angles["right_elbow_joint"] = -r_elbow_ang

        r_roll = np.arctan2(r_u[0], -r_u[1])
        r_pitch = -np.arctan2(r_u[2], -r_u[1])
        r_yaw = compute_yaw(r_u, r_f)

        joint_angles["right_shoulder_roll_joint"]  = r_roll
        joint_angles["right_shoulder_pitch_joint"] = r_pitch
        joint_angles["right_shoulder_yaw_joint"]   = r_yaw

        # --- WRIST ANGLES (unchanged) ---
        def palm_norm(u, f, prev):
            pn = np.cross(u, f)
            norm = np.linalg.norm(pn)
            if norm > 1e-6:
                pn /= norm
            if prev is not None and np.dot(prev, pn) < 0:
                pn = -pn
            return pn

        l_pn = palm_norm(l_u, l_f, self.prev_palm_normal['left'])
        self.prev_palm_normal['left'] = l_pn
        r_pn = palm_norm(r_u, r_f, self.prev_palm_normal['right'])
        self.prev_palm_normal['right'] = r_pn

        l_wr, l_wp, l_wy = self.compute_wrist_angles(l_f, l_pn)
        r_wr, r_wp, r_wy = self.compute_wrist_angles(r_f, r_pn)

        joint_angles["left_wrist_roll_joint"]  = 0
        joint_angles["left_wrist_pitch_joint"] = 0
        joint_angles["left_wrist_yaw_joint"]   = 0
        joint_angles["right_wrist_roll_joint"]  = 0
        joint_angles["right_wrist_pitch_joint"] = 0
        joint_angles["right_wrist_yaw_joint"]   = 0

        # --- 4) apply joint limits ---
        for joint, ang in joint_angles.items():
            if joint in self.joint_limits:
                mn, mx = self.joint_limits[joint]
                joint_angles[joint] = np.clip(ang, mn, mx)

        return joint_angles

"""