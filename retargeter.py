from robot_model import RobotModel


class Retargeter:
    def __init__(self):
        self.robot_model = RobotModel()
        self.is_calibrated = False

        self.prev_angles = {}
        self.alpha = 0.7
        
    def process_frame(self, landmarks):
        """
        Process a single frame of pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of calibrated joint angles for the robot
        """
        if not self.is_calibrated:
            print("Warning: System not calibrated. Results may be inaccurate.")
            
        # Calculate raw joint angles using IK
        joint_angles = self.robot_model.inverse_kinematics(landmarks)
        
        # Apply calibration if available
        if self.is_calibrated:
            joint_angles = self.robot_model.apply_calibration(joint_angles)

        if not self.prev_angles:
            self.prev_angles = joint_angles.copy()

        smoothed_angles = {}
        for joint, value in joint_angles.items():
            prev = self.prev_angles.get(joint, 0.0)
            smooth_val = self.alpha * prev + (1.0 - self.alpha)*value
            smoothed_angles[joint] = smooth_val
            self.prev_angles[joint] = smooth_val
            
        return smoothed_angles
        
    def calibrate(self, landmarks):
        """Calibrate using the current pose landmarks"""
        self.robot_model.calibrate(landmarks)
        self.is_calibrated = True
        
    def generate_csv_row(self, timestamp, joint_angles):
        """
        Format joint angles as a CSV row for MuJoCo.
        
        Args:
            timestamp: Time in seconds
            joint_angles: Dictionary of joint angles
            
        Returns:
            List of values for CSV row
        """
        # Order matters! Must match expected CSV format
        ordered_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint",
            "left_elbow_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint",
            "right_elbow_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint"
        ]
        
        row = [f"{timestamp:.1f}"]
        for joint in ordered_joints:
            row.append(f"{joint_angles.get(joint, 0.0):.4f}")
            
        return row