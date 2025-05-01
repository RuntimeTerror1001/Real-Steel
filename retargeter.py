import time
from robot_model import RobotModel
import math

class OneEuroFilter:
    def __init__(self, init_freq=30.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq       = init_freq
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.last_val   = {}
        self.last_d     = {}
        self.last_time  = {}

    def _alpha(self, cutoff, dt):
        # α = 1 / (1 + (1/(2π·cutoff·dt)))
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, key, x, t):
        """
        key:        unique joint name (string)
        x:          raw angle (float)
        t:          current timestamp (float, seconds)
        returns:    filtered angle
        """
        # first call?
        if key not in self.last_val:
            self.last_val[key]  = x
            self.last_d[key]    = 0.0
            self.last_time[key] = t

        # compute dt
        dt = t - self.last_time[key]
        if dt <= 0.0:
            # no time advance → nothing to do
            return self.last_val[key]

        # estimate derivative of x
        d_x = (x - self.last_val[key]) / dt
        # filter the derivative
        a_d   = self._alpha(self.d_cutoff, dt)
        d_hat = a_d * d_x + (1 - a_d) * self.last_d[key]

        # adapt cutoff
        cutoff = self.min_cutoff + self.beta * abs(d_hat)
        a      = self._alpha(cutoff, dt)

        # filter signal
        x_hat = a * x + (1 - a) * self.last_val[key]

        # store for next time
        self.last_val[key]   = x_hat
        self.last_d[key]     = d_hat
        self.last_time[key]  = t

        return x_hat
    
class Retargeter:
    def __init__(self, FPS):
        self.robot_model = RobotModel()
        self.is_calibrated = False

        self.angle_filter = OneEuroFilter(
            init_freq=FPS,        # e.g. 20 or 30
            min_cutoff=1.0,       # suppress noise below ~1 Hz
            beta=0.01,            # adjust responsiveness
            d_cutoff=1.0          # derivative smoothing
        )
        
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
        raw_angles = self.robot_model.inverse_kinematics(landmarks)
        
        # Apply calibration if available
        if self.is_calibrated:
            raw_angles = self.robot_model.apply_calibration(raw_angles)

        now = time.time()
        filtered_angles = {}
        for joint, angle in raw_angles.items():
            filtered_angles[joint] = self.angle_filter.filter(joint, angle, now)

        # 4) Return the filtered result
        return filtered_angles
        
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
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        row = [f"{timestamp:.1f}"]
        for joint in ordered_joints:
            row.append(f"{joint_angles.get(joint, 0.0):.4f}")
            
        return row