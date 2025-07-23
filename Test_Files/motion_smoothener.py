import numpy as np
import csv
import pandas as pd

class MotionSmoothener:
    def __init__(self):
        # Define the required joint order for the CSV
        self.joint_order = [
            'timestamp',
            'left_shoulder_pitch_joint', 'left_shoulder_yaw_joint', 'left_shoulder_roll_joint',
            'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_yaw_joint', 'right_shoulder_roll_joint',
            'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint'
        ]

        # Joint limits (in radians)
        self.joint_limits = {
            'left_shoulder_pitch_joint': (-2.0, 2.0),
            'left_shoulder_yaw_joint': (-2.0, 2.0),
            'left_shoulder_roll_joint': (-2.0, 2.0),
            'left_elbow_joint': (0.0, 2.5),
            'left_wrist_pitch_joint': (-2.0, 2.0),
            'left_wrist_yaw_joint': (-2.0, 2.0),
            'left_wrist_roll_joint': (-2.0, 2.0),
            'right_shoulder_pitch_joint': (-2.0, 2.0),
            'right_shoulder_yaw_joint': (-2.0, 2.0),
            'right_shoulder_roll_joint': (-2.0, 2.0),
            'right_elbow_joint': (0.0, 2.5),
            'right_wrist_pitch_joint': (-2.0, 2.0),
            'right_wrist_yaw_joint': (-2.0, 2.0),
            'right_wrist_roll_joint': (-2.0, 2.0)
        }

        # Motion constraints
        self.max_joint_velocity = 2.0  # rad/s
        self.max_joint_acceleration = 1.0  # rad/s^2
        self.time_step = 0.1  # seconds

        # Known working poses
        self.initial_pose = {
            'left_shoulder_pitch_joint': 0.2,
            'left_shoulder_roll_joint': 0.2,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.28,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.2,
            'right_shoulder_roll_joint': -0.2,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.28,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0
        }

        self.block_pose = {
            'left_shoulder_pitch_joint': -1.25,
            'left_shoulder_roll_joint': 0.2,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.32,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': -1.15,
            'right_shoulder_roll_joint': -0.1,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.32,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0
        }

    def safe_copy(d):
        return d.copy() if d is not None else {}
    def check_joint_limits(self, pose):
        """Check if all joint angles are within their limits."""
        for joint, value in pose.items():
            if joint == 'timestamp':
                continue
            min_val, max_val = self.joint_limits[joint]
            if value < min_val or value > max_val:
                return False, joint, value, min_val, max_val
        return True, None, None, None, None

    def enforce_joint_limits(self, pose):
        """Clip joint angles to their limits."""
        clipped_pose = self.safe_copy(pose)
        for joint, value in pose.items():
            if joint == 'timestamp':
                continue
            min_val, max_val = self.joint_limits[joint]
            clipped_pose[joint] = np.clip(value, min_val, max_val)
        return clipped_pose

    def check_motion_constraints(self, prev_pose, current_pose, prev_velocity=None):
        """Check if motion between poses satisfies velocity and acceleration constraints."""
        if prev_pose is None:
            return True, None

        # Calculate joint velocities
        current_velocity = {}
        for joint in prev_pose.keys():
            if joint == 'timestamp':
                continue
            current_velocity[joint] = (current_pose[joint] - prev_pose[joint]) / self.time_step

        # Check velocity constraints
        for joint, velocity in current_velocity.items():
            if abs(velocity) > self.max_joint_velocity:
                return False, f"Velocity constraint violated for {joint}: {velocity} rad/s"

        # Check acceleration constraints if previous velocity is provided
        if prev_velocity is not None:
            for joint in current_velocity.keys():
                if joint in prev_velocity:
                    acceleration = (current_velocity[joint] - prev_velocity[joint]) / self.time_step
                    if abs(acceleration) > self.max_joint_acceleration:
                        return False, f"Acceleration constraint violated for {joint}: {acceleration} rad/s^2"

        return True, current_velocity

    def interpolate_poses(self, start_pose, end_pose, num_frames, start_time=0.0):
        """Interpolate between two poses using cosine interpolation with motion constraints."""
        frames = []
        prev_pose = None
        prev_velocity = None

        for frame in range(num_frames):
            t = frame / num_frames
            # Smooth interpolation using cosine
            t_smooth = (1 - np.cos(t * np.pi)) / 2
            
            # Interpolate between poses
            current_pose = {'timestamp': round(start_time + frame * 0.1, 2)}
            for joint in start_pose.keys():
                if joint != 'timestamp':
                    start_val = start_pose[joint]
                    end_val = end_pose[joint]
                    current_pose[joint] = start_val + (end_val - start_val) * t_smooth
            
            # Enforce joint limits
            current_pose = self.enforce_joint_limits(current_pose)
            
            # Check motion constraints
            if prev_pose is not None:
                valid, velocity = self.check_motion_constraints(prev_pose, current_pose, prev_velocity)
                if not valid:
                    print(f"Warning: {velocity}")
                    # Adjust the pose to satisfy constraints (simplified approach)
                    for joint in current_pose.keys():
                        if joint == 'timestamp':
                            continue
                        max_change = self.max_joint_velocity * self.time_step
                        current_pose[joint] = prev_pose[joint] + np.clip(
                            current_pose[joint] - prev_pose[joint], 
                            -max_change, 
                            max_change
                        )
                prev_velocity = velocity
            else:
                prev_velocity = None
            
            frames.append(current_pose)
            prev_pose = current_pose

        return frames

    def load_pose_from_csv(self, csv_path, timestamp=0.0):
        """Load a pose from a CSV file at a specific timestamp."""
        df = pd.read_csv(csv_path)
        pose_data = df[df['timestamp'] == timestamp].iloc[0]
        pose = {col: pose_data[col] for col in df.columns if col != 'timestamp'}
        
        # Validate joint limits
        valid, joint, value, min_val, max_val = self.check_joint_limits(pose)
        if not valid:
            print(f"Warning: Joint {joint} value {value} outside limits [{min_val}, {max_val}]")
            pose = self.enforce_joint_limits(pose)
        
        return pose

    def generate_motion_sequence(self, sequence, output_csv):
        """
        Generate a smooth motion sequence from a list of (pose, num_frames) tuples.
        sequence: list of tuples (pose_dict, num_frames)
        """
        results = []
        current_time = 0.0
        
        for i in range(len(sequence)-1):
            start_pose, num_frames = sequence[i]
            end_pose = sequence[i+1][0]
            
            # Validate poses
            for pose in [start_pose, end_pose]:
                valid, joint, value, min_val, max_val = self.check_joint_limits(pose)
                if not valid:
                    print(f"Warning: Joint {joint} value {value} outside limits [{min_val}, {max_val}]")
                    pose = self.enforce_joint_limits(pose)
            
            # Generate interpolated frames
            frames = self.interpolate_poses(start_pose, end_pose, num_frames, current_time)
            results.extend(frames)
            current_time += num_frames * 0.1

        # Write to CSV with correct column order
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.joint_order)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"Motion sequence written to {output_csv}")
        return results

    def generate_smooth_block_motion(self, output_csv='smooth_block_motion.csv'):
        """Generate a smooth motion sequence: initial -> block -> initial"""
        sequence = [
            (self.initial_pose, 10),  # Hold initial pose
            (self.block_pose, 20),    # Move to block
            (self.initial_pose, 20)   # Return to initial
        ]
        return self.generate_motion_sequence(sequence, output_csv)

    def create_custom_sequence(self, poses, frames_between, output_csv='custom_motion.csv'):
        """
        Create a custom motion sequence from a list of poses and frames between them.
        poses: list of pose dictionaries
        frames_between: list of number of frames between poses
        """
        if len(poses) != len(frames_between) + 1:
            raise ValueError("Number of poses must be one more than number of frame counts")
        
        sequence = [(pose, frames) for pose, frames in zip(poses, frames_between + [0])]
        return self.generate_motion_sequence(sequence, output_csv)

    def validate_csv(self, csv_path):
        """Validate a CSV file to ensure all joint angles are within limits."""
        df = pd.read_csv(csv_path)
        valid = True
        
        for joint in self.joint_limits:
            if joint in df.columns:
                min_val, max_val = self.joint_limits[joint]
                if df[joint].min() < min_val or df[joint].max() > max_val:
                    print(f"Warning: {joint} has values outside limits [{min_val}, {max_val}]")
                    valid = False
        
        if valid:
            print(f"CSV file {csv_path} is valid with all joint angles within limits.")
        else:
            print(f"CSV file {csv_path} has out-of-range joint angles.")
        
        return valid

if __name__ == "__main__":
    # Example usage
    smoothener = MotionSmoothener()
    
    # Generate basic block motion
    smoothener.generate_smooth_block_motion("test_motion.csv")
    
    # Validate the generated CSV
    smoothener.validate_csv("test_motion.csv")
    
    # Example of custom sequence
    custom_poses = [
        smoothener.initial_pose,
        smoothener.block_pose,
        smoothener.initial_pose
    ]
    frames = [15, 25]  # Frames between poses
    smoothener.create_custom_sequence(custom_poses, frames, "custom_motion.csv")
    
    # Validate the custom sequence CSV
    smoothener.validate_csv("custom_motion.csv") 