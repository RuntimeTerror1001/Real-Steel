import numpy as np
import csv
from ik_analytical3d import IKAnalytical3D

class IKAnalytical3DTester:
    def __init__(self, upper_arm_length=0.3, lower_arm_length=0.3, csv_path='ik_test_results.csv'):
        self.ik = IKAnalytical3D(upper_arm_length, lower_arm_length)
        self.csv_path = csv_path
        self.test_cases = []

    def add_test_case(self, shoulder, elbow, wrist, description=None):
        self.test_cases.append({
            'shoulder': np.array(shoulder),
            'elbow': np.array(elbow),
            'wrist': np.array(wrist),
            'description': description or ''
        })

    def run_tests(self):
        results = []
        for idx, case in enumerate(self.test_cases):
            try:
                angles = self.ik.solve(case['shoulder'], case['elbow'], case['wrist'])
                result = {
                    'test_case': idx + 1,
                    'description': case['description'],
                }
                for joint, angle in angles.items():
                    min_limit, max_limit = self.ik.joint_limits[joint]
                    if not (min_limit <= angle <= max_limit):
                        print(f"Warning: {joint} angle {angle} out of bounds, clipping.")
                    angle = np.clip(angle, min_limit, max_limit)
                    angle_deg = np.degrees(angle)
                    result[joint] = angle_deg
                results.append(result)
                print(f"Test {idx+1}: {case['description']}\n  Angles: {result}")
            except Exception as e:
                print(f"Test {idx+1} failed: {e}")
        self._write_csv(results)

    def _write_csv(self, results, csv_path=None):
        if not results:
            print("No results to write.")
            return
        fieldnames = list(results[0].keys())
        path = csv_path if csv_path else self.csv_path
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Results written to {path}")

        joint_names = list(self.ik.joint_limits.keys())
        for row in results:
            for joint in joint_names:
                if joint in row:
                    min_limit, max_limit = self.ik.joint_limits[joint]
                    min_limit_deg, max_limit_deg = np.degrees(min_limit), np.degrees(max_limit)
                    if (row[joint] < min_limit_deg) or (row[joint] > max_limit_deg):
                        print(f"ERROR: {joint} value {row[joint]} out of bounds!")

    def generate_jab_motion_csv(self, csv_path='jab_motion.csv', num_frames=30):
        """Generate a smooth jab motion, compute joint angles, and output to CSV in the required format."""
        # Define start and end positions for the left arm
        shoulder = np.array([0, 0, 0])
        elbow_start = np.array([0.08, 0, -0.18])
        elbow_end = np.array([0.03, 0, -0.28])
        wrist_start = np.array([0.18, 0, -0.32])
        wrist_end = np.array([0.0, 0, -0.55])
        arc_amplitude = 0.02
        arc = arc_amplitude * np.sin(np.linspace(0, np.pi, num_frames))
        elbow_traj = np.linspace(elbow_start, elbow_end, num_frames)
        wrist_traj = np.linspace(wrist_start, wrist_end, num_frames)
        for i in range(num_frames):
            elbow_traj[i][1] += arc[i]
            wrist_traj[i][1] += arc[i]
        # Required CSV columns and order
        csv_columns = [
            'timestamp',
            'left_shoulder_pitch_joint', 'left_shoulder_yaw_joint', 'left_shoulder_roll_joint',
            'left_elbow_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_wrist_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_yaw_joint', 'right_shoulder_roll_joint',
            'right_elbow_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_wrist_roll_joint'
        ]
        left_joint_names = [
            'shoulder_pitch', 'shoulder_yaw', 'shoulder_roll',
            'elbow', 'wrist_pitch', 'wrist_yaw', 'wrist_roll'
        ]
        results = []
        for i in range(num_frames):
            try:
                angles = self.ik.solve(shoulder, elbow_traj[i], wrist_traj[i])
                # Enforce joint limits and convert to degrees
                left_angles = {}
                for j, joint in enumerate(left_joint_names):
                    min_limit, max_limit = self.ik.joint_limits[joint]
                    angle = angles[joint]
                    if not (min_limit <= angle <= max_limit):
                        print(f"Warning: {joint} angle {angle} out of bounds, clipping.")
                    angle = np.clip(angle, min_limit, max_limit)
                    left_angles[joint] = np.degrees(angle)
                # Fill right arm with zeros (or default pose)
                row = {
                    'timestamp': round(i * 0.1, 2),
                    'left_shoulder_pitch_joint': left_angles['shoulder_pitch'],
                    'left_shoulder_yaw_joint': left_angles['shoulder_yaw'],
                    'left_shoulder_roll_joint': left_angles['shoulder_roll'],
                    'left_elbow_joint': left_angles['elbow'],
                    'left_wrist_pitch_joint': left_angles['wrist_pitch'],
                    'left_wrist_yaw_joint': left_angles['wrist_yaw'],
                    'left_wrist_roll_joint': left_angles['wrist_roll'],
                    'right_shoulder_pitch_joint': 0.0,
                    'right_shoulder_yaw_joint': 0.0,
                    'right_shoulder_roll_joint': 0.0,
                    'right_elbow_joint': 0.0,
                    'right_wrist_pitch_joint': 0.0,
                    'right_wrist_yaw_joint': 0.0,
                    'right_wrist_roll_joint': 0.0
                }
                results.append(row)
            except Exception as e:
                print(f"Frame {i+1} IK failed: {e}")
        # Write to CSV in the required order
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Jab motion CSV written to {csv_path} in required format.")

if __name__ == "__main__":
    tester = IKAnalytical3DTester()
    # Add a few test cases (shoulder, elbow, wrist positions in meters)
    tester.add_test_case([0,0,0], [0,0,-0.3], [0,0,-0.6], "Straight arm down")
    tester.add_test_case([0,0,0], [0.1,0,-0.25], [0.2,0,-0.5], "Bent arm to the right")
    tester.add_test_case([0,0,0], [-0.1,0,-0.25], [-0.2,0,-0.5], "Bent arm to the left")
    tester.add_test_case([0,0,0], [0,0.1,-0.25], [0,0.2,-0.5], "Bent arm forward")
    tester.add_test_case([0,0,0], [0,-0.1,-0.25], [0,-0.2,-0.5], "Bent arm backward")
    tester.run_tests()
    # Generate jab motion CSV
    tester.generate_jab_motion_csv() 