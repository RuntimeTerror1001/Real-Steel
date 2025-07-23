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
                print(f"\nTest {idx+1}: {case['description']}")
                print(f"  Input Shoulder: {case['shoulder']}")
                print(f"  Input Elbow:    {case['elbow']}")
                print(f"  Input Wrist:    {case['wrist']}")
                # --- Detailed IK step debug ---
                # 1. Shoulder and wrist vectors
                shoulder = case['shoulder']
                elbow = case['elbow']
                wrist = case['wrist']
                local_elbow = elbow - shoulder
                local_wrist = wrist - shoulder
                print(f"  Local Elbow:    {local_elbow}")
                print(f"  Local Wrist:    {local_wrist}")
                # 2. Shoulder angles
                try:
                    sp, sy, sr = self.ik._shoulder_angles(local_elbow, local_wrist)
                    print(f"  Shoulder Angles (rad): pitch={sp:.3f}, yaw={sy:.3f}, roll={sr:.3f}")
                except Exception as e:
                    print(f"  [Shoulder angle calc error: {e}]")
                # 3. Elbow angle
                try:
                    elb = self.ik._elbow_angle(shoulder, elbow, wrist)
                    print(f"  Elbow Angle (rad): {elb:.3f}")
                except Exception as e:
                    print(f"  [Elbow angle calc error: {e}]")
                # 4. Wrist angles
                try:
                    wp, wy, wr = self.ik._wrist_angles(elbow, wrist)
                    print(f"  Wrist Angles (rad): pitch={wp:.3f}, yaw={wy:.3f}, roll={wr:.3f}")
                except Exception as e:
                    print(f"  [Wrist angle calc error: {e}]")
                # --- IK solve and FK validation ---
                angles = self.ik.solve(shoulder, elbow, wrist)
                result = {
                    'test_case': idx + 1,
                    'description': case['description'],
                }
                for joint, angle in angles.items():
                    min_limit, max_limit = self.ik.joint_limits[joint]
                    if not (min_limit <= angle <= max_limit):
                        print(f"  [Warning] {joint} angle {angle:.3f} out of bounds ({min_limit:.3f}, {max_limit:.3f})")
                    angle = np.clip(angle, min_limit, max_limit)
                    angle_deg = np.degrees(angle)
                    result[joint] = angle_deg
                print(f"  IK Angles (deg): { {k: round(v,2) for k,v in result.items() if k not in ['test_case','description']} }")
                # FK validation
                if hasattr(self.ik, 'forward_kinematics'):
                    fk_wrist, _ = self.ik.forward_kinematics(angles)
                    print(f"  FK Wrist:       {fk_wrist}")
                    pos_err = np.linalg.norm(fk_wrist - wrist)
                    print(f"  FK Error:       {pos_err:.4f} m")
                else:
                    print("  [FK not implemented in IKAnalytical3D]")
                results.append(result)
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

    def test_forward_kinematics_initial_pose(self):
        """Test FK with the initial pose from the CSV file"""
        print("\n" + "="*60)
        print("FORWARD KINEMATICS VERIFICATION")
        print("="*60)
        
        # Initial pose from CSV (left arm)
        initial_angles = {
            'shoulder_pitch': 0.2,
            'shoulder_roll': 0.2,
            'shoulder_yaw': 0.0,
            'elbow': 1.28,
            'wrist_roll': 0.0,
            'wrist_pitch': 0.0,
            'wrist_yaw': 0.0
        }
        
        print(f"Testing FK with initial pose:")
        for joint, angle in initial_angles.items():
            print(f"  {joint}: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
        
        # Test FK step by step
        print(f"\nLink lengths: L1={self.ik.L1:.4f}m, L2={self.ik.L2:.4f}m")
        
        # Extract joint angles
        sp = initial_angles['shoulder_pitch']
        sy = initial_angles['shoulder_yaw']
        sr = initial_angles['shoulder_roll']
        elb = initial_angles['elbow']
        wp = initial_angles['wrist_pitch']
        wy = initial_angles['wrist_yaw']
        wr = initial_angles['wrist_roll']
        
        print(f"\nStep-by-step transformations:")
        
        # Calculate transformation matrices
        T_shoulder = self.ik.transform_matrix(sp, 0, 0, np.pi/2)
        print(f"T_shoulder (pitch={sp:.3f}):")
        print(T_shoulder)
        
        T_yaw = self.ik.transform_matrix(sy, 0, 0, 0)
        print(f"\nT_yaw (yaw={sy:.3f}):")
        print(T_yaw)
        
        T_roll = self.ik.transform_matrix(sr, 0, 0, 0)
        print(f"\nT_roll (roll={sr:.3f}):")
        print(T_roll)
        
        T_elbow = self.ik.transform_matrix(elb, 0, self.ik.L1, 0)
        print(f"\nT_elbow (elbow={elb:.3f}, L1={self.ik.L1:.4f}):")
        print(T_elbow)
        
        T_wrist_pitch = self.ik.transform_matrix(wp, 0, self.ik.L2, 0)
        print(f"\nT_wrist_pitch (pitch={wp:.3f}, L2={self.ik.L2:.4f}):")
        print(T_wrist_pitch)
        
        T_wrist_yaw = self.ik.transform_matrix(wy, 0, 0, 0)
        print(f"\nT_wrist_yaw (yaw={wy:.3f}):")
        print(T_wrist_yaw)
        
        T_wrist_roll = self.ik.transform_matrix(wr, 0, 0, 0)
        print(f"\nT_wrist_roll (roll={wr:.3f}):")
        print(T_wrist_roll)
        
        # Calculate final transformation
        T_final = T_shoulder @ T_yaw @ T_roll @ T_elbow @ T_wrist_pitch @ T_wrist_yaw @ T_wrist_roll
        print(f"\nT_final (complete chain):")
        print(T_final)
        
        # Extract position and orientation
        position = T_final[:3, 3]
        orientation = T_final[:3, :3]
        
        print(f"\nFinal Results:")
        print(f"  Wrist Position: {position}")
        print(f"  Wrist Orientation Matrix:")
        print(orientation)
        
        # Check if position makes sense
        print(f"\nAnalysis:")
        print(f"  Distance from origin: {np.linalg.norm(position):.4f}m")
        print(f"  X component: {position[0]:.4f}m")
        print(f"  Y component: {position[1]:.4f}m")
        print(f"  Z component: {position[2]:.4f}m")
        
        return position, orientation

    def test_fk_ik_round_trip(self):
        """Test FK → IK → FK round-trip to find inconsistencies"""
        print("\n" + "="*60)
        print("FK → IK → FK ROUND-TRIP DEBUGGING")
        print("="*60)
        
        # Use the initial pose as our test case
        initial_angles = {
            'shoulder_pitch': 0.2,
            'shoulder_roll': 0.2,
            'shoulder_yaw': 0.0,
            'elbow': 1.28,
            'wrist_roll': 0.0,
            'wrist_pitch': 0.0,
            'wrist_yaw': 0.0
        }
        
        print("STEP 1: Original joint angles")
        for joint, angle in initial_angles.items():
            print(f"  {joint}: {angle:.6f} rad ({np.degrees(angle):.3f}°)")
        
        # STEP 2: Run FK to get wrist position
        print(f"\nSTEP 2: Forward Kinematics")
        fk_position, fk_orientation = self.ik.forward_kinematics(initial_angles)
        print(f"  FK Wrist Position: {fk_position}")
        print(f"  FK Wrist Orientation:")
        print(fk_orientation)
        
        # STEP 3: Use FK result as IK input
        print(f"\nSTEP 3: Inverse Kinematics")
        print(f"  IK Input - Shoulder: [0, 0, 0]")
        print(f"  IK Input - Elbow: {fk_position * 0.5}")  # Midpoint
        print(f"  IK Input - Wrist: {fk_position}")
        
        try:
            ik_angles = self.ik.solve(
                np.array([0, 0, 0]),  # shoulder
                fk_position * 0.5,    # elbow (midpoint)
                fk_position           # wrist
            )
            
            print(f"  IK Output angles:")
            for joint, angle in ik_angles.items():
                print(f"    {joint}: {angle:.6f} rad ({np.degrees(angle):.3f}°)")
            
            # STEP 4: Run FK again on IK result
            print(f"\nSTEP 4: FK of IK result")
            fk2_position, fk2_orientation = self.ik.forward_kinematics(ik_angles)
            print(f"  FK2 Wrist Position: {fk2_position}")
            
            # STEP 5: Compare results
            print(f"\nSTEP 5: Comparison")
            print(f"  Original FK position: {fk_position}")
            print(f"  FK2 position:        {fk2_position}")
            
            pos_error = np.linalg.norm(fk_position - fk2_position)
            print(f"  Position error: {pos_error:.6f} m")
            
            # Compare joint angles
            print(f"\n  Joint angle differences:")
            for joint in initial_angles.keys():
                if joint in ik_angles:
                    angle_diff = abs(initial_angles[joint] - ik_angles[joint])
                    angle_diff_deg = np.degrees(angle_diff)
                    print(f"    {joint}: {angle_diff:.6f} rad ({angle_diff_deg:.3f}°)")
            
            # STEP 6: Detailed analysis
            print(f"\nSTEP 6: Analysis")
            if pos_error < 1e-3:
                print(f"  ✅ SUCCESS: FK/IK round-trip works (error < 1mm)")
            elif pos_error < 1e-2:
                print(f"  ⚠️  WARNING: Moderate error ({pos_error*1000:.1f}mm)")
            else:
                print(f"  ❌ FAILURE: Large error ({pos_error*1000:.1f}mm)")
                
            return pos_error, ik_angles
            
        except Exception as e:
            print(f"  ❌ IK failed: {e}")
            return float('inf'), None

    def debug_simple_case(self):
        """Debug with a very simple case: all angles = 0"""
        print("\n" + "="*60)
        print("SIMPLE CASE DEBUG: All angles = 0")
        print("="*60)
        
        # Simple case: all angles = 0
        simple_angles = {
            'shoulder_pitch': 0.0,
            'shoulder_roll': 0.0,
            'shoulder_yaw': 0.0,
            'elbow': 0.0,
            'wrist_roll': 0.0,
            'wrist_pitch': 0.0,
            'wrist_yaw': 0.0
        }
        
        print("Simple case: all joint angles = 0")
        
        # Run FK
        fk_pos, fk_ori = self.ik.forward_kinematics(simple_angles)
        print(f"FK result: {fk_pos}")
        
        # Expected result for all zeros - FIXED to expect Z-axis extension
        expected_pos = np.array([0.0, 0.0, self.ik.L1 + self.ik.L2])  # Should be straight along Z
        print(f"Expected:  {expected_pos}")
        
        error = np.linalg.norm(fk_pos - expected_pos)
        print(f"Error: {error:.6f} m")
        
        return error

    def debug_coordinate_system(self):
        """Debug the coordinate system transformation"""
        print("\n" + "="*60)
        print("COORDINATE SYSTEM DEBUG")
        print("="*60)
        
        # Test the initial rotation matrix
        T_initial = self.ik.transform_matrix(np.pi/2, 0, 0, 0)  # Changed to +90°
        print("Initial rotation matrix (+90° around Y):")
        print(T_initial)
        
        # Test what happens to a unit vector along X after this rotation
        unit_x = np.array([1, 0, 0, 1])  # Unit vector along X
        rotated_x = T_initial @ unit_x
        print(f"Unit X vector [1,0,0] after rotation: {rotated_x[:3]}")
        
        # Test what happens to a unit vector along Z after this rotation
        unit_z = np.array([0, 0, 1, 1])  # Unit vector along Z
        rotated_z = T_initial @ unit_z
        print(f"Unit Z vector [0,0,1] after rotation: {rotated_z[:3]}")
        
        # Test the complete chain with all zeros
        simple_angles = {
            'shoulder_pitch': 0.0,
            'shoulder_roll': 0.0,
            'shoulder_yaw': 0.0,
            'elbow': 0.0,
            'wrist_roll': 0.0,
            'wrist_pitch': 0.0,
            'wrist_yaw': 0.0
        }
        
        # Build the chain step by step
        T_shoulder = self.ik.transform_matrix(0, 0, 0, 0)
        T_yaw = self.ik.transform_matrix(0, 0, 0, 0)
        T_roll = self.ik.transform_matrix(0, 0, 0, 0)
        T_elbow = self.ik.transform_matrix(0, 0, self.ik.L1, 0)
        T_wrist_pitch = self.ik.transform_matrix(0, 0, self.ik.L2, 0)
        T_wrist_yaw = self.ik.transform_matrix(0, 0, 0, 0)
        T_wrist_roll = self.ik.transform_matrix(0, 0, 0, 0)
        
        T_final = T_initial @ T_shoulder @ T_yaw @ T_roll @ T_elbow @ T_wrist_pitch @ T_wrist_yaw @ T_wrist_roll
        position = T_final[:3, 3]
        print(f"Final position with all zeros: {position}")
        
        return position

if __name__ == "__main__":
    tester = IKAnalytical3DTester()
    
    # Debug coordinate system first
    tester.debug_coordinate_system()
    
    # Test forward kinematics with initial pose
    tester.test_forward_kinematics_initial_pose()
    
    # Debug simple case
    tester.debug_simple_case()
    
    # Test FK → IK → FK round-trip
    tester.test_fk_ik_round_trip()
    
    # Add a few test cases (shoulder, elbow, wrist positions in meters)
    tester.add_test_case([0,0,0], [0,0,0.3], [0,0,0.6], "Straight arm up (Z-axis)")
    tester.add_test_case([0,0,0], [0.1,0,0.25], [0.2,0,0.5], "Bent arm to the right (Z-axis)")
    tester.add_test_case([0,0,0], [-0.1,0,0.25], [-0.2,0,0.5], "Bent arm to the left (Z-axis)")
    tester.add_test_case([0,0,0], [0,0.1,0.25], [0,0.2,0.5], "Bent arm forward (Z-axis)")
    tester.add_test_case([0,0,0], [0,-0.1,0.25], [0,-0.2,0.5], "Bent arm backward (Z-axis)")
    # New: fully extended arm
    tester.add_test_case([0,0,0], [0,0,0.15], [0,0,0.3], "Fully extended arm (Z-axis)")
    # New: 90 degree bent
    tester.add_test_case([0,0,0], [0.15,0,0.15], [0.3,0,0.3], "90 degree bent right (Z-axis)")
    # New: across body
    tester.add_test_case([0,0,0], [-0.15,0,0.15], [-0.3,0,0.3], "Across body left (Z-axis)")
    # New: colinear points (edge case)
    tester.add_test_case([0,0,0], [0,0,0.3], [0,0,0.6], "Colinear (edge case, Z-axis)")
    # New: problematic pose (elbow at same point as shoulder)
    tester.add_test_case([0,0,0], [0,0,0], [0,0,0.6], "Elbow at shoulder (edge case)")
    # New: problematic pose (wrist at same point as elbow)
    tester.add_test_case([0,0,0], [0,0,0.3], [0,0,0.3], "Wrist at elbow (edge case)")
    # New: out-of-reach pose
    tester.add_test_case([0,0,0], [0,0,0.3], [0,0,1.0], "Out of reach (Z-axis)")
    tester.run_tests()
    tester.generate_jab_motion_csv() 