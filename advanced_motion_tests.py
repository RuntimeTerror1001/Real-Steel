import numpy as np
import pandas as pd
from motion_validator import MotionValidator
import matplotlib.pyplot as plt

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\nTesting edge cases...")
    
    joint_limits = {
        'right_shoulder_pitch': (-2.0, 2.0),
        'right_elbow': (0, 2.27)
    }
    validator = MotionValidator(joint_limits)
    
    # Test exactly at limits
    boundary_angles = {
        'right_shoulder_pitch': 2.0,  # At max limit
        'right_elbow': 0.0   # At min limit
    }
    is_valid, violations = validator.validate_joint_limits(boundary_angles)
    print(f"Boundary angles test - is_valid: {is_valid}, violations: {violations}")
    
    # Test missing joint
    incomplete_angles = {
        'right_shoulder_pitch': 1.0
        # right_elbow missing
    }
    is_valid, violations = validator.validate_joint_limits(incomplete_angles)
    print(f"Missing joint test - is_valid: {is_valid}, violations: {violations}")

def test_complex_motion():
    """Test more complex motion patterns"""
    print("\nTesting complex motion patterns...")
    
    # Create complex motion pattern
    timestamps = np.arange(0, 2, 0.1)
    complex_motion = np.sin(timestamps * 2 * np.pi) + 0.5 * np.sin(timestamps * 4 * np.pi)
    
    motion_data = pd.DataFrame({
        'timestamp': timestamps,
        'right_shoulder_pitch': complex_motion,
        'right_elbow': np.abs(np.sin(timestamps * 3 * np.pi))  # Always positive
    })
    
    # Visualize the motion
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, complex_motion, label='shoulder_pitch')
    plt.plot(timestamps, np.abs(np.sin(timestamps * 3 * np.pi)), label='elbow')
    plt.title('Complex Motion Pattern')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.savefig('complex_motion.png')
    plt.close()
    
    joint_limits = {
        'right_shoulder_pitch': (-2.0, 2.0),
        'right_elbow': (0, 2.27)
    }
    validator = MotionValidator(joint_limits, max_velocity=3.0, max_acceleration=2.0)
    
    # Test validation
    is_smooth, violations = validator.validate_motion_smoothness(motion_data)
    print(f"Complex motion test - is_smooth: {is_smooth}, violations: {violations}")

def test_rapid_transitions():
    """Test motion with rapid transitions"""
    print("\nTesting rapid transitions...")
    
    timestamps = np.arange(0, 1, 0.1)
    # Create a motion with sudden acceleration
    rapid_motion = np.zeros_like(timestamps)
    rapid_motion[5:] = 1.0  # Sudden jump at t=0.5s
    
    motion_data = pd.DataFrame({
        'timestamp': timestamps,
        'joint1': rapid_motion
    })
    
    validator = MotionValidator({'joint1': (-2, 2)}, max_velocity=1.0, max_acceleration=0.5)
    is_smooth, violations = validator.validate_motion_smoothness(motion_data)
    print(f"Rapid transition test - is_smooth: {is_smooth}, violations: {violations}")

def main():
    print("Running advanced motion tests...")
    
    test_edge_cases()
    test_complex_motion()
    test_rapid_transitions()
    
    print("\nAdvanced tests completed!")

if __name__ == "__main__":
    main() 