import numpy as np
import pandas as pd
from motion_validator import MotionValidator

def test_joint_limits():
    """Test joint limits validation"""
    print("\nTesting joint limits validation...")
    
    # Define joint limits
    joint_limits = {
        'right_shoulder_pitch': (-2.0, 2.0),
        'right_elbow': (0, 2.27)
    }
    
    validator = MotionValidator(joint_limits)
    
    # Test valid angles
    valid_angles = {
        'right_shoulder_pitch': 1.0,
        'right_elbow': 1.5
    }
    is_valid, violations = validator.validate_joint_limits(valid_angles)
    print(f"Valid angles test - is_valid: {is_valid}, violations: {violations}")
    
    # Test invalid angles
    invalid_angles = {
        'right_shoulder_pitch': 2.5,  # Exceeds limit
        'right_elbow': -0.5  # Below limit
    }
    is_valid, violations = validator.validate_joint_limits(invalid_angles)
    print(f"Invalid angles test - is_valid: {is_valid}, violations: {violations}")

def test_motion_smoothness():
    """Test motion smoothness validation"""
    print("\nTesting motion smoothness validation...")
    
    # Create test motion data
    timestamps = np.arange(0, 1, 0.1)
    smooth_motion = np.sin(timestamps * 2 * np.pi)  # Smooth sinusoidal motion
    jerky_motion = np.array([0, 2, -2, 2, -2, 0, 0, 0, 0, 0])  # Jerky motion
    
    # Create test dataframes
    smooth_df = pd.DataFrame({
        'timestamp': timestamps,
        'joint1': smooth_motion
    })
    
    jerky_df = pd.DataFrame({
        'timestamp': timestamps,
        'joint1': jerky_motion
    })
    
    validator = MotionValidator({'joint1': (-3, 3)}, max_velocity=2.0, max_acceleration=1.0)
    
    # Test smooth motion
    is_smooth, violations = validator.validate_motion_smoothness(smooth_df)
    print(f"Smooth motion test - is_smooth: {is_smooth}, violations: {violations}")
    
    # Test jerky motion
    is_smooth, violations = validator.validate_motion_smoothness(jerky_df)
    print(f"Jerky motion test - is_smooth: {is_smooth}, violations: {violations}")

def test_fk_validation():
    """Test forward kinematics validation"""
    print("\nTesting FK validation...")
    
    # Define joint limits and link lengths
    joint_limits = {
        'right_shoulder_pitch': (-2.0, 2.0),
        'right_shoulder_roll': (-1.5, 1.5),
        'right_shoulder_yaw': (-2.0, 2.0),
        'right_elbow': (0, 2.27)
    }
    
    link_lengths = {
        'upper_arm': 0.3,  # 30cm
        'forearm': 0.25    # 25cm
    }
    
    validator = MotionValidator(joint_limits, link_lengths=link_lengths)
    
    # Test cases
    test_cases = [
        {
            'name': 'Rest position',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            },
            'expected_pos': np.array([0.55, 0, 0])  # Full extension along x-axis
        },
        {
            'name': 'Raised arm',
            'angles': {
                'right_shoulder_pitch': np.pi/2,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': 0
            },
            'expected_pos': np.array([0, 0, 0.55])  # Full extension along z-axis
        },
        {
            'name': '90-degree elbow',
            'angles': {
                'right_shoulder_pitch': 0,
                'right_shoulder_roll': 0,
                'right_shoulder_yaw': 0,
                'right_elbow': np.pi/2
            },
            'expected_pos': np.array([0.3, 0.25, 0])  # L-shape in x-y plane
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\n{test['name']}:")
        # Test FK computation
        pos, rot = validator.compute_fk(test['angles'])
        print(f"Expected position: {test['expected_pos']}")
        print(f"Computed position: {pos}")
        
        # Test FK-IK validation
        is_valid, error = validator.validate_fk_ik(test['angles'], test['expected_pos'])
        print(f"Validation result - is_valid: {is_valid}, error: {error:.6f} meters")

def test_csv_validation():
    """Test CSV file validation"""
    print("\nTesting CSV validation...")
    
    # Create test motion data
    timestamps = np.arange(0, 1, 0.1)
    motion_data = pd.DataFrame({
        'timestamp': timestamps,
        'right_shoulder_pitch': np.sin(timestamps * 2 * np.pi),
        'right_elbow': np.cos(timestamps * 2 * np.pi) * 0.5 + 1.0  # Keep within limits
    })
    
    # Save test data
    test_input = "test_motion.csv"
    test_output = "validated_motion.csv"
    motion_data.to_csv(test_input, index=False)
    
    # Create validator
    joint_limits = {
        'right_shoulder_pitch': (-2.0, 2.0),
        'right_elbow': (0, 2.27)
    }
    validator = MotionValidator(joint_limits)
    
    # Test validation
    validated_df = validator.validate_from_csv(test_input, test_output)
    print(f"Validated motion shape: {validated_df.shape}")
    print(f"Columns: {validated_df.columns.tolist()}")
    
    # Clean up test files
    import os
    if os.path.exists(test_input):
        os.remove(test_input)
    if os.path.exists(test_output):
        os.remove(test_output)

def test_stream_validation():
    """Test stream validation"""
    print("\nTesting stream validation...")
    
    # Create test stream data
    stream_data = [
        {'timestamp': 0.0, 'joint1': 0.0, 'joint2': 1.0},
        {'timestamp': 0.1, 'joint1': 0.5, 'joint2': 1.2},
        {'timestamp': 0.2, 'joint1': 1.0, 'joint2': 1.4}
    ]
    
    validator = MotionValidator({'joint1': (-2, 2), 'joint2': (0, 2)})
    
    # Test validation
    validated_df = validator.validate_from_stream(stream_data)
    print(f"Validated stream shape: {validated_df.shape}")
    print(f"Columns: {validated_df.columns.tolist()}")

def main():
    print("Running MotionValidator tests...")
    
    # Run all tests
    test_joint_limits()
    test_motion_smoothness()
    test_fk_validation()
    test_csv_validation()
    test_stream_validation()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 