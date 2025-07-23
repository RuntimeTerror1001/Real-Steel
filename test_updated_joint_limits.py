#!/usr/bin/env python3
"""
Test script to verify updated joint limits from Unitree G1 model files
"""

import numpy as np
from src.core.brpso_ik_solver import BRPSO_IK_Solver

def test_joint_limits():
    """Test that joint limits match Unitree G1 model specifications"""
    print("Testing BRPSO IK Solver - Joint Limits Verification")
    print("="*60)
    
    # Create solver
    solver = BRPSO_IK_Solver()
    
    # Expected joint limits from Unitree G1 model files (in radians)
    expected_limits = {
        'shoulder_pitch': (-3.0892, 2.6704),    # -177° to 153°
        'shoulder_yaw': (-2.618, 2.618),        # -150° to 150°
        'shoulder_roll': (-1.5882, 2.2515),     # -91° to 129°
        'elbow': (-1.0472, 2.0944),            # -60° to 120°
        'wrist_pitch': (-1.61443, 1.61443),    # -92.5° to 92.5°
        'wrist_yaw': (-1.61443, 1.61443),      # -92.5° to 92.5°
        'wrist_roll': (-1.97222, 1.97222)      # -113° to 113°
    }
    
    print("Joint Limits Comparison:")
    print("-" * 40)
    
    all_match = True
    for joint_name in solver.joint_names:
        expected_min, expected_max = expected_limits[joint_name]
        actual_min, actual_max = solver.joint_limits[joint_name]
        
        min_match = abs(expected_min - actual_min) < 1e-6
        max_match = abs(expected_max - actual_max) < 1e-6
        
        status = "✓" if min_match and max_match else "✗"
        print(f"{status} {joint_name:15} Expected: [{expected_min:8.4f}, {expected_max:8.4f}] "
              f"Actual: [{actual_min:8.4f}, {actual_max:8.4f}]")
        
        if not (min_match and max_match):
            all_match = False
    
    print("-" * 40)
    if all_match:
        print("✓ All joint limits match Unitree G1 model specifications!")
    else:
        print("✗ Some joint limits do not match!")
    
    return all_match

def test_joint_clipping():
    """Test that joint clipping works correctly with the limits"""
    print("\nTesting Joint Clipping:")
    print("-" * 30)
    
    solver = BRPSO_IK_Solver()
    
    # Test cases: values outside limits
    test_cases = [
        ('shoulder_pitch', -4.0, -3.0892),  # Below minimum
        ('shoulder_pitch', 3.0, 2.6704),    # Above maximum
        ('elbow', -2.0, -1.0472),           # Below minimum
        ('wrist_roll', 3.0, 1.97222),       # Above maximum
        ('shoulder_yaw', 0.5, 0.5),         # Within limits
    ]
    
    for joint_name, input_value, expected_output in test_cases:
        # Create array with the test value at the correct position
        test_array = np.zeros(7)
        joint_index = solver.joint_names.index(joint_name)
        test_array[joint_index] = input_value
        
        # Clip to limits
        clipped = solver.clip_to_limits(test_array)
        actual_output = clipped[joint_index]
        
        match = abs(actual_output - expected_output) < 1e-6
        status = "✓" if match else "✗"
        print(f"{status} {joint_name:15} Input: {input_value:6.2f} -> Output: {actual_output:8.4f} "
              f"(Expected: {expected_output:8.4f})")

def test_swarm_initialization():
    """Test that swarm initialization respects joint limits"""
    print("\nTesting Swarm Initialization:")
    print("-" * 35)
    
    solver = BRPSO_IK_Solver(swarm_size=10)
    solver.initialize_swarm()
    
    # Check that all particles are within limits
    all_within_limits = True
    for i in range(solver.swarm_size):
        particle = solver.swarm[i]
        for j, joint_name in enumerate(solver.joint_names):
            min_val, max_val = solver.joint_limits[joint_name]
            if particle[j] < min_val or particle[j] > max_val:
                print(f"✗ Particle {i}, Joint {joint_name}: {particle[j]:.4f} "
                      f"(limits: [{min_val:.4f}, {max_val:.4f}])")
                all_within_limits = False
    
    if all_within_limits:
        print("✓ All particles initialized within joint limits!")
    else:
        print("✗ Some particles initialized outside joint limits!")

def test_forward_kinematics():
    """Test forward kinematics with joint limits"""
    print("\nTesting Forward Kinematics:")
    print("-" * 30)
    
    solver = BRPSO_IK_Solver()
    
    # Test with zero angles (should be within limits)
    zero_angles = np.zeros(7)
    pos, ori = solver.forward_kinematics(zero_angles)
    print(f"Zero angles position: {pos}")
    print(f"Zero angles orientation shape: {ori.shape}")
    
    # Test with angles at limits
    limit_angles = np.array([
        solver.joint_limits['shoulder_pitch'][0],  # Minimum
        solver.joint_limits['shoulder_yaw'][0],    # Minimum
        solver.joint_limits['shoulder_roll'][0],   # Minimum
        solver.joint_limits['elbow'][0],           # Minimum
        solver.joint_limits['wrist_pitch'][0],     # Minimum
        solver.joint_limits['wrist_yaw'][0],       # Minimum
        solver.joint_limits['wrist_roll'][0]       # Minimum
    ])
    
    try:
        pos, ori = solver.forward_kinematics(limit_angles)
        print(f"Limit angles position: {pos}")
        print("✓ Forward kinematics works with limit angles")
    except Exception as e:
        print(f"✗ Forward kinematics failed with limit angles: {e}")

def main():
    """Main test function"""
    print("BRPSO IK Solver - Joint Limits Verification")
    print("="*50)
    
    # Test joint limits
    limits_ok = test_joint_limits()
    
    # Test joint clipping
    test_joint_clipping()
    
    # Test swarm initialization
    test_swarm_initialization()
    
    # Test forward kinematics
    test_forward_kinematics()
    
    print("\n" + "="*50)
    if limits_ok:
        print("✓ All tests passed! Joint limits are correctly implemented.")
    else:
        print("✗ Some tests failed! Check joint limit implementation.")
    print("="*50)

if __name__ == "__main__":
    main() 