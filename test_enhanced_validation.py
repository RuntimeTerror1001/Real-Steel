#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced validation and logging functionality.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add the src/main directory to the path
sys.path.append('src/main')

from main import (
    setup_logging, 
    validate_log_file, 
    initialize_pose_mirror, 
    list_validation_logs, 
    generate_validation_report
)

def test_log_validation():
    """Test the log file validation functionality"""
    print("\n=== Testing Log File Validation ===")
    
    # Test with a non-existent file
    is_valid, error_msg, details = validate_log_file("nonexistent_file.csv")
    print(f"Non-existent file test: {is_valid} - {error_msg}")
    
    # Test with an empty file
    with open("test_empty.csv", "w") as f:
        pass
    
    is_valid, error_msg, details = validate_log_file("test_empty.csv")
    print(f"Empty file test: {is_valid} - {error_msg}")
    
    # Test with a valid file
    valid_data = {
        'Timestamp': ['2024-01-01 12:00:00.000', '2024-01-01 12:00:01.000'],
        'Position_Error': [0.001, 0.002],
        'Angle_Error': [0.01, 0.02],
        'Current_Motion': ['Guard', 'Left Jab'],
        'Right_Shoulder_Pitch': [0.1, 0.2],
        'Right_Shoulder_Roll': [0.1, 0.2],
        'Right_Shoulder_Yaw': [0.0, 0.1],
        'Right_Elbow': [0.6, 0.7],
        'Left_Shoulder_Pitch': [0.1, 0.2],
        'Left_Shoulder_Roll': [-0.1, -0.2],
        'Left_Shoulder_Yaw': [0.0, -0.1],
        'Left_Elbow': [0.6, 0.7]
    }
    
    df = pd.DataFrame(valid_data)
    df.to_csv("test_valid.csv", index=False)
    
    is_valid, error_msg, details = validate_log_file("test_valid.csv")
    print(f"Valid file test: {is_valid} - {error_msg}")
    if is_valid:
        print(f"Validation details: {details}")
    
    # Clean up test files
    for file in ["test_empty.csv", "test_valid.csv"]:
        if os.path.exists(file):
            os.remove(file)

def test_pose_mirror_initialization():
    """Test the PoseMirror initialization with error handling"""
    print("\n=== Testing PoseMirror Initialization ===")
    
    pose_mirror, success, error_msg = initialize_pose_mirror()
    print(f"Initialization test: {success} - {error_msg}")
    
    if success:
        print("PoseMirror initialized successfully")
        # Test that robot retargeter is available
        if hasattr(pose_mirror, 'robot_retargeter'):
            print("Robot retargeter is available")
        else:
            print("Warning: Robot retargeter not found")

def test_validation_logs_listing():
    """Test the validation logs listing functionality"""
    print("\n=== Testing Validation Logs Listing ===")
    
    # Create a test validation logs directory
    log_dir = "validation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create some test log files
    test_files = [
        "ik_fk_validation_20240101_120000.csv",
        "ik_fk_validation_20240101_130000.csv",
        "ik_fk_validation_20240101_140000.csv"
    ]
    
    for file in test_files:
        file_path = os.path.join(log_dir, file)
        # Create minimal valid CSV
        with open(file_path, 'w') as f:
            f.write("Timestamp,Position_Error,Angle_Error,Current_Motion\n")
            f.write("2024-01-01 12:00:00.000,0.001,0.01,Guard\n")
    
    # Test listing
    list_validation_logs()
    
    # Clean up
    for file in test_files:
        file_path = os.path.join(log_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)

def test_validation_report_generation():
    """Test the validation report generation functionality"""
    print("\n=== Testing Validation Report Generation ===")
    
    # Create a test validation log file
    log_dir = "validation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    test_file = os.path.join(log_dir, "test_validation_report.csv")
    
    # Create test data with various error levels
    test_data = {
        'Timestamp': [f'2024-01-01 12:00:{i:02d}.000' for i in range(10)],
        'Position_Error': [0.001 + i*0.002 for i in range(10)],
        'Angle_Error': [0.01 + i*0.005 for i in range(10)],
        'Current_Motion': ['Guard', 'Left Jab', 'Guard', 'Right Cross', 'Guard', 
                          'Left Hook', 'Guard', 'Right Uppercut', 'Guard', 'Guard'],
        'Right_Shoulder_Pitch': [0.1 + i*0.02 for i in range(10)],
        'Right_Shoulder_Roll': [0.1 + i*0.01 for i in range(10)],
        'Right_Shoulder_Yaw': [0.0 + i*0.01 for i in range(10)],
        'Right_Elbow': [0.6 + i*0.02 for i in range(10)],
        'Left_Shoulder_Pitch': [0.1 + i*0.02 for i in range(10)],
        'Left_Shoulder_Roll': [-0.1 - i*0.01 for i in range(10)],
        'Left_Shoulder_Yaw': [0.0 - i*0.01 for i in range(10)],
        'Left_Elbow': [0.6 + i*0.02 for i in range(10)]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv(test_file, index=False)
    
    # Test report generation (without visualization)
    print(f"Testing report generation for {test_file}")
    result = generate_validation_report(test_file)
    
    if result:
        print("Report generation successful")
        stats = result.get('stats', {})
        print(f"Total samples: {stats.get('total_samples', 0)}")
        print(f"Average position error: {stats.get('avg_position_error', 0):.6f}m")
        print(f"Average angle error: {stats.get('avg_angle_error', 0):.6f}rad")
    else:
        print("Report generation failed")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

def main():
    """Run all tests"""
    print("Testing Enhanced Validation and Logging Functionality")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logging(level=logging.DEBUG)
    logger.info("Starting enhanced validation tests")
    
    try:
        # Run all tests
        test_log_validation()
        test_pose_mirror_initialization()
        test_validation_logs_listing()
        test_validation_report_generation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        logger.info("All tests completed successfully")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        logger.error(f"Test failed: {str(e)}", exc_info=True)
    
    finally:
        # Clean up any remaining test files
        test_files = ["test_empty.csv", "test_valid.csv"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        print("\nTest files cleaned up")

if __name__ == "__main__":
    main() 