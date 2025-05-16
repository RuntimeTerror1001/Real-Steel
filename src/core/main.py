#!/usr/bin/env python3
"""
Main script for controlling PoseMirror3D with Robot Retargeting.
This is a visualization-only version for demonstrating retargeting.
"""

import os
import sys
import argparse
import pygame
from datetime import datetime

from pose_mirror_retargeting import PoseMirror3DWithRetargeting, VisualRobotRetargeter

def main():
    """Main entry point for the Real-Steel motion retargeting visualization system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-Steel Motion Retargeting Visualization')
    parser.add_argument('--validation-report', type=str, help='Generate validation report from a specified log file')
    parser.add_argument('--validation-dir', action='store_true', help='List all available validation log files')
    args = parser.parse_args()
    
    # Handle special commands
    if args.validation_dir:
        list_validation_logs()
        return
    
    if args.validation_report:
        generate_validation_report(args.validation_report)
        return
    
    # Initialize the system
    pose_mirror = PoseMirror3DWithRetargeting()
    
    try:
        # Start the system
        print("Starting Real-Steel motion retargeting visualization...")
        print("Controls:")
        print("  - Press 'Q' to quit")
        print("  - Press 'S' to start/stop recording (visual effect only)")
        print("  - Press 'P' to pause motion (helps when taking screenshots)")
        print("  - Press 'V' to toggle IK-FK validation error display")
        
        # Run the system
        pose_mirror.run()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        print("System shutdown complete.")

def list_validation_logs():
    """List all available validation log files"""
    log_dir = "validation_logs"
    
    if not os.path.exists(log_dir):
        print(f"No validation logs directory found at {log_dir}")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    
    if not log_files:
        print(f"No validation log files found in {log_dir}")
        return
    
    print(f"\nAvailable validation log files in {log_dir}:")
    for i, log_file in enumerate(sorted(log_files)):
        file_path = os.path.join(log_dir, log_file)
        file_size = os.path.getsize(file_path) / 1024  # size in KB
        file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {log_file} ({file_size:.1f} KB, {file_date})")
    
    print("\nTo generate a report from a log file, run:")
    print("python main.py --validation-report validation_logs/FILENAME.csv")

def generate_validation_report(log_file):
    """Generate a validation report from a specified log file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Generating validation report from {log_file}...")
    
    # Create a temporary PoseMirror3D system just for report generation
    # This ensures we have access to all the visualization methods
    from pose_mirror_retargeting import PoseMirror3DWithRetargeting
    
    # Initialize the system but don't run it
    system = PoseMirror3DWithRetargeting()
    
    # Set up the validation log path
    system.robot_retargeter.validation_log_path = log_file
    
    # Generate and display the report using the robot retargeter
    system.robot_retargeter.generate_validation_report(show_plot=True)

if __name__ == "__main__":
    main()