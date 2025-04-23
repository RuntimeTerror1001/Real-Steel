#!/usr/bin/env python3
"""
Main script for controlling PoseMirror3D with Robot Retargeting.
This allows upper body boxing motion to be detected, retargeted to a robot figure,
and recorded to CSV files.
"""

import os
import argparse
from datetime import datetime

from pose_mirror_retargeting import PoseMirror3DWithRetargeting

def main():
    """Main entry point for the Real-Steel motion retargeting system."""
    # Initialize the system
    pose_mirror = PoseMirror3DWithRetargeting()
    
    try:
        # Start the system with automatic recording
        print("Starting Real-Steel motion retargeting system...")
        print("Controls:")
        print("  - Press 'Q' to quit")
        print("  - Press 'R' to reset calibration")
        print("  - Press 'S' to start/stop recording")
        print("  - Left/Right arrows to adjust rotation")
        
        # Run the system
        pose_mirror.run()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        print("System shutdown complete.")

if __name__ == "__main__":
    main()