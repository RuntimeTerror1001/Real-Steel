#!/usr/bin/env python3
"""
Enhanced Real-Steel Main System
Interactive flow selection for IK solver and execution mode with visual recording
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pose_mirror_retargeting import PoseMirror3DWithRetargeting
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are installed and paths are correct")
    sys.exit(1)

def print_banner():
    """Print the Real-Steel main banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           REAL-STEEL MAIN                                   ║
    ║                    Enhanced Motion Retargeting                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Interactive IK solver and execution mode selection                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def select_ik_flow():
    """Interactive IK solver flow selection"""
    print("\n🔧 SELECT IK SOLVER FLOW:")
    print("1. ⚡ Analytical IK - Fast geometric solver (recommended for real-time)")
    print("2. 🧠 BRPSO IK - Optimization-based solver (higher accuracy)")
    print("3. 🔄 Dual Mode - Start with Analytical, allow runtime switching")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == '1':
                return 'analytical', False
            elif choice == '2':
                return 'brpso', False
            elif choice == '3':
                return 'analytical', True  # Start with analytical, enable dual mode
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def select_execution_mode():
    """Interactive execution mode selection"""
    print("\n🎯 SELECT EXECUTION MODE:")
    print("1. 📹 Live Camera - Real-time motion capture from webcam")
    print("2. 📁 File Input - Process pre-recorded motion data")
    print("3. 🎮 Demo Mode - Simulated motion for testing")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == '1':
                return 'live', None
            elif choice == '2':
                file_path = input("Enter CSV file path: ").strip()
                return 'file', file_path
            elif choice == '3':
                return 'demo', None
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def main():
    """Enhanced main entry point with interactive flow selection"""
    parser = argparse.ArgumentParser(description='Enhanced Real-Steel Motion Retargeting System')
    parser.add_argument('--ik-backend', choices=['analytical', 'brpso'], help='IK solver backend')
    parser.add_argument('--dual-mode', action='store_true', help='Enable dual mode IK switching')
    parser.add_argument('--mode', choices=['live', 'file', 'demo'], help='Execution mode')
    parser.add_argument('--input-file', help='Input CSV file for file mode')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index')
    parser.add_argument('--silent', action='store_true', help='Silent mode with minimal output')
    
    args = parser.parse_args()
    
    if not args.silent:
        print_banner()
    
    # Interactive flow selection if not specified via command line
    if args.ik_backend and args.mode:
        ik_backend = args.ik_backend
        dual_mode = args.dual_mode
        execution_mode = args.mode
        input_file = args.input_file
    else:
        if not args.silent:
            ik_backend, dual_mode = select_ik_flow()
            execution_mode, input_file = select_execution_mode()
        else:
            # Default values for silent mode
            ik_backend = 'analytical'
            dual_mode = False
            execution_mode = 'demo'
            input_file = None
    
    # Setup logging
    log_level = logging.WARNING if args.silent else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/motion_retargeting.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    if not args.silent:
        print(f"\n🎯 STARTING REAL-STEEL SYSTEM:")
        print(f"   📋 IK Solver: {ik_backend.upper()}")
        print(f"   🔄 Dual Mode: {'ENABLED' if dual_mode else 'DISABLED'}")
        print(f"   🎮 Execution Mode: {execution_mode.upper()}")
        if input_file:
            print(f"   📁 Input File: {input_file}")
        print(f"   📹 Camera Index: {args.camera_index}")
    
    try:
        # Initialize the enhanced system
        system = PoseMirror3DWithRetargeting(
            ik_solver_backend=ik_backend,
            camera_index=args.camera_index
        )
        
        if not args.silent:
            print("\n✅ System initialized successfully!")
            print("\n🎮 RUNTIME CONTROLS:")
            print("   Q / ESC      - Quit application")
            print("   S            - Start/Stop CSV recording (with visual feedback)")
            print("   P            - Pause/Resume motion processing")
            if dual_mode:
                print("   I            - Switch IK solver (Analytical ↔ BRPSO)")
            print("   V            - Toggle visualizations")
            print("   C            - Calibrate pose alignment")
            print("\n🔴 Press 'S' to start recording with visual indicators!")
            print("=" * 60)
        
        # Start the system
        logger.info(f"Starting Real-Steel with {ik_backend} IK, {execution_mode} mode")
        system.run()
        
    except KeyboardInterrupt:
        if not args.silent:
            print("\n⚠️ System interrupted by user")
        logger.info("System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if not args.silent:
            print("\n👋 Real-Steel system shutdown complete")
        logger.info("Real-Steel system shutdown complete")

if __name__ == "__main__":
    main()