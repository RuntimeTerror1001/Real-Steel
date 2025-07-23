#!/usr/bin/env python3
"""
Real-Steel Motion Retargeting System Launcher
Provides options for different logging levels and execution modes.
"""

import sys
import os
import argparse

# Add src/core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

def print_banner():
    """Print the Real-Steel launcher banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        REAL-STEEL LAUNCHER                                  ║
    ║                   Motion Retargeting System                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Select your preferred execution mode and logging level                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def set_logging_levels(level):
    """Set logging levels for all components"""
    if level == 'silent':
        # Minimal output
        from ik_analytical3d import set_debug_logging as set_analytical_debug
        from brpso_ik_solver import set_debug_logging as set_brpso_debug
        set_analytical_debug(False)
        set_brpso_debug(False)
        print("🔇 SILENT MODE: Minimal logging enabled")
        
    elif level == 'normal':
        # Standard output
        from ik_analytical3d import set_debug_logging as set_analytical_debug
        from brpso_ik_solver import set_debug_logging as set_brpso_debug
        set_analytical_debug(False)
        set_brpso_debug(False)
        print("📢 NORMAL MODE: Standard logging enabled")
        
    elif level == 'verbose':
        # Detailed output
        from ik_analytical3d import set_debug_logging as set_analytical_debug
        from brpso_ik_solver import set_debug_logging as set_brpso_debug
        set_analytical_debug(True)
        set_brpso_debug(True)
        print("🔊 VERBOSE MODE: Detailed method call logging enabled")
        
    elif level == 'debug':
        # Full debug output
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        from ik_analytical3d import set_debug_logging as set_analytical_debug
        from brpso_ik_solver import set_debug_logging as set_brpso_debug
        set_analytical_debug(True)
        set_brpso_debug(True)
        print("🐛 DEBUG MODE: Full debug logging enabled")

def interactive_mode():
    """Interactive mode for selecting options"""
    print_banner()
    
    print("\n🎯 SELECT EXECUTION MODE:")
    print("1. 📊 Demo Mode - Run feature demonstration")
    print("2. 🎮 Live Mode - Real-time motion retargeting")
    print("3. 🧪 Test Mode - Run system tests")
    print("4. 📖 Help - Show detailed usage")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1-4): ").strip()
            if mode_choice in ['1', '2', '3', '4']:
                break
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    print("\n🔊 SELECT LOGGING LEVEL:")
    print("1. 🔇 Silent - Minimal output")
    print("2. 📢 Normal - Standard output (recommended)")
    print("3. 🔊 Verbose - Detailed method calls")
    print("4. 🐛 Debug - Full debug information")
    
    while True:
        try:
            log_choice = input("\nEnter your choice (1-4): ").strip()
            if log_choice in ['1', '2', '3', '4']:
                break
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Set logging level
    log_levels = {'1': 'silent', '2': 'normal', '3': 'verbose', '4': 'debug'}
    set_logging_levels(log_levels[log_choice])
    
    # Execute selected mode
    if mode_choice == '1':
        print("\n🚀 Starting Demo Mode...")
        run_demo()
    elif mode_choice == '2':
        print("\n🚀 Starting Live Mode...")
        run_live_mode()
    elif mode_choice == '3':
        print("\n🚀 Starting Test Mode...")
        run_tests()
    elif mode_choice == '4':
        show_help()

def run_demo():
    """Run the demo mode"""
    import subprocess
    try:
        subprocess.run([sys.executable, 'demo_enhanced_features.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
    except FileNotFoundError:
        print("❌ Demo script not found. Please ensure demo_enhanced_features.py exists.")

def run_live_mode():
    """Run live mode with IK solver selection"""
    print("\n🔧 SELECT IK SOLVER:")
    print("1. ⚡ Analytical IK (fast, deterministic)")
    print("2. 🧠 BRPSO IK (optimization-based)")
    print("3. 🔄 Both (start with analytical, allow runtime switching)")
    
    while True:
        try:
            ik_choice = input("\nEnter your choice (1-3): ").strip()
            if ik_choice in ['1', '2', '3']:
                break
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Prepare arguments
    args = []
    if ik_choice == '1':
        args = ['--ik-backend', 'analytical']
    elif ik_choice == '2':
        args = ['--ik-backend', 'brpso']
    else:  # Both
        args = ['--ik-backend', 'analytical', '--dual-mode']
        print("💡 Runtime switching enabled: Press 'I' during execution to switch solvers")
    
    import subprocess
    try:
        cmd = [sys.executable, 'src/core/main.py'] + args
        print(f"\n🚀 Starting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Live mode failed: {e}")
    except FileNotFoundError:
        print("❌ Main script not found. Please ensure src/core/main.py exists.")

def run_tests():
    """Run system tests"""
    print("\n🧪 Running Real-Steel system tests...")
    
    # Test imports
    try:
        print("Testing imports...")
        from pose_mirror_retargeting import PoseMirror3DWithRetargeting
        from robot_retargeter import RobotRetargeter
        print("✅ Import test passed")
        
        # Test system initialization
        print("Testing system initialization...")
        system = PoseMirror3DWithRetargeting(ik_solver_backend='analytical')
        print(f"✅ System initialized with {len(system.robot_retargeter.joint_angles)} joint angles")
        
        # Test IK solver switching
        print("Testing IK solver switching...")
        if hasattr(system.robot_retargeter, 'switch_ik_solver'):
            original = system.robot_retargeter.ik_solver_backend
            switched = system.robot_retargeter.switch_ik_solver()
            print(f"✅ IK solver switched from {original} to {switched}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """Show detailed help information"""
    help_text = """
    🔧 REAL-STEEL MOTION RETARGETING SYSTEM HELP
    
    📊 DEMO MODE:
    - Runs automated demonstration of all features
    - Shows dual-pipeline IK comparison
    - Generates sample motion data
    - Creates visualization plots
    
    🎮 LIVE MODE:
    - Real-time motion retargeting from camera
    - MediaPipe pose detection
    - 14-joint robot motion output
    - CSV recording with timestamps
    
    🧪 TEST MODE:
    - Verifies system components
    - Tests IK solver functionality
    - Validates imports and initialization
    
    🔊 LOGGING LEVELS:
    - Silent: Minimal output for production use
    - Normal: Standard informational messages
    - Verbose: Detailed method call tracking
    - Debug: Full debug information with timing
    
    🎯 COMMAND LINE USAGE:
    python real_steel_launcher.py            # Interactive mode
    python real_steel_launcher.py --demo     # Direct demo mode
    python real_steel_launcher.py --live     # Direct live mode
    python real_steel_launcher.py --test     # Direct test mode
    
    🎮 RUNTIME CONTROLS (Live Mode):
    Q / ESC      - Quit application
    S            - Start/Stop CSV recording (with visual feedback)
    P            - Pause/Resume motion processing
    I            - Switch IK solver (Analytical ↔ BRPSO)
    V            - Toggle visualizations
    C            - Calibrate pose alignment
    
    📁 OUTPUT FILES:
    - recordings/robot_motion_TIMESTAMP.csv - Motion data
    - logs/motion_retargeting.log - System logs
    - Various visualization plots
    """
    print(help_text)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Real-Steel Motion Retargeting System Launcher')
    parser.add_argument('--demo', action='store_true', help='Run demo mode directly')
    parser.add_argument('--live', action='store_true', help='Run live mode directly')
    parser.add_argument('--test', action='store_true', help='Run test mode directly')
    parser.add_argument('--help-detailed', action='store_true', help='Show detailed help')
    parser.add_argument('--log-level', choices=['silent', 'normal', 'verbose', 'debug'], 
                       default='normal', help='Set logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    set_logging_levels(args.log_level)
    
    # Execute based on arguments
    if args.help_detailed:
        show_help()
    elif args.demo:
        run_demo()
    elif args.live:
        run_live_mode()
    elif args.test:
        run_tests()
    else:
        interactive_mode()

if __name__ == "__main__":
    main() 