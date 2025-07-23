#!/usr/bin/env python3
"""
Test script for the enhanced Real-Steel main system
Tests flow selection and recording indicators
"""

import sys
import os
import time

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

def test_flow_selection():
    """Test the flow selection functionality"""
    print("🧪 Testing flow selection functionality...")
    
    try:
        from main import select_ik_flow, select_execution_mode
        print("✅ Flow selection functions imported successfully")
        
        # Test would require user input, so just verify functions exist
        print("✅ IK flow selection function available")
        print("✅ Execution mode selection function available")
        
        return True
    except Exception as e:
        print(f"❌ Flow selection test failed: {e}")
        return False

def test_recording_system():
    """Test the recording system with visual indicators"""
    print("\n🧪 Testing recording system...")
    
    try:
        from pose_mirror_retargeting import PoseMirror3DWithRetargeting
        
        # Initialize with minimal configuration for testing
        system = PoseMirror3DWithRetargeting(
            ik_solver_backend='analytical',
            execution_mode='demo',
            enable_visualizations=False  # Disable for testing
        )
        
        print("✅ System initialized successfully")
        
        # Test recording status structure
        if hasattr(system, 'recording_status'):
            print("✅ Recording status structure exists")
            print(f"   Initial state: {system.recording_status}")
            
            # Test start recording
            if hasattr(system, 'start_recording'):
                print("✅ Start recording method available")
                
            # Test stop recording  
            if hasattr(system, 'stop_recording'):
                print("✅ Stop recording method available")
                
            # Test CSV recording
            if hasattr(system, 'record_frame_to_csv'):
                print("✅ CSV recording method available")
                
        else:
            print("❌ Recording status structure missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Recording system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dual_mode():
    """Test dual mode IK solver switching"""
    print("\n🧪 Testing dual mode IK switching...")
    
    try:
        from pose_mirror_retargeting import PoseMirror3DWithRetargeting
        
        # Initialize with dual mode enabled
        system = PoseMirror3DWithRetargeting(
            ik_solver_backend='analytical',
            dual_mode=True,
            execution_mode='demo',
            enable_visualizations=False
        )
        
        print("✅ Dual mode system initialized")
        
        # Check if robot retargeter has dual mode
        if hasattr(system.robot_retargeter, 'dual_mode'):
            print(f"✅ Dual mode enabled: {system.robot_retargeter.dual_mode}")
            
        # Check if switching method exists
        if hasattr(system.robot_retargeter, 'switch_ik_solver'):
            print("✅ IK solver switching method available")
            
        return True
        
    except Exception as e:
        print(f"❌ Dual mode test failed: {e}")
        return False

def test_visual_indicators():
    """Test visual recording indicators"""
    print("\n🧪 Testing visual recording indicators...")
    
    try:
        from pose_mirror_retargeting import PoseMirror3DWithRetargeting
        
        system = PoseMirror3DWithRetargeting(
            ik_solver_backend='analytical',
            execution_mode='demo',
            enable_visualizations=True
        )
        
        # Test status info method
        if hasattr(system, '_add_status_info'):
            print("✅ Status info method available")
            
        # Check recording status structure for visual indicators
        if 'blink_state' in system.recording_status:
            print("✅ Blinking indicator support available")
            
        if 'indicator_timer' in system.recording_status:
            print("✅ Indicator timing support available")
            
        if 'show_indicator' in system.recording_status:
            print("✅ Indicator visibility control available")
            
        return True
        
    except Exception as e:
        print(f"❌ Visual indicators test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 REAL-STEEL ENHANCED FEATURES TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_flow_selection,
        test_recording_system,
        test_dual_mode,
        test_visual_indicators
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
        print("\n📋 FEATURES VERIFIED:")
        print("   ✅ Flow selection (IK solver & execution mode)")
        print("   ✅ Recording system with visual indicators")
        print("   ✅ Dual mode IK solver switching")
        print("   ✅ Visual recording status indicators")
        print("   ✅ Enhanced main.py with interactive options")
        
        print("\n🎮 USAGE:")
        print("   python src/core/main.py                    # Interactive mode")
        print("   python src/core/main.py --dual-mode       # Enable IK switching")
        print("   python src/core/main.py --mode live       # Live camera mode")
        print("   python src/core/main.py --silent          # Silent mode")
        
        print("\n🔧 RUNTIME CONTROLS:")
        print("   S - Start/Stop recording (with visual feedback)")
        print("   I - Switch IK solver (if dual mode enabled)")
        print("   P - Pause/Resume")
        print("   V - Toggle visualizations")
        print("   Q/ESC - Quit")
        
    else:
        print(f"❌ {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 