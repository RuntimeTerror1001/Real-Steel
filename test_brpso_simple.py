#!/usr/bin/env python3
"""
Simple test script for BRPSO IK solver
"""

import numpy as np
import time
from src.core.brpso_ik_solver import BRPSO_IK_Solver

def test_brpso_basic():
    """Test basic BRPSO functionality"""
    print("Testing BRPSO IK Solver - Basic Functionality")
    print("="*50)
    
    # Create solver
    solver = BRPSO_IK_Solver(
        swarm_size=20,
        max_iterations=50,
        position_tolerance=1e-3  # 1mm tolerance
    )
    
    # Test targets
    test_targets = [
        np.array([0.15, 0.0, 0.1]),   # Forward reach
        np.array([0.1, 0.15, 0.1]),   # Side reach
        np.array([0.1, 0.0, 0.25]),   # High reach
    ]
    
    for i, target in enumerate(test_targets):
        print(f"\nTest {i+1}: Target = {target}")
        
        try:
            # Solve IK
            start_time = time.time()
            solution = solver.solve(target)
            solve_time = time.time() - start_time
            
            print(f"  Success: {solution['converged']}")
            print(f"  Position error: {solution['position_error']:.6f}m")
            print(f"  Solve time: {solve_time:.4f}s")
            print(f"  Iterations: {solution['iterations']}")
            
            # Validate solution
            achieved_pos, _ = solver.forward_kinematics(
                np.array([solution['joint_angles'][joint] for joint in solver.joint_names])
            )
            validation_error = np.linalg.norm(achieved_pos - target)
            print(f"  Validation error: {validation_error:.6f}m")
            
        except Exception as e:
            print(f"  Error: {e}")

def test_brpso_configurations():
    """Test different BRPSO configurations"""
    print("\n\nTesting BRPSO IK Solver - Different Configurations")
    print("="*60)
    
    target = np.array([0.15, 0.0, 0.1])  # Forward reach
    
    configs = [
        {'name': 'Fast', 'swarm_size': 15, 'max_iterations': 30},
        {'name': 'Balanced', 'swarm_size': 25, 'max_iterations': 60},
        {'name': 'Accurate', 'swarm_size': 40, 'max_iterations': 100}
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} Configuration ---")
        
        solver = BRPSO_IK_Solver(
            swarm_size=config['swarm_size'],
            max_iterations=config['max_iterations'],
            position_tolerance=1e-3
        )
        
        try:
            start_time = time.time()
            solution = solver.solve(target)
            solve_time = time.time() - start_time
            
            print(f"  Swarm size: {config['swarm_size']}")
            print(f"  Max iterations: {config['max_iterations']}")
            print(f"  Position error: {solution['position_error']:.6f}m")
            print(f"  Solve time: {solve_time:.4f}s")
            print(f"  Iterations used: {solution['iterations']}")
            print(f"  Converged: {solution['converged']}")
            
        except Exception as e:
            print(f"  Error: {e}")

def test_brpso_forward_kinematics():
    """Test forward kinematics consistency"""
    print("\n\nTesting BRPSO Forward Kinematics")
    print("="*40)
    
    solver = BRPSO_IK_Solver()
    
    # Test with zero angles
    zero_angles = np.zeros(7)
    pos, ori = solver.forward_kinematics(zero_angles)
    print(f"Zero angles position: {pos}")
    print(f"Zero angles orientation shape: {ori.shape}")
    
    # Test with some non-zero angles
    test_angles = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1])
    pos, ori = solver.forward_kinematics(test_angles)
    print(f"Test angles position: {pos}")
    print(f"Test angles orientation shape: {ori.shape}")

def main():
    """Main test function"""
    print("BRPSO IK Solver Test Suite")
    print("="*30)
    
    # Test basic functionality
    test_brpso_basic()
    
    # Test different configurations
    test_brpso_configurations()
    
    # Test forward kinematics
    test_brpso_forward_kinematics()
    
    print("\n" + "="*30)
    print("Test completed!")

if __name__ == "__main__":
    main() 