#!/usr/bin/env python3
"""
Final comprehensive comparison between BRPSO and Analytical IK solvers
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from src.core.brpso_ik_solver import BRPSO_IK_Solver
from src.core.ik_analytical3d import IKAnalytical3D

def create_test_targets():
    """Create diverse test targets for IK comparison"""
    targets = []
    
    # Simple forward reach
    targets.append({
        'name': 'Forward Reach',
        'position': np.array([0.15, 0.0, 0.1]),
        'description': 'Straight forward, moderate reach'
    })
    
    # High reach
    targets.append({
        'name': 'High Reach',
        'position': np.array([0.1, 0.0, 0.25]),
        'description': 'High position, moderate forward'
    })
    
    # Side reach
    targets.append({
        'name': 'Side Reach',
        'position': np.array([0.1, 0.15, 0.1]),
        'description': 'Side position, moderate reach'
    })
    
    # Low reach
    targets.append({
        'name': 'Low Reach',
        'position': np.array([0.1, 0.0, -0.05]),
        'description': 'Low position, moderate forward'
    })
    
    # Far reach
    targets.append({
        'name': 'Far Reach',
        'position': np.array([0.25, 0.0, 0.1]),
        'description': 'Maximum forward reach'
    })
    
    # Complex position
    targets.append({
        'name': 'Complex Position',
        'position': np.array([0.18, 0.12, 0.08]),
        'description': 'Combined forward, side, and height'
    })
    
    # Edge case - very close
    targets.append({
        'name': 'Close Position',
        'position': np.array([0.05, 0.0, 0.05]),
        'description': 'Very close to shoulder'
    })
    
    # Edge case - very far
    targets.append({
        'name': 'Far Position',
        'position': np.array([0.3, 0.0, 0.1]),
        'description': 'Beyond maximum reach'
    })
    
    return targets

def test_analytical_ik(targets):
    """Test analytical IK solver"""
    print("\n" + "="*60)
    print("TESTING ANALYTICAL IK SOLVER")
    print("="*60)
    
    ik_analytical = IKAnalytical3D()
    results = []
    
    for target in targets:
        print(f"\nTesting: {target['name']}")
        print(f"Target: {target['position']}")
        print(f"Description: {target['description']}")
        
        start_time = time.time()
        
        try:
            # Create shoulder, elbow, wrist positions for IK solver
            shoulder = np.array([0.0, 0.0, 0.0])
            
            # Estimate elbow position based on target
            target_dir = target['position'] / (np.linalg.norm(target['position']) + 1e-8)
            elbow = shoulder + target_dir * ik_analytical.L1 * 0.8
            
            # Use target as wrist position
            wrist = target['position']
            
            # Solve IK
            joint_angles = ik_analytical.solve(shoulder, elbow, wrist)
            
            # Validate solution
            achieved_pos, _ = ik_analytical.forward_kinematics(joint_angles)
            position_error = np.linalg.norm(achieved_pos - target['position'])
            
            solve_time = time.time() - start_time
            
            result = {
                'target_name': target['name'],
                'target_position': target['position'],
                'joint_angles': joint_angles,
                'achieved_position': achieved_pos,
                'position_error': position_error,
                'solve_time': solve_time,
                'success': position_error < 0.01  # 1cm tolerance
            }
            
            print(f"✓ Success - Error: {position_error:.6f}m, Time: {solve_time:.4f}s")
            print(f"  Joint angles: {joint_angles}")
            
        except Exception as e:
            solve_time = time.time() - start_time
            result = {
                'target_name': target['name'],
                'target_position': target['position'],
                'joint_angles': None,
                'achieved_position': None,
                'position_error': float('inf'),
                'solve_time': solve_time,
                'success': False,
                'error': str(e)
            }
            print(f"✗ Failed - {e}")
        
        results.append(result)
    
    return results

def test_brpso_ik(targets):
    """Test BRPSO IK solver with optimized settings"""
    print("\n" + "="*60)
    print("TESTING BRPSO IK SOLVER (OPTIMIZED)")
    print("="*60)
    
    # Use the best configuration from previous tests
    configs = [
        {'name': 'Fast', 'swarm_size': 20, 'max_iterations': 40, 'tolerance': 5e-3},
        {'name': 'Balanced', 'swarm_size': 25, 'max_iterations': 60, 'tolerance': 3e-3},
        {'name': 'Accurate', 'swarm_size': 35, 'max_iterations': 80, 'tolerance': 2e-3}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} Configuration ---")
        print(f"Swarm size: {config['swarm_size']}, Max iterations: {config['max_iterations']}")
        print(f"Tolerance: {config['tolerance']*1000:.1f}mm")
        
        brpso_solver = BRPSO_IK_Solver(
            swarm_size=config['swarm_size'],
            max_iterations=config['max_iterations'],
            position_tolerance=config['tolerance']
        )
        
        config_results = []
        
        for target in targets:
            print(f"\nTesting: {target['name']}")
            print(f"Target: {target['position']}")
            
            # Solve IK
            solution = brpso_solver.solve(target['position'])
            
            result = {
                'target_name': target['name'],
                'target_position': target['position'],
                'config_name': config['name'],
                'joint_angles': solution['joint_angles'],
                'position_error': solution['position_error'],
                'solve_time': solution['solve_time'],
                'iterations': solution['iterations'],
                'converged': solution['converged'],
                'success': solution['position_error'] < config['tolerance']
            }
            
            if result['success']:
                print(f"✓ Success - Error: {solution['position_error']:.6f}m, Time: {solution['solve_time']:.4f}s")
                print(f"  Iterations: {solution['iterations']}, Converged: {solution['converged']}")
            else:
                print(f"✗ Failed - Error: {solution['position_error']:.6f}m, Time: {solution['solve_time']:.4f}s")
            
            config_results.append(result)
        
        all_results.extend(config_results)
    
    return all_results

def compare_results(analytical_results, brpso_results):
    """Compare and analyze results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print("="*60)
    
    # Filter successful results
    successful_analytical = [r for r in analytical_results if r['success']]
    successful_brpso = [r for r in brpso_results if r['success']]
    
    print(f"\nAnalytical IK Results:")
    print(f"  Total tests: {len(analytical_results)}")
    print(f"  Successful: {len(successful_analytical)}")
    print(f"  Success rate: {len(successful_analytical)/len(analytical_results)*100:.1f}%")
    
    if successful_analytical:
        analytical_errors = [r['position_error'] for r in successful_analytical]
        analytical_times = [r['solve_time'] for r in successful_analytical]
        
        print(f"  Average error: {np.mean(analytical_errors):.6f}m")
        print(f"  Min error: {np.min(analytical_errors):.6f}m")
        print(f"  Max error: {np.max(analytical_errors):.6f}m")
        print(f"  Average time: {np.mean(analytical_times):.4f}s")
        print(f"  Min time: {np.min(analytical_times):.4f}s")
        print(f"  Max time: {np.max(analytical_times):.4f}s")
    
    print(f"\nBRPSO IK Results:")
    print(f"  Total tests: {len(brpso_results)}")
    print(f"  Successful: {len(successful_brpso)}")
    print(f"  Success rate: {len(successful_brpso)/len(brpso_results)*100:.1f}%")
    
    if successful_brpso:
        brpso_errors = [r['position_error'] for r in successful_brpso]
        brpso_times = [r['solve_time'] for r in successful_brpso]
        
        print(f"  Average error: {np.mean(brpso_errors):.6f}m")
        print(f"  Min error: {np.min(brpso_errors):.6f}m")
        print(f"  Max error: {np.max(brpso_errors):.6f}m")
        print(f"  Average time: {np.mean(brpso_times):.4f}s")
        print(f"  Min time: {np.min(brpso_times):.4f}s")
        print(f"  Max time: {np.max(brpso_times):.4f}s")
    
    # Compare by configuration
    configs = ['Fast', 'Balanced', 'Accurate']
    for config in configs:
        config_results = [r for r in successful_brpso if r['config_name'] == config]
        if config_results:
            errors = [r['position_error'] for r in config_results]
            times = [r['solve_time'] for r in config_results]
            print(f"\n  {config} config: {len(config_results)} successful")
            print(f"    Average error: {np.mean(errors):.6f}m")
            print(f"    Average time: {np.mean(times):.4f}s")
            print(f"    Success rate: {len(config_results)/8*100:.1f}%")  # 8 targets per config

def plot_final_comparison(analytical_results, brpso_results):
    """Create comprehensive comparison plots"""
    print("\nGenerating comprehensive comparison plots...")
    
    # Filter successful results
    successful_analytical = [r for r in analytical_results if r['success']]
    successful_brpso = [r for r in brpso_results if r['success']]
    
    if not successful_analytical or not successful_brpso:
        print("Not enough successful results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Error comparison
    analytical_errors = [r['position_error'] for r in successful_analytical]
    brpso_errors = [r['position_error'] for r in successful_brpso]
    
    ax1.hist(analytical_errors, alpha=0.7, label='Analytical', bins=15, color='blue')
    ax1.hist(brpso_errors, alpha=0.7, label='BRPSO', bins=15, color='red')
    ax1.set_xlabel('Position Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Position Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Time comparison
    analytical_times = [r['solve_time'] for r in successful_analytical]
    brpso_times = [r['solve_time'] for r in successful_brpso]
    
    ax2.hist(analytical_times, alpha=0.7, label='Analytical', bins=15, color='blue')
    ax2.hist(brpso_times, alpha=0.7, label='BRPSO', bins=15, color='red')
    ax2.set_xlabel('Solve Time (s)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Solve Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error vs Time scatter
    ax3.scatter(analytical_times, analytical_errors, alpha=0.7, label='Analytical', s=80, color='blue')
    ax3.scatter(brpso_times, brpso_errors, alpha=0.7, label='BRPSO', s=80, color='red')
    ax3.set_xlabel('Solve Time (s)')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Error vs Time Trade-off')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # BRPSO config comparison
    configs = ['Fast', 'Balanced', 'Accurate']
    colors = ['red', 'blue', 'green']
    
    for i, config in enumerate(configs):
        config_results = [r for r in successful_brpso if r['config_name'] == config]
        if config_results:
            errors = [r['position_error'] for r in config_results]
            times = [r['solve_time'] for r in config_results]
            ax4.scatter(times, errors, alpha=0.8, label=config, color=colors[i], s=100)
    
    ax4.set_xlabel('Solve Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('BRPSO Configuration Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('brpso_final_comparison.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'brpso_final_comparison.png'")

def main():
    """Main test function"""
    print("BRPSO vs Analytical IK - Final Comprehensive Comparison")
    print("="*70)
    
    # Create test targets
    targets = create_test_targets()
    print(f"Created {len(targets)} test targets")
    
    # Test analytical IK
    analytical_results = test_analytical_ik(targets)
    
    # Test BRPSO IK
    brpso_results = test_brpso_ik(targets)
    
    # Compare results
    compare_results(analytical_results, brpso_results)
    
    # Create plots
    try:
        plot_final_comparison(analytical_results, brpso_results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\n" + "="*70)
    print("FINAL COMPARISON COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main() 