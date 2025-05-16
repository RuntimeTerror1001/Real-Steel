#!/usr/bin/env python3
"""
IK-FK Validation Module
Validates the consistency between forward and inverse kinematics solutions for a 7-DOF robotic arm.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class JointLimits:
    """Joint angle limits in radians."""
    shoulder_pitch: Tuple[float, float] = (-3.0892, 2.6704)  # -177° to 153°
    shoulder_yaw: Tuple[float, float] = (-2.618, 2.618)      # -150° to 150°
    shoulder_roll: Tuple[float, float] = (-1.5882, 2.2515)   # -91° to 129°
    elbow: Tuple[float, float] = (0.0, 2.27)                # 0-130°
    wrist_pitch: Tuple[float, float] = (-1.61443, 1.61443)  # -92.5° to 92.5°
    wrist_yaw: Tuple[float, float] = (-1.61443, 1.61443)    # -92.5° to 92.5°
    wrist_roll: Tuple[float, float] = (-1.97222, 1.97222)   # -113° to 113°

class IKFKValidator:
    def __init__(self, 
                 solver: Any,
                 tolerance: float = 0.01,
                 joint_limits: Optional[JointLimits] = None):
        """
        Initialize the IK-FK validator.
        
        Args:
            solver: IK solver instance with solve() and forward_kinematics() methods
            tolerance: Maximum allowed error between FK and IK solutions (in meters)
            joint_limits: Joint angle limits (defaults to standard Unitree G1 limits)
        """
        self.solver = solver
        self.tolerance = tolerance
        self.joint_limits = joint_limits or JointLimits()
        self.validation_history: List[Dict] = []
        
        # Define standard test configurations
        self.test_configurations = {
            'zero_position': {
                'shoulder_pitch': 0.0,
                'shoulder_yaw': 0.0,
                'shoulder_roll': 0.0,
                'elbow': 0.0,
                'wrist_pitch': 0.0,
                'wrist_yaw': 0.0,
                'wrist_roll': 0.0
            },
            'forward_reach': {
                'shoulder_pitch': 0.5,
                'shoulder_yaw': 0.0,
                'shoulder_roll': 0.0,
                'elbow': 1.0,
                'wrist_pitch': 0.0,
                'wrist_yaw': 0.0,
                'wrist_roll': 0.0
            },
            'side_reach': {
                'shoulder_pitch': 0.0,
                'shoulder_yaw': 1.0,
                'shoulder_roll': 0.0,
                'elbow': 1.0,
                'wrist_pitch': 0.0,
                'wrist_yaw': 0.0,
                'wrist_roll': 0.0
            },
            'upward_reach': {
                'shoulder_pitch': -0.5,
                'shoulder_yaw': 0.0,
                'shoulder_roll': 0.0,
                'elbow': 1.0,
                'wrist_pitch': 0.0,
                'wrist_yaw': 0.0,
                'wrist_roll': 0.0
            },
            'complex_pose': {
                'shoulder_pitch': 0.3,
                'shoulder_yaw': 0.4,
                'shoulder_roll': 0.2,
                'elbow': 1.2,
                'wrist_pitch': 0.3,
                'wrist_yaw': 0.2,
                'wrist_roll': 0.1
            }
        }
        
    def generate_random_configuration(self) -> Dict[str, float]:
        """Generate a random but valid joint configuration."""
        return {
            'shoulder_pitch': np.random.uniform(*self.joint_limits.shoulder_pitch),
            'shoulder_yaw': np.random.uniform(*self.joint_limits.shoulder_yaw),
            'shoulder_roll': np.random.uniform(*self.joint_limits.shoulder_roll),
            'elbow': np.random.uniform(*self.joint_limits.elbow),
            'wrist_pitch': np.random.uniform(*self.joint_limits.wrist_pitch),
            'wrist_yaw': np.random.uniform(*self.joint_limits.wrist_yaw),
            'wrist_roll': np.random.uniform(*self.joint_limits.wrist_roll)
        }
    
    def validate_configuration(self, joint_angles: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Validate a single joint configuration.
        
        Args:
            joint_angles: Dictionary of joint angles
            
        Returns:
            Tuple of (is_valid, position_error, angle_errors)
        """
        try:
            # Forward kinematics to get target position
            target_pos, _ = self.solver.forward_kinematics(joint_angles)
            
            # Set up IK inputs
            shoulder = np.array([0, 0, 0])
            elbow = shoulder + np.array([0, 0, self.solver.L1])
            wrist = shoulder + target_pos
            
            # Solve IK
            ik_angles = self.solver.solve(shoulder, elbow, wrist)
            
            # Forward kinematics with IK solution
            fk_pos, _ = self.solver.forward_kinematics(ik_angles)
            
            # Calculate position error
            position_error = np.linalg.norm(fk_pos - target_pos)
            
            # Calculate joint angle errors
            angle_errors = {
                joint: abs(np.degrees(ik_angles[joint] - joint_angles[joint]))
                for joint in joint_angles
            }
            
            is_valid = position_error < self.tolerance
            
            # Log validation result
            validation_result = {
                'joint_angles': joint_angles,
                'ik_angles': ik_angles,
                'target_position': target_pos,
                'fk_position': fk_pos,
                'position_error': position_error,
                'angle_errors': angle_errors,
                'is_valid': is_valid,
                'timestamp': datetime.now().isoformat()
            }
            self.validation_history.append(validation_result)
            
            if not is_valid:
                logger.warning(f"Validation failed - Position error: {position_error:.6f} m")
            
            return is_valid, position_error, angle_errors
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False, float('inf'), {}
    
    def run_validation_suite(self, num_tests: int = 100) -> None:
        """
        Run a comprehensive validation suite.
        
        Args:
            num_tests: Number of random test configurations to generate
        """
        logger.info(f"Starting validation suite with {num_tests} tests...")
        
        # First run standard test configurations
        logger.info("Running standard test configurations...")
        for name, config in self.test_configurations.items():
            logger.info(f"\nTesting {name}:")
            is_valid, pos_error, angle_errors = self.validate_configuration(config)
            logger.info(f"Valid: {is_valid}")
            logger.info(f"Position Error: {pos_error:.6f} m")
            logger.info("Joint Angle Errors (degrees):")
            for joint, error in angle_errors.items():
                logger.info(f"  {joint}: {error:.2f}°")
        
        # Then run random configurations
        logger.info(f"\nRunning {num_tests} random configurations...")
        for i in range(num_tests):
            if (i + 1) % 10 == 0:
                logger.info(f"Running test {i + 1}/{num_tests}")
            
            joint_angles = self.generate_random_configuration()
            self.validate_configuration(joint_angles)
    
    def generate_validation_report(self, show_plot: bool = True) -> None:
        """
        Generate a comprehensive validation report.
        
        Args:
            show_plot: Whether to display validation plots
        """
        if not self.validation_history:
            logger.warning("No validation history available")
            return
        
        # Calculate statistics
        position_errors = [result['position_error'] for result in self.validation_history]
        valid_count = sum(1 for result in self.validation_history if result['is_valid'])
        
        # Print report
        logger.info("\nValidation Report:")
        logger.info(f"Total tests: {len(self.validation_history)}")
        logger.info(f"Passed tests: {valid_count}")
        logger.info(f"Failed tests: {len(self.validation_history) - valid_count}")
        logger.info(f"Success rate: {valid_count/len(self.validation_history)*100:.1f}%")
        logger.info(f"\nPosition Error Statistics (m):")
        logger.info(f"  Average: {np.mean(position_errors):.6f}")
        logger.info(f"  Maximum: {np.max(position_errors):.6f}")
        logger.info(f"  Minimum: {np.min(position_errors):.6f}")
        logger.info(f"  Std Dev: {np.std(position_errors):.6f}")
        
        # Calculate joint angle error statistics
        joint_errors = {joint: [] for joint in self.joint_limits.__dataclass_fields__}
        for result in self.validation_history:
            for joint, error in result['angle_errors'].items():
                joint_errors[joint].append(error)
        
        logger.info("\nJoint Angle Error Statistics (degrees):")
        for joint, errors in joint_errors.items():
            logger.info(f"\n{joint}:")
            logger.info(f"  Average: {np.mean(errors):.2f}")
            logger.info(f"  Maximum: {np.max(errors):.2f}")
            logger.info(f"  Minimum: {np.min(errors):.2f}")
            logger.info(f"  Std Dev: {np.std(errors):.2f}")
        
        if show_plot:
            self._plot_validation_results()
    
    def _plot_validation_results(self) -> None:
        """Generate comprehensive validation plots."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Position error histogram
            plt.subplot(2, 2, 1)
            position_errors = [result['position_error'] for result in self.validation_history]
            plt.hist(position_errors, bins=30)
            plt.axvline(self.tolerance, color='r', linestyle='--', 
                       label=f'Tolerance ({self.tolerance} m)')
            plt.xlabel('Position Error (m)')
            plt.ylabel('Frequency')
            plt.title('Position Error Distribution')
            plt.legend()
            plt.grid(True)
            
            # Joint angle error boxplot
            plt.subplot(2, 2, 2)
            joint_errors = []
            joint_names = []
            for result in self.validation_history:
                for joint, error in result['angle_errors'].items():
                    joint_errors.append(error)
                    joint_names.append(joint)
            
            error_df = pd.DataFrame({
                'Joint': joint_names,
                'Error (degrees)': joint_errors
            })
            
            error_df.boxplot(column='Error (degrees)', by='Joint', ax=plt.gca())
            plt.title('Joint Angle Error Distribution')
            plt.suptitle('')  # Remove automatic suptitle
            plt.grid(True)
            
            # Position error over time
            plt.subplot(2, 2, 3)
            timestamps = [i for i in range(len(position_errors))]
            plt.plot(timestamps, position_errors, 'b-', alpha=0.5)
            plt.axhline(self.tolerance, color='r', linestyle='--', 
                       label=f'Tolerance ({self.tolerance} m)')
            plt.xlabel('Test Number')
            plt.ylabel('Position Error (m)')
            plt.title('Position Error Over Tests')
            plt.legend()
            plt.grid(True)
            
            # Cumulative success rate
            plt.subplot(2, 2, 4)
            cumulative_success = np.cumsum([1 if result['is_valid'] else 0 
                                          for result in self.validation_history])
            cumulative_rate = cumulative_success / np.arange(1, len(self.validation_history) + 1)
            plt.plot(range(len(cumulative_rate)), cumulative_rate * 100, 'g-')
            plt.xlabel('Test Number')
            plt.ylabel('Cumulative Success Rate (%)')
            plt.title('Cumulative Success Rate')
            plt.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('ik_fk_validation_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Validation plots saved as 'ik_fk_validation_summary.png'")
            
            # Export detailed results to CSV
            results_df = pd.DataFrame(self.validation_history)
            results_df.to_csv('validation_results.csv', index=False)
            logger.info("Detailed results saved as 'validation_results.csv'")
            
        except ImportError as e:
            logger.error(f"Failed to generate plots: {str(e)}")

def main():
    """Example usage of the IKFKValidator"""
    from ik_analytical3d import IKAnalytical3DRefined
    
    # Create solver instance with our custom parameters
    solver = IKAnalytical3DRefined(
        upper_arm_length=0.1032,  # Unitree G1 upper arm length
        lower_arm_length=0.1,     # Unitree G1 lower arm length
        position_tolerance=1e-6,
        orientation_tolerance=1e-4
    )
    
    # Create validator instance
    validator = IKFKValidator(
        solver=solver,
        tolerance=0.01,  # 1cm position error tolerance
    )
    
    # Run validation suite
    validator.run_validation_suite(num_tests=100)
    
    # Generate report
    validator.generate_validation_report(show_plot=True)

if __name__ == "__main__":
    main() 