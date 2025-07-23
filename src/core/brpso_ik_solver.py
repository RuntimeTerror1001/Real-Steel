# brpso_ik_solver.py
"""
Binary Real Particle Swarm Optimization (BRPSO) IK Solver
Based on Ghosh et al. (IFAC 2022) methodology for 7-DOF robotic arm
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# Global logging control for method calls
DEBUG_METHOD_CALLS = False  # Set to True for verbose method call logging

def log_method_call(cls_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if DEBUG_METHOD_CALLS:
                print(f"[{cls_name}] -> {func.__name__}()")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def set_debug_logging(enabled=True):
    """Enable or disable verbose method call logging"""
    global DEBUG_METHOD_CALLS
    DEBUG_METHOD_CALLS = enabled
    print(f"BRPSO IK debug logging: {'ENABLED' if enabled else 'DISABLED'}")

class BRPSO_IK_Solver:
    def __init__(
        self,
        upper_arm_length: float = 0.1032,
        lower_arm_length: float = 0.1,
        swarm_size: int = 30,
        max_iterations: int = 100,
        position_tolerance: float = 1e-4,
        c1: float = 2.0,  # Cognitive parameter
        c2: float = 2.0,  # Social parameter
        w: float = 0.7,   # Inertia weight
        w_min: float = 0.4,
        w_max: float = 0.9
    ):
        """
        Initialize BRPSO IK Solver
        
        Args:
            upper_arm_length: Length from shoulder to elbow (m)
            lower_arm_length: Length from elbow to wrist (m)
            swarm_size: Number of particles in swarm
            max_iterations: Maximum optimization iterations
            position_tolerance: Position error tolerance (m)
            c1, c2: PSO learning parameters
            w: Inertia weight
        """
        print('[IKAlternative] -> __init__()')
        # Arm dimensions
        self.L1 = upper_arm_length
        self.L2 = lower_arm_length
        
        # PSO parameters
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_min = w_min
        self.w_max = w_max
        
        # Joint limits (from Unitree G1 model files - in radians)
        self.joint_limits = {
            'shoulder_pitch': (-3.0892, 2.6704),    # -177° to 153°
            'shoulder_yaw': (-2.618, 2.618),        # -150° to 150°
            'shoulder_roll': (-1.5882, 2.2515),     # -91° to 129°
            'elbow': (-1.0472, 2.0944),            # -60° to 120°
            'wrist_pitch': (-1.61443, 1.61443),    # -92.5° to 92.5°
            'wrist_yaw': (-1.61443, 1.61443),      # -92.5° to 92.5°
            'wrist_roll': (-1.97222, 1.97222)      # -113° to 113°
        }
        
        # Joint names in order
        self.joint_names = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll', 
                           'elbow', 'wrist_pitch', 'wrist_yaw', 'wrist_roll']
        
        # Initialize swarm
        self.swarm = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # Performance tracking
        self.convergence_history = []
        self.solve_times = []
    
    def initialize_swarm(self, initial_guess: Optional[np.ndarray] = None):
        """Initialize particle swarm with random positions within joint limits"""
        print('[IKAlternative] -> initialize_swarm()')
        # Initialize positions
        self.swarm = np.zeros((self.swarm_size, 7))
        
        # Smart initialization: use some particles near zero (home position)
        # and some with random values
        num_home_particles = max(1, self.swarm_size // 4)
        
        # Initialize some particles near home position
        for i in range(num_home_particles):
            self.swarm[i] = np.random.uniform(-0.1, 0.1, 7)  # Near zero
        
        # Initialize remaining particles randomly
        for i in range(num_home_particles, self.swarm_size):
            for j, joint in enumerate(self.joint_names):
                min_val, max_val = self.joint_limits[joint]
                self.swarm[i, j] = np.random.uniform(min_val, max_val)
        
        # If initial guess provided, set one particle to it
        if initial_guess is not None:
            self.swarm[0] = self.clip_to_limits(initial_guess)
        
        # Initialize velocities
        self.velocities = np.zeros((self.swarm_size, 7))
        for i, joint in enumerate(self.joint_names):
            min_val, max_val = self.joint_limits[joint]
            velocity_range = (max_val - min_val) * 0.05  # 5% of joint range for stability
            self.velocities[:, i] = np.random.uniform(-velocity_range, velocity_range, self.swarm_size)
        
        # Initialize personal bests
        self.personal_best_positions = self.swarm.copy()
        self.personal_best_fitness = np.full(self.swarm_size, float('inf'))
        
        # Initialize global best
        self.global_best_position = np.zeros(7)
        self.global_best_fitness = float('inf')
    
    @log_method_call('IKAlternative')
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forward kinematics using DH parameters
        
        Args:
            joint_angles: Array of 7 joint angles [sp, sy, sr, el, wp, wy, wr]
            
        Returns:
            Tuple of (position, orientation_matrix)
        """
        print('[IKAlternative] -> forward_kinematics()')
        sp, sy, sr, el, wp, wy, wr = joint_angles
        
        # DH transformation matrices
        def transform_matrix(theta, d, a, alpha):
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            return np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
        
        # DH chain for 7-DOF arm
        T = transform_matrix(sy, 0, 0, np.pi/2)      # Shoulder Yaw
        T = T @ transform_matrix(sp, 0, 0, -np.pi/2)  # Shoulder Pitch
        T = T @ transform_matrix(sr, 0, 0, np.pi/2)   # Shoulder Roll
        T = T @ transform_matrix(el, 0, self.L1, 0)   # Elbow
        T = T @ transform_matrix(wp, 0, self.L2, np.pi/2)  # Wrist Pitch
        T = T @ transform_matrix(wy, 0, 0, -np.pi/2)  # Wrist Yaw
        T = T @ transform_matrix(wr, 0, 0, 0)         # Wrist Roll
        
        position = T[:3, 3]
        orientation = T[:3, :3]
        
        return position, orientation
    
    @log_method_call('IKAlternative')
    def objective_function(self, joint_angles: np.ndarray, target_position: np.ndarray) -> float:
        """
        Objective function: Euclidean distance between target and achieved position
        
        Args:
            joint_angles: Current joint angles
            target_position: Target end-effector position
            
        Returns:
            Position error (Euclidean distance)
        """
        print('[IKAlternative] -> objective_function()')
        try:
            achieved_position, _ = self.forward_kinematics(joint_angles)
            error = np.linalg.norm(achieved_position - target_position)
            return error
        except:
            return float('inf')  # Penalty for invalid configurations
    
    @log_method_call('IKAlternative')
    def clip_to_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clip joint angles to their limits"""
        print('[IKAlternative] -> clip_to_limits()')
        clipped = joint_angles.copy()
        for i, joint in enumerate(self.joint_names):
            min_val, max_val = self.joint_limits[joint]
            clipped[i] = np.clip(clipped[i], min_val, max_val)
        return clipped
    
    def solve(self, target_position: np.ndarray, 
              initial_guess: Optional[np.ndarray] = None,
              target_orientation: Optional[np.ndarray] = None) -> Dict:
        """
        Solve IK using BRPSO
        
        Args:
            target_position: Target end-effector position [x, y, z]
            initial_guess: Optional initial joint angle guess
            target_orientation: Optional target orientation matrix
            
        Returns:
            Dictionary with solution and metadata
        """
        print('[IKAlternative] -> solve()')
        start_time = time.time()
        
        # Initialize swarm with optional initial guess
        self.initialize_swarm(initial_guess)
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all particles
            for i in range(self.swarm_size):
                fitness = self.objective_function(self.swarm[i], target_position)
                
                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_positions[i] = self.swarm[i].copy()
                    self.personal_best_fitness[i] = fitness
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_position = self.swarm[i].copy()
                    self.global_best_fitness = fitness
            
            # Store convergence history
            self.convergence_history.append(self.global_best_fitness)
            
            # Check convergence
            if self.global_best_fitness < self.position_tolerance:
                break
            
            # Update particles
            self._update_particles(iteration)
        
        # Prepare solution
        solution = {
            'joint_angles': {
                joint: self.global_best_position[i] 
                for i, joint in enumerate(self.joint_names)
            },
            'position_error': self.global_best_fitness,
            'iterations': iteration + 1,
            'converged': self.global_best_fitness < self.position_tolerance,
            'solve_time': time.time() - start_time,
            'convergence_history': self.convergence_history.copy()
        }
        
        self.solve_times.append(solution['solve_time'])
        return solution
    
    def _update_particles(self, iteration: int):
        """Update particle positions and velocities"""
        # Adaptive inertia weight
        w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
        
        # Random coefficients
        r1 = np.random.rand(self.swarm_size, 7)
        r2 = np.random.rand(self.swarm_size, 7)
        
        # Update velocities
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.swarm)
        social = self.c2 * r2 * (self.global_best_position - self.swarm)
        
        self.velocities = w * self.velocities + cognitive + social
        
        # Update positions
        self.swarm += self.velocities
        
        # Clip to joint limits
        for i in range(self.swarm_size):
            self.swarm[i] = self.clip_to_limits(self.swarm[i])
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        print('[IKAlternative] -> get_performance_stats()')
        if not self.solve_times:
            return {}
        
        return {
            'avg_solve_time': np.mean(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'std_solve_time': np.std(self.solve_times),
            'total_solves': len(self.solve_times)
        }
    
    def validate_solution(self, joint_angles: Dict, target_position: np.ndarray) -> Tuple[bool, float]:
        """
        Validate IK solution using forward kinematics
        
        Args:
            joint_angles: Dictionary of joint angles
            target_position: Target position
            
        Returns:
            Tuple of (is_valid, position_error)
        """
        print('[IKAlternative] -> validate_solution()')
        # Convert to array
        angles_array = np.array([joint_angles[joint] for joint in self.joint_names])
        
        # Calculate FK
        achieved_position, _ = self.forward_kinematics(angles_array)
        position_error = np.linalg.norm(achieved_position - target_position)
        
        is_valid = position_error < self.position_tolerance
        return is_valid, position_error 