import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from robot_retargeter import RobotRetargeter

class PoseMirror3DWithRetargeting:
    def __init__(self, window_size=(1280, 720)):
        """Initialize the PoseMirror3D system with robot retargeting."""
        self.window_size = window_size
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Initialize visualization
        self._setup_visualization()
        
        # Initialize robot retargeter
        self.robot_retargeter = RobotRetargeter(recording_freq=20)
        self.robot_retargeter.ax_robot = self.ax_robot
        self.robot_retargeter.fig_robot = self.fig
        
        # Initialize pose tracking variables
        self.pose_history = []
        self.smoothing_window = 5
        self.current_rotation_angle = 0
        self.initial_angle_set = False
        self.angle_offset = 0
        self.smoothing_factor = 0.8
        self.recent_chest_vectors = []
        self.max_history = 5
        self.joint_angles = {}
        
    def _setup_visualization(self):
        """Set up all visualization components."""
        # Initialize matplotlib
        plt.ion()
        self.fig = plt.figure(figsize=(20, 6))
        self.fig.canvas.manager.set_window_title('Motion Retargeting Visualization')
        
        # Create subplots
        self.ax_human = self.fig.add_subplot(131, projection='3d')
        self.ax_robot = self.fig.add_subplot(132, projection='3d')
        self.ax_angles = self.fig.add_subplot(133)
        
        # Configure subplot titles
        self.ax_human.set_title('Human Pose')
        self.ax_robot.set_title('Robot Simulation')
        self.ax_angles.set_title('Joint Angles')
        
        # Set up 3D plot parameters
        self.view_limits = 0.8
        for ax in [self.ax_human, self.ax_robot]:
            ax.set_xlim([-self.view_limits, self.view_limits])
            ax.set_ylim([-self.view_limits, self.view_limits])
            ax.set_zlim([-self.view_limits, self.view_limits])
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.view_init(elev=0, azim=270)
            ax.grid(True)
        
        plt.tight_layout()
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("3D Pose Mirror - With Robot Retargeting")
        
    def calculate_body_plane_angle(self, landmarks):
        """Calculate the angle between the body plane and camera plane."""
        if not landmarks:
            return 0
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Calculate chest midpoint
        chest_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Calculate chest-to-nose vector
        chest_to_nose_x = -(nose.x - chest_mid_x)
        chest_to_nose_z = -(nose.y - chest_mid_y)
        
        # Normalize vector
        magnitude = math.sqrt(chest_to_nose_x**2 + chest_to_nose_z**2)
        if magnitude > 0:
            chest_to_nose_x /= magnitude
            chest_to_nose_z /= magnitude
        
        # Update vector history
        self.recent_chest_vectors.append((chest_to_nose_x, chest_to_nose_z))
        if len(self.recent_chest_vectors) > self.max_history:
            self.recent_chest_vectors.pop(0)
        
        # Calculate average vector
        avg_x = sum(v[0] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
        avg_z = sum(v[1] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
        
        # Calculate angle
        raw_angle = math.degrees(math.atan2(avg_x, avg_z))
        
        # Set initial reference angle
        if not self.initial_angle_set and len(self.recent_chest_vectors) >= 3:
            self.angle_offset = raw_angle
            self.initial_angle_set = True
            
        # Calculate relative angle
        relative_angle = raw_angle - self.angle_offset
        
        # Apply dead zone
        if abs(relative_angle) < 10:
            relative_angle = 0
                
        # Normalize angle
        relative_angle = ((relative_angle + 180) % 360) - 180
        
        # Apply smoothing
        self.current_rotation_angle = (self.current_rotation_angle * self.smoothing_factor + 
                                     relative_angle * (1 - self.smoothing_factor))
        
        # Final normalization
        self.current_rotation_angle = ((self.current_rotation_angle + 180) % 360) - 180
            
        return self.current_rotation_angle

    def update_visualization(self, results):
        """Update all visualizations with the latest pose data."""
        if not results.pose_world_landmarks:
            return

        # Clear all plots
        self.ax_human.clear()
        self.ax_robot.clear()
        self.ax_angles.clear()

        # Update human pose visualization
        self._update_human_pose(results.pose_world_landmarks)
        
        # Update robot visualization
        self.robot_retargeter.update_robot_plot(self.ax_robot)
        
        # Update joint angles visualization
        if self.joint_angles:
            self._update_joint_angles()

        # Refresh the display
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_human_pose(self, world_landmarks):
        """Update the human pose visualization."""
        landmarks = world_landmarks.landmark
        
        # Extract coordinates with correct mapping
        x_coords = [-lm.x for lm in landmarks]  # Flip x for right-positive
        y_coords = [-lm.z for lm in landmarks]  # Use -z for up
        z_coords = [-lm.y for lm in landmarks]  # Use -y for forward

        # Plot landmarks
        self.ax_human.scatter3D(x_coords, z_coords, y_coords, c='b', marker='o')

        # Plot connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            self.ax_human.plot3D([x_coords[start_idx], x_coords[end_idx]],
                               [z_coords[start_idx], z_coords[end_idx]],
                               [y_coords[start_idx], y_coords[end_idx]], 'b-')

        # Set plot properties
        self.ax_human.set_xlabel('X (Left-Right)')
        self.ax_human.set_ylabel('Z (Forward-Back)')
        self.ax_human.set_zlabel('Y (Up-Down)')
        self.ax_human.view_init(elev=0, azim=270)
        self.ax_human.set_box_aspect([1,1,1])

    def _update_joint_angles(self):
        """Update the joint angles visualization."""
        joint_limits = self.robot_retargeter.joint_limits
        
        # Prepare data
        joints = list(self.joint_angles.keys())
        values = [math.degrees(self.joint_angles[j]) for j in joints]
        y_pos = np.arange(len(joints))
        
        # Create bar plot
        bars = self.ax_angles.barh(y_pos, values, align='center', height=0.5)
        
        # Add joint limits and styling
        for i, (joint, bar) in enumerate(zip(joints, bars)):
            if joint in joint_limits:
                min_deg = math.degrees(joint_limits[joint][0])
                max_deg = math.degrees(joint_limits[joint][1])
                
                # Add limit lines
                self.ax_angles.axvline(x=min_deg, color='r', linestyle='--', alpha=0.5)
                self.ax_angles.axvline(x=max_deg, color='r', linestyle='--', alpha=0.5)
                
                # Color bars based on limits
                bar.set_color('red' if values[i] < min_deg or values[i] > max_deg else 'blue')
                
                # Add limit labels
                self.ax_angles.text(min_deg, i, f'{min_deg:.0f}°', va='center', ha='right', fontsize=6)
                self.ax_angles.text(max_deg, i, f'{max_deg:.0f}°', va='center', ha='left', fontsize=6)
        
        # Customize appearance
        self.ax_angles.set_yticks(y_pos)
        self.ax_angles.set_yticklabels([j.replace('_joint', '').replace('_', ' ') for j in joints], fontsize=8)
        self.ax_angles.set_xlabel('Angle (degrees)')
        self.ax_angles.grid(True, alpha=0.3)
        
        # Set reasonable x-axis limits
        max_abs_val = max(abs(min(values)), abs(max(values)))
        self.ax_angles.set_xlim(-max_abs_val * 1.2, max_abs_val * 1.2)

    def update_robot_state(self, results):
        """Update robot state based on pose detection results."""
        if not results.pose_world_landmarks:
            return

        # Retarget pose and calculate joint angles
        self.robot_retargeter.retarget_pose(results.pose_world_landmarks, self.current_rotation_angle)
        
        # Update joint angles
        self.joint_angles = {
            **self.robot_retargeter.calculate_joint_angles("left"),
            **self.robot_retargeter.calculate_joint_angles("right")
        }
        self.robot_retargeter.joint_angles.update(self.joint_angles)

        # Update visualization
        self.update_visualization(results)

    def run(self):
        """Main run loop for pose detection and visualization."""
        cap = cv2.VideoCapture(0)
        for _ in range(5):
            cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Failed to capture frame from camera.")
                    continue

                # Process image
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)

                # Update robot state and visualization
                self.update_robot_state(results)

                # Check for exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            plt.close('all')

    def smooth_joint_angles(self):
        """Smooth the joint angles using exponential smoothing."""
        alpha = 0.7  # Smoothing factor (0 < alpha < 1)
        previous_joint_angles = self.joint_angles.copy()
        for joint in self.joint_angles:
            self.joint_angles[joint] = (
                alpha * self.joint_angles[joint] +
                (1 - alpha) * previous_joint_angles[joint]
            )