import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from robot_retargeter import RobotRetargeter

class PoseMirror3DWithRetargeting:
    def __init__(self, window_size=(640, 480)):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        
        # Initialize main pygame window
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("3D Pose Mirror - With Robot Retargeting")
        
        # Initialize matplotlib for 3D visualization
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Set initial view angle for forward-facing
        # Changed to match figure orientation
        self.ax.view_init(elev=0, azim=0)
        
        # Initialize the robot retargeter
        self.robot_retargeter = RobotRetargeter(recording_freq=20)  # 20Hz recording
        
        # Style parameters
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        
        # Drawing styles
        self.landmark_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        self.connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=2
        )
        
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.scale = 200
        
        # Store the current rotation angle with a sensible default
        self.current_rotation_angle = 0
        self.initial_angle_set = False
        self.angle_offset = 0
        
        # Smoothing factor for angle (higher = more smoothing)
        self.smoothing_factor = 0.8
        
        # Maintain a history of recent chest vectors for stability
        self.recent_chest_vectors = []
        self.max_history = 5  # Number of frames to keep in history
        
    def calculate_body_plane_angle(self, landmarks):
        """
        Calculate the angle between the body plane and camera plane using chest orientation.
        Updated for correct MediaPipe coordinate system.
        """
        if not landmarks:
            return 0
        
        # Get key upper body landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # We'll also use the nose as a reference point for the front of the body
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Calculate the midpoint between shoulders (represents the center of the chest)
        chest_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        chest_mid_z = (left_shoulder.z + right_shoulder.z) / 2
        
        # Based on the debug output and coordinate mapping:
        # X needs to be flipped, Y is mapped to Z, Z is mapped to Y
        # So chest-to-nose vector in our target coordinates would be:
        # [-(nose.x - chest_mid_x), -(nose.z - chest_mid_z), -(nose.y - chest_mid_y)]
        
        # But for angle calculation, we only need the horizontal plane (X and Z in our mapping)
        chest_to_nose_x = -(nose.x - chest_mid_x)  # Flipped X
        chest_to_nose_z = -(nose.y - chest_mid_y)  # Y mapped to Z
        
        # Normalize this vector
        magnitude = math.sqrt(chest_to_nose_x**2 + chest_to_nose_z**2)
        if magnitude > 0:
            chest_to_nose_x /= magnitude
            chest_to_nose_z /= magnitude
        
        # Add to history for smoothing
        self.recent_chest_vectors.append((chest_to_nose_x, chest_to_nose_z))
        if len(self.recent_chest_vectors) > self.max_history:
            self.recent_chest_vectors.pop(0)
        
        # Average the recent vectors for stability
        avg_x = sum(v[0] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
        avg_z = sum(v[1] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
        
        # Calculate the angle between chest normal vector and the Z-axis
        # For our mapped coordinates, forward is positive Z
        raw_angle = math.degrees(math.atan2(avg_x, avg_z))
        
        # Set an initial reference angle if not set
        if not self.initial_angle_set:
            # Only calibrate when the person is reasonably stable and detected well
            if len(self.recent_chest_vectors) >= 3:  # Wait for enough history
                self.angle_offset = raw_angle
                self.initial_angle_set = True
            
        # Calculate relative angle from initial position
        relative_angle = raw_angle - self.angle_offset
        
        # Normalize angle to stay within -180 to 180 degrees
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360
        
        # Apply smoothing (weighted moving average)
        self.current_rotation_angle = self.current_rotation_angle * self.smoothing_factor + relative_angle * (1 - self.smoothing_factor)
        
        # Apply another normalization after smoothing
        while self.current_rotation_angle > 180:
            self.current_rotation_angle -= 360
        while self.current_rotation_angle < -180:
            self.current_rotation_angle += 360
            
        return self.current_rotation_angle
    
    def print_pose_info(self, results):
        """
        Print information about MediaPipe pose world landmarks to understand coordinate system.
        Add this to the PoseMirror3DWithRetargeting class for debugging.
        """
        if not results.pose_world_landmarks:
            return
            
        # Get some key landmarks
        landmarks = results.pose_world_landmarks.landmark
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        print("\n--- MediaPipe World Landmarks Raw Data ---")
        print(f"Nose: x={nose.x:.4f}, y={nose.y:.4f}, z={nose.z:.4f}")
        print(f"Left Shoulder: x={left_shoulder.x:.4f}, y={left_shoulder.y:.4f}, z={left_shoulder.z:.4f}")
        print(f"Right Shoulder: x={right_shoulder.x:.4f}, y={right_shoulder.y:.4f}, z={right_shoulder.z:.4f}")
        
        # Calculate shoulder width vector (left to right)
        shoulder_vector = [
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
            right_shoulder.z - left_shoulder.z
        ]
        print(f"Shoulder vector (left to right): [{shoulder_vector[0]:.4f}, {shoulder_vector[1]:.4f}, {shoulder_vector[2]:.4f}]")
        
        # Chest to nose vector (approximating forward direction)
        chest_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        chest_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        chest_mid_z = (left_shoulder.z + right_shoulder.z) / 2
        
        forward_vector = [
            nose.x - chest_mid_x,
            nose.y - chest_mid_y,
            nose.z - chest_mid_z
        ]
        print(f"Forward vector (chest to nose): [{forward_vector[0]:.4f}, {forward_vector[1]:.4f}, {forward_vector[2]:.4f}]")
        
        # Call this method in your run loop where results.pose_world_landmarks is checked
        
    def update_3d_plot(self, results):
        """
        Visualization with correct mapping based on debug output.
        MediaPipe world landmark coordinates:
        - x: positive is left (needs to be flipped)
        - y: negative is upward/inward
        - z: negative is forward/upward
        
        Mapping to standard view:
        - X: positive right (flip MediaPipe's x)
        - Y: positive up (using MediaPipe's z, negated)
        - Z: positive forward (using MediaPipe's y, negated)
        """
        self.ax.clear()
        
        # Set up the 3D axes
        self.ax.set_xlabel('X (Left-Right)')
        self.ax.set_ylabel('Y (Up-Down)')
        self.ax.set_zlabel('Z (Forward-Back)')
        
        # Set fixed limits
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)
        
        # Draw grid
        self.ax.grid(True)
        
        if results.pose_world_landmarks:
            # Extract landmarks 
            landmarks = results.pose_world_landmarks.landmark
            
            # Map coordinates to standard view
            x = []  # Right is positive
            y = []  # Up is positive
            z = []  # Forward is positive
            
            for landmark in landmarks:
                # Apply the mapping based on the debug output
                x.append(-landmark.x)    # Flip x to make right positive
                y.append(-landmark.z)    # Use -z for up
                z.append(-landmark.y)    # Use -y for forward
            
            # Plot landmarks
            self.ax.scatter(x, y, z, c='r', marker='o')
            
            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(x) and end_idx < len(x):
                    self.ax.plot([x[start_idx], x[end_idx]],
                            [y[start_idx], y[end_idx]],
                            [z[start_idx], z[end_idx]], 'b-')
        
        # View from the front
        self.ax.view_init(elev=15, azim=270)
        
        # Add coordinate axis markers
        origin = [0, 0, 0]
        self.ax.quiver(*origin, 0.2, 0, 0, color='r', label='X')
        self.ax.quiver(*origin, 0, 0.2, 0, color='g', label='Y')
        self.ax.quiver(*origin, 0, 0, 0.2, color='b', label='Z')
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        cap = cv2.VideoCapture(0)
        
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            rotation_angle = 0
            if results.pose_world_landmarks:
                #self.print_pose_info(results)
                rotation_angle = self.calculate_body_plane_angle(results.pose_world_landmarks.landmark)
                
                # Retarget to robot model
                self.robot_retargeter.retarget_pose(results.pose_world_landmarks, rotation_angle)
                self.robot_retargeter.update_robot_plot()
                
                # Record if we're recording
                self.robot_retargeter.record_frame()
            
            # Update main pygame window
            if results.pose_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style
                )
                
                # Convert frame to pygame surface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
                self.screen.blit(frame_surface, (0,0))
                
                # Render rotation angle on screen
                angle_text = self.font.render(f"Rotation: {rotation_angle:.1f}°", True, self.RED)
                self.screen.blit(angle_text, (10, 10))
                
                # Render direction indicator
                direction = "Right" if rotation_angle > 10 else "Left" if rotation_angle < -10 else "Center"
                direction_text = self.font.render(f"Facing: {direction}", True, self.BLUE)
                self.screen.blit(direction_text, (10, 50))
                
                # Render calibration status
                calibration_text = self.font.render(
                    "Calibrated" if self.initial_angle_set else "Calibrating... face camera", 
                    True, 
                    (0, 255, 0) if self.initial_angle_set else (255, 165, 0)
                )
                self.screen.blit(calibration_text, (10, 90))
                
                # Render recording status
                recording_text = self.font.render(
                    f"Recording @ {self.robot_retargeter.recording_freq}Hz" if self.robot_retargeter.is_recording else "Not Recording", 
                    True, 
                    self.RED if self.robot_retargeter.is_recording else self.WHITE
                )
                self.screen.blit(recording_text, (10, 130))
                
                # Draw keyboard commands
                key_commands = [
                    "R - Reset calibration",
                    "S - Start/stop recording",
                    "Q - Quit",
                    "Arrow keys - Rotate view"
                ]
                
                for i, cmd in enumerate(key_commands):
                    cmd_text = self.small_font.render(cmd, True, self.WHITE)
                    self.screen.blit(cmd_text, (10, self.window_size[1] - 110 + i*25))
                
                # Draw chest direction vector on screen for debugging
                if self.recent_chest_vectors:
                    center_x, center_y = self.window_size[0] - 100, 100
                    avg_x = sum(v[0] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
                    avg_z = sum(v[1] for v in self.recent_chest_vectors) / len(self.recent_chest_vectors)
                    line_length = 50
                    end_x = center_x + avg_x * line_length
                    end_y = center_y + avg_z * line_length
                    pygame.draw.circle(self.screen, self.BLACK, (center_x, center_y), 52)
                    pygame.draw.circle(self.screen, self.WHITE, (center_x, center_y), 50)
                    pygame.draw.line(self.screen, self.RED, (center_x, center_y), (end_x, end_y), 3)
                
                pygame.display.flip()
            
            # Update 3D visualization
            if results.pose_world_landmarks:
                self.update_3d_plot(results)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset calibration
                        self.initial_angle_set = False
                        self.current_rotation_angle = 0
                        self.recent_chest_vectors = []
                    elif event.key == pygame.K_s:
                        # Toggle recording
                        if self.robot_retargeter.is_recording:
                            self.robot_retargeter.stop_recording()
                        else:
                            # Generate filename with timestamp
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            self.robot_retargeter.start_recording(f"robot_motion_{timestamp}.csv")
                    elif event.key == pygame.K_q:
                        running = False
                    # View control keys
                    elif event.key == pygame.K_LEFT:
                        current_azim = self.ax.azim
                        self.ax.view_init(elev=self.ax.elev, azim=current_azim + 10)
                    elif event.key == pygame.K_RIGHT:
                        current_azim = self.ax.azim
                        self.ax.view_init(elev=self.ax.elev, azim=current_azim - 10)
                    elif event.key == pygame.K_UP:
                        current_elev = self.ax.elev
                        self.ax.view_init(elev=current_elev + 10, azim=self.ax.azim)
                    elif event.key == pygame.K_DOWN:
                        current_elev = self.ax.elev
                        self.ax.view_init(elev=current_elev - 10, azim=self.ax.azim)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Make sure to stop recording if we're quitting
        if self.robot_retargeter.is_recording:
            self.robot_retargeter.stop_recording()
            
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        plt.close()