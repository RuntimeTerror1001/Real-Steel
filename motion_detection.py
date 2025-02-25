import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt

class PoseMirror3D:
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
        pygame.display.set_caption("3D Pose Mirror")
        
        # Initialize matplotlib for 3D visualization
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Set initial view angle for forward-facing
        self.ax.view_init(elev=0, azim=90)
        
        # Style parameters
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
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
        self.scale = 200
        
    def update_3d_plot(self, results):
        """Update the 3D matplotlib visualization"""
        self.ax.clear()
        
        # Set up the 3D axes
        self.ax.set_xlabel('Z')  # Swap axes for forward-facing orientation
        self.ax.set_ylabel('X')
        self.ax.set_zlabel('Y')
        
        # Set fixed axes limits for consistent visualization
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)
        
        # Draw the coordinate grid
        self.ax.grid(True)
        
        if results.pose_world_landmarks:
            # Extract landmark coordinates and remap for forward-facing orientation
            landmarks = results.pose_world_landmarks.landmark
            x = [-landmark.z for landmark in landmarks]  # Negative Z for x-axis
            y = [landmark.x for landmark in landmarks]   # X for y-axis
            z = [-landmark.y for landmark in landmarks]  # Negative Y for z-axis
            
            # Plot landmarks
            self.ax.scatter(x, y, z, c='r', marker='o')
            
            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                self.ax.plot([x[start_idx], x[end_idx]],
                           [y[start_idx], y[end_idx]],
                           [z[start_idx], z[end_idx]], 'b-')
        
        # Maintain view angle
        self.ax.view_init(elev=0, azim=0)
        
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
                        # Reset view to forward-facing
                        self.ax.view_init(elev=0, azim=90)
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
                
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        plt.close()

if __name__ == "__main__":
    mirror = PoseMirror3D()
    mirror.run()