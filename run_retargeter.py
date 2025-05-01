import cv2
import mediapipe as mp
import pygame
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import csv
from datetime import datetime

from robot_model import RobotModel
from retargeter import Retargeter

class PoseMirrorWithRetargeting:
    def __init__(self, window_size=(1280, 720), recording_freq=10):
        # Set up MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )

        # self.hands = mp.solutions.hands.Hands(
        #     static_image_mode = False,
        #     max_num_hands = 2,
        #     min_detection_confidence = 0.7,
        #     min_tracking_confidence = 0.7
        # )
        
        # Initialize pygame display
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Pose Mirror - Improved Retargeting")
        
        # Initialize plot for robot visualization
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Initialize the new retargeter
        self.retargeter = Retargeter(10)
        
        # Recording settings
        self.is_recording = False
        self.recording_freq = recording_freq
        self.csv_file = None
        self.csv_writer = None
        self.frame_counter = 0
        self.last_record_time = 0
        self.record_interval = 1.0 / recording_freq
        
        # UI elements
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255)
        }

        # for calibration
        self.calibrating = False
        self.calib_request_t = 0
        self.calib_delay_ms = 5000

        # for recording
        self.recording_requested   = False
        self.record_request_t      = 0
        self.record_delay_ms       = 5000  # ms
        
    def start_recording(self, filename=None):
        """Start recording joint angles to CSV file."""
        if self.is_recording:
            print("Already recording")
            return
            
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"robot_motion_{timestamp}.csv"
            
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header
        header = ["timestamp"]
        ordered_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        header.extend(ordered_joints)
        self.csv_writer.writerow(header)
        
        self.is_recording = True
        self.start_time = time.time()
        self.last_record_time = self.start_time
        self.frame_counter = 0
        
        print(f"Recording started to {filename} at {self.recording_freq}Hz")
        
    def stop_recording(self):
        """Stop recording and close the CSV file."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        print("Recording stopped")
        
    def record_frame(self, joint_angles):
        """Records current joint angles to CSV."""
        if not self.is_recording:
            return
            
        current_time = time.time()
        # Check if it's time to record based on desired frequency
        if current_time - self.last_record_time >= self.record_interval:
            # Calculate timestamp
            timestamp = self.frame_counter * (1.0 / self.recording_freq)
            
            # Generate row using the retargeter
            row = self.retargeter.generate_csv_row(timestamp, joint_angles)
            
            # Write to CSV
            self.csv_writer.writerow(row)
            self.last_record_time = current_time
            self.frame_counter += 1
    
    def visualize_robot(self, joint_angles):
        """Visualize the robot using joint angles."""
        # Clear the plot
        self.ax.clear()
        
        # Set labels and limits (X=Forward, Y=Right, Z=Up)
        self.ax.set_xlabel('Z (Forward →)')
        self.ax.set_ylabel('X (Right →)')
        self.ax.set_zlabel('Y (Up ↑)')
        
        limit = 0.4
        self.ax.set_xlim3d(-limit, limit)
        self.ax.set_ylim3d(-limit, limit)
        self.ax.set_zlim3d(-limit, limit)
        self.ax.grid(True)
        
        # Get joint positions using forward kinematics
        positions = self.retargeter.robot_model.forward_kinematics(joint_angles)
        
        # Helper to remap robot coords → plot coords
        def to_plot(p):
            x_r, y_r, z_r = p
            return (-z_r, x_r , -y_r)
        
        # Draw the robot skeleton
        if 'torso' in positions and 'left_shoulder' in positions and 'right_shoulder' in positions:
            # Draw torso
            self.ax.scatter(*to_plot(positions['torso']), c='black', marker='o', s=50)
            
            # Connect shoulders
            ls = to_plot(positions['left_shoulder'])
            rs = to_plot(positions['right_shoulder'])
            self.ax.plot(
                [ls[0], rs[0]],
                [ls[1], rs[1]],
                [ls[2], rs[2]],
                'k-', linewidth=3
            )
            
            # Draw left arm
            if 'left_elbow' in positions and 'left_wrist' in positions:
                sh, el, wr = (to_plot(positions[j]) for j in 
                            ['left_shoulder', 'left_elbow', 'left_wrist'])
                
                # Upper arm
                self.ax.plot(
                    [sh[0], el[0]],
                    [sh[1], el[1]],
                    [sh[2], el[2]],
                    color='green', linewidth=3
                )
                # Forearm
                self.ax.plot(
                    [el[0], wr[0]],
                    [el[1], wr[1]],
                    [el[2], wr[2]],
                    color='green', linewidth=3
                )
                # Joints
                self.ax.scatter(*sh, c='green', marker='o', s=80)
                self.ax.scatter(*el, c='green', marker='o', s=80)
                self.ax.scatter(*wr, c='green', marker='o', s=80)
            
            # Draw right arm
            if 'right_elbow' in positions and 'right_wrist' in positions:
                sh, el, wr = (to_plot(positions[j]) for j in 
                            ['right_shoulder', 'right_elbow', 'right_wrist'])
                
                # Upper arm
                self.ax.plot(
                    [sh[0], el[0]],
                    [sh[1], el[1]],
                    [sh[2], el[2]],
                    color='blue', linewidth=3
                )
                # Forearm
                self.ax.plot(
                    [el[0], wr[0]],
                    [el[1], wr[1]],
                    [el[2], wr[2]],
                    color='blue', linewidth=3
                )
                # Joints
                self.ax.scatter(*sh, c='blue', marker='o', s=80)
                self.ax.scatter(*el, c='blue', marker='o', s=80)
                self.ax.scatter(*wr, c='blue', marker='o', s=80)
        
        # Set view angle
        self.ax.view_init(elev=0, azim=0)
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Main loop to run the system."""
        cap = cv2.VideoCapture(0)
        
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            #hand_results = self.hands.process(frame_rgb)

            #hand_world = {'Left': None, 'Right': None}
            
            # Process landmarks if detected
            if results.pose_landmarks and results.pose_world_landmarks: #and hand_results.multi_hand_world_landmarks:
                # for lm_set, hand_class in zip(
                #     hand_results.multi_hand_world_landmarks,
                #     hand_results.multi_handedness
                # ):
                #     label = hand_class.classification[0].label #Left or Right
                #     hand_world[label] = lm_set.landmark
                
                # Calculate joint angles using our new retargeter
                joint_angles = self.retargeter.process_frame(results.pose_world_landmarks.landmark) #,hand_world)
                
                # Visualize robot
                self.visualize_robot(joint_angles)
                
                # Record if recording is enabled
                self.record_frame(joint_angles)
                
                # Draw pose landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                # Convert frame for pygame display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
                self.screen.blit(frame_surface, (0,0))
                
                # Display status information
                calibration_text = self.font.render(
                    "Calibrated" if self.retargeter.is_calibrated else "Not Calibrated", 
                    True, 
                    self.colors["green"] if self.retargeter.is_calibrated else self.colors["red"]
                )
                self.screen.blit(calibration_text, (10, 10))
                
                recording_text = self.font.render(
                    f"Recording @ {self.recording_freq}Hz" if self.is_recording else "Not Recording", 
                    True, 
                    self.colors["red"] if self.is_recording else self.colors["white"]
                )
                self.screen.blit(recording_text, (10, 50))
                
                # Draw key commands
                commands = [
                    "C - Calibrate",
                    "R - Start/stop recording",
                    "Q - Quit",
                    "Arrow keys - Rotate view"
                ]
                for i, cmd in enumerate(commands):
                    cmd_text = self.small_font.render(cmd, True, self.colors["white"])
                    self.screen.blit(cmd_text, (10, self.window_size[1] - 110 + i*25))
                
                # ——— non‐blocking calibration countdown ———
                now = pygame.time.get_ticks()
                if self.calibrating:
                    elapsed = now - self.calib_request_t
                    remaining_s = max(0, (self.calib_delay_ms - elapsed)//1000 + 1)
                    # draw it on screen
                    txt = self.small_font.render(f"Calibrating in {remaining_s}s", True, self.colors["red"])
                    self.screen.blit(txt, (10, 90))
                    if elapsed >= self.calib_delay_ms:
                        # fire!
                        print("\nRunning calibration now…")
                        self.retargeter.calibrate(results.pose_world_landmarks.landmark)
                        self.retargeter.prev_angles = {}
                        self.calibrating = False

                # ——— non‐blocking recording countdown ———
                if self.recording_requested:
                    elapsed = now - self.record_request_t
                    remaining_s = max(0, (self.record_delay_ms - elapsed)//1000 + 1)
                    txt = self.small_font.render(
                        ("Stopping" if self.is_recording else "Starting") +
                        f" recording in {remaining_s}s", 
                        True, self.colors["blue"]
                    )
                    self.screen.blit(txt, (10, 130))
                    if elapsed >= self.record_delay_ms:
                        # toggle now
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                        self.recording_requested = False

                pygame.display.flip()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        # Calibrate
                        if results.pose_world_landmarks:
                            print("Calibrating...")
                            print("Calibration requested — hold still for 5 s")
                            self.calibrating     = True
                            self.calib_request_t = pygame.time.get_ticks()
                    elif event.key == pygame.K_r:
                        # Toggle recording
                        print("Recording requested - hold pose for 5 s")
                        self.recording_requested = True
                        self.record_request_t = pygame.time.get_ticks()
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
        
        # Clean up when done
        if self.is_recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        plt.close()

if __name__ == "__main__":
    app = PoseMirrorWithRetargeting()
    app.run()