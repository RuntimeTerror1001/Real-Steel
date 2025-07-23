#!/usr/bin/env python3
"""
calibrate_scale.py

Measure the average MediaPipe shoulder→elbow distance over a number of frames
and compute a suggested scale factor to map it to the robot's configured upper_arm_length.
"""

import cv2
import numpy as np
import mediapipe as mp

# Number of frames to sample and robot's upper arm length (meters)
NUM_FRAMES = 100
ROBOT_UPPER_ARM_LENGTH = 0.30  # adjust if your robot uses a different value


def main():
    distances = []
    # Initialize MediaPipe Pose
    with mp.solutions.pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as pose:
        cap = cv2.VideoCapture(0)
        for i in range(NUM_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_world_landmarks:
                lm = results.pose_world_landmarks.landmark
                # Right arm landmarks
                s = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                e = lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
                # 3D positions in MP world space
                p_s = np.array([s.x, s.y, s.z])
                p_e = np.array([e.x, e.y, e.z])
                distances.append(np.linalg.norm(p_e - p_s))
        cap.release()

    if not distances:
        print("No valid shoulder/elbow landmarks detected.")
        return

    mean_dist = float(np.mean(distances))
    suggested_scale = ROBOT_UPPER_ARM_LENGTH / mean_dist
    print(f"Mean MP shoulder→elbow distance: {mean_dist:.6f}")
    print(f"Suggested scale factor: {suggested_scale:.3f} (vs. current 0.300)")


if __name__ == '__main__':
    main() 