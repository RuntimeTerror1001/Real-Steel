import cv2 
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def normalize_coords(landmarks, width, height):
    left_shldr = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width, 
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y*height])
    
    right_shldr = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*height])
    
    center = (left_shldr+right_shldr)/2

    norm_landmarks = []
    for landmark in landmarks:
        x,y = (landmark.x * width) - center[0], height - (landmark.y * height) - center[1]  
        norm_landmarks.append((x,y)) 

    return np.array(norm_landmarks)

def rotate_landmarks(landmarks, angle):
    rotation_matrix = np.array([
        [np.cos(angle),-np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    rotated_landmarks = np.dot(landmarks, rotation_matrix)
    return rotated_landmarks 

cap=cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        height, width, _ = frame.shape

        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame.flags.writeable=False

        results = pose.process(original_frame)

        original_frame.flags.writeable=True
        original_frame=cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)

        processed_frame = np.zeros_like(original_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            norm_landmarks = normalize_coords(landmarks, width, height)

            left_shoulder = norm_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = norm_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_vector = right_shoulder - left_shoulder
            angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])

            rotated_landmarks = rotate_landmarks(norm_landmarks, -angle)

            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection

                if start_idx < len(rotated_landmarks) and end_idx < len(rotated_landmarks):
                    x1, y1 = rotated_landmarks[start_idx]
                    x2, y2 = rotated_landmarks[end_idx]

                    if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                        cv2.line(processed_frame, 
                                (int(x1 + width // 2), int(y1 + height // 2)), 
                                (int(x2 + width // 2), int(y2 + height // 2)), 
                                (0, 0, 255), 3)

            # Ensure valid keypoints before drawing circles
            for x, y in rotated_landmarks:
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(processed_frame, (int(x + width // 2), int(y + height // 2)), 5, (0, 255, 0), -1)


            combined_frame = np.hstack((original_frame, processed_frame))

            combined_frame = cv2.resize(combined_frame, (width*2, height))

        mp_drawing.draw_landmarks(original_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Real Steel Motion Detection', combined_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()