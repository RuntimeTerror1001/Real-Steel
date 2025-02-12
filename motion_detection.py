import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Open3D for 3D rendering
def initialize_3d_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Humanoid Motion', width=800, height=600)
    return vis

# Function to update the 3D humanoid figure based on landmarks
def update_3d_humanoid(vis, landmarks_3d, connections):
    # Clear previous frame
    vis.clear_geometries()

    # Create point cloud for landmarks
    points = o3d.utility.Vector3dVector(landmarks_3d)
    point_cloud = o3d.geometry.PointCloud(points)
    point_cloud.paint_uniform_color([1, 0, 0])  # Red color for points

    # Create lines for connections
    lines = o3d.utility.Vector2iVector(connections)
    line_set = o3d.geometry.LineSet()
    line_set.points = points
    line_set.lines = lines
    line_set.paint_uniform_color([0, 1, 0])  # Green color for lines

    # Add geometries to the visualizer
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    # Update the visualizer
    vis.update_geometry(point_cloud)
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()

# Main loop
def main():
    cap = cv2.VideoCapture(0)
    vis = initialize_3d_visualizer()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with MediaPipe Pose
            height, width, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Extract 3D landmarks
                landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

                # Rotate to frontal view (optional, based on shoulder angle)
                left_shoulder = landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                shoulder_vector = right_shoulder - left_shoulder
                angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])

                rotation_matrix = np.array([
                    [np.cos(-angle), -np.sin(-angle), 0],
                    [np.sin(-angle), np.cos(-angle), 0],
                    [0, 0, 1]
                ])
                landmarks_3d = np.dot(landmarks_3d, rotation_matrix)

                # Convert POSE_CONNECTIONS to a NumPy array of shape (N, 2)
                connections = np.array([[conn[0], conn[1]] for conn in mp_pose.POSE_CONNECTIONS])

                # Update the 3D humanoid figure
                update_3d_humanoid(vis, landmarks_3d, connections)

            # Display the original frame
            cv2.imshow('Real-Time Motion Detection', image)

            # Exit on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == "__main__":
    main()