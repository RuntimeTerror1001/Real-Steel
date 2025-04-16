#!/usr/bin/env python3
import os
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import math
import time

def load_model(xml_path):
    """Load the MuJoCo model from an XML file."""
    original_dir = os.getcwd()
    try:
        abs_xml_path = os.path.abspath(xml_path)
        xml_dir = os.path.dirname(abs_xml_path)
        xml_filename = os.path.basename(abs_xml_path)
        print(f"Changing directory to: {xml_dir}")
        os.chdir(xml_dir)
        print(f"Loading XML file: {xml_filename}")
        xml = open(xml_filename, 'r').read()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    finally:
        print(f"Changing back to original directory: {original_dir}")
        os.chdir(original_dir)

def process_motion_data(csv_path):
    """Process the CSV file to extract joint angle data."""
    df = pd.read_csv(csv_path)
    print("Available columns in CSV:", df.columns.tolist())
    timestamps = df['timestamp'].values
    joint_trajectories = {}
    for joint in df.columns:
        if joint == 'timestamp':
            continue
        try:
            angles = np.array([float(angle) for angle in df[joint].values])
            min_limit, max_limit = self.ik.joint_limits[joint]  # radians
            angles = np.clip(angles, min_limit, max_limit)
            angles_deg = np.degrees(angles)
            joint_trajectories[joint] = angles
            print(f"Processed joint {joint} - Sample angle: {angles[0]:.4f} rad ({angles_deg[0]:.1f}°)")
        except Exception as e:
            print(f"Warning: Could not process data for joint {joint}: {e}")
    return joint_trajectories, timestamps

def create_joint_mapping(model, joint_names_from_csv):
    """Create a mapping from CSV joint names to MuJoCo joint indices."""
    joint_mapping = {}
    mujoco_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("MuJoCo joints:", mujoco_joints)
    for joint_name in joint_names_from_csv:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            joint_mapping[joint_name] = joint_id
            print(f"Mapped {joint_name} to MuJoCo joint ID {joint_id}")
        else:
            print(f"Warning: Could not find MuJoCo joint ID for {joint_name}")
    return joint_mapping

def run_simulation(model, data, joint_trajectories, joint_mapping, timestamps):
    """Run the MuJoCo simulation using the processed motion data."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        # Set initial pose using the model's keyframe (assume key_qpos is defined)
        data.qpos[:] = model.key_qpos[0]
        
        # Set up camera parameters
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.5
        viewer.sync()
        time.sleep(1.0)
        
        target_fps = 30  # Frames per second
        frame_time = 1.0 / target_fps
        current_frame = 0
        total_frames = len(timestamps)
        print(f"Starting simulation with {total_frames} frames of motion data...")
        print(f"Running at {target_fps} FPS")
        print(f"Camera settings: distance={viewer.cam.distance}, azimuth={viewer.cam.azimuth}, elevation={viewer.cam.elevation}")
        print("\nYou can adjust the camera view manually in the MuJoCo viewer:")
        print("  - Hold right mouse button and move to rotate")
        print("  - Scroll to zoom in/out")
        print("  - Hold middle mouse button and move to pan")
        print("  - Press ESC to exit")
        
        last_frame_time = time.time()
        
        while current_frame < total_frames and viewer.is_running():
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            if elapsed >= frame_time:
                last_frame_time = current_time
                if current_frame % 30 == 0:
                    print(f"Processing frame {current_frame}/{total_frames} ({current_frame / total_frames * 100:.1f}%)")
                
                # Update each joint angle from CSV data
                for joint_name, joint_id in joint_mapping.items():
                    if joint_name in joint_trajectories:
                        angle = joint_trajectories[joint_name][current_frame]
                        print(f"Frame {current_frame}, Joint {joint_name}: {angle:.3f} rad ({math.degrees(angle):.1f}°)")

                        joint_adr = model.jnt_qposadr[joint_id]
                        data.qpos[joint_adr] = angle
                
                # Propagate the changes in the simulation
                mujoco.mj_forward(model, data)
                viewer.sync()
                current_frame += 1
            else:
                time.sleep(0.001)
        
        print("Simulation complete.")
        print("Playback finished. Viewer will remain open until closed.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)

def main():
    xml_path = "unitree_g1/g1.xml"  # MuJoCo model path
    csv_path = "test.csv"           # CSV with recorded motion data
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file {xml_path} not found!")
        return
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    print(f"Loading MuJoCo model from {xml_path}...")
    model, data = load_model(xml_path)
    
    print(f"Processing motion data from {csv_path}...")
    joint_trajectories, timestamps = process_motion_data(csv_path)
    
    joint_names = list(joint_trajectories.keys())
    joint_mapping = create_joint_mapping(model, joint_names)
    
    print("Running simulation...")
    run_simulation(model, data, joint_trajectories, joint_mapping, timestamps)

if __name__ == "__main__":
    main()
