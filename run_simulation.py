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
    # Store the original working directory
    original_dir = os.getcwd()
    
    try:
        # Get the absolute path to the XML file
        abs_xml_path = os.path.abspath(xml_path)
        
        # Get the directory containing the XML file
        xml_dir = os.path.dirname(abs_xml_path)
        
        # Get just the filename part
        xml_filename = os.path.basename(abs_xml_path)
        
        # Change to that directory so relative paths work
        print(f"Changing directory to: {xml_dir}")
        os.chdir(xml_dir)
        
        # Now load the XML file (use just the filename since we've changed directories)
        print(f"Loading XML file: {xml_filename}")
        xml = open(xml_filename, 'r').read()
        
        # Load the model
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        return model, data
    finally:
        # Always change back to the original directory when done
        print(f"Changing back to original directory: {original_dir}")
        os.chdir(original_dir)

def process_motion_data(csv_path):
    """Process the CSV file to extract joint angle data."""
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Print column names to help with debugging
    print("Available columns in CSV:", df.columns.tolist())
    
    # Get timestamps from the first column
    timestamps = df['timestamp'].values
    
    # Create a dictionary to store joint angles over time
    joint_trajectories = {}
    
    # Process each joint column (skip the timestamp column)
    for joint in df.columns:
        if joint == 'timestamp':
            continue
        
        # Convert string values to float angles (radians)
        try:
            angles = np.array([float(angle) for angle in df[joint].values])
            joint_trajectories[joint] = angles
            print(f"Processed joint {joint} - Sample angle: {angles[0]:.4f} rad ({math.degrees(angles[0]):.1f}°)")
        except Exception as e:
            print(f"Warning: Could not process data for joint {joint}: {e}")
    
    return joint_trajectories, timestamps

def create_joint_mapping(model, joint_names_from_csv):
    """Create a mapping from CSV joint names to MuJoCo joint indices."""
    joint_mapping = {}
    
    # Print MuJoCo joint names for reference
    mujoco_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print("MuJoCo joints:", mujoco_joints)
    
    # Direct mapping - your CSV joint names match the MuJoCo joint names
    for joint_name in joint_names_from_csv:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:  # Valid joint ID
            joint_mapping[joint_name] = joint_id
            print(f"Mapped {joint_name} to MuJoCo joint ID {joint_id}")
        else:
            print(f"Warning: Could not find MuJoCo joint ID for {joint_name}")
    
    return joint_mapping

def run_simulation(model, data, joint_trajectories, joint_mapping, timestamps):
    """Run the MuJoCo simulation using the processed motion data."""
    # Initialize the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial state
        mujoco.mj_resetData(model, data)
        
        # Set up camera position to show the full robot
        viewer.cam.distance = 3.0      # Distance from target - increase to zoom out
        viewer.cam.azimuth = 180.0      # Camera rotation around z axis in degrees
        viewer.cam.elevation = -20.0   # Camera elevation in degrees
        viewer.cam.lookat[0] = 0.0     # Target x position
        viewer.cam.lookat[1] = 0.0     # Target y position
        viewer.cam.lookat[2] = 0.5     # Target z position
        
        # Apply the camera settings
        viewer.sync()
        
        # Wait for the viewer to initialize
        time.sleep(1.0)
        
        # Use a fixed frame rate instead of trying to match real-time exactly
        target_fps = 30  # Target frames per second
        frame_time = 1.0 / target_fps
        
        current_frame = 0
        total_frames = len(timestamps)
        
        print(f"Starting simulation with {total_frames} frames of motion data...")
        print(f"Running at {target_fps} FPS")
        print(f"Camera settings: distance={viewer.cam.distance}, azimuth={viewer.cam.azimuth}, elevation={viewer.cam.elevation}")
        
        # Simple instructions for adjusting view manually
        print("\nYou can adjust the camera view manually in the MuJoCo viewer:")
        print("  - Hold right mouse button and move to rotate")
        print("  - Scroll to zoom in/out")
        print("  - Hold middle mouse button and move to pan")
        print("  - Press ESC to exit")
        
        last_frame_time = time.time()
        
        while current_frame < total_frames and viewer.is_running():
            # Calculate elapsed time since last frame
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # If it's time to show the next frame
            if elapsed >= frame_time:
                # Update time tracker
                last_frame_time = current_time
                
                # Print progress occasionally
                if current_frame % 30 == 0:
                    print(f"Processing frame {current_frame}/{total_frames} ({current_frame / total_frames * 100:.1f}%)")
                
                # Set joint angles in MuJoCo
                for joint_name, joint_id in joint_mapping.items():
                    if joint_name in joint_trajectories:
                        angle = joint_trajectories[joint_name][current_frame]
                        joint_adr = model.jnt_qposadr[joint_id]
                        data.qpos[joint_adr] = angle
                
                # Step the simulation
                mujoco.mj_forward(model, data)
                
                # Update the viewer
                viewer.sync()
                
                # Move to next frame
                current_frame += 1
            else:
                # Small sleep to prevent high CPU usage
                time.sleep(0.001)
        
        print("Simulation complete.")
        
        # Keep the viewer open until the user exits
        print("Playback finished. Viewer will remain open until closed.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)

def main():
    # File paths
    xml_path = "unitree_g1/g1.xml"  # Unitree G1 model
    csv_path = "test.csv"  # Your recorded motion data
    
    # Check if files exist
    if not os.path.exists(xml_path):
        print(f"Error: XML file {xml_path} not found!")
        return
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return
    
    # Load the model
    print(f"Loading MuJoCo model from {xml_path}...")
    model, data = load_model(xml_path)
    
    # Process the motion data
    print(f"Processing motion data from {csv_path}...")
    joint_trajectories, timestamps = process_motion_data(csv_path)
    
    # Create joint mapping
    joint_names = list(joint_trajectories.keys())
    joint_mapping = create_joint_mapping(model, joint_names)
    
    # Run the simulation
    print("Running simulation...")
    run_simulation(model, data, joint_trajectories, joint_mapping, timestamps)

if __name__ == "__main__":
    main()