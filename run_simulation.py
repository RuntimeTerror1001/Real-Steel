import os
import numpy as np
import pandas as pd
import math
import time
import platform
import mujoco
import mujoco.viewer

def load_model(xml_path):
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
    df = pd.read_csv(csv_path)
    print("Available columns in CSV:", df.columns.tolist())
    timestamps = df['timestamp'].values
    joint_trajectories = {}
    for joint in df.columns:
        if joint == 'timestamp':
            continue
        try:
            angles = np.array([float(angle) for angle in df[joint].values])
            joint_trajectories[joint] = angles
            print(f"Processed joint {joint} - Sample angle: {angles[0]:.4f} rad ({math.degrees(angles[0]):.1f}°)")
        except Exception as e:
            print(f"Warning: Could not process data for joint {joint}: {e}")
    return joint_trajectories, timestamps

def create_joint_mapping(model, joint_names_from_csv):
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
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"Error launching viewer: {e}")
        return

    mujoco.mj_resetData(model, data)
    if hasattr(model, 'key_qpos'):
        data.qpos[:] = model.key_qpos[0]

    frame_time = 1.0 / 30
    last_frame_time = time.time()
    current_frame = 0
    total_frames = len(timestamps)

    print(f"Starting simulation with {total_frames} frames of motion data...")
    try:
        while current_frame < total_frames and viewer.is_running():
            current_time = time.time()
            elapsed = current_time - last_frame_time

            if elapsed >= frame_time:
                last_frame_time = current_time

                for joint_name, joint_id in joint_mapping.items():
                    if joint_name in joint_trajectories:
                        angle = joint_trajectories[joint_name][current_frame]
                        joint_adr = model.jnt_qposadr[joint_id]
                        data.qpos[joint_adr] = angle

                mujoco.mj_forward(model, data)
                viewer.sync()
                current_frame += 1
            else:
                time.sleep(0.001)

        print("Simulation complete.")
        print("Viewer will remain open until closed.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
    finally:
        viewer.close()

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python run_simulation.py <xml_path> <csv_path>")
        return

    xml_path = sys.argv[1]
    csv_path = sys.argv[2]

    if not os.path.exists(xml_path):
        print(f"Error: XML file {xml_path} not found!")
        return
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found!")
        return

    model, data = load_model(xml_path)
    joint_trajectories, timestamps = process_motion_data(csv_path)
    joint_names = list(joint_trajectories.keys())
    joint_mapping = create_joint_mapping(model, joint_names)
    run_simulation(model, data, joint_trajectories, joint_mapping, timestamps)

if __name__ == "__main__":
    main()                                                                                                                                                                              
