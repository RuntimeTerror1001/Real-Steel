#!/usr/bin/env python3
import os
import mujoco
import numpy as np
import pandas as pd
import math

def validate_simulator_config(xml_path, csv_path):
    results = []

    if not os.path.exists(xml_path):
        results.append(("XML File", False, f"XML file not found at {xml_path}"))
        return results

    if not os.path.exists(csv_path):
        results.append(("CSV File", False, f"CSV file not found at {csv_path}"))
        return results

    # Load model
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    xml_file = os.path.basename(xml_path)
    cwd = os.getcwd()
    try:
        os.chdir(xml_dir)
        xml = open(xml_file, 'r').read()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    finally:
        os.chdir(cwd)

    # Check MuJoCo version
    mujoco_version = mujoco.__version__
    results.append(("MuJoCo Version", True, f"{mujoco_version}"))

    # Joint name check
    mujoco_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    results.append(("MuJoCo Joint Count", True, f"{len(mujoco_joint_names)} joints found"))

    # CSV columns
    df = pd.read_csv(csv_path)
    csv_columns = df.columns.tolist()
    if 'timestamp' in csv_columns:
        csv_columns.remove('timestamp')

    matched, unmatched = [], []
    for joint in csv_columns:
        if joint in mujoco_joint_names:
            matched.append(joint)
        else:
            unmatched.append(joint)
    results.append(("CSV Joint Matching", len(unmatched) == 0,
                    f"{len(matched)} matched, {len(unmatched)} unmatched: {unmatched}"))

    # Angle value range
    try:
        sample_angles = df[csv_columns].iloc[0].astype(float).values
        deg_angles = np.degrees(sample_angles)
        if np.any(np.abs(deg_angles) > 180):
            results.append(("Angle Units Check", False, "Angles appear to be in degrees"))
        else:
            results.append(("Angle Units Check", True, "Angles likely in radians"))
    except Exception as e:
        results.append(("Angle Units Check", False, f"Failed to parse angles: {e}"))

    # qpos sanity check
    qpos_range = (np.min(data.qpos), np.max(data.qpos))
    results.append(("qpos Range", True, f"{qpos_range}"))

    return results

if __name__ == "__main__":
    xml_path = "unitree_mujoco/unitree_robots/g1/scene_with_hands.xml"
    csv_path = "src/simulation/test.csv"
    results = validate_simulator_config(xml_path, csv_path)
    
    print("\n=== MuJoCo Simulator Configuration Check ===")
    for check, passed, detail in results:
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {detail}") 