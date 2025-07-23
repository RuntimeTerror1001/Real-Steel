#!/usr/bin/env python3
"""
verify_joint_angles.py

Script to verify that the wrist and other joint angles in a CSV
were correctly calculated by doing a forward–inverse kinematics consistency check.
"""

import sys
import pandas as pd
import numpy as np
from ik_analytical3d import IKAnalytical3D


def compute_elbow_wrist_positions(solver, angles):
    sp = angles['shoulder_pitch']
    sy = angles['shoulder_yaw']
    sr = angles['shoulder_roll']
    elb = angles['elbow']
    wp = angles['wrist_pitch']
    wy = angles['wrist_yaw']
    wr = angles['wrist_roll']
    # assume shoulder at origin
    T_shoulder = solver.transform_matrix(sp, 0, 0, np.pi/2)
    T_yaw = solver.transform_matrix(sy, 0, 0, 0)
    T_roll = solver.transform_matrix(sr, 0, 0, 0)
    T_elbow = solver.transform_matrix(elb, 0, solver.L1, 0)
    T_elbow_full = T_shoulder @ T_yaw @ T_roll @ T_elbow
    elbow_pos = T_elbow_full[:3,3]
    T_wrist_pitch = solver.transform_matrix(wp, 0, solver.L2, 0)
    T_wrist_yaw = solver.transform_matrix(wy, 0, 0, 0)
    T_wrist_roll = solver.transform_matrix(wr, 0, 0, 0)
    T_final = T_elbow_full @ T_wrist_pitch @ T_wrist_yaw @ T_wrist_roll
    wrist_pos = T_final[:3,3]
    orientation = T_final[:3,:3]
    return elbow_pos, wrist_pos, orientation


def verify_angles(csv_path):
    df = pd.read_csv(csv_path)
    solver = IKAnalytical3D()
    joint_names = ['shoulder_pitch', 'shoulder_yaw', 'shoulder_roll', 'elbow', 'wrist_pitch', 'wrist_yaw', 'wrist_roll']
    errors = {jn: [] for jn in joint_names}

    for idx, row in df.iterrows():
        # build angles dictionary from CSV
        angles = {jn: row[f'right_{jn}_joint'] for jn in joint_names}
        # compute FK positions and orientation
        elbow_pos, wrist_pos, orientation = compute_elbow_wrist_positions(solver, angles)
        # solve IK with target orientation
        ik_sol = solver.solve(np.zeros(3), elbow_pos, wrist_pos, target_orientation=orientation)
        # collect abs error per joint
        for jn in joint_names:
            err = abs(ik_sol[jn] - angles[jn])
            errors[jn].append(err)

    # summarize results
    print("IK Consistency Check:")
    for jn in joint_names:
        max_err = np.max(errors[jn])
        mean_err = np.mean(errors[jn])
        print(f"  {jn}: max error = {max_err:.3e} rad, mean error = {mean_err:.3e} rad")

    # determine overall pass/fail
    overall_ok = all(np.max(errors[jn]) < 1e-6 for jn in joint_names)
    if overall_ok:
        print("All joint angles are perfectly consistent (within numerical tolerance).")
        sys.exit(0)
    else:
        print("Some joint angles deviated beyond tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 verify_joint_angles.py <csv_path>")
        sys.exit(1)
    verify_angles(sys.argv[1]) 