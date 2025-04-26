#!/usr/bin/env python3
"""
audit_transform.py

Compare IKAnalytical3D.forward_kinematics output to MuJoCo's body positions
for a handful of hard-coded joint configurations to catch any DH axis or sign mismatches.
"""
import numpy as np
import mujoco

def load_mujoco_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    return model, data

# Define a few test joint-angle sets
test_sets = [
    {'shoulder_pitch':0, 'shoulder_yaw':0, 'shoulder_roll':0, 'elbow':0, 'wrist_pitch':0, 'wrist_yaw':0, 'wrist_roll':0},
    {'shoulder_pitch':np.pi/6, 'shoulder_yaw':-np.pi/8, 'shoulder_roll':np.pi/8,
     'elbow':np.pi/4, 'wrist_pitch':np.pi/12, 'wrist_yaw':-np.pi/12, 'wrist_roll':np.pi/6},
    {'shoulder_pitch':-np.pi/4,'shoulder_yaw':np.pi/4, 'shoulder_roll':-np.pi/6,
     'elbow':np.pi/2, 'wrist_pitch':-np.pi/6,'wrist_yaw':np.pi/8,   'wrist_roll':-np.pi/4}
]

joint_list = ['shoulder_pitch','shoulder_yaw','shoulder_roll','elbow','wrist_pitch','wrist_yaw','wrist_roll']

def main():
    xml_path = 'unitree_g1/g1_with_hands.xml'
    model, data = load_mujoco_model(xml_path)
    solver = mujoco.wrapper.MjWrapper(model) if False else None
    # Actually import your analytic solver
    from ik_analytical3d import IKAnalytical3D
    solver = IKAnalytical3D()

    print('Comparing FK positions: Solver vs MuJoCo')
    for idx, q in enumerate(test_sets):
        # 1. Compute solver FK
        fk_pos, _ = solver.forward_kinematics(q)
        # 2. Set MuJoCo qpos
        data.qpos[:] = 0
        for joint in joint_list:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = q[joint]
        mujoco.mj_forward(model, data)
        # 3. Read body position from MuJoCo (use body 'right_wrist')
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist')
        if bid < 0:
            print('Could not find body "right_wrist"; skipping MuJoCo comparison')
            continue
        mj_pos = data.body_xpos[bid]
        diff = fk_pos - mj_pos
        print(f'Test {idx}: diff = {diff[0]:+.6f}, {diff[1]:+.6f}, {diff[2]:+.6f}')

if __name__ == '__main__':
    main() 