import numpy as np
import pytest
from ik_analytical3d import IKAnalytical3D

# Helper to wrap angles into [-pi, pi]
def wrap_angle(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi

# Compute elbow position from joint angles
def compute_elbow_pos(solver, q):
    sp  = q['shoulder_pitch']
    sy  = q['shoulder_yaw']
    sr  = q['shoulder_roll']
    elb = q['elbow']
    # Transform shoulder pitch, yaw, roll, then elbow link
    T_sp  = solver.transform_matrix(sp, 0, 0, np.pi/2)
    T_sy  = solver.transform_matrix(sy, 0, 0, 0)
    T_sr  = solver.transform_matrix(sr, 0, 0, 0)
    T_elb = solver.transform_matrix(elb, 0, solver.L1, 0)
    T_e   = T_sp @ T_sy @ T_sr @ T_elb
    return T_e[:3,3]

# Test poses: neutral, elbow 90°, mixed angles
@pytest.fixture(scope='module')
def solver():
    return IKAnalytical3D()

test_sets = [
    {'shoulder_pitch':0,                 'shoulder_yaw':0,                 'shoulder_roll':0,
     'elbow':0,                          'wrist_pitch':0,                  'wrist_yaw':0,    'wrist_roll':0},

    {'shoulder_pitch':0,                 'shoulder_yaw':0,                 'shoulder_roll':0,
     'elbow':np.pi/2,                   'wrist_pitch':0,                  'wrist_yaw':0,    'wrist_roll':0},

    {'shoulder_pitch':np.pi/6,           'shoulder_yaw':-np.pi/8,          'shoulder_roll':np.pi/8,
     'elbow':np.pi/3,                   'wrist_pitch':-np.pi/12,          'wrist_yaw':np.pi/12,'wrist_roll':np.pi/6},
]

@pytest.mark.parametrize('q_truth', test_sets)
def test_round_trip(solver, q_truth):
    # Compute FK positions and orientation
    elbow_pos, wrist_pos, orientation = None, None, None
    elbow_pos = compute_elbow_pos(solver, q_truth)
    wrist_pos, orientation = solver.forward_kinematics(q_truth)

    # Solve IK using FK results
    q_est = solver.solve(np.zeros(3), elbow_pos, wrist_pos, target_orientation=orientation)

    # Compare each joint angle
    for joint, true_val in q_truth.items():
        est_val = q_est[joint]
        diff = abs(wrap_angle(est_val - true_val))
        assert diff < 1e-6, f"Joint {joint} differs by {diff:.2e} rad" 