import numpy as np

def angle_diff(a, b):
    """Smallest signed difference between two angles."""
    return ((a - b + np.pi) % (2*np.pi)) - np.pi

class KinematicValidator:
    """
    Validate round-trip consistency of IKAnalytical3D using its own FK and IK methods.
    """
    def __init__(self, solver, error_tolerance=1e-3):
        """
        Args:
            solver: instance of IKAnalytical3D
        """
        self.solver = solver
        self.error_tolerance = error_tolerance
        # DH parameters up to elbow: (joint, d, a, alpha)
        self._elbow_dh = [
            ('shoulder_pitch', 0,          0,      np.pi/2),
            ('shoulder_yaw',   0,          0,      0),
            ('shoulder_roll',  0,          0,      0),
            ('elbow',          0,    solver.L1,      0)
        ]

    def compute_positions(self, joint_angles: dict) -> dict:
        """
        Compute positions of shoulder, elbow, and wrist (end-effector) given joint angles.
        Returns:
            dict with keys 'shoulder','elbow','wrist'
        """
        # Shoulder at origin
        T = np.eye(4)
        positions = {'shoulder': np.zeros(3)}
        # Build transform up to elbow
        for name, d, a, alpha in self._elbow_dh:
            theta = joint_angles[name]
            T = T @ self.solver.transform_matrix(theta, d, a, alpha)
        positions['elbow'] = T[:3, 3]
        # Full FK for wrist
        wrist_pos, wrist_ori = self.solver.forward_kinematics(joint_angles)
        positions['wrist'] = wrist_pos
        positions['wrist_ori'] = wrist_ori
        return positions

    def round_trip_errors(self, joint_angles: dict) -> dict:
        """
        Compute absolute difference between original joint angles and those recovered by IK
        from the positions computed via FK.
        Returns:
            dict mapping each joint to its absolute error in radians.
        """
        # 1) FK positions
        pos = self.compute_positions(joint_angles)
        # 2) IK from FK outputs
        q_est = self.solver.solve(
            pos['shoulder'],
            pos['elbow'],
            pos['wrist'],
            target_orientation=pos['wrist_ori']
        )
        # 3) errors per joint
        errors = {}
        for joint, true_val in joint_angles.items():
            est = q_est.get(joint)
            if est is None:
                continue
            errors[joint] = abs(est - true_val)
        return errors

    def validate_sequence(self, sequence: list) -> tuple[list, int]:
        """
        Given a list of joint-angle dicts, returns:
          - errors_seq: list of {joint:err} for frames with no clipping
          - skipped_frames: count of frames clipped at joint limits
        """
        errors_seq = []
        skipped_frames = 0
        for angles in sequence:
            # FK→IK round-trip
            pos = self.compute_positions(angles)
            q_est = self.solver.solve(
                pos['shoulder'], pos['elbow'], pos['wrist'],
                target_orientation=pos['wrist_ori']
            )
            # compute errors and detect clipping
            clipped = False
            frame_errors = {}
            for joint, orig in angles.items():
                est = q_est.get(joint)
                err = abs(angle_diff(orig, est))
                frame_errors[joint] = err
                lo, hi = self.solver.joint_limits[joint]
                if est <= lo or est >= hi:
                    clipped = True
            if clipped:
                skipped_frames += 1
            else:
                errors_seq.append(frame_errors)
        return errors_seq, skipped_frames

    def summarize(self, errors_seq: list, skipped_frames: int = 0) -> dict:
        """
        Given a list of {'frame':i,'errors':{joint:err}}, computes per-joint max & RMS errors.
        Returns:
            dict: joint -> {'max':..., 'rms':...}, plus 'skipped_frames'
        """
        if not errors_seq:
            return {'skipped_frames': skipped_frames}
        joints = list(errors_seq[0].keys())
        stats = {}
        for j in joints:
            vals = [frame[j] for frame in errors_seq]
            arr  = np.array(vals)
            stats[j] = {
                'max': np.max(arr),
                'rms': float(np.sqrt(np.mean(arr**2)))
            }
        stats['skipped_frames'] = skipped_frames
        return stats

    def is_sequence_valid(self, errors_seq: list, skipped_frames: int = 0) -> bool:
        """
        Returns True only if:
          • no frames were clipped, AND
          • every error ≤ error_tolerance
        """
        if skipped_frames > 0:
            return False
        for frame in errors_seq:
            for err in frame.values():
                if err > self.error_tolerance:
                    return False
        return True 