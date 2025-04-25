import numpy as np

# Rotation matrix helpers
def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# Intrinsic Y-X-Z Euler extraction
def compute_euler_yxz(R):
    """
    Extract intrinsic Y–X–Z Euler angles from rotation matrix R:
      R = R_z(gamma) @ R_x(beta) @ R_y(alpha)

    Returns:
      alpha = rotation about local Y
      beta  = rotation about local X
      gamma = rotation about local Z
    """
    # Beta = arcsin(R[2,1])
    beta  = np.arcsin(np.clip(R[2,1], -1.0, 1.0))
    # Alpha = atan2(-R[2,0], R[2,2])
    alpha = np.arctan2(-R[2,0], R[2,2])
    # Gamma = atan2(-R[0,1], R[1,1])
    gamma = np.arctan2(-R[0,1], R[1,1])
    return alpha, beta, gamma


if __name__ == "__main__":
    # Test single-axis rotations
    tests = [
        ("pitch only", (0.3, 0.0, 0.0)),
        ("roll only",  (0.0, 0.4, 0.0)),
        ("yaw only",   (0.0, 0.0, 0.5)),
    ]

    print("Testing intrinsic Y–X–Z Euler extraction:")
    for name, (alpha_exp, beta_exp, gamma_exp) in tests:
        # Build R = R_z(gamma) @ R_x(beta) @ R_y(alpha)
        R = rotation_matrix_z(gamma_exp) @ rotation_matrix_x(beta_exp) @ rotation_matrix_y(alpha_exp)
        alpha, beta, gamma = compute_euler_yxz(R)
        print(f"{name:10s} -> expected (α,β,γ)=({alpha_exp:.2f},{beta_exp:.2f},{gamma_exp:.2f}), "
              f"got ({alpha:.2f},{beta:.2f},{gamma:.2f})")
