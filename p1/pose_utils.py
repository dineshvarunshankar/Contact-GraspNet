"""Convert contact predictions to SE(3) grasp poses."""

import numpy as np

# Panda wrist-to-fingertip distance in meters.
GRIPPER_DEPTH = 0.1034


def contact_to_grasp_pose(contact_pts, base_dirs, approach_dirs, widths):
    """Return (N, 4, 4) poses in the input coordinate frame.

    Points and directions have shape (N, 3); widths have shape (N,) in meters.
    Approach and baseline directions must be nonzero and nonparallel.
    """
    N = len(contact_pts)
    grasps = np.eye(4)[None].repeat(N, axis=0)

    z = approach_dirs / (np.linalg.norm(approach_dirs, axis=1, keepdims=True) + 1e-8)

    x_raw = base_dirs / (np.linalg.norm(base_dirs, axis=1, keepdims=True) + 1e-8)
    y = np.cross(z, x_raw)
    y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)

    # Orthogonalize the baseline against the approach direction.
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    grasps[:, :3, 0] = x
    grasps[:, :3, 1] = y
    grasps[:, :3, 2] = z

    # Shift from the contact point to the gripper wrist.
    grasps[:, :3, 3] = (
        contact_pts
        + (widths[:, None] / 2.0) * x
        - GRIPPER_DEPTH * z
    )

    return grasps
