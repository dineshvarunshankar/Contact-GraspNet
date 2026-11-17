"""Draw Panda grasp poses and point clouds with Open3D."""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# Panda local frame: z is approach; x is finger-to-finger. Lengths in meters.
_APPROACH_DEPTH  = 0.0584   # wrist to finger root along z
_GRIPPER_DEPTH   = 0.1034   # wrist to fingertip along z


def _gripper_control_points(opening):
    """Return the seven control points of the gripper wireframe."""
    hw = opening / 2.0

    wrist        = [0,   0, 0]
    baseline_mid = [0,   0, _APPROACH_DEPTH]
    left_root    = [-hw, 0, _APPROACH_DEPTH]
    left_tip     = [-hw, 0, _GRIPPER_DEPTH]
    right_root   = [ hw, 0, _APPROACH_DEPTH]
    right_tip    = [ hw, 0, _GRIPPER_DEPTH]

    return np.array([
        wrist,
        baseline_mid,
        left_root,
        left_tip,
        left_root,
        right_root,
        right_tip,
    ], dtype=np.float64)


def _gripper_lines_in_world(grasp_pose, opening):
    """Transform the wireframe into the grasp pose coordinate frame."""
    pts_local = _gripper_control_points(opening)

    pts_world = (grasp_pose[:3, :3] @ pts_local.T).T + grasp_pose[:3, 3]

    N = len(pts_world)
    connections = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1)

    return pts_world, connections


def visualize_grasps(pc, pred_grasps_cam, scores,
                     pc_colors=None, topk=50,
                     gripper_openings=None, default_opening=0.08,
                     save_path=None, show=True):
    """Draw top-k grasps per segment, with optional PNG output.

    Poses, scores, and openings are keyed by segment ID. Openings use meters.
    Point colors are RGB values in [0, 255]. Single-segment grasps use score colors.
    """
    if topk <= 0:
        raise ValueError("topk must be positive")

    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255.0)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(pcd)

    cm_segments = plt.get_cmap("rainbow")
    cm_scores   = plt.get_cmap("viridis")

    seg_keys = [k for k, grasps in pred_grasps_cam.items() if len(grasps)]
    single_segment = (len(seg_keys) == 1)

    for seg_idx, k in enumerate(seg_keys):
        grasps_k = pred_grasps_cam[k]
        scores_k = scores[k]

        n = min(topk, len(grasps_k))
        top_idcs = np.argsort(scores_k)[::-1][:n]
        grasps_k = grasps_k[top_idcs]
        scores_k = scores_k[top_idcs]

        openings_k = (
            gripper_openings[k][top_idcs]
            if gripper_openings is not None and k in gripper_openings
            else np.full(n, default_opening)
        )

        if single_segment:
            s_min, s_max = scores_k.min(), scores_k.max()
            span = s_max - s_min if s_max > s_min else 1.0
            grasp_colors = [
                cm_scores((s - s_min) / span)[:3] for s in scores_k
            ]
        else:
            base_color = cm_segments(seg_idx / max(len(seg_keys) - 1, 1))[:3]
            grasp_colors = [base_color] * n

        all_pts   = []
        all_conns = []
        all_colors = []
        offset = 0

        for g, opening, color in zip(grasps_k, openings_k, grasp_colors):
            pts, conns = _gripper_lines_in_world(g, opening)
            all_pts.append(pts)
            all_conns.append(conns + offset)
            all_colors.extend([list(color)] * len(conns))
            offset += len(pts)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        line_set.lines  = o3d.utility.Vector2iVector(np.vstack(all_conns))
        line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        geometries.append(line_set)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        render = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = "defaultUnlit"
        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0
        for idx, geom in enumerate(geometries):
            if isinstance(geom, o3d.geometry.LineSet):
                render.scene.add_geometry(f"geom_{idx}", geom, mat_line)
            else:
                render.scene.add_geometry(f"geom_{idx}", geom, mat_pc)
        render.scene.set_background([1, 1, 1, 1])
        bounds = render.scene.bounding_box
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
        eye    = center + np.array([0, 0, -extent])
        up     = np.array([0, -1, 0])
        render.setup_camera(60.0, center.astype(np.float32),
                            eye.astype(np.float32),
                            up.astype(np.float32))
        img = render.render_to_image()
        img_np = np.asarray(img)
        from PIL import Image as PILImage
        PILImage.fromarray(img_np).save(save_path)
        print(f"Saved render to {save_path}")

    if show:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="CGN Predicted Grasps",
            width=1280, height=720,
        )
