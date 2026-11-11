#!/opt/shared/Prog/anaconda3/envs/rise/bin/python
"""Online RGBD → raw17 sparse voxel preprocessor for Hybrid64 V3.1 deployment.

Requires the ``rise`` conda environment (torch + pytorch3d + MinkowskiEngine + open3d):
    /opt/shared/Prog/anaconda3/envs/rise/bin/python

Reuses the exact raw17 aggregation logic from the training/data-export pipeline.
This MUST produce identical features to the offline precomputed .pt files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import IMG_MEAN, IMG_STD, WORKSPACE_MAX, WORKSPACE_MIN


# ---------------------------------------------------------------------------
# Raw17 feature layout (must match export_realworld_rgbd_voxel_raw17.py)
# ---------------------------------------------------------------------------
RAW17_LAYOUT = (
    "xyz_norm3 + log_density1 + rgb_mean3 + offset_mean_vox3 + "
    "cov6_vox2_6 + log_count1"
)
RAW17_DIM = 17


def _cov_to_6d(cov: "np.ndarray") -> "np.ndarray":
    """Extract upper-triangle of 3×3 covariance matrices: N×3×3 → N×6."""
    return np.stack(
        [
            cov[:, 0, 0],
            cov[:, 1, 1],
            cov[:, 2, 2],
            cov[:, 0, 1],
            cov[:, 0, 2],
            cov[:, 1, 2],
        ],
        axis=-1,
    )


def _finite_inside_mask(
    xyz: "np.ndarray",
    workspace_min: "np.ndarray",
    workspace_max: "np.ndarray",
) -> "np.ndarray":
    finite = np.isfinite(xyz).all(axis=1)
    inside = finite & ((xyz >= workspace_min) & (xyz <= workspace_max)).all(axis=1)
    return inside


def rgbd_to_camera_point_cloud(
    colors: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    voxel_size: float,
    depth_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RealSense RGBD → camera-frame point cloud.

    Args:
        colors:  H×W×3 uint8 RGB image.
        depths:  H×W float32 depth image in metres (RealSense already returns metres).
        intrinsics: 3×4 array [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]].
        voxel_size: Voxel size for downsampling (metres).
        depth_scale: Depth divisor; RealSense returns metres so depth_scale=1.0.

    Returns:
        points: (N, 3) float32 camera-frame XYZ.
        colors: (N, 3) float32 colours in [0, 1].
    """
    import open3d as o3d

    h, w = depths.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    color_o3d = o3d.geometry.Image(colors.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depths.astype(np.float32))

    cam_intr = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=float(depth_scale),
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intr)
    cloud = cloud.voxel_down_sample(float(voxel_size))

    points = np.asarray(cloud.points, dtype=np.float32)
    cloud_colors = np.asarray(cloud.colors, dtype=np.float32)
    return points, cloud_colors


def workspace_crop_camera_frame(
    points: np.ndarray,
    colors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop points and colours to the camera-frame workspace."""
    ws_min = np.asarray(WORKSPACE_MIN, dtype=np.float32)
    ws_max = np.asarray(WORKSPACE_MAX, dtype=np.float32)
    inside = _finite_inside_mask(points, ws_min, ws_max)
    return points[inside].astype(np.float32), colors[inside].astype(np.float32)


def _maybe_subsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_num_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_num_points <= 0 or points.shape[0] <= max_num_points:
        return points, colors
    idx = np.linspace(0, points.shape[0] - 1, num=max_num_points, dtype=np.int64)
    return points[idx], colors[idx]


def aggregate_points_raw17(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    max_num_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate camera-frame RGB-D points into raw17 sparse voxels.

    This is a numpy-based reimplementation of the PyTorch version in
    ``tools/export_realworld_rgbd_voxel_raw17.py:aggregate_points_raw17``.
    Both versions must produce numerically identical output.

    Args:
        points:  (N, 3)  float32  camera-frame XYZ.
        colors:  (N, 3)  float32  ImageNet-normalized RGB.
        voxel_size:  Scalar  (m).
        max_num_points:  If >0, subsample before aggregation.

    Returns:
        coords:    (M, 3)  int32  centred signed voxel indices.
        feats:     (M, 17) float32  raw17 features.
        input_xyz: (M, 3)  float32  mean XYZ per voxel.
    """
    points, colors = _maybe_subsample(points, colors, max_num_points)

    empty_coords = np.empty((0, 3), dtype=np.int32)
    empty_feats = np.empty((0, RAW17_DIM), dtype=np.float32)
    empty_xyz = np.empty((0, 3), dtype=np.float32)

    if points.shape[0] == 0:
        return empty_coords, empty_feats, empty_xyz

    ws_min = np.asarray(WORKSPACE_MIN, dtype=np.float32)
    ws_max = np.asarray(WORKSPACE_MAX, dtype=np.float32)
    ws_center = 0.5 * (ws_min + ws_max)
    ws_extent = ws_max - ws_min
    vs = float(voxel_size)

    # --- colour normalization (must happen *before* aggregation) ---------
    colors = ((colors - IMG_MEAN) / IMG_STD).astype(np.float32)

    # --- filter points whose *voxel centre* lies inside workspace --------
    coords = np.floor((points - ws_center) / vs).astype(np.int32)
    voxel_centers_per_point = ws_center + (coords.astype(np.float32) + 0.5) * vs
    center_ok = ((voxel_centers_per_point >= ws_min) & (voxel_centers_per_point <= ws_max)).all(axis=1)
    coords = coords[center_ok]
    points = points[center_ok]
    colors = colors[center_ok]

    if points.shape[0] == 0:
        return empty_coords, empty_feats, empty_xyz

    # --- group by unique voxel -------------------------------------------
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    nvox = unique_coords.shape[0]

    count = np.bincount(inverse, minlength=nvox).astype(np.float32).reshape(-1, 1)
    count_safe = np.maximum(count, 1.0)

    # Keep float32 accumulation to match the offline torch.scatter_add_
    # exporter used to create the training payloads.
    sum_xyz = np.zeros((nvox, 3), dtype=np.float32)
    sum_rgb = np.zeros((nvox, 3), dtype=np.float32)
    # build per-point outer products for covariance
    outer = np.einsum("ni,nj->nij", points, points).reshape(-1, 9)
    sum_outer = np.zeros((nvox, 9), dtype=np.float32)

    np.add.at(sum_xyz, inverse, points)
    np.add.at(sum_rgb, inverse, colors)
    np.add.at(sum_outer, inverse, outer)

    input_xyz = (sum_xyz / count_safe).astype(np.float32)
    rgb_mean = (sum_rgb / count_safe).astype(np.float32)

    mean_outer = (sum_outer / count_safe.flatten()[:, None]).reshape(nvox, 3, 3)
    cov = mean_outer - np.einsum("ni,nj->nij", input_xyz, input_xyz)
    cov = 0.5 * (cov + cov.transpose(0, 2, 1))
    cov6 = _cov_to_6d(cov) / (vs * vs + 1e-12)

    # --- raw17 features --------------------------------------------------
    voxel_centers = ws_center + (unique_coords.astype(np.float32) + 0.5) * vs
    xyz_norm = (voxel_centers - ws_min) / (ws_extent + 1e-8) * 2.0 - 1.0
    offset_mean_vox = (input_xyz - voxel_centers) / vs

    log_density = np.log1p(count).astype(np.float32)
    log_count = np.log1p(count).astype(np.float32)

    feats = np.concatenate(
        [
            xyz_norm.astype(np.float32),          # 0-2
            log_density,                           # 3
            rgb_mean.astype(np.float32),           # 4-6
            offset_mean_vox.astype(np.float32),    # 7-9
            cov6.astype(np.float32),               # 10-15
            log_count,                             # 16
        ],
        axis=-1,
    )
    assert feats.shape[-1] == RAW17_DIM, f"Expected raw17, got {feats.shape[-1]}"

    return unique_coords.astype(np.int32), feats.astype(np.float32), input_xyz.astype(np.float32)


def preprocess_observation(
    colors: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    current_tcp: np.ndarray,   # unused, kept for API symmetry
    voxel_size: float = 0.005,
    max_num_points: int = 0,
    depth_scale: float = 1.0,
) -> dict:
    """
    Complete observation preprocessing: RealSense RGBD → raw17 sparse voxels.

    Args:
        colors:       H×W×3 uint8 RGB.
        depths:       H×W float32 depth (metres).
        intrinsics:   3×4 array.
        current_tcp:  Unused (kept for API compatibility).
        voxel_size:   Scalar (m).
        max_num_points: Max points before aggregation (0 = no limit).
        depth_scale:  Depth divisor (1.0 when depths are already metres).

    Returns:
        Dict with keys:
            input_coords   [M, 3] int32
            input_feats    [M, 17] float32
            input_xyz      [M, 3] float32
            M              int
    """
    # 1. RGBD → camera-frame point cloud
    points, cloud_colors = rgbd_to_camera_point_cloud(
        colors, depths, intrinsics, voxel_size=voxel_size, depth_scale=depth_scale,
    )

    # 2. Workspace crop (camera frame)
    points, cloud_colors = workspace_crop_camera_frame(points, cloud_colors)

    if points.shape[0] == 0:
        return {
            "input_coords": np.empty((0, 3), dtype=np.int32),
            "input_feats": np.empty((0, RAW17_DIM), dtype=np.float32),
            "input_xyz": np.empty((0, 3), dtype=np.float32),
            "M": 0,
        }

    # 3. Aggregate into raw17 sparse voxels
    coords, feats, input_xyz = aggregate_points_raw17(
        points, cloud_colors, voxel_size=voxel_size, max_num_points=max_num_points,
    )

    return {
        "input_coords": coords,
        "input_feats": feats,
        "input_xyz": input_xyz,
        "M": int(coords.shape[0]),
    }
