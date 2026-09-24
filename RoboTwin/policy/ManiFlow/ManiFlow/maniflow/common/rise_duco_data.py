"""Shared RISE/DUCO preprocessing primitives for ManiFlow.

The offline converter and the future online deployment adapter should import
this module instead of maintaining separate RGB-D preprocessing code paths.
The first data-contract version intentionally excludes force/torque data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import warnings

import numpy as np
from PIL import Image


SCHEMA_VERSION = "rise_duco_maniflow_v1"
DEFAULT_CAMERA_SERIAL = "243222076209"
DEFAULT_INTRINSICS = np.asarray(
    [
        [911.1273, 0.0, 644.1116],
        [0.0, 910.7914, 364.0461],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DEFAULT_WORKSPACE_MIN = np.asarray([-0.32, -0.34, 0.16], dtype=np.float32)
DEFAULT_WORKSPACE_MAX = np.asarray([0.34, 0.18, 0.62], dtype=np.float32)
REQUIRED_MODALITIES = ("color", "depth", "joint", "gripper_command")


class DatasetContractError(RuntimeError):
    """Raised when source data violates the declared conversion contract."""


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    path: Path
    camera_serial: str
    camera_path: Path
    frame_ids: Tuple[int, ...]
    finish_time_ms: Optional[int]
    warnings: Tuple[str, ...]


def _demo_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[-1]), path.name
    except ValueError:
        return 2**31 - 1, path.name


def _frame_stems(directory: Path, suffix: str) -> set[int]:
    result = set()
    for path in directory.glob(f"*{suffix}"):
        try:
            result.add(int(path.stem))
        except ValueError as exc:
            raise DatasetContractError(f"Non-integer frame name: {path}") from exc
    return result


def discover_episodes(
    dataset_root: Path,
    camera_serial: Optional[str] = DEFAULT_CAMERA_SERIAL,
    strict_metadata: bool = False,
) -> List[EpisodeSpec]:
    """Discover and strictly synchronize RISE episode modalities by timestamp."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetContractError(f"Dataset root does not exist: {dataset_root}")

    demo_paths = sorted(dataset_root.glob("demonstration_*"), key=_demo_sort_key)
    if not demo_paths:
        raise DatasetContractError(f"No demonstration_* directories under {dataset_root}")

    episodes: List[EpisodeSpec] = []
    for demo_path in demo_paths:
        episode_warnings: List[str] = []
        if camera_serial:
            camera_path = demo_path / f"cam_{camera_serial}"
            if not camera_path.is_dir():
                raise DatasetContractError(
                    f"{demo_path.name}: missing camera directory {camera_path.name}"
                )
            resolved_serial = camera_serial
        else:
            camera_paths = sorted(path for path in demo_path.glob("cam_*") if path.is_dir())
            if len(camera_paths) != 1:
                raise DatasetContractError(
                    f"{demo_path.name}: expected exactly one cam_* directory, got {len(camera_paths)}"
                )
            camera_path = camera_paths[0]
            resolved_serial = camera_path.name.removeprefix("cam_")

        modality_stems: Dict[str, set[int]] = {}
        for modality in REQUIRED_MODALITIES:
            modality_path = camera_path / modality
            if not modality_path.is_dir():
                raise DatasetContractError(
                    f"{demo_path.name}: missing modality directory {modality_path}"
                )
            suffix = ".png" if modality in ("color", "depth") else ".npy"
            modality_stems[modality] = _frame_stems(modality_path, suffix)

        reference = modality_stems[REQUIRED_MODALITIES[0]]
        for modality in REQUIRED_MODALITIES[1:]:
            if modality_stems[modality] != reference:
                missing = sorted(reference - modality_stems[modality])[:5]
                extra = sorted(modality_stems[modality] - reference)[:5]
                raise DatasetContractError(
                    f"{demo_path.name}: timestamp mismatch for {modality}; "
                    f"missing={missing}, extra={extra}"
                )
        if not reference:
            raise DatasetContractError(f"{demo_path.name}: episode has no synchronized frames")

        metadata_path = demo_path / "metadata.json"
        finish_time_ms: Optional[int] = None
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if "finish_time" in metadata:
                    finish_time_ms = int(metadata["finish_time"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetContractError(
                    f"{demo_path.name}: invalid metadata.json: {exc}"
                ) from exc
        elif strict_metadata:
            raise DatasetContractError(f"{demo_path.name}: metadata.json is required")
        else:
            episode_warnings.append("metadata.json missing; using all synchronized frames")

        frame_ids = sorted(reference)
        if finish_time_ms is not None:
            frame_ids = [frame_id for frame_id in frame_ids if frame_id <= finish_time_ms]
        if not frame_ids:
            raise DatasetContractError(
                f"{demo_path.name}: no frames remain after finish_time filtering"
            )

        timestamp_path = demo_path / "timestamp.txt"
        if not timestamp_path.is_file():
            if strict_metadata:
                raise DatasetContractError(f"{demo_path.name}: timestamp.txt is required")
            episode_warnings.append("timestamp.txt missing; per-frame filename timestamps are used")

        episodes.append(
            EpisodeSpec(
                name=demo_path.name,
                path=demo_path,
                camera_serial=resolved_serial,
                camera_path=camera_path,
                frame_ids=tuple(frame_ids),
                finish_time_ms=finish_time_ms,
                warnings=tuple(episode_warnings),
            )
        )
    return episodes


def scan_episodes(episodes: Sequence[EpisodeSpec], validate_arrays: bool = True) -> Dict:
    """Return a JSON-serializable integrity report for discovered episodes."""
    frame_counts = []
    deltas_ms: List[int] = []
    warnings_out: List[Dict[str, str]] = []
    joint_min = np.full(6, np.inf, dtype=np.float64)
    joint_max = np.full(6, -np.inf, dtype=np.float64)
    gripper_values = set()

    for episode in episodes:
        frame_counts.append(len(episode.frame_ids))
        if len(episode.frame_ids) > 1:
            deltas_ms.extend(np.diff(np.asarray(episode.frame_ids, dtype=np.int64)).tolist())
        for message in episode.warnings:
            warnings_out.append({"episode": episode.name, "message": message})

        if validate_arrays:
            for frame_id in episode.frame_ids:
                joint = load_joint(episode, frame_id)
                gripper = load_gripper(episode, frame_id)
                joint_min = np.minimum(joint_min, joint)
                joint_max = np.maximum(joint_max, joint)
                gripper_values.add(float(gripper))

    delta_array = np.asarray(deltas_ms, dtype=np.float64)
    report = {
        "schema_version": SCHEMA_VERSION,
        "episodes": len(episodes),
        "frames_total": int(sum(frame_counts)),
        "frames_min": int(min(frame_counts)),
        "frames_max": int(max(frame_counts)),
        "frames_median": float(np.median(frame_counts)),
        "warnings": warnings_out,
    }
    if delta_array.size:
        report["frame_delta_ms"] = {
            "min": float(delta_array.min()),
            "median": float(np.median(delta_array)),
            "p95": float(np.percentile(delta_array, 95)),
            "max": float(delta_array.max()),
            "mean": float(delta_array.mean()),
        }
    if validate_arrays:
        report["joint_min"] = joint_min.tolist()
        report["joint_max"] = joint_max.tolist()
        report["gripper_raw_values"] = sorted(gripper_values)
    return report


def load_joint(episode: EpisodeSpec, frame_id: int) -> np.ndarray:
    path = episode.camera_path / "joint" / f"{frame_id}.npy"
    joint = np.load(path, allow_pickle=False).reshape(-1)
    if joint.shape != (6,):
        raise DatasetContractError(f"Expected 6 joint values in {path}, got {joint.shape}")
    if not np.isfinite(joint).all():
        raise DatasetContractError(f"Non-finite joint values in {path}: {joint}")
    return joint.astype(np.float32, copy=False)


def load_gripper(episode: EpisodeSpec, frame_id: int) -> float:
    path = episode.camera_path / "gripper_command" / f"{frame_id}.npy"
    value = np.load(path, allow_pickle=False).reshape(-1)
    if value.shape != (1,):
        raise DatasetContractError(f"Expected one gripper value in {path}, got {value.shape}")
    if not np.isfinite(value[0]):
        raise DatasetContractError(f"Non-finite gripper value in {path}: {value}")
    return float(value[0])


def build_agent_state(
    joint: np.ndarray,
    gripper_raw: float,
    gripper_max_command: float = 1000.0,
) -> np.ndarray:
    """Build [q1..q6, normalized_gripper], where 0=closed and 1=open."""
    if gripper_max_command <= 0:
        raise ValueError("gripper_max_command must be positive")
    tolerance = max(1e-6, 1e-6 * gripper_max_command)
    if gripper_raw < -tolerance or gripper_raw > gripper_max_command + tolerance:
        raise DatasetContractError(
            f"gripper command {gripper_raw} outside [0, {gripper_max_command}]"
        )
    gripper = np.clip(gripper_raw / gripper_max_command, 0.0, 1.0)
    return np.concatenate([np.asarray(joint, dtype=np.float32), [gripper]]).astype(np.float32)


def _numpy_farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic NumPy FPS fallback used when PyTorch3D is unavailable."""
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        farthest = int(np.argmax(min_distance))
        selected[index] = farthest
        distance = np.sum((points - points[farthest]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


class RGBDPointCloudPreprocessor:
    """Convert an aligned RGB-D frame to fixed-size camera-frame XYZRGB."""

    def __init__(
        self,
        intrinsics: np.ndarray = DEFAULT_INTRINSICS,
        workspace_min: np.ndarray = DEFAULT_WORKSPACE_MIN,
        workspace_max: np.ndarray = DEFAULT_WORKSPACE_MAX,
        depth_scale: float = 1000.0,
        voxel_size: float = 0.005,
        num_points: int = 1024,
        sampling: str = "fps",
        device: str = "auto",
    ):
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float32)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float32)
        self.depth_scale = float(depth_scale)
        self.voxel_size = float(voxel_size)
        self.num_points = int(num_points)
        self.sampling = str(sampling)
        self.device = str(device)
        self._torch = None
        self._pytorch3d_fps = None
        self._warned_numpy_fallback = False

        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3,3), got {self.intrinsics.shape}")
        if self.workspace_min.shape != (3,) or self.workspace_max.shape != (3,):
            raise ValueError("workspace bounds must each have shape (3,)")
        if np.any(self.workspace_max <= self.workspace_min):
            raise ValueError("workspace_max must be greater than workspace_min")
        if self.depth_scale <= 0 or self.voxel_size <= 0 or self.num_points <= 0:
            raise ValueError("depth_scale, voxel_size and num_points must be positive")
        if self.sampling not in ("fps", "uniform"):
            raise ValueError("sampling must be 'fps' or 'uniform'")

    def contract_dict(self) -> Dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "point_frame": "camera_optical_opencv_x_right_y_down_z_forward",
            "point_layout": ["x_m", "y_m", "z_m", "r_0_1", "g_0_1", "b_0_1"],
            "intrinsics": self.intrinsics.tolist(),
            "workspace_min": self.workspace_min.tolist(),
            "workspace_max": self.workspace_max.tolist(),
            "depth_scale": self.depth_scale,
            "voxel_size": self.voxel_size,
            "num_points": self.num_points,
            "sampling": self.sampling,
        }

    def load_files(self, color_path: Path, depth_path: Path) -> np.ndarray:
        with Image.open(color_path) as color_image:
            color = np.asarray(color_image.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as depth_image:
            depth = np.asarray(depth_image, dtype=np.float32)
        return self(color, depth)

    def _sample_indices(self, points: np.ndarray) -> np.ndarray:
        count = points.shape[0]
        if count < self.num_points:
            return np.resize(np.arange(count, dtype=np.int64), self.num_points)
        if count == self.num_points:
            return np.arange(count, dtype=np.int64)
        if self.sampling == "uniform":
            return np.linspace(0, count - 1, self.num_points, dtype=np.int64)

        try:
            if self._pytorch3d_fps is None:
                import torch
                from pytorch3d.ops import sample_farthest_points

                self._torch = torch
                self._pytorch3d_fps = sample_farthest_points
            torch = self._torch
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            tensor = torch.from_numpy(points).unsqueeze(0).to(device=device, dtype=torch.float32)
            _, indices = self._pytorch3d_fps(
                tensor,
                K=[self.num_points],
                random_start_point=False,
            )
            return indices[0].detach().cpu().numpy().astype(np.int64, copy=False)
        except Exception as exc:
            if not self._warned_numpy_fallback:
                warnings.warn(
                    f"PyTorch3D FPS unavailable ({type(exc).__name__}: {exc}); "
                    "falling back to deterministic NumPy FPS, which is slower.",
                    RuntimeWarning,
                )
                self._warned_numpy_fallback = True
            return _numpy_farthest_point_indices(points, self.num_points)

    def __call__(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if color.ndim != 3 or color.shape[2] != 3:
            raise DatasetContractError(f"Expected HxWx3 RGB image, got {color.shape}")
        if depth.ndim != 2:
            raise DatasetContractError(f"Expected HxW depth image, got {depth.shape}")
        if color.shape[:2] != depth.shape:
            raise DatasetContractError(
                f"RGB/depth resolution mismatch: {color.shape[:2]} vs {depth.shape}"
            )

        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "open3d is required for point-cloud conversion; use the maniflow environment"
            ) from exc

        height, width = depth.shape
        fx, fy = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])
        cx, cy = float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color, dtype=np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32))
        camera = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=float(self.workspace_max[2] + 0.05),
            convert_rgb_to_intensity=False,
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32)

        finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        inside = finite & (points >= self.workspace_min).all(axis=1)
        inside &= (points <= self.workspace_max).all(axis=1)
        points = points[inside]
        colors = colors[inside]
        if points.shape[0] == 0:
            raise DatasetContractError(
                "No valid point remains after workspace cropping; check depth scale, intrinsics, and bounds"
            )

        indices = self._sample_indices(points)
        result = np.concatenate([points[indices], colors[indices]], axis=1).astype(np.float32)
        if result.shape != (self.num_points, 6):
            raise AssertionError(f"Unexpected point-cloud shape {result.shape}")
        if not np.isfinite(result).all():
            raise DatasetContractError("Non-finite value in processed point cloud")
        return result
