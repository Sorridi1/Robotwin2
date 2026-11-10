#!/usr/bin/env python3
"""Convert synchronized RISE/DUCO demonstrations to ManiFlow Zarr format."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFLOW_PACKAGE_ROOT = SCRIPT_DIR.parent / "ManiFlow"
if str(MANIFLOW_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MANIFLOW_PACKAGE_ROOT))

from maniflow.common.rise_duco_data import (  # noqa: E402
    DEFAULT_CAMERA_SERIAL,
    DEFAULT_INTRINSICS,
    DEFAULT_WORKSPACE_MAX,
    DEFAULT_WORKSPACE_MIN,
    DatasetContractError,
    RGBDPointCloudPreprocessor,
    SCHEMA_VERSION,
    build_agent_state,
    discover_episodes,
    load_gripper,
    load_joint,
    scan_episodes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RISE/DUCO RGB-D demonstrations to ManiFlow point-cloud Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="RISE dataset root")
    parser.add_argument("--output", type=Path, help="Destination .zarr directory")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL)
    parser.add_argument("--action-offset", type=int, default=1, help="Action label frame offset")
    parser.add_argument("--gripper-max-command", type=float, default=1000.0)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--sampling", choices=("fps", "uniform"), default="fps")
    parser.add_argument("--device", default="auto", help="FPS device: auto, cpu, or cuda")
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        default=[
            float(DEFAULT_INTRINSICS[0, 0]),
            float(DEFAULT_INTRINSICS[1, 1]),
            float(DEFAULT_INTRINSICS[0, 2]),
            float(DEFAULT_INTRINSICS[1, 2]),
        ],
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_WORKSPACE_MIN.tolist(),
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_WORKSPACE_MAX.tolist(),
    )
    parser.add_argument("--scan-only", action="store_true", help="Validate source without writing Zarr")
    parser.add_argument(
        "--quick-scan",
        action="store_true",
        help="Skip loading every joint/gripper array during scan",
    )
    parser.add_argument("--strict-metadata", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Testing-only episode limit")
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Testing-only source frame limit applied before action offset",
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true", help="Keep old output as timestamped backup")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _intrinsics(values: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = values
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _create_arrays(root, num_points: int, chunk_size: int):
    import zarr

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    data = root.create_group("data")
    meta = root.create_group("meta")
    arrays = {
        "point_cloud": data.create_dataset(
            "point_cloud",
            shape=(0, num_points, 6),
            chunks=(chunk_size, num_points, 6),
            dtype="float32",
            compressor=compressor,
        ),
        "state": data.create_dataset(
            "state",
            shape=(0, 7),
            chunks=(chunk_size, 7),
            dtype="float32",
            compressor=compressor,
        ),
        "action": data.create_dataset(
            "action",
            shape=(0, 7),
            chunks=(chunk_size, 7),
            dtype="float32",
            compressor=compressor,
        ),
        "timestamp_ms": data.create_dataset(
            "timestamp_ms",
            shape=(0,),
            chunks=(max(chunk_size, 256),),
            dtype="int64",
            compressor=compressor,
        ),
        "episode_id": data.create_dataset(
            "episode_id",
            shape=(0,),
            chunks=(max(chunk_size, 256),),
            dtype="int32",
            compressor=compressor,
        ),
        "episode_ends": meta.create_dataset(
            "episode_ends",
            shape=(0,),
            chunks=(max(1, chunk_size),),
            dtype="int64",
            compressor=compressor,
        ),
    }
    return arrays


def _resize_first_axis(array, size: int) -> None:
    array.resize((size,) + array.shape[1:])


def _array_is_finite(array, batch_size: int = 256) -> bool:
    """Validate a Zarr array without loading the full dataset into RAM."""
    for start in range(0, array.shape[0], batch_size):
        stop = min(start + batch_size, array.shape[0])
        if not np.isfinite(array[start:stop]).all():
            return False
    return True


def convert(args: argparse.Namespace, episodes, source_report: Dict) -> Dict:
    if args.output is None:
        raise DatasetContractError("--output is required unless --scan-only is used")
    if args.action_offset < 1:
        raise DatasetContractError("--action-offset must be >= 1")
    if args.chunk_size < 1:
        raise DatasetContractError("--chunk-size must be >= 1")

    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required; run this converter in the maniflow environment") from exc

    intrinsics = _intrinsics(args.intrinsics)
    preprocessor = RGBDPointCloudPreprocessor(
        intrinsics=intrinsics,
        workspace_min=np.asarray(args.workspace_min, dtype=np.float32),
        workspace_max=np.asarray(args.workspace_max, dtype=np.float32),
        depth_scale=args.depth_scale,
        voxel_size=args.voxel_size,
        num_points=args.num_points,
        sampling=args.sampling,
        device=args.device,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace safely")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    backup_path: Optional[Path] = None
    episode_reports: List[Dict] = []
    total = 0
    try:
        root = zarr.group(str(temp_dir), overwrite=True)
        arrays = _create_arrays(root, args.num_points, args.chunk_size)

        for episode_index, episode in enumerate(episodes):
            frame_ids = list(episode.frame_ids)
            if args.max_frames_per_episode is not None:
                frame_ids = frame_ids[: args.max_frames_per_episode]
            usable = len(frame_ids) - args.action_offset
            if usable < 1:
                raise DatasetContractError(
                    f"{episode.name}: {len(frame_ids)} frames are insufficient for "
                    f"action_offset={args.action_offset}"
                )

            point_cloud = np.empty((usable, args.num_points, 6), dtype=np.float32)
            state = np.empty((usable, 7), dtype=np.float32)
            action = np.empty((usable, 7), dtype=np.float32)
            timestamps = np.asarray(frame_ids[:usable], dtype=np.int64)

            for local_index in range(usable):
                obs_frame = frame_ids[local_index]
                action_frame = frame_ids[local_index + args.action_offset]
                color_path = episode.camera_path / "color" / f"{obs_frame}.png"
                depth_path = episode.camera_path / "depth" / f"{obs_frame}.png"
                point_cloud[local_index] = preprocessor.load_files(color_path, depth_path)
                state[local_index] = build_agent_state(
                    load_joint(episode, obs_frame),
                    load_gripper(episode, obs_frame),
                    args.gripper_max_command,
                )
                action[local_index] = build_agent_state(
                    load_joint(episode, action_frame),
                    load_gripper(episode, action_frame),
                    args.gripper_max_command,
                )
                if (local_index + 1) % 25 == 0 or local_index + 1 == usable:
                    print(
                        f"[{episode_index + 1}/{len(episodes)}] {episode.name}: "
                        f"{local_index + 1}/{usable}",
                        end="\r",
                        flush=True,
                    )
            print()

            new_total = total + usable
            for key in ("point_cloud", "state", "action", "timestamp_ms", "episode_id"):
                _resize_first_axis(arrays[key], new_total)
            arrays["point_cloud"][total:new_total] = point_cloud
            arrays["state"][total:new_total] = state
            arrays["action"][total:new_total] = action
            arrays["timestamp_ms"][total:new_total] = timestamps
            arrays["episode_id"][total:new_total] = episode_index
            _resize_first_axis(arrays["episode_ends"], episode_index + 1)
            arrays["episode_ends"][episode_index] = new_total

            episode_reports.append(
                {
                    "episode": episode.name,
                    "source_frames": len(frame_ids),
                    "converted_samples": usable,
                    "first_timestamp_ms": int(frame_ids[0]),
                    "last_timestamp_ms": int(frame_ids[-1]),
                    "warnings": list(episode.warnings),
                }
            )
            total = new_total

        contract = preprocessor.contract_dict()
        contract.update(
            {
                "source_root": str(args.input.expanduser().resolve()),
                "camera_serial": args.camera_serial,
                "observation_layout": [
                    "joint_1_rad",
                    "joint_2_rad",
                    "joint_3_rad",
                    "joint_4_rad",
                    "joint_5_rad",
                    "joint_6_rad",
                    "gripper_0_closed_1_open",
                ],
                "action_layout": [
                    "joint_1_target_rad",
                    "joint_2_target_rad",
                    "joint_3_target_rad",
                    "joint_4_target_rad",
                    "joint_5_target_rad",
                    "joint_6_target_rad",
                    "gripper_target_0_closed_1_open",
                ],
                "action_offset_frames": args.action_offset,
                "gripper_max_command": args.gripper_max_command,
                "force_torque_used": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        root.attrs.update(contract)

        validation = {
            "point_cloud_shape": list(arrays["point_cloud"].shape),
            "state_shape": list(arrays["state"].shape),
            "action_shape": list(arrays["action"].shape),
            "episode_ends": arrays["episode_ends"][:].tolist(),
            "finite_point_cloud": _array_is_finite(arrays["point_cloud"]),
            "finite_state": _array_is_finite(arrays["state"]),
            "finite_action": _array_is_finite(arrays["action"]),
        }
        expected_ends = np.cumsum([item["converted_samples"] for item in episode_reports])
        if not np.array_equal(arrays["episode_ends"][:], expected_ends):
            raise AssertionError("episode_ends validation failed")
        if not all(validation[key] for key in ("finite_point_cloud", "finite_state", "finite_action")):
            raise DatasetContractError("Converted Zarr contains non-finite values")

        if output.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = output.with_name(f"{output.name}.backup-{timestamp}")
            if backup_path.exists():
                raise FileExistsError(f"Backup path already exists: {backup_path}")
            os.replace(output, backup_path)
        try:
            os.replace(temp_dir, output)
        except Exception:
            if backup_path is not None and backup_path.exists() and not output.exists():
                os.replace(backup_path, output)
            raise

        return {
            "schema_version": SCHEMA_VERSION,
            "source_scan": source_report,
            "contract": contract,
            "episodes": episode_reports,
            "validation": validation,
            "output": str(output),
            "previous_output_backup": str(backup_path) if backup_path else None,
        }
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def main() -> int:
    args = _parse_args()
    try:
        episodes = discover_episodes(
            args.input,
            camera_serial=args.camera_serial,
            strict_metadata=args.strict_metadata,
        )
        if args.max_episodes is not None:
            if args.max_episodes < 1:
                raise DatasetContractError("--max-episodes must be >= 1")
            episodes = episodes[: args.max_episodes]
        source_report = scan_episodes(episodes, validate_arrays=not args.quick_scan)
        print(json.dumps(source_report, indent=2, ensure_ascii=False))

        if args.scan_only:
            if args.report:
                _write_json(args.report.expanduser().resolve(), source_report)
            return 0

        report = convert(args, episodes, source_report)
        report_path = (
            args.report.expanduser().resolve()
            if args.report
            else args.output.expanduser().resolve().with_suffix(".conversion_report.json")
        )
        _write_json(report_path, report)
        print(f"Converted dataset: {report['output']}")
        print(f"Validation report: {report_path}")
        if report["previous_output_backup"]:
            print(f"Previous output retained at: {report['previous_output_backup']}")
        return 0
    except Exception as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
