#!/opt/shared/Prog/anaconda3/envs/rise/bin/python
"""Action Executor for Hybrid64 V3.1 deployment.

Requires the ``rise`` conda environment (torch + pytorch3d + MinkowskiEngine + open3d):
    /opt/shared/Prog/anaconda3/envs/rise/bin/python

Bridges the policy adapter output [20, 10] (camera-frame, normalized,
rotation_6d) to robot base-frame axis-angle commands via the existing
Agent hardware interface.

Responsibilities:
- Denormalize camera-frame model output
- Convert camera-frame → robot base-frame (via Projector)
- rotation_6d → axis_angle conversion
- Gripper width mapping
- Safety validation (workspace, step limits, finite checks)
- Dispatch to Agent.set_tcp_pose / set_gripper_width

This module MUST NOT import the policy adapter or build any model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.projector import Projector
from utils.constants import (
    GRIPPER_THRESHOLD,
    MAX_GRIPPER_WIDTH,
    SAFE_EPS,
    SAFE_WORKSPACE_MAX,
    SAFE_WORKSPACE_MIN,
    TRANS_MAX,
    TRANS_MIN,
)
from utils.transformation import xyz_rot_transform

# ---------------------------------------------------------------------------
# Agent API quaternion convention: scalar-last [x, y, z, qx, qy, qz, qw].
# Original RISE dataset files use scalar-first [x, y, z, qw, qx, qy, qz].
# ---------------------------------------------------------------------------
QUATERNION_CONVENTION: str = "xyzw"  # scalar-last


# ---------------------------------------------------------------------------
# Action denormalization
# ---------------------------------------------------------------------------
def denormalize_action_chunk(
    action_norm: np.ndarray,
) -> np.ndarray:
    """Denormalize camera-frame action from [-1, 1] to physical units.

    Args:
        action_norm: [H, 10] normalized action (camera frame, rotation_6d).

    Returns:
        [H, 10] denormalized action:
          position in metres, rotation_6d unchanged, gripper in metres.
    """
    action = action_norm.copy().astype(np.float64)
    # position
    trans_range = np.asarray(TRANS_MAX, dtype=np.float64) - np.asarray(TRANS_MIN, dtype=np.float64)
    action[:, :3] = (action[:, :3] + 1.0) / 2.0 * trans_range + np.asarray(TRANS_MIN, dtype=np.float64)
    # gripper
    action[:, 9] = (action[:, 9] + 1.0) / 2.0 * float(MAX_GRIPPER_WIDTH)
    return action.astype(np.float64)


def decode_binary_gripper_widths(
    gripper_norm: np.ndarray,
    *,
    threshold_norm: float = -0.2,
    closed_width_m: float = 0.0,
    open_width_m: float = 0.095,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode the binary gripper labels used by the training dataset."""
    signal = np.asarray(gripper_norm, dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Non-finite normalized gripper signal: {signal}")
    closed = signal < float(threshold_norm)
    widths = np.where(closed, float(closed_width_m), float(open_width_m))
    return widths.astype(np.float64), closed


# ---------------------------------------------------------------------------
# Camera → base transform
# ---------------------------------------------------------------------------
class CameraBaseTransform:
    """Wraps the Projector for camera↔base TCP conversion."""

    def __init__(self, camera_serial: str = "243222076209"):
        self.projector = Projector(None)
        self.camera_serial = str(camera_serial)

    def tcp_camera_to_base(
        self,
        tcp_camera: np.ndarray,
        rotation_rep: str = "rotation_6d",
    ) -> np.ndarray:
        """Convert TCP pose from camera frame to robot base frame.

        Args:
            tcp_camera: [..., D] pose(s) in camera frame.
            rotation_rep: Rotation representation string.

        Returns:
            [..., D] pose(s) in base frame.
        """
        return self.projector.project_tcp_to_base_coord(
            tcp_camera,
            cam=self.camera_serial,
            rotation_rep=rotation_rep,
        )

    def tcp_base_to_camera(
        self,
        tcp_base: np.ndarray,
        rotation_rep: str = "rotation_6d",
    ) -> np.ndarray:
        """Inverse: base → camera."""
        return self.projector.project_tcp_to_camera_coord(
            tcp_base,
            cam=self.camera_serial,
            rotation_rep=rotation_rep,
        )

    @property
    def T_camera_base(self) -> np.ndarray:
        return np.asarray(self.projector.cam_to_base.get(self.camera_serial, np.eye(4)),
                          dtype=np.float64)


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------
class SafetyValidator:
    """Stateless safety checks.  All return (is_safe: bool, message: str)."""

    def __init__(
        self,
        safe_workspace_min: np.ndarray = SAFE_WORKSPACE_MIN,
        safe_workspace_max: np.ndarray = SAFE_WORKSPACE_MAX,
        safe_eps: float = SAFE_EPS,
        max_per_step_translation: float = 0.1,     # metres
        max_per_step_rotation_rad: float = 0.3,     # ~17°
        max_first_step_jump: float = 0.06,          # metres
        gripper_min_m: float = 0.0,
        gripper_max_m: float = 0.095,
    ):
        self.workspace_min = np.asarray(safe_workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(safe_workspace_max, dtype=np.float64)
        self.eps = float(safe_eps)
        self.max_translation = float(max_per_step_translation)
        self.max_rotation = float(max_per_step_rotation_rad)
        self.max_first_jump = float(max_first_step_jump)
        self.gripper_min = float(gripper_min_m)
        self.gripper_max = float(gripper_max_m)

    def check_workspace(self, positions: np.ndarray) -> Tuple[bool, str]:
        """Check all positions stay inside safe workspace."""
        lo = self.workspace_min + self.eps
        hi = self.workspace_max - self.eps
        violations = (positions < lo) | (positions > hi)
        if violations.any():
            idx = np.where(violations.any(axis=1))[0]
            return False, f"Workspace violation at steps {idx.tolist()}: pos={positions[idx]}"
        return True, "ok"

    def check_finite(self, action: np.ndarray) -> Tuple[bool, str]:
        if not np.all(np.isfinite(action)):
            n_bad = action.size - np.isfinite(action).sum()
            return False, f"Non-finite values in action: {n_bad}/{action.size}"
        return True, "ok"

    def check_step_limits(
        self,
        action: np.ndarray,
        current_tcp_base: np.ndarray,
        check_rotation: bool = True,
    ) -> Tuple[bool, str]:
        """Check max per-step translation and rotation."""
        positions = action[:, :3]
        position_sequence = np.concatenate([current_tcp_base[None, :3], positions], axis=0)
        diffs = np.linalg.norm(np.diff(position_sequence, axis=0), axis=1)
        max_diff = float(diffs.max())
        if max_diff > self.max_translation:
            step = int(np.argmax(diffs))
            return False, (
                f"Step {step}: translation delta {max_diff:.4f}m > "
                f"{self.max_translation:.4f}"
            )
        if not check_rotation:
            return True, "ok"
        # rotation check (via angular distance between rotation_6d)
        previous_rotation = current_tcp_base[3:]
        for i in range(action.shape[0]):
            ang = _rotation_6d_angular_distance(action[i, 3:9], previous_rotation)
            if ang > self.max_rotation:
                return False, f"Step {i}: rotation delta {ang:.4f}rad > {self.max_rotation:.4f}"
            previous_rotation = action[i, 3:9]
        return True, "ok"

    def check_first_step(
        self,
        action: np.ndarray,
        current_tcp_base: np.ndarray,
    ) -> Tuple[bool, str]:
        diff = np.linalg.norm(action[0, :3] - current_tcp_base[:3])
        if diff > self.max_first_jump:
            return False, f"First-step jump {diff:.4f}m > {self.max_first_jump:.4f}"
        return True, "ok"

    def check_gripper(self, widths: np.ndarray) -> Tuple[bool, str]:
        violations = (widths < self.gripper_min) | (widths > self.gripper_max)
        if violations.any():
            return False, f"Gripper width out of bounds: {widths[violations]}"
        return True, "ok"


def _rotation_6d_angular_distance(rot6d_1: np.ndarray, rot6d_2: np.ndarray) -> float:
    """Angular distance (radians) between two rotation_6d representations."""
    from utils.transformation import rotation_transform

    def _to_mat(r6d):
        return rotation_transform(r6d.astype(np.float64), from_rep="rotation_6d", to_rep="matrix")

    m1 = _to_mat(rot6d_1)
    m2 = _to_mat(rot6d_2)
    diff = m1 @ m2.T
    trace = np.trace(diff)
    cos_val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_val))


def _tcp_pose_to_base_rot6d(
    tcp_base: np.ndarray,
    rotation_rep: Optional[str] = None,
) -> np.ndarray:
    """Normalize Agent/base TCP pose to [x,y,z,rotation_6d]."""
    pose = np.asarray(tcp_base, dtype=np.float64).reshape(-1)
    if rotation_rep is None:
        if pose.shape[0] == 9:
            rotation_rep = "rotation_6d"
        elif pose.shape[0] == 7:
            rotation_rep = "quaternion_xyzw"
        elif pose.shape[0] == 6:
            rotation_rep = "axis_angle"
        else:
            raise ValueError(f"Cannot infer TCP rotation representation from shape {pose.shape}")

    if rotation_rep == "rotation_6d":
        if pose.shape[0] != 9:
            raise ValueError(f"rotation_6d TCP must have 9 values, got {pose.shape[0]}")
        return pose
    if rotation_rep == "axis_angle":
        if pose.shape[0] != 6:
            raise ValueError(f"axis_angle TCP must have 6 values, got {pose.shape[0]}")
        return xyz_rot_transform(pose, from_rep="axis_angle", to_rep="rotation_6d")
    if rotation_rep in {"quaternion_xyzw", "quaternion"}:
        if pose.shape[0] != 7:
            raise ValueError(f"quaternion TCP must have 7 values, got {pose.shape[0]}")
        # Agent.get_tcp_pose returns scalar-last [x,y,z,qx,qy,qz,qw], while
        # pytorch3d expects scalar-first quaternion [x,y,z,qw,qx,qy,qz].
        quat_wxyz = pose[[0, 1, 2, 6, 3, 4, 5]]
        return xyz_rot_transform(quat_wxyz, from_rep="quaternion", to_rep="rotation_6d")
    raise ValueError(f"Unsupported current_tcp_base rotation_rep={rotation_rep!r}")


def clip_tcp_to_safe_workspace(action_tcp: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Clip base-frame TCP xyz to the same safe workspace used by eval_duco.py."""
    before_clip = np.asarray(action_tcp, dtype=np.float64).copy()
    after_clip = before_clip.copy()
    after_clip[..., :3] = np.clip(
        after_clip[..., :3],
        SAFE_WORKSPACE_MIN + SAFE_EPS,
        SAFE_WORKSPACE_MAX - SAFE_EPS,
    )
    clip_delta = np.linalg.norm(before_clip[..., :3] - after_clip[..., :3], axis=-1)
    clipped_count = int(np.sum(clip_delta > 1e-9))
    max_clip_delta = float(np.max(clip_delta)) if clip_delta.size else 0.0
    return after_clip, clipped_count, max_clip_delta


# ---------------------------------------------------------------------------
# Action Executor
# ---------------------------------------------------------------------------
class ActionExecutor:
    """Converts model output to robot commands with safety gating.

    This class holds the *interface* to the hardware Agent but does NOT
    import or construct one — the caller must inject the Agent instance.
    """

    def __init__(
        self,
        camera_serial: str = "243222076209",
        safety: Optional[SafetyValidator] = None,
        enable_robot_motion: bool = False,
        gripper_threshold_norm: float = -0.2,
        gripper_closed_width_m: float = 0.0,
        gripper_open_width_m: float = 0.095,
    ):
        self.transform = CameraBaseTransform(camera_serial)
        self.safety = safety or SafetyValidator()
        self.enable_robot_motion = bool(enable_robot_motion)
        self.gripper_threshold_norm = float(gripper_threshold_norm)
        self.gripper_closed_width_m = float(gripper_closed_width_m)
        self.gripper_open_width_m = float(gripper_open_width_m)
        if not (
            self.safety.gripper_min <= self.gripper_closed_width_m
            <= self.gripper_open_width_m <= self.safety.gripper_max
        ):
            raise ValueError(
                "Binary gripper widths must lie inside the safety range: "
                f"closed={self.gripper_closed_width_m}, "
                f"open={self.gripper_open_width_m}, "
                f"safe=[{self.safety.gripper_min}, {self.safety.gripper_max}]"
            )
        self._agent = None
        self._plan: Optional[Dict[str, Any]] = None

    def set_agent(self, agent: Any) -> None:
        """Inject the hardware Agent (must have set_tcp_pose, set_gripper_width)."""
        self._agent = agent

    # ------------------------------------------------------------------
    # Pure computation (no robot)
    # ------------------------------------------------------------------
    def prepare_commands(
        self,
        action_norm: np.ndarray,
        current_tcp_base: Optional[np.ndarray] = None,
        current_grip_width: Optional[float] = None,
        current_tcp_rotation_rep: Optional[str] = None,
        check_rotation: bool = True,
        clip_workspace: bool = False,
        anchor_first_position: bool = False,
    ) -> Dict[str, Any]:
        """Convert [20, 10] normalized camera-frame action to base-frame commands.

        Returns a dict with the planned commands and all intermediate values
        for inspection / logging.  Does NOT send anything to the robot.
        """
        # --- contract validation -----------------------------------------
        assert action_norm.shape == (20, 10), f"Expected [20,10], got {action_norm.shape}"

        # --- denormalize -------------------------------------------------
        action_cam_continuous = denormalize_action_chunk(action_norm)
        action_cam = action_cam_continuous.copy()

        grip_widths, gripper_closed = decode_binary_gripper_widths(
            action_norm[:, 9],
            threshold_norm=self.gripper_threshold_norm,
            closed_width_m=self.gripper_closed_width_m,
            open_width_m=self.gripper_open_width_m,
        )
        action_cam[:, 9] = grip_widths

        # --- split position / rotation / gripper -------------------------
        tcp_cam_rot6d = action_cam[:, :9]       # [H, 9]
        grip_widths_continuous = action_cam_continuous[:, 9]  # diagnostics only

        # --- camera → base -----------------------------------------------
        tcp_base_rot6d = self.transform.tcp_camera_to_base(
            tcp_cam_rot6d, rotation_rep="rotation_6d"
        )  # [H, 9]
        tcp_base_rot6d_before_anchor = tcp_base_rot6d.copy()
        anchor_offset = np.zeros(3, dtype=np.float64)

        current_tcp_base_rot6d = None
        if current_tcp_base is not None:
            current_tcp_base_rot6d = _tcp_pose_to_base_rot6d(
                current_tcp_base,
                rotation_rep=current_tcp_rotation_rep,
            )

        if anchor_first_position:
            if current_tcp_base_rot6d is None:
                raise ValueError("anchor_first_position requires current_tcp_base")
            # Absolute predictions can have a sizeable first-waypoint error. Remove
            # that discontinuity while preserving the policy's terminal waypoint.
            anchor_offset = current_tcp_base_rot6d[:3] - tcp_base_rot6d[0, :3]
            anchor_weights = np.linspace(1.0, 0.0, tcp_base_rot6d.shape[0])[:, None]
            tcp_base_rot6d[:, :3] += anchor_weights * anchor_offset[None, :]

        tcp_base_rot6d_before_clip = tcp_base_rot6d.copy()
        clipped_count = 0
        max_clip_delta = 0.0
        if clip_workspace:
            tcp_base_rot6d, clipped_count, max_clip_delta = clip_tcp_to_safe_workspace(
                tcp_base_rot6d
            )

        # --- safety checks -----------------------------------------------
        checks: Dict[str, Tuple[bool, str]] = {}
        checks["finite"] = self.safety.check_finite(action_cam_continuous)
        checks["workspace"] = self.safety.check_workspace(tcp_base_rot6d[:, :3])
        checks["gripper"] = self.safety.check_gripper(grip_widths)

        if current_tcp_base is not None:
            checks["step_limits"] = self.safety.check_step_limits(
                tcp_base_rot6d,
                current_tcp_base_rot6d,
                check_rotation=check_rotation,
            )
            checks["first_step"] = self.safety.check_first_step(
                tcp_base_rot6d, current_tcp_base_rot6d
            )

        all_safe = all(v[0] for v in checks.values())

        # --- gripper change detection ------------------------------------
        gripper_change_steps: list[int] = []
        if current_grip_width is not None:
            if abs(float(grip_widths[0]) - float(current_grip_width)) > GRIPPER_THRESHOLD:
                gripper_change_steps.append(0)
        for i in range(1, len(grip_widths)):
            if abs(float(grip_widths[i]) - float(grip_widths[i - 1])) > GRIPPER_THRESHOLD:
                gripper_change_steps.append(i)

        plan = {
            "action_norm_camera": action_norm,
            "action_denorm_camera": action_cam,
            "action_denorm_camera_continuous_gripper": action_cam_continuous,
            "tcp_base_rot6d_before_anchor": tcp_base_rot6d_before_anchor,
            "anchor_first_position": bool(anchor_first_position),
            "anchor_position_offset": anchor_offset,
            "tcp_base_rot6d_before_clip": tcp_base_rot6d_before_clip,
            "tcp_base_rot6d": tcp_base_rot6d,
            "gripper_widths": grip_widths,
            "gripper_widths_continuous": grip_widths_continuous,
            "gripper_closed": gripper_closed,
            "gripper_threshold_norm": self.gripper_threshold_norm,
            "gripper_change_steps": gripper_change_steps,
            "workspace_clipped_count": clipped_count,
            "workspace_max_clip_delta": max_clip_delta,
            "checks": {k: {"safe": v[0], "message": v[1]} for k, v in checks.items()},
            "all_safe": all_safe,
        }
        self._plan = plan
        return plan

    # ------------------------------------------------------------------
    # Robot execution (gated)
    # ------------------------------------------------------------------
    def execute_step(
        self,
        step_index: int,
        plan: Optional[Dict[str, Any]] = None,
        blocking: bool = True,
    ) -> bool:
        """Execute a single step from a prepared plan.

        Returns True if the command was sent, False if blocked by safety.
        """
        if plan is None:
            plan = self._plan
        if plan is None:
            raise RuntimeError("No plan prepared. Call prepare_commands() first.")

        if not plan.get("all_safe", False):
            print(f"[ActionExecutor] ❌ Safety check failed — not executing.")
            return False

        if not self.enable_robot_motion:
            print(f"[ActionExecutor] 🔒 Robot motion disabled (enable_robot_motion=False).")
            return False

        if self._agent is None:
            raise RuntimeError("No Agent injected. Call set_agent() first.")

        step = int(step_index)
        tcp = plan["tcp_base_rot6d"][step].copy()
        width = float(plan["gripper_widths"][step])

        # rotation_6d → axis_angle for Duco
        tcp_axis = xyz_rot_transform(tcp, from_rep="rotation_6d", to_rep="axis_angle")

        # --- check if gripper change needed ------------------------------
        if step in plan.get("gripper_change_steps", []):
            print(f"[ActionExecutor] Step {step}: gripper → {width:.4f}m")
            self._agent.set_gripper_width(width, blocking=blocking)

        print(f"[ActionExecutor] Step {step}: TCP={np.array2string(tcp_axis, precision=4)}")
        self._agent.set_tcp_pose(tcp_axis, rotation_rep="axis_angle", blocking=blocking)
        return True


# ---------------------------------------------------------------------------
# CLI for offline testing
# ---------------------------------------------------------------------------
def parse_executor_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Action Executor — offline test & inspection.")
    p.add_argument("--action-npy", default="",
                   help="Path to .npy file containing [20,10] model output for dry-run.")
    p.add_argument("--camera-serial", default="243222076209")
    p.add_argument("--enable-robot-motion", action="store_true", default=False)
    p.add_argument("--confirm-motion", default="", help="Must be 'YES' to enable motion.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_executor_args(argv)

    enable = args.enable_robot_motion and args.confirm_motion == "YES"

    executor = ActionExecutor(
        camera_serial=args.camera_serial,
        enable_robot_motion=enable,
    )

    # Print the camera→base transform for inspection
    T = executor.transform.T_camera_base
    print("T_camera_base:")
    print(np.array2string(T, precision=6, suppress_small=True))

    if args.action_npy:
        action_norm = np.load(args.action_npy)
        if action_norm.ndim == 3:
            action_norm = action_norm[0]
        plan = executor.prepare_commands(action_norm)
        print("\n=== Prepared plan ===")
        print(f"All safe: {plan['all_safe']}")
        for k, v in plan["checks"].items():
            print(f"  {k}: {v}")
        print(f"Gripper change steps: {plan['gripper_change_steps']}")
        print("First 3 base-frame TCP (rotation_6d):")
        for i in range(min(3, plan["tcp_base_rot6d"].shape[0])):
            print(f"  step {i:2d}: {np.array2string(plan['tcp_base_rot6d'][i], precision=4, suppress_small=True)}")
    else:
        # dummy test
        action_norm = np.zeros((20, 10), dtype=np.float32)
        action_norm[:, 2] = 0.0  # z
        action_norm[:, 3] = 1.0  # rot_6d [1,0,0, 0,1,0] = identity
        action_norm[:, 7] = 1.0
        plan = executor.prepare_commands(action_norm)
        print("\n=== Dummy plan ===")
        print(f"All safe: {plan['all_safe']}")


if __name__ == "__main__":
    main()
