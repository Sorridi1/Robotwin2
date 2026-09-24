#!/usr/bin/env python3
"""Real-robot deployment loop for bsxyz_mainline on Duco.

This script keeps the hardware lifecycle and motion execution style from
``eval_duco.py`` but replaces the policy path with:

    RGB-D -> raw17 preprocessing -> bsxyz_mainline policy -> safety preview

It does not modify ``eval_duco.py``; it is a separate deployment entrypoint
for the mainline checkpoint format.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch

from deployment.action_executor import ActionExecutor
from deployment.hybrid64_v31_policy_adapter import Hybrid64V31PolicyAdapter
from deployment.online_obs_preprocessor import preprocess_observation
from eval_duco_agent import Agent
from utils.constants import ROBOT_IP, FORCE_PORT
from utils.ensemble import EnsembleBuffer
from utils.training import set_seed
from utils.transformation import rotation_transform, xyz_rot_transform


def _build_agent(args):
    return Agent(
        robot_ip=args.robot_ip,
        pc_force_port=args.pc_force_port,
        camera_serial=args.camera_serial,
        num_obs_force=args.num_obs_force,
        initial_gripper_closed=True,
    )


def _prepare_observation(agent, voxel_size: float, max_num_points: int = 0):
    colors, depths = agent.get_observation()
    obs = preprocess_observation(
        colors=colors,
        depths=depths,
        intrinsics=agent.intrinsics,
        current_tcp=agent.get_tcp_pose(),
        voxel_size=voxel_size,
        max_num_points=max_num_points,
    )
    if obs["M"] <= 0:
        return None
    return obs, colors, depths


def _save_rgb_and_depth(step: int, colors: np.ndarray, depths: np.ndarray,
                        debug_dir: str) -> None:
    """Save RGB image and depth heatmap as PNG files."""
    rgb_path = os.path.join(debug_dir, f"step_{step:03d}_rgb.png")
    depth_path = os.path.join(debug_dir, f"step_{step:03d}_depth.png")

    # RGB: OpenCV expects BGR, so convert
    cv2.imwrite(rgb_path, cv2.cvtColor(colors, cv2.COLOR_RGB2BGR))

    # Depth: normalize to [0, 255] and apply a colormap
    d = depths.copy()
    finite = np.isfinite(d)
    if finite.any():
        d_min, d_max = d[finite].min(), d[finite].max()
        if d_max > d_min:
            d_clipped = np.clip(d, d_min, d_max)
            d_norm = ((d_clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(depth_path, d_color)
    np.save(os.path.join(debug_dir, f"step_{step:03d}_depth_m.npy"), depths)


def _save_trajectory_plot(step: int, action_norm: np.ndarray, plan: dict,
                          debug_dir: str) -> None:
    """Save a 3D trajectory visualization of the predicted action chunk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use the base-frame trajectory (already denormalized and transformed)
    tcp_base = plan["tcp_base_rot6d"]  # [20, 9]
    positions = tcp_base[:, :3]  # [20, 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory as a connected line with markers
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "b-", linewidth=2, label="Predicted TCP trajectory")
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c="green", s=120, marker="o", label="Start (step 0)")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c="red", s=120, marker="X", label="End (step 19)")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c="blue", s=20, alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Step {step:03d} — Predicted Trajectory (base frame)\n"
                 f"all_safe={plan['all_safe']}")
    ax.legend()

    # Equal aspect ratio on XY plane helps visual interpretation
    x_vals, y_vals = positions[:, 0], positions[:, 1]
    z_vals = positions[:, 2]
    max_range_xy = max(x_vals.ptp(), y_vals.ptp())
    if max_range_xy > 0:
        ax.set_box_aspect([x_vals.ptp() / max_range_xy,
                           y_vals.ptp() / max_range_xy,
                           z_vals.ptp() / max_range_xy if z_vals.ptp() > 0 else 1.0])

    fig.tight_layout()
    traj_path = os.path.join(debug_dir, f"step_{step:03d}_traj.png")
    fig.savefig(traj_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_plan_arrays(
    step: int,
    action_norm: np.ndarray,
    plan: dict,
    current_tcp: np.ndarray,
    observation: dict,
    debug_dir: str,
) -> None:
    """Save numeric policy and execution outputs for offline diagnosis."""
    np.savez_compressed(
        os.path.join(debug_dir, f"step_{step:03d}_plan.npz"),
        current_tcp_base=np.asarray(current_tcp),
        action_norm_camera=np.asarray(action_norm),
        action_denorm_camera=np.asarray(plan["action_denorm_camera"]),
        tcp_base_before_anchor=np.asarray(plan["tcp_base_rot6d_before_anchor"]),
        anchor_position_offset=np.asarray(plan["anchor_position_offset"]),
        tcp_base_before_clip=np.asarray(plan["tcp_base_rot6d_before_clip"]),
        tcp_base=np.asarray(plan["tcp_base_rot6d"]),
        gripper_widths_continuous=np.asarray(plan["gripper_widths_continuous"]),
        gripper_widths_command=np.asarray(plan["gripper_widths"]),
        input_coords=np.asarray(observation["input_coords"]),
        input_feats=np.asarray(observation["input_feats"]),
        input_xyz=np.asarray(observation["input_xyz"]),
    )


def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep="rotation_6d",
        to_rep="matrix",
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep="rotation_6d",
        to_rep="matrix",
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)


def discretize_rotation(rot_begin, rot_end, rot_step_size=np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps


def _longest_true_run(mask: np.ndarray) -> tuple:
    """Return ``(start, length)`` for the longest contiguous true run."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = 0
    best_length = 0
    run_start = 0
    run_length = 0
    for i, value in enumerate(values):
        if value:
            if run_length == 0:
                run_start = i
            run_length += 1
            if run_length > best_length:
                best_start = run_start
                best_length = run_length
        else:
            run_length = 0
    return int(best_start), int(best_length)


def _plan_debounced_gripper_commands(
    plan: dict,
    *,
    current_width: float,
    closed_width_threshold: float,
    min_run_steps: int,
) -> dict:
    """Debounce binary model output without imposing task-specific phases."""
    predicted_closed = np.asarray(plan["gripper_closed"], dtype=bool).reshape(-1)
    predicted_widths = np.asarray(plan["gripper_widths"], dtype=float).reshape(-1)
    if predicted_widths.shape != predicted_closed.shape:
        raise ValueError("gripper_closed and gripper_widths must have the same shape")
    stable_closed = float(current_width) <= float(closed_width_threshold)
    commands = {}
    i = 0
    while i < predicted_closed.size:
        run_state = bool(predicted_closed[i])
        j = i + 1
        while j < predicted_closed.size and bool(predicted_closed[j]) == run_state:
            j += 1
        run_length = j - i
        if run_state != stable_closed and run_length >= int(min_run_steps):
            commands[int(i)] = float(predicted_widths[i])
            stable_closed = run_state
        i = j

    open_start, open_run = _longest_true_run(~predicted_closed)
    closed_start, closed_run = _longest_true_run(predicted_closed)
    return {
        "commands": commands,
        "open_run_start": open_start,
        "open_run_length": open_run,
        "closed_run_start": closed_start,
        "closed_run_length": closed_run,
    }


def evaluate(args):
    if args.gripper_min_run_steps < 1:
        raise ValueError("--gripper_min_run_steps must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = Hybrid64V31PolicyAdapter(
        checkpoint_path=args.ckpt,
        device=str(device),
        eval_seed=args.seed,
    )
    report = adapter.load_checkpoint_strict()
    print("Checkpoint loaded.")
    print(report)

    executor = ActionExecutor(
        camera_serial=args.camera_serial,
        enable_robot_motion=False,
        gripper_threshold_norm=float(adapter.args_dict.get("grip_threshold", -0.2)),
    )

    agent = None
    try:
        agent = _build_agent(args)
        ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)
        fixed_home_axis_angle = np.array(agent.ready_pose, dtype=np.float32)
        last_rot = xyz_rot_transform(
            fixed_home_axis_angle,
            from_rep="axis_angle",
            to_rep="rotation_6d",
        )[3:]
        prev_width = None
        pending_gripper_commands = {}
        replan_tcp_history = []

        if args.debug_dir is not None:
            os.makedirs(args.debug_dir, exist_ok=True)
            print(f"[Debug] Saving debug output to {os.path.abspath(args.debug_dir)}")

        with torch.inference_mode():
            for t in range(args.max_steps):
                if t % args.num_inference_step == 0:
                    result = _prepare_observation(
                        agent=agent,
                        voxel_size=args.voxel_size,
                        max_num_points=args.max_num_points,
                    )
                    if result is None:
                        print("Warning: empty raw17 observation, skipping.")
                        continue
                    obs, colors, depths = result

                    current_tcp = agent.get_tcp_pose()
                    current_grip_width = (
                        prev_width if prev_width is not None else agent.get_gripper_width()
                    )
                    inference_seed = args.seed + t if args.vary_inference_seed else args.seed
                    action_norm = adapter.predict_action_chunk(obs, seed=inference_seed)
                    plan = executor.prepare_commands(
                        action_norm,
                        current_tcp_base=current_tcp,
                        current_grip_width=current_grip_width,
                        check_rotation=not args.lock_rotation,
                        clip_workspace=True,
                        anchor_first_position=args.anchor_chunk_start,
                    )

                    if args.debug_dir is not None:
                        _save_rgb_and_depth(t, colors, depths, args.debug_dir)
                        _save_trajectory_plot(t, action_norm, plan, args.debug_dir)
                        _save_plan_arrays(t, action_norm, plan, current_tcp, obs, args.debug_dir)

                    positions = plan["tcp_base_rot6d"][:, :3]
                    first_delta = positions[0] - current_tcp[:3]
                    horizon_delta = positions[-1] - current_tcp[:3]
                    axis_span = np.ptp(positions, axis=0)
                    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
                    gripper_schedule = _plan_debounced_gripper_commands(
                        plan,
                        current_width=current_grip_width,
                        closed_width_threshold=args.gripper_closed_width,
                        min_run_steps=args.gripper_min_run_steps,
                    )
                    pending_gripper_commands = {
                        int(t + step): float(width)
                        for step, width in gripper_schedule["commands"].items()
                    }

                    print(
                        f"[t={t}] raw17_tokens={obs['M']} "
                        f"all_safe={plan['all_safe']} "
                        f"safe_z=[{executor.safety.workspace_min[2] + executor.safety.eps:.4f}, "
                        f"{executor.safety.workspace_max[2] - executor.safety.eps:.4f}] "
                        f"pred_z_min/max={plan['tcp_base_rot6d'][:, 2].min():.4f}/"
                        f"{plan['tcp_base_rot6d'][:, 2].max():.4f} "
                        f"clipped_count={plan['workspace_clipped_count']} "
                        f"max_clip_delta={plan['workspace_max_clip_delta']:.6f} "
                        f"anchor_offset={np.round(plan['anchor_position_offset'], 4).tolist()} "
                        f"inference_seed={inference_seed} "
                        f"tcp_now={np.round(current_tcp[:3], 4).tolist()} "
                        f"first_delta={np.round(first_delta, 4).tolist()} "
                        f"horizon_delta={np.round(horizon_delta, 4).tolist()} "
                        f"axis_span={np.round(axis_span, 4).tolist()} "
                        f"path_length={path_length:.4f}m "
                        f"gripper_raw_min/max="
                        f"{plan['gripper_widths_continuous'].min():.5f}/"
                        f"{plan['gripper_widths_continuous'].max():.5f}m "
                        f"gripper_cmd={np.unique(plan['gripper_widths']).tolist()} "
                        f"gripper_change_steps={plan['gripper_change_steps']} "
                        f"open_run={gripper_schedule['open_run_length']}@"
                        f"{gripper_schedule['open_run_start']} "
                        f"closed_run={gripper_schedule['closed_run_length']}@"
                        f"{gripper_schedule['closed_run_start']} "
                        f"scheduled_gripper={pending_gripper_commands} "
                        f"ensemble_mode={args.ensemble_mode}"
                    )

                    replan_tcp_history.append(current_tcp[:3].copy())
                    if len(replan_tcp_history) >= args.stall_window:
                        recent = np.asarray(replan_tcp_history[-args.stall_window:])
                        travelled = float(np.linalg.norm(np.diff(recent, axis=0), axis=1).sum())
                        net_progress = float(np.linalg.norm(recent[-1] - recent[0]))
                        if travelled >= args.stall_min_travel and net_progress <= args.stall_max_progress:
                            print(
                                f"[PolicyStall] last {args.stall_window} replans: "
                                f"travelled={travelled:.4f}m but net_progress={net_progress:.4f}m; "
                                "policy is oscillating in a local region"
                            )

                    if not plan["all_safe"]:
                        failed_checks = {
                            name: check["message"]
                            for name, check in plan["checks"].items()
                            if not check["safe"]
                        }
                        raise RuntimeError(f"Unsafe action plan; robot motion aborted: {failed_checks}")

                    action = np.concatenate(
                        [plan["tcp_base_rot6d"], plan["gripper_widths"][..., np.newaxis]],
                        axis=-1,
                    )
                    ensemble_buffer.add_action(action, t)

                step_action = ensemble_buffer.get_action()
                if step_action is None:
                    continue

                step_tcp = step_action[:-1].copy()
                step_width = float(step_action[-1])

                if args.lock_rotation:
                    final_pose_6d = np.concatenate([step_tcp[:3], fixed_home_axis_angle[3:]])
                    print(
                        "执行动作 (TCP位置更新, 旋转锁定):",
                        final_pose_6d,
                        " Model Gripper Width:",
                        step_width,
                    )
                    agent.set_tcp_pose(final_pose_6d, rotation_rep="axis_angle", blocking=False)
                elif args.discretize_rotation:
                    rot_steps = discretize_rotation(last_rot, step_tcp[3:], np.pi / 16)
                    last_rot = step_tcp[3:].copy()
                    for rot in rot_steps:
                        step_tcp_i = step_tcp.copy()
                        step_tcp_i[3:] = rot
                        print("执行动作 (预测rotation_6d离散插值):", step_tcp_i, " Gripper Width:", step_width)
                        agent.set_tcp_pose(step_tcp_i, rotation_rep="rotation_6d", blocking=True)
                else:
                    print("执行动作 (预测rotation_6d):", step_tcp, " Gripper Width:", step_width)
                    agent.set_tcp_pose(step_tcp, rotation_rep="rotation_6d", blocking=True)

                time.sleep(args.control_interval)
                if args.debug_dir is not None:
                    actual_tcp = agent.get_tcp_pose()
                    tracking_delta = actual_tcp[:3] - step_tcp[:3]
                    print(
                        f"[Track t={t}] cmd_xyz={np.round(step_tcp[:3], 5).tolist()} "
                        f"actual_xyz={np.round(actual_tcp[:3], 5).tolist()} "
                        f"error_xyz={np.round(tracking_delta, 5).tolist()} "
                        f"error_norm={np.linalg.norm(tracking_delta):.5f}m"
                    )
                if int(t) in pending_gripper_commands:
                    command_width = float(pending_gripper_commands.pop(int(t)))
                    print(
                        f"执行夹爪命令: step={t}, width={command_width:.5f} m"
                    )
                    agent.set_gripper_width(command_width, blocking=True)
                    prev_width = command_width
    finally:
        if agent is not None:
            agent.stop()


def add_boolean_optional_argument(parser, name, default, help=None):
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument("--" + dest, dest=dest, action="store_true", help=help)
    parser.add_argument("--no_" + dest, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Explicit bsxyz_mainline checkpoint path; no stale-model default is used.",
    )
    parser.add_argument("--robot_ip", type=str, default=ROBOT_IP, required=False)
    parser.add_argument("--pc_force_port", type=int, default=FORCE_PORT, required=False)
    parser.add_argument("--camera_serial", type=str, default="243222076209", required=False)
    parser.add_argument("--num_obs_force", type=int, default=100, required=False)
    parser.add_argument("--num_inference_step", type=int, default=20, required=False)
    parser.add_argument("--max_steps", type=int, default=1000, required=False)
    parser.add_argument("--seed", type=int, default=233, required=False)
    parser.add_argument("--voxel_size", type=float, default=0.005, required=False)
    parser.add_argument("--max_num_points", type=int, default=0, required=False)
    parser.add_argument("--control_interval", type=float, default=0.2, required=False)
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="If set, save debug images (RGB, depth, trajectory) per inference step.")
    parser.add_argument("--ensemble_mode", type=str, default="new", required=False,
                        help="Temporal ensemble mode: new, old, avg, act, or hato.")
    add_boolean_optional_argument(
        parser,
        "anchor_chunk_start",
        default=False,
        help="Anchor each predicted chunk to the measured TCP and decay the correction to zero at the horizon.",
    )
    add_boolean_optional_argument(
        parser,
        "vary_inference_seed",
        default=False,
        help="Use a different diffusion seed at each replan; fixed-seed deployment is the default.",
    )
    parser.add_argument("--stall_window", type=int, default=5)
    parser.add_argument("--stall_min_travel", type=float, default=0.05)
    parser.add_argument("--stall_max_progress", type=float, default=0.02)
    parser.add_argument(
        "--gripper_min_run_steps",
        type=int,
        default=3,
        help="Ignore binary gripper pulses shorter than this many consecutive model steps.",
    )
    add_boolean_optional_argument(parser, "lock_rotation", default=True, help="Lock rollout rotation to HOME_POSE.")
    parser.add_argument("--discretize_rotation", action="store_true", default=False)
    parser.add_argument("--gripper_closed_width", type=float, default=0.03, required=False,
                        help="Width threshold in meters for interpreting measured gripper state.")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
