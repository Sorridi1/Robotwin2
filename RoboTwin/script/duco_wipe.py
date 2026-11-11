"""Independent native GUI / original replay / learned Duco policy inference."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Staging tests also use the real RoboTwin interfaces.
sys.path.insert(0, str(ROOT))
sys.path.append('/home/ljj/code/Robotwin2/RoboTwin')

import numpy as np
import yaml
from duco_sim.common import DEFAULT_RUN, add_geometry_path, write_new_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['view', 'replay', 'infer', 'check'])
    parser.add_argument('--config', type=Path, default=ROOT / 'task_config/duco_wipe_whiteboard.yml')
    parser.add_argument('--demo', type=int, default=50)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--action-steps', type=int, default=4, help='Execute this many predictions before replanning; 0 uses full returned chunk')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tool-mount', choices=['raw', 'estimated'])
    parser.add_argument('--tool-calibration', type=Path)
    parser.add_argument('--scene-profile', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory; existing dirs are never overwritten')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--tracking-error-limit', type=float, default=.35,
                        help='Stop after 5 consecutive steps exceeding this joint error in radians')
    args = parser.parse_args()
    if not 1 <= args.demo <= 100 or args.steps < 1 or args.action_steps < 0 or not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.tracking_error_limit) or args.tracking_error_limit <= 0:
        parser.error('Invalid demo, steps, action-steps or speed')
    if args.mode == 'view' and args.headless:
        parser.error('view opens the official native GUI; it cannot be headless')
    if args.output is None:
        args.output = ROOT / 'outputs/duco_wipe' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.output.exists():
        parser.error(f'Refusing to overwrite output directory: {args.output}')
    return args


def check_policy(args, config):
    """Real source RGB-D -> identical preprocessing -> checkpoint, no simulator."""
    from duco_sim.common import add_policy_path
    add_policy_path()
    from duco_sim.policy import DucoManiFlow
    from duco_wipe_data import load_recording
    from maniflow.common.rise_duco_data import RGBDPointCloudPreprocessor
    model = DucoManiFlow(args.run_dir, args.checkpoint, args.device, args.seed)
    processor = model.configure_preprocessor(RGBDPointCloudPreprocessor())
    recording = load_recording(config['dataset_root'], args.demo)
    for index in range(model.history.maxlen):
        stamp = recording.timestamp_ms[index]
        cloud = processor.load_files(recording.camera_path / 'color' / f'{stamp}.png', recording.camera_path / 'depth' / f'{stamp}.png')
        obs = dict(pointcloud=cloud, joint_action={'vector': np.r_[recording.joints[index], recording.gripper[index] / 1000.]})
        model.update_obs(obs)
    actions = model.predict()
    report = dict(mode='checkpoint_source_check', policy=model.report, returned_action_shape=list(actions.shape),
                  finite=bool(np.isfinite(actions).all()), simulation_tested=False, task_success_not_evaluated=True)
    write_new_json(args.output / 'report.json', report)
    np.save(args.output / 'predicted_actions.npy', actions, allow_pickle=False)
    print(json.dumps(report, indent=2), flush=True)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    config.update(demo=args.demo, max_steps=args.steps, seed=args.seed, preprocess_device=args.device)
    for key in ['tool_mount', 'tool_calibration', 'scene_profile']:
        if getattr(args, key) is not None:
            config[key] = str(getattr(args, key))
    add_geometry_path(config['geometry_root'])
    if args.mode == 'check':
        check_policy(args, config)
        return
    # Actual RoboTwin task subclass, not a ManiSkill environment.
    from envs.duco_wipe_whiteboard import duco_wipe_whiteboard
    env = duco_wipe_whiteboard()
    model = None
    requested, applied, states, inference_seconds = [], [], [], []
    eraser_poses, measured_qpos = [], []
    reason = 'step_limit'
    try:
        env.setup_demo(**config)
        print('[duco mount]', json.dumps(env.tool_calibration), flush=True)
        print('[duco task] single arm; 7D joint action; 10Hz; real contacts; no hardware driver', flush=True)
        if args.mode == 'infer':
            from duco_sim.policy import DucoManiFlow
            model = DucoManiFlow(args.run_dir, args.checkpoint, args.device, args.seed)
            env.preprocessor = model.configure_preprocessor(env.preprocessor)
        viewer = None if args.headless else env.render_human()
        if viewer is not None:
            viewer.paused = args.pause
        if args.mode == 'view':
            print('[duco GUI] Native SAPIEN Viewer. Mouse orbit/pan/zoom; official scene/entity/articulation/control panels.', flush=True)
            while not viewer.closed:
                env.render_human()
                time.sleep(.01)
            reason = 'viewer_closed'
        else:
            observation = env.get_obs()
            # Save a diagnostic camera image, never use it as a GUI replacement.
            from PIL import Image
            args.output.mkdir(parents=True, exist_ok=False)
            Image.fromarray(observation['observation']['head_camera']['rgb']).save(args.output / 'duco_initial_rgb.png')
            np.save(args.output / 'duco_initial_pointcloud.npy', observation['pointcloud'], allow_pickle=False)
            if model:
                model.update_obs(observation)
            from duco_wipe_data import sample_trajectory
            pending = []
            bad_tracking_steps = 0
            for step in range(args.steps):
                wall_start = time.perf_counter()
                if viewer is not None:
                    if viewer.closed:
                        reason = 'viewer_closed'
                        break
                    env.render_human()  # Official pause still allows inspection and camera movement.
                if model:
                    if not pending:
                        started = time.perf_counter()
                        chunk = model.predict()
                        inference_seconds.append(time.perf_counter() - started)
                        count = len(chunk) if args.action_steps == 0 else min(args.action_steps, len(chunk))
                        pending = list(chunk[:count])
                    action = pending.pop(0)
                else:
                    t = min((step + 1) / env.control_freq, env.recording.duration)
                    q, gripper = sample_trajectory(env.recording, env.recording.joints, t)
                    action = np.r_[q, gripper / 1000.]
                requested.append(np.asarray(action).copy())
                applied.append(env.take_action(action))
                states.append(env.get_state())
                measured_qpos.append(env.robot.get_qpos().copy())
                eraser_poses.append(np.r_[env.eraser.pose.p, env.eraser.pose.q])
                tracking_error = float(np.max(np.abs(states[-1][:6] - applied[-1][:6])))
                bad_tracking_steps = bad_tracking_steps + 1 if tracking_error > args.tracking_error_limit else 0
                observation = env.get_obs()
                if model:
                    model.update_obs(observation)
                if step % 20 == 0:
                    print('[duco step]', step + 1, json.dumps(env.task_metrics()), flush=True)
                if env.check_success():
                    reason = 'task_success'
                    break
                if bad_tracking_steps >= 5 or tracking_error > 2 * args.tracking_error_limit:
                    reason = 'excessive_joint_tracking_error'
                    print('[duco safety] Stopping: sustained or severe joint tracking error.', flush=True)
                    break
                if not model and (step + 1) / env.control_freq >= env.recording.duration:
                    reason = 'source_end'
                    break
                if viewer is not None:
                    env.render_human()
                    time.sleep(max(0., 1 / env.control_freq / args.speed - (time.perf_counter() - wall_start)))
        report = dict(mode=args.mode, stop_reason=reason, source_demo=str(env.recording.path),
                      steps=env.take_action_cnt, simulation_seconds=env.take_action_cnt / env.control_freq,
                      task_metrics=env.task_metrics(), scene=env.profile, tool_calibration=env.tool_calibration,
                      policy=None if model is None else model.report, learned_policy_executed=bool(model and applied),
                      requested_execution_chunk_steps=args.action_steps if model else None,
                      action_retargeting=False, scripted_task_trajectory=False, object_attachment=False,
                      safety_filter=dict(joint_limits=True, gripper_range=[0, 1],
                                         max_joint_step_rad=config['robot']['max_joint_step_rad'],
                                         tracking_error_stop_rad=args.tracking_error_limit,
                                         hard_tracking_error_stop_rad=2 * args.tracking_error_limit,
                                         raw_and_applied_actions_saved=True),
                      preprocessing=env.preprocessor.contract_dict(),
                      inference_seconds=inference_seconds,
                      limitations=['Tool mounting is provisional unless physically calibrated.',
                                   'RGB-D scene estimates and simple object shapes have a real-to-sim visual gap.',
                                   'Wiping uses real contact coverage, not pigment physics.',
                                   'Successful checkpoint loading or rollout completion does not establish task success.'])
        write_new_json(args.output / 'report.json', report)
        if requested:
            np.savez(args.output / 'duco_rollout.npz', requested=np.asarray(requested), applied=np.asarray(applied),
                     measured_state=np.asarray(states), measured_qpos=np.asarray(measured_qpos),
                     active_joint_names=np.array(env.active_names), eraser_pose_world=np.asarray(eraser_poses),
                     time_s=np.arange(1, len(states) + 1) / env.control_freq)
            Image.fromarray(observation['observation']['head_camera']['rgb']).save(args.output / 'duco_final_rgb.png')
        print('[duco result]', json.dumps(report['task_metrics']), '\n[duco output]', args.output, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
