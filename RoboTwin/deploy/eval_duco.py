import torch
import argparse
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import time

from copy import deepcopy
from pathlib import Path
import threading
from easydict import EasyDict as edict

from policy import FoAR
# ------------------ 修改 1: 引入适配好的 Duco Agent ------------------
from eval_duco_agent import Agent
# -------------------------------------------------------------------
from utils.constants import *
from dataset.constants import *
from utils.training import set_seed
from dataset.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform
from utils.constants import ROBOT_IP, ROBOT_PORT

default_args = edict({
    "ckpt": None,
    "crop_in_base": True,
    "num_action": 20,
    "num_inference_step": 20,
    "num_obs_force": 100,
    "num_motion_calc_steps": 5,
    "cls_threshold": 0.9,
    "force_threshold": 8.0,
    "torque_threshold": 5.0,
    "force_torque_freq": 100.0,
    "epsilon": 0.006,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 600,
    "seed": 233,
    "vis": False,
    "discretize_rotation": False,
    "ensemble_mode": "new",
    "lock_rotation": True,
    # 可以在这里添加默认 IP 配置，方便管理
    "robot_ip": ROBOT_IP,
    "pc_force_port": 2011,
    "camera_serial": "243222076209",
    "record_video": True,
    "record_video_dir": "/media/Elements/Data/FoAR/inference/recordings",
    "record_fps": 30,
    "control_interval": 0.2,
    "hold_gripper_after_close": True,
    "gripper_closed_width": 0.03,
})


def add_boolean_optional_argument(parser, name, default, help=None):
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument("--" + dest, dest=dest, action="store_true", help=help)
    parser.add_argument("--no_" + dest, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def make_recording_path(output_dir, serial):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"realsense_{serial}_{timestamp}.mp4"


class RealSenseObservationRecorder:
    """
    Record the already-open inference RealSense at camera rate.

    This thread is the only reader of agent.camera.pipeline while active. The
    policy loop consumes the latest cached RGB-D frame from get_observation().
    """
    def __init__(self, camera, output_dir, serial, fps=30):
        import cv2

        self.cv2 = cv2
        self.camera = camera
        self.serial = str(serial)
        self.fps = int(fps)
        self.output_path = make_recording_path(output_dir, self.serial)
        self.writer = None
        self.frame_size = None
        self.frames_written = 0
        self.latest_colors = None
        self.latest_depths = None
        self.error = None
        self.running = True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def _open_writer(self, bgr_frame):
        height, width = bgr_frame.shape[:2]
        self.frame_size = (width, height)
        fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = self.cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            float(self.fps),
            self.frame_size,
        )
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
        print(
            f"Recording inference observation video: {self.output_path} "
            f"({width}x{height}@{self.fps}fps, camera={self.serial})"
        )

    def _record_loop(self):
        try:
            while self.running:
                colors, depths = self.camera.get_rgbd_image()
                colors = colors.copy()
                depths = depths.copy()

                bgr_frame = self.cv2.cvtColor(colors.astype(np.uint8), self.cv2.COLOR_RGB2BGR)
                if self.writer is None:
                    self._open_writer(bgr_frame)
                elif bgr_frame.shape[1::-1] != self.frame_size:
                    bgr_frame = self.cv2.resize(bgr_frame, self.frame_size, interpolation=self.cv2.INTER_AREA)

                self.writer.write(bgr_frame)
                self.frames_written += 1

                with self.cond:
                    self.latest_colors = colors
                    self.latest_depths = depths
                    self.cond.notify_all()
        except Exception as exc:
            with self.cond:
                self.error = exc
                self.cond.notify_all()

    def get_observation(self, timeout=2.0):
        deadline = time.time() + timeout
        with self.cond:
            while self.latest_colors is None and self.error is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for RealSense recording frame.")
                self.cond.wait(timeout=remaining)
            if self.error is not None:
                raise RuntimeError(f"RealSense recording thread failed: {self.error}") from self.error
            return self.latest_colors.copy(), self.latest_depths.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        print(f"Video recorder stopped: {self.output_path} ({self.frames_written} frames)")


def make_policy(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = FoAR(
        num_action=args.num_action,
        input_dim=6,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        num_obs_force=args.num_obs_force
    ).to(device)
    return policy


def mask_point_cloud(points, colors, projector=None, cam_id="243222076209", crop_in_base=True):
    if not crop_in_base:
        x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
    else:
        points = projector.project_point_to_base_coord(points, cam=cam_id)
        x_mask = ((points[:, 0] >= WORKSPACE_BASE_MIN[0]) & (points[:, 0] <= WORKSPACE_BASE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_BASE_MIN[1]) & (points[:, 1] <= WORKSPACE_BASE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_BASE_MIN[2]) & (points[:, 2] <= WORKSPACE_BASE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
        points = projector.project_point_to_camera_coord(points, cam=cam_id)

    return points, colors


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size=0.005, projector=None, crop_in_base=True):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)
    # mask point cloud
    points, colors = mask_point_cloud(points, colors, projector=projector, crop_in_base=crop_in_base)
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis=-1).astype(np.float32)
    return cloud_final


def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch


def create_input(colors, depths, cam_intrinsics, voxel_size=0.005, projector=None, crop_in_base=True):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size=voxel_size, projector=projector,
                               crop_in_base=crop_in_base)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype=np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud


def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action


def _normalize_force(force_list):
    ''' force_list: [T, 6]'''
    force_list = (force_list - FORCE_MIN) / (FORCE_MAX - FORCE_MIN) * 2 - 1
    return force_list


def clip_action_tcp_to_safe_workspace(action_tcp):
    before_clip = action_tcp.copy()
    after_clip = action_tcp.copy()
    after_clip[..., :3] = np.clip(
        after_clip[..., :3],
        SAFE_WORKSPACE_MIN + SAFE_EPS,
        SAFE_WORKSPACE_MAX - SAFE_EPS
    )
    clip_delta = np.linalg.norm(before_clip[..., :3] - after_clip[..., :3], axis=-1)
    clipped_count = int(np.sum(clip_delta > 1e-9))
    max_clip_delta = float(np.max(clip_delta)) if clip_delta.size else 0.0
    return after_clip, clipped_count, max_clip_delta


def detect_gripper_change_steps(gripper_widths, threshold, current_or_prev_gripper_width=None):
    widths = np.asarray(gripper_widths, dtype=np.float64).reshape(-1)
    change_steps = []
    if len(widths) == 0:
        return change_steps
    if current_or_prev_gripper_width is not None:
        if abs(widths[0] - float(current_or_prev_gripper_width)) > threshold:
            change_steps.append(0)
    for i in range(1, len(widths)):
        if abs(widths[i] - widths[i - 1]) > threshold:
            change_steps.append(i)
    return change_steps


def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep="rotation_6d",
        to_rep="matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep="rotation_6d",
        to_rep="matrix"
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


def evaluate(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading policy ...")
    force_policy = make_policy(args)
    n_parameters = sum(p.numel() for p in force_policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    force_policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    force_policy.eval()
    print("Checkpoint {} loaded.".format(args.ckpt))
    print(
        f"Eval config: camera_serial={args.camera_serial}, crop_in_base={args.crop_in_base}, "
        f"num_obs_force={args.num_obs_force}, record_video={args.record_video}, "
        f"lock_rotation={args.lock_rotation}"
    )

    recorder = None
    agent = None
    try:
        agent = Agent(
            robot_ip=args.robot_ip,
            pc_force_port=args.pc_force_port,
            camera_serial=args.camera_serial,
            num_obs_force=args.num_obs_force,
            initial_gripper_closed=True,
        )
        if args.record_video:
            recorder = RealSenseObservationRecorder(
                camera=agent.camera,
                output_dir=args.record_video_dir,
                serial=args.camera_serial,
                fps=args.record_fps,
            )

        projector = Projector(None)
        ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)
        force_ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)
        img_process = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        last_rot = np.array(agent.ready_rot_6d, dtype=np.float32)
        fixed_home_rot_6d = np.array(agent.ready_rot_6d, dtype=np.float32)
        prop_value = 0.0
        prev_width = None
        pending_gripper_steps = set()
        gripper_closed_latched = False

        with torch.inference_mode():
            for t in range(args.max_steps):
                if t % args.num_inference_step == 0:
                    if recorder is not None:
                        raw_colors, raw_depths = recorder.get_observation()
                    else:
                        raw_colors, raw_depths = agent.get_observation()
                    colors = raw_colors.copy()
                    depths = raw_depths.copy()

                    if np.isnan(depths).any() or np.isinf(depths).any():
                        print("Warning: Depth contains NaN/Inf, skipping frame.")
                        continue

                    coords, feats, cloud = create_input(
                        colors,
                        depths,
                        cam_intrinsics=agent.intrinsics,
                        voxel_size=args.voxel_size,
                        projector=projector,
                        crop_in_base=args.crop_in_base
                    )
                    if cloud.shape[0] == 0:
                        print("Warning: point cloud is empty after crop, skipping frame.")
                        continue

                    feats, coords = feats.to(device), coords.to(device)
                    cloud_data = ME.SparseTensor(feats, coords)
                    tcp = agent.get_tcp_pose()
                    force_torque_base = deepcopy(agent.get_force_torque_history(freq=args.num_obs_force))

                    force_torque_cam = []
                    for i in range(args.num_obs_force):
                        if i < len(force_torque_base):
                            force_torque_cam.append(
                                projector.project_force_to_camera_coord(tcp, force_torque_base[i], cam=args.camera_serial)
                            )
                        else:
                            force_torque_cam.append(np.zeros(6))

                    force_torque_cam = np.asarray(force_torque_cam, dtype=np.float32)
                    force_torque_normalized = _normalize_force(force_torque_cam.copy())
                    force_torque_normalized = torch.from_numpy(force_torque_normalized).float()
                    force_torque_normalized = force_torque_normalized.unsqueeze(0).to(device)

                    color_list = img_process(colors).unsqueeze(0).to(device)
                    prop, pred_raw_action = force_policy(
                        force_torque_normalized,
                        color_list,
                        cloud_data,
                        actions=None,
                        contact=None,
                        batch_size=1
                    )
                    prop_value = float(torch.as_tensor(prop).detach().cpu().reshape(-1)[0])
                    pred_raw_action = pred_raw_action.squeeze(0).cpu().numpy()

                    action = unnormalize_action(pred_raw_action)
                    if args.vis:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                        pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                        tcp_vis_list = []
                        for raw_tcp in action:
                            tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                            tcp_vis_list.append(tcp_vis)
                        o3d.visualization.draw_geometries([pcd, *tcp_vis_list])

                    action_tcp = projector.project_tcp_to_base_coord(
                        action[..., :-1],
                        cam=args.camera_serial,
                        rotation_rep="rotation_6d"
                    )
                    action_width = action[..., -1]
                    action_tcp, clipped_count, max_clip_delta = clip_action_tcp_to_safe_workspace(action_tcp)
                    action = np.concatenate([action_tcp, action_width[..., np.newaxis]], axis=-1)
                    gripper_change_steps = detect_gripper_change_steps(
                        action_width,
                        GRIPPER_THRESHOLD,
                        current_or_prev_gripper_width=prev_width
                    )
                    pending_gripper_steps = {int(t + step) for step in gripper_change_steps}

                    print(
                        f"接触概率: {prop_value:.4f}; "
                        f"safe_z=[{SAFE_WORKSPACE_MIN[2] + SAFE_EPS:.4f}, {SAFE_WORKSPACE_MAX[2] - SAFE_EPS:.4f}], "
                        f"pred_z_min/max={action[:, 2].min():.4f}/{action[:, 2].max():.4f}, "
                        f"clipped_count={clipped_count}, max_clip_delta={max_clip_delta:.6f}, "
                        f"gripper_change_steps={gripper_change_steps}"
                    )
                    if prop_value < args.cls_threshold:
                        ensemble_buffer.add_action(action, t)
                    else:
                        cur_force_value, cur_torque_value = agent.get_force_torque_value()
                        if cur_force_value < args.force_threshold and cur_torque_value < args.torque_threshold:
                            distance = np.mean(action[:args.num_motion_calc_steps, :3], axis=0) - agent.get_tcp_pose()[:3]
                            dist_norm = np.linalg.norm(distance)
                            if dist_norm > 1e-6:
                                action[:, :3] = action[:, :3] + distance / dist_norm * args.epsilon
                                action[:, :-1], clipped_count, max_clip_delta = clip_action_tcp_to_safe_workspace(action[:, :-1])
                                print(
                                    f"reactive action re-clipped: clipped_count={clipped_count}, "
                                    f"max_clip_delta={max_clip_delta:.6f}, "
                                    f"z_min/max={action[:, 2].min():.4f}/{action[:, 2].max():.4f}"
                                )
                        force_ensemble_buffer.add_action(action, t)

                if prop_value < args.cls_threshold:
                    step_action = ensemble_buffer.get_action()
                    force_ensemble_buffer.get_action()
                else:
                    ensemble_buffer.get_action()
                    step_action = force_ensemble_buffer.get_action()

                if step_action is None:
                    continue

                step_tcp = step_action[:-1].copy()
                step_width = float(step_action[-1])

                if args.lock_rotation:
                    final_pose_6d = np.concatenate([step_tcp[:3], fixed_home_rot_6d[3:]])
                    print("执行动作 (TCP位置更新, 旋转锁定):", final_pose_6d, " Gripper Width:", step_width)
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
                if int(t) in pending_gripper_steps:
                    if (
                        args.hold_gripper_after_close
                        and gripper_closed_latched
                        and step_width > args.gripper_closed_width
                    ):
                        print(
                            f"跳过夹爪打开命令: latched closed, requested width={step_width:.5f} m"
                        )
                    else:
                        print(f"执行夹爪命令: step={t}, width={step_width:.5f} m")
                        agent.set_gripper_width(step_width, blocking=True)
                        prev_width = step_width
                        if step_width <= args.gripper_closed_width:
                            gripper_closed_latched = True
                    pending_gripper_steps.discard(int(t))
    finally:
        if recorder is not None:
            recorder.stop()
        if agent is not None:
            agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='policy checkpoint path', required=True)
    # parser.add_argument('--calib', action='store', type=str, help='calibration path', required=True)
    add_boolean_optional_argument(
        parser,
        "crop_in_base",
        default=default_args.crop_in_base,
        help='whether to crop point cloud in base coordinate'
    )
    parser.add_argument('--num_action', action='store', type=int, help='number of action steps', required=False,
                        default=20)
    parser.add_argument('--num_inference_step', action='store', type=int, help='number of inference query steps',
                        required=False, default=20)
    parser.add_argument('--num_obs_force', action='store', type=int, help='width of force window', required=False,
                        default=100)
    parser.add_argument('--num_motion_calc_steps', action='store', type=int, help='number of motion calculation steps',
                        required=False, default=5)
    parser.add_argument('--cls_threshold', action='store', type=float, help='threshold for future contact probability',
                        required=False, default=0.9)
    parser.add_argument('--force_threshold', action='store', type=float, help='force threshold', required=False,
                        default=8.0)
    parser.add_argument('--torque_threshold', action='store', type=float, help='torque threshold', required=False,
                        default=5.0)
    parser.add_argument('--force_torque_freq', action='store', type=float, help='force torque freq', required=False,
                        default=100.0)
    parser.add_argument('--epsilon', action='store', type=int, help='epsilon for reactive control', required=False,
                        default=0.006)
    parser.add_argument('--voxel_size', action='store', type=float, help='voxel size', required=False, default=0.005)
    parser.add_argument('--obs_feature_dim', action='store', type=int, help='observation feature dimension',
                        required=False, default=512)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden dimension', required=False, default=512)
    parser.add_argument('--nheads', action='store', type=int, help='number of heads', required=False, default=8)
    parser.add_argument('--num_encoder_layers', action='store', type=int, help='number of encoder layers',
                        required=False, default=4)
    parser.add_argument('--num_decoder_layers', action='store', type=int, help='number of decoder layers',
                        required=False, default=1)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='feedforward dimension', required=False,
                        default=2048)
    parser.add_argument('--dropout', action='store', type=float, help='dropout ratio', required=False, default=0.1)
    parser.add_argument('--max_steps', action='store', type=int, help='max steps for evaluation', required=False,
                        default=300)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=233)
    parser.add_argument('--vis', action='store_true', help='add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action='store_true', help='whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action='store', type=str, help='temporal ensemble mode', required=False,
                        default='new')
    add_boolean_optional_argument(
        parser,
        "lock_rotation",
        default=default_args.lock_rotation,
        help='lock rollout rotation to HOME_POSE axis-angle; pass --no_lock_rotation to use predicted rotation_6d'
    )

    # ------------------ 修改 3: 增加 IP 和 端口 参数 (可选) ------------------
    import utils.constants as constants

    parser.add_argument('--robot_ip', action='store', type=str, help='Robot IP address', required=False,
                        default=constants.ROBOT_IP)
    parser.add_argument('--pc_force_port', action='store', type=int, help='UDP port for force data', required=False,
                        default=constants.FORCE_PORT)
    parser.add_argument('--camera_serial', action='store', type=str, help='RealSense camera serial', required=False,
                        default=default_args.camera_serial)
    add_boolean_optional_argument(
        parser,
        "record_video",
        default=default_args.record_video,
        help='record RGB observation frames from the inference RealSense camera'
    )
    parser.add_argument('--record_video_dir', action='store', type=str, help='video output directory', required=False,
                        default=default_args.record_video_dir)
    parser.add_argument('--record_fps', action='store', type=int, help='saved video FPS metadata', required=False,
                        default=default_args.record_fps)
    parser.add_argument('--control_interval', action='store', type=float, help='sleep after each sent TCP target',
                        required=False, default=default_args.control_interval)
    add_boolean_optional_argument(
        parser,
        "hold_gripper_after_close",
        default=default_args.hold_gripper_after_close,
        help='keep gripper closed after the first close command; pass --no_hold_gripper_after_close to allow reopening'
    )
    parser.add_argument('--gripper_closed_width', action='store', type=float,
                        help='width threshold in meters for latching gripper as closed',
                        required=False, default=default_args.gripper_closed_width)

    evaluate(vars(parser.parse_args()))
