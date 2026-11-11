import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
from scipy.spatial.transform import Rotation
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        voxel_size = 0.005,
        cam_ids = ['750612070851'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.vis = vis
        
        self.all_demos = sorted(os.listdir(self.data_path))
        # DEBUG
        self.num_demos = len(self.all_demos)
        # self.num_demos = 2

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.projectors = {}
        
        for i in range(self.num_demos):

            demo_path = os.path.join(self.data_path, self.all_demos[i])
            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # metadata
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                # get calib timestamps
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
        
    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        center = clouds[-1][..., :3].mean(axis = 0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        tcps[..., :3] -= center
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps[..., :3] += center

        return clouds, tcps

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        # print("宽度前",tcp_list[:, -1])
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        # print("宽度后",tcp_list[:, -1])
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id):
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.#120

        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')
        
        # load camera projector by calib timestamp
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            # print('color_id',os.path.join(color_dir, "{}.png".format(frame_id)))

            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)


        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)

            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            if index < 3 and i == 0:
                print("  valid after workspace:", int(mask.sum()))
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            # print('tcp_id',frame_id)
            # tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            tcp_raw = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))
            tcp_raw = np.asarray(tcp_raw).reshape(-1)   # 保证一维
            xyz = tcp_raw[:3]

            #new
            qx, qy, qz,qw =tcp_raw[3:]
            # R_flat = tcp_raw[3:12]
            # R_mat = R_flat.reshape(3,3)
            # # 转四元数（SciPy 返回 [qx,qy,qz,qw]）
            # qx, qy, qz, qw = Rotation.from_matrix(R_mat).as_quat()


            tcp = np.array([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz], dtype=np.float32)
            # print("frame_id投影前的tcp",tcp)


            # print("GT",tcp)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            
            # projected_tcp=[ 0.18844119, -0.00284205,  0.62785938,  0.26420109,  0.02284793,  0.36668259, -0.89175088]
            # projected_tcp=[ 0.0000844119, 0.000000284205,  1.1800011,  0.26420109,  0.02284793,  0.36668259, -0.89175088]
            # projected_tcp=[ 0.21466789,  0.002276,   0.58080907,  0.68464936, -0.26898689, -0.21512515,-0.64235697]
            # print("投影后的tcp",projected_tcp)

            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        
        # action_tcps.append([ 0.0000844119, 0.000000284205,  1.1800011,  0.26420109,  0.02284793,  0.36668259, -0.89175088])
        # action_grippers.append(0.89175088)
        # action_tcps.append([ 0.0000844119, 0.000000284205,  0.00011,  0.26420109,  0.02284793,  0.36668259, -0.89175088])
        # action_grippers.append(0.89175088)
        # action_tcps.append([ 0.0000844119, 0.500000284205,  1.18000011,  0.26420109,  0.02284793,  0.36668259, -0.89175088])
        # action_grippers.append(0.89175088)
        # action_tcps.append([ 0.0000844119, 0.500000284205,  0.00011,  0.26420109,  0.02284793,  0.36668259, -0.89175088])
        # action_grippers.append(0.89175088)
        # action_tcps.append([ 0.0600001, 0.500000284205,  1.18000011,  0.26420109,  0.02284793,  0.36668259, -0.89175088])
        # action_grippers.append(0.89175088)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)
        
        # point augmentations
        # DEBUG
        # if self.split == 'train' and self.aug:
        #    clouds, action_tcps = self._augmentation(clouds, action_tcps)
        # print("augment的tcp",action_tcps)
        
        # visualization
        # # if self.vis:
        # # print("强制可视化")
        # points = clouds[-1][..., :3]
        # # print("point range", points.min(axis=0), points.max(axis=0))

        # # 点云
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pc_colors = np.clip(colors * IMG_STD + IMG_MEAN, 0.0, 1.0)
        # pcd.colors = o3d.utility.Vector3dVector(pc_colors)

        # # ====== 将 TCP 轨迹变成彩色点并追加到点云（无需 mesh）======
        # # action_tcps: [..., 7] (xyz + quaternion)，先转 4x4 矩阵
        # # print('before',action_tcps)
        # action_tcps_vis = xyz_rot_transform(action_tcps, from_rep="quaternion", to_rep="matrix")
        # # print('after',action_tcps_vis)

        # axis_len   = 0.05   # 每个坐标轴长度（可调）
        # num_samp   = 20     # 每条轴离散的点数（可调）
        # tcp_pts    = []
        # tcp_colors = []

        # for T_pose  in action_tcps_vis:
        #     R = T_pose [:3, :3]
        #     t = T_pose [:3,  3]
        #     # 三条轴在世界系的方向
        #     x_dir, y_dir, z_dir = R[:, 0], R[:, 1], R[:, 2]

        #     # 在三条轴上均匀采样点（不含原点，避免太亮）
        #     for dir_vec, col in [(x_dir, (1, 0, 0)), (y_dir, (0, 1, 0)), (z_dir, (0, 0, 1))]:
        #         for s in np.linspace(0.0, 1.0, num_samp, endpoint=True)[1:]:
        #             p = t + s * axis_len * dir_vec
        #             tcp_pts.append(p)
        #             tcp_colors.append(col)

        # # 追加到点云（点云里就包含 TCP 轨迹了）
        # #"""
        # if len(tcp_pts) > 0:
        #     tcp_pts    = np.asarray(tcp_pts, dtype=np.float64)
        #     tcp_colors = np.asarray(tcp_colors, dtype=np.float64)
        #     all_pts    = np.vstack([np.asarray(pcd.points), tcp_pts])
        #     all_cols   = np.vstack([np.asarray(pcd.colors), tcp_colors])
        #     pcd.points = o3d.utility.Vector3dVector(all_pts)
        #     pcd.colors = o3d.utility.Vector3dVector(all_cols)
        # #"""
        # # ====== 仍可交互可视化：点云 + 线框盒 +（可选）坐标系网格 ======
        # # 盒子改线框，避免遮挡
        # bbox3d_1 = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
        # lines1 = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox3d_1)
        # lines1.paint_uniform_color([1, 0, 0])

        # bbox3d_2 = o3d.geometry.AxisAlignedBoundingBox(TRANS_MIN, TRANS_MAX)
        # lines2 = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox3d_2)
        # lines2.paint_uniform_color([0, 1, 0])

        # # # （可选）窗口里依然画真正的坐标系网格，观感更好，但不会写入文件
        # traj = []
        # for T_pose in action_tcps_vis:
        #     #print("test------------------", T_pose)
        #     # T_pose[2,-1] =0.1 #-0.3
        #     # T_pose[1,-1] =0 #-0.3
        #     traj.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(T_pose))

        # o3d.visualization.draw_geometries([pcd.voxel_down_sample(self.voxel_size), lines1, lines2, *traj])

        # # ====== 保存：单个 PLY 文件，已经包含 TCP 轨迹的彩色点 ======
        # o3d.io.write_point_cloud("scene_with_tcp.ply", pcd)
        # print("已保存：scene_with_tcp.ply（含点云 + 彩色 TCP 轨迹）")




        # #o3d.visualization.draw_geometries([pcd.voxel_down_sample(self.voxel_size), bbox3d_1, bbox3d_2])

        
        # input('..')

        # rotation transformation (to 6d)
        

        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)
        # print("转换后的tcp",actions)
        # input()
        # normalization
        actions_normalized = self._normalize_tcp(actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095
