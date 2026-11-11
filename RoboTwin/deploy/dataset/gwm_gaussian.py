import glob
import os

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.constants import MAX_GRIPPER_WIDTH, TRANS_MAX, TRANS_MIN, WORKSPACE_MAX, WORKSPACE_MIN
from utils.transformation import xyz_rot_transform

TO_TENSOR_KEYS = ["action", "action_normalized", "num_voxels"]


class GWMGaussianDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        num_action=20,
        voxel_size=0.005,
        train_ratio=0.9,
        use_xyz_feat=True,
    ):
        self.path = path
        self.split = split
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.use_xyz_feat = use_xyz_feat

        files = sorted(glob.glob(os.path.join(path, "*.pt")))
        n_train = max(1, int(len(files) * train_ratio))
        self.files = files[:n_train] if split == "train" else files[n_train:]

        self.samples = []
        for fp in self.files:
            item = torch.load(fp, map_location="cpu")

            xyz = self._to_numpy(item["gauss_xyz"])
            feat = self._to_numpy(item["gauss_feat"])
            action = self._to_numpy(item["action"])

            if xyz.ndim == 4 and xyz.shape[0] == 1:
                xyz = xyz[0]
            if feat.ndim == 4 and feat.shape[0] == 1:
                feat = feat[0]
            if action.ndim == 3 and action.shape[0] == 1:
                action = action[0]

            action = action.astype(np.float32)
            action_is_normalized = self._action_is_normalized(item)

            if action.shape[-1] == 8:
                if action_is_normalized:
                    raise ValueError(f"{fp} stores normalized action but still uses 8D quaternion action.")
                tcp_quat = action[:, :7].astype(np.float32)
                tcp_rot6d = xyz_rot_transform(tcp_quat, from_rep="quaternion", to_rep="rotation_6d")
                grip = action[:, 7:8].astype(np.float32)
                action = np.concatenate([tcp_rot6d, grip], axis=-1)
            elif action.shape[-1] != 10:
                raise ValueError(f"Unsupported action dim {action.shape[-1]} in {fp}")

            T = min(len(xyz), len(feat), len(action))
            xyz = xyz[:T]
            feat = feat[:T]
            action = action[:T]

            if action_is_normalized:
                action_normalized = action.astype(np.float32)
            else:
                action_normalized = self._normalize_raw_action(action)

            for t in range(T - num_action + 1):
                self.samples.append(
                    {
                        "xyz": xyz[t],
                        "feat": feat[t],
                        "action": action[t : t + num_action],
                        "action_normalized": action_normalized[t : t + num_action],
                        "source": fp,
                        "action_is_normalized": action_is_normalized,
                    }
                )

        self.input_dim = 32 + (3 if use_xyz_feat else 0)

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _action_is_normalized(item):
        if "action_is_normalized" in item:
            return bool(item["action_is_normalized"])
        action_frame = item.get("action_frame", "")
        if isinstance(action_frame, str) and "normalized" in action_frame.lower():
            return True
        return False

    @staticmethod
    def _normalize_raw_action(action):
        action = action.copy().astype(np.float32)
        action[:, :3] = (action[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN + 1e-8) * 2.0 - 1.0
        action[:, -1] = action[:, -1] / (MAX_GRIPPER_WIDTH + 1e-8) * 2.0 - 1.0
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _crop(xyz, feat):
        mask = (
            (xyz[:, 0] >= WORKSPACE_MIN[0])
            & (xyz[:, 0] <= WORKSPACE_MAX[0])
            & (xyz[:, 1] >= WORKSPACE_MIN[1])
            & (xyz[:, 1] <= WORKSPACE_MAX[1])
            & (xyz[:, 2] >= WORKSPACE_MIN[2])
            & (xyz[:, 2] <= WORKSPACE_MAX[2])
        )
        return xyz[mask], feat[mask]

    def _voxel_pool(self, xyz, feat):
        xyz, feat = self._crop(xyz, feat)
        if len(xyz) == 0:
            return (
                np.zeros((1, 3), np.int32),
                np.zeros((1, feat.shape[-1]), np.float32),
                np.zeros((1, 3), np.float32),
            )

        coords = np.floor((xyz - WORKSPACE_MIN) / self.voxel_size).astype(np.int32)
        uniq, inv = np.unique(coords, axis=0, return_inverse=True)

        pooled_feat = np.zeros((len(uniq), feat.shape[-1]), np.float32)
        pooled_xyz = np.zeros((len(uniq), 3), np.float32)

        np.add.at(pooled_feat, inv, feat)
        np.add.at(pooled_xyz, inv, xyz)

        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float32)[:, None]
        pooled_feat /= np.maximum(counts, 1.0)
        pooled_xyz /= np.maximum(counts, 1.0)

        return uniq, pooled_feat, pooled_xyz

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        coords_int, pooled_feat, pooled_xyz = self._voxel_pool(sample["xyz"], sample["feat"])

        return {
            "input_coords_list": [coords_int],
            "input_feats_list": [pooled_feat.astype(np.float32)],
            "input_xyz_list": [pooled_xyz.astype(np.float32)],
            "action": torch.from_numpy(sample["action"]).float(),
            "action_normalized": torch.from_numpy(sample["action_normalized"]).float(),
            "num_voxels": torch.tensor(coords_int.shape[0], dtype=torch.int32),
            "source": sample["source"],
            "action_is_normalized": sample["action_is_normalized"],
        }


def collate_fn(batch):
    ret = {}
    for key in batch[0]:
        if key in TO_TENSOR_KEYS:
            ret[key] = torch.stack([b[key] for b in batch], 0)
        elif key in {"input_coords_list", "input_feats_list", "input_xyz_list"}:
            ret[key] = [b[key][0] for b in batch]
        else:
            ret[key] = [b[key] for b in batch]

    coords_batch, feats_batch = ME.utils.sparse_collate(ret["input_coords_list"], ret["input_feats_list"])
    _, xyz_batch = ME.utils.sparse_collate(ret["input_coords_list"], ret["input_xyz_list"])

    ret["input_coords_list"] = coords_batch
    ret["input_feats_list"] = feats_batch
    ret["input_xyz_list"] = xyz_batch
    return ret
