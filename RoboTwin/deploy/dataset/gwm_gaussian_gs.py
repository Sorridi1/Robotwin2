import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GWMGaussianVoxelDataset(Dataset):
    """
    Dataset for precomputed Gaussian-to-voxel sparse observations.

    Each item returns variable-length sparse voxel features:
        input_coords:      [M, 3] centered voxel xyz, int32, without batch column
        input_feats:       [M, F]
        input_xyz:         [M, 3] robot_base xyz for each sparse voxel/token
        action_normalized: [H, 10]

    Notes:
        input_xyz is required for action-grounded Gaussian affordance supervision.
        It should be produced by tools/precompute_gs_voxels.py after the affordance update.
    """

    def __init__(self, path, require_input_xyz=False):
        self.path = os.path.expanduser(path)
        self.require_input_xyz = bool(require_input_xyz)

        files = sorted(glob.glob(os.path.join(self.path, "*.pt")))
        self.payloads = []
        self.index = []

        for fp in files:
            item = torch.load(fp, map_location="cpu")
            windows = item.get("windows", [])
            if not windows:
                continue

            self._validate_payload_contract(item, fp)

            payload_id = len(self.payloads)
            self.payloads.append(item)

            for win_id in range(len(windows)):
                self.index.append((payload_id, win_id))

        if not self.index:
            raise ValueError(f"No precomputed windows found in {self.path}")

        first = self.payloads[0]["windows"][0]
        self.feature_dim = int(
            self.payloads[0].get("feature_dim", first["feats"].shape[-1])
        )
        self.has_input_xyz = "input_xyz" in first

        if self.require_input_xyz and not self.has_input_xyz:
            raise ValueError(
                f"{self.path} does not contain input_xyz. "
                f"Please rerun tools/precompute_gs_voxels.py with the affordance update."
            )

    def _validate_payload_contract(self, payload, fp):
        """Reject mixed precompute contracts before they reach training."""
        if not self.payloads:
            return

        reference = self.payloads[0]
        fields = (
            "feature_dim",
            "output_frame",
            "action_frame",
            "point_xyz_frame",
            "tcp_source_layout",
            "action_rotation_layout",
            "action_rotation_source",
        )
        for field in fields:
            reference_value = reference.get(field)
            payload_value = payload.get(field)
            if reference_value != payload_value:
                raise ValueError(
                    f"Mixed precompute payloads in {self.path}: {field} differs in {fp}; "
                    f"expected {reference_value!r}, got {payload_value!r}. "
                    "Re-export the directory from a clean, consistent pipeline."
                )

        for field in ("workspace_min", "workspace_max"):
            reference_value = reference.get(field)
            payload_value = payload.get(field)
            if (reference_value is None) != (payload_value is None):
                raise ValueError(
                    f"Mixed precompute payloads in {self.path}: {field} is missing in only "
                    f"one payload ({fp})."
                )
            if reference_value is not None and not np.allclose(
                np.asarray(reference_value), np.asarray(payload_value), rtol=0.0, atol=1e-7
            ):
                raise ValueError(
                    f"Mixed precompute payloads in {self.path}: {field} differs in {fp}; "
                    f"expected {np.asarray(reference_value)}, got {np.asarray(payload_value)}."
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        payload_id, win_id = self.index[idx]
        payload = self.payloads[payload_id]
        w = payload["windows"][win_id]

        item = {
            "input_coords": w["coords"].int(),
            "input_feats": w["feats"].float(),
            "action_normalized": w["action_normalized"].float(),
            "source": w.get("source", payload.get("source", "")),
            "t": int(w.get("t", -1)),
        }
        meta = w.get("meta", {}) if isinstance(w.get("meta", {}), dict) else {}
        if "demo_id" in meta:
            item["demo_id"] = meta["demo_id"]
        if "episode_id" in meta:
            item["episode_id"] = meta["episode_id"]
        if "traj_id" in meta:
            item["traj_id"] = meta["traj_id"]
        if "obs_frame_id" in meta:
            item["window_start"] = int(meta["obs_frame_id"])
        # Frame IDs are capture timestamps in the real-world export. Repair
        # phase must instead be the demo-local window index.
        item["phase_t"] = int(item["t"])
        action_frame_ids = meta.get("action_frame_ids", None)
        if action_frame_ids:
            item["action_frame_ids"] = [int(x) for x in action_frame_ids]
            item["window_end"] = int(action_frame_ids[-1])
        item["action_space"] = "normalized"
        item["action_is_normalized"] = True
        frame = payload.get("action_frame", payload.get("output_frame", None))
        if frame is not None:
            item["coordinate_frame"] = str(frame)

        if "input_xyz" in w:
            item["input_xyz"] = w["input_xyz"].float()
        elif self.require_input_xyz:
            raise KeyError(
                f"Window missing input_xyz in {item['source']} t={item['t']}. "
                f"Please rerun voxel precompute."
            )

        if "action_raw" in w:
            item["action_raw"] = w["action_raw"].float()

        return item


def collate_sparse_voxel(batch):
    coords_list, feats_list, actions = [], [], []
    sources, ts = [], []
    metadata_keys = [
        "demo_id", "episode_id", "traj_id",
        "window_start", "window_end", "phase_t", "phase_progress",
        "action_frame_ids",
        "action_space", "action_is_normalized", "coordinate_frame",
    ]
    metadata_values = {key: [] for key in metadata_keys}

    has_input_xyz = "input_xyz" in batch[0]
    has_action_raw = "action_raw" in batch[0]

    xyz_list = []
    xyz_batch_list = []
    action_raws = []

    for b, item in enumerate(batch):
        coords = item["input_coords"].int()
        feats = item["input_feats"].float()

        batch_col = torch.full((coords.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([batch_col, coords], dim=1))
        feats_list.append(feats)

        actions.append(item["action_normalized"].float())
        sources.append(item.get("source", ""))
        ts.append(int(item.get("t", -1)))
        for key in metadata_keys:
            metadata_values[key].append(item.get(key, None))

        if has_input_xyz:
            xyz = item["input_xyz"].float()
            xyz_list.append(xyz)
            xyz_batch_list.append(
                torch.full((xyz.shape[0],), b, dtype=torch.long)
            )

        if has_action_raw:
            action_raws.append(item["action_raw"].float())

    out = {
        "input_coords": torch.cat(coords_list, dim=0).int(),
        "input_feats": torch.cat(feats_list, dim=0).float(),
        "action_normalized": torch.stack(actions, dim=0).float(),
        "source": sources,
        "t": ts,
    }

    if has_input_xyz:
        out["input_xyz"] = torch.cat(xyz_list, dim=0).float()
        out["input_batch"] = torch.cat(xyz_batch_list, dim=0).long()

    if has_action_raw:
        out["action_raw"] = torch.stack(action_raws, dim=0).float()

    for key, values in metadata_values.items():
        if any(v is not None for v in values):
            out[key] = values

    return out
