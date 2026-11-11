"""Inference-only loader: archived architecture, EMA and checkpoint normalizer."""
from collections import deque
import json
from pathlib import Path
import numpy as np

from .common import add_policy_path


class DucoManiFlow:
    def __init__(self, run_dir, checkpoint=None, device='cuda:0', seed=0):
        add_policy_path()
        import torch
        import dill
        import hydra
        from omegaconf import OmegaConf
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; --device cpu is supported but slow. No package changes are needed.')
        torch.manual_seed(seed)
        np.random.seed(seed)
        run_dir = Path(run_dir).resolve(strict=True)
        if checkpoint is None:
            paths = sorted((run_dir / 'checkpoints').glob('*.ckpt'))
            if len(paths) != 1:
                raise ValueError(f'Found {len(paths)} checkpoints; select one explicitly with --checkpoint')
            checkpoint = paths[0]
        self.checkpoint = str(Path(checkpoint).resolve(strict=True))
        # User-specified local training checkpoint; never load untrusted downloads.
        payload = torch.load(self.checkpoint, map_location='cpu', pickle_module=dill, mmap=True)
        self.cfg = payload['cfg']
        shape = self.cfg.task.shape_meta
        if list(shape.obs.point_cloud.shape) != [1024, 6] or list(shape.obs.agent_pos.shape) != [7] or list(shape.action.shape) != [7]:
            raise ValueError('Checkpoint is not the single-arm Duco XYZRGB/7D policy')
        if not self.cfg.policy.use_pc_color or self.cfg.policy.language_conditioned:
            raise ValueError('Expected the archived non-language XYZRGB checkpoint')
        self.model = hydra.utils.instantiate(self.cfg.policy)
        key = 'ema_model' if self.cfg.training.use_ema else 'model'
        if key not in payload['state_dicts']:
            raise ValueError(f'Missing checkpoint weights: {key}')
        incompatible = self.model.load_state_dict(payload['state_dicts'][key], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Checkpoint architecture mismatch: {incompatible}')
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.history = deque(maxlen=int(self.cfg.n_obs_steps))
        attrs_path = Path(self.cfg.task.dataset.zarr_path) / '.zattrs'
        if not attrs_path.is_file():
            raise FileNotFoundError(f'Training point-cloud contract missing: {attrs_path}')
        self.contract = json.loads(attrs_path.read_text())
        if self.contract['schema_version'] != 'rise_duco_maniflow_v1' or self.contract['point_frame'] != 'camera_optical_opencv_x_right_y_down_z_forward':
            raise ValueError('Unexpected training data schema/frame')
        self.report = dict(checkpoint=self.checkpoint, weight_set=key, strict_load=True,
                           n_obs_steps=int(self.cfg.n_obs_steps), horizon=int(self.cfg.horizon),
                           configured_n_action_steps=int(self.cfg.n_action_steps), device=str(self.device),
                           checkpoint_normalizer_loaded=True, training_contract_path=str(attrs_path),
                           action_layout=self.contract['action_layout'], data_contract=self.contract)
        del payload

    def configure_preprocessor(self, preprocessor):
        from maniflow.common.rise_duco_data import RGBDPointCloudPreprocessor
        c = self.contract
        kwargs = {k: c[k] for k in ['intrinsics', 'workspace_min', 'workspace_max', 'depth_scale', 'voxel_size', 'num_points', 'sampling']}
        if not np.allclose(preprocessor.intrinsics, c['intrinsics'], atol=1e-4):
            raise ValueError('Simulation camera intrinsics differ from checkpoint training intrinsics')
        return RGBDPointCloudPreprocessor(**kwargs, device=str(self.device))

    def update_obs(self, observation):
        cloud = np.asarray(observation['pointcloud'], dtype=np.float32)
        state = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
        if cloud.shape != (1024, 6) or state.shape != (7,) or not np.isfinite(cloud).all() or not np.isfinite(state).all():
            raise ValueError('Invalid live simulation observation')
        if np.any(cloud[:, 3:] < 0) or np.any(cloud[:, 3:] > 1):
            raise ValueError('Point colors must be RGB in [0,1]')
        item = dict(point_cloud=cloud.copy(), agent_pos=state.copy())
        if not self.history:
            for _ in range(self.history.maxlen):
                self.history.append(item)
        else:
            self.history.append(item)

    def predict(self):
        if not self.history:
            raise RuntimeError('Call update_obs before predict')
        torch = self.torch
        obs = {key: torch.as_tensor(np.stack([o[key] for o in self.history])[None], device=self.device)
               for key in ['point_cloud', 'agent_pos']}
        with torch.inference_mode():
            result = self.model.predict_action(obs)
        actions = result['action'][0].detach().cpu().numpy()
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0 or not np.isfinite(actions).all():
            raise RuntimeError('Model returned invalid action chunk')
        return actions
