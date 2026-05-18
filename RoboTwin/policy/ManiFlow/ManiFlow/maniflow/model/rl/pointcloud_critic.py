from typing import Dict, Optional
import copy

import torch
import torch.nn as nn

from maniflow.common.pytorch_util import dict_apply
from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder


class PointcloudCritic(nn.Module):
    """Independent point-cloud value network for RoboTwin PPO fine-tuning."""

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int,
        encoder_output_dim: int = 128,
        use_pc_color: bool = True,
        pointnet_type: str = "pointnet",
        downsample_points: bool = True,
        pointcloud_encoder_cfg: Optional[dict] = None,
        hidden_dims=(256, 256),
        activation: str = "relu",
        pool: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.use_pc_color = use_pc_color
        self.pool = pool
        self.normalizer: Optional[LinearNormalizer] = None

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])
        encoder_cfg = copy.deepcopy(pointcloud_encoder_cfg)
        if encoder_cfg is None:
            encoder_cfg = {
                "in_channels": 6 if use_pc_color else 3,
                "out_channels": encoder_output_dim,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
                "num_points": obs_dict["point_cloud"][0],
                "pointwise": True,
            }

        self.obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            downsample_points=downsample_points,
        )
        feat_dim = self.obs_encoder.output_shape()

        if activation == "mish":
            act_fn = nn.Mish
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        layers = []
        last_dim = feat_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, dim), act_fn()])
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1))
        self.value_head = nn.Sequential(*layers)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = copy.deepcopy(normalizer)
        self.normalizer.requires_grad_(False)

    def _normalize_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.normalizer is None:
            nobs = dict(obs_dict)
        else:
            nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        return nobs

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self._normalize_obs(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        to = min(self.n_obs_steps, value.shape[1])

        enc_in = dict_apply(
            nobs,
            lambda x: x[:, :to].reshape(-1, *x.shape[2:]).to(
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            ),
        )
        feat = self.obs_encoder(enc_in)
        if feat.ndim == 3:
            feat = feat.reshape(batch_size, -1, feat.shape[-1])
        else:
            feat = feat.reshape(batch_size, to, feat.shape[-1])

        if self.pool == "max":
            pooled = feat.max(dim=1).values
        else:
            pooled = feat.mean(dim=1)
        return self.value_head(pooled).squeeze(-1)
