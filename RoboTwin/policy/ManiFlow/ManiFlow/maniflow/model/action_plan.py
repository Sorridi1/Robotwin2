from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedCoarseActionOperator(nn.Module):
    """Parameter-free temporal coarse target C(A) for normalized actions."""

    def __init__(
        self,
        mode: str = "lowpass",
        kernel_size: int = 3,
        preserve_dims: Iterable[int] = (),
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("lowpass", "identity"):
            raise ValueError(f"Unsupported plan target mode={mode}")
        if int(kernel_size) < 1 or int(kernel_size) % 2 != 1:
            raise ValueError(
                f"plan_lowpass_kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.mode = mode
        self.kernel_size = int(kernel_size)
        self.preserve_dims = tuple(int(dim) for dim in preserve_dims)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,H,Da], got {tuple(actions.shape)}")
        action_dim = actions.shape[-1]
        if any(dim < 0 or dim >= action_dim for dim in self.preserve_dims):
            raise ValueError(
                f"preserve_dims={self.preserve_dims} is invalid for action_dim={action_dim}"
            )
        if self.mode == "identity" or self.kernel_size == 1:
            return actions.clone()

        # AvgPool1d operates on [B, C, H], so action dimensions are independent channels.
        action_channels = actions.transpose(1, 2)
        padding = self.kernel_size // 2
        padded = F.pad(action_channels, (padding, padding), mode="replicate")
        coarse = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        coarse = coarse.transpose(1, 2)
        if self.preserve_dims:
            preserve_mask = torch.zeros(
                action_dim, device=actions.device, dtype=torch.bool
            )
            preserve_mask[list(self.preserve_dims)] = True
            coarse = torch.where(preserve_mask.view(1, 1, -1), actions, coarse)
        if coarse.shape != actions.shape:
            raise RuntimeError(
                f"Coarse operator changed shape from {tuple(actions.shape)} to {tuple(coarse.shape)}"
            )
        return coarse


class FutureActionPlanPredictor(nn.Module):
    """Small pooled-visual MLP predicting [B,H,Da] coarse future actions."""

    def __init__(
        self,
        visual_dim: int,
        horizon: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        dims = [int(visual_dim), *[int(dim) for dim in hidden_dims]]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.Mish()])
        layers.append(nn.Linear(dims[-1], self.horizon * self.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, vis_cond: torch.Tensor) -> torch.Tensor:
        if vis_cond.ndim != 3:
            raise ValueError(f"Expected vis_cond [B,L,Do], got {tuple(vis_cond.shape)}")
        pooled_visual = vis_cond.mean(dim=1)
        plan = self.mlp(pooled_visual)
        return plan.reshape(vis_cond.shape[0], self.horizon, self.action_dim)
