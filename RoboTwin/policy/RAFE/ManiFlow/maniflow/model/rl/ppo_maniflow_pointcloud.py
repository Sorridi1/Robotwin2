from typing import Dict, Iterable, List, Optional, Tuple
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from maniflow.common.pytorch_util import dict_apply
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy


def _expand_flow_time(time: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    if sample.ndim != 3:
        raise ValueError(f"Expected sample tensor [B,H,Da], got {tuple(sample.shape)}")
    if not torch.is_tensor(time):
        time = torch.as_tensor(time, device=sample.device, dtype=sample.dtype)
    time = time.to(device=sample.device, dtype=sample.dtype)
    if time.ndim == 0:
        time = time.expand(sample.shape[0])
    if time.ndim != 1 or time.shape[0] != sample.shape[0]:
        raise ValueError(f"Expected time scalar or [B], got {tuple(time.shape)}")
    return time.view(sample.shape[0], 1, 1)


def compute_expert_interpolation_state(
    x0: torch.Tensor,
    action: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Raw normalized expert Flow Matching path; one X0 for every time step."""
    if x0.shape != action.shape or x0.ndim != 3:
        raise ValueError(
            f"Expected matching x0/action [B,H,Da], got {tuple(x0.shape)} and {tuple(action.shape)}"
        )
    t = _expand_flow_time(time, x0)
    return (1.0 - t) * x0 + t * action.to(device=x0.device, dtype=x0.dtype)


def diagonal_gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Elementwise Gaussian NLL with a fixed mean, evaluated in float32.

    Keeping this calculation outside half precision avoids overflow of small
    sigma squared. Only the sigma argument receives gradients.
    """
    if target.shape != mean.shape or target.shape != sigma.shape:
        raise ValueError("Gaussian target, mean and sigma must have matching shapes")
    if not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise FloatingPointError("Expert interpolation sigma must be finite and strictly positive")
    residual = target.detach().float() - mean.detach().float()
    std = sigma.float()
    nll = 0.5 * ((residual / std).square() + 2.0 * std.log() + math.log(2.0 * math.pi))
    if not torch.isfinite(nll).all():
        raise FloatingPointError("Expert interpolation Gaussian NLL is NaN or Inf")
    return nll


class TimeConditionedNoiseHead(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        action_horizon: int,
        action_dim: int,
        time_dim: int = 16,
        hidden_dims=(128, 128),
        min_std: float = 0.01,
        max_std: float = 0.05,
        activation: str = "tanh",
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.min_std = min_std
        self.max_std = max_std

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        if activation == "mish":
            act_fn = nn.Mish
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        layers = []
        last_dim = cond_dim + time_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, dim), act_fn()])
            last_dim = dim
        layers.append(nn.Linear(last_dim, action_horizon * action_dim))
        self.mlp_logvar = nn.Sequential(*layers)

    def forward(self, cond_emb: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(cond_emb.shape[0]).to(device=cond_emb.device, dtype=cond_emb.dtype)
        time_emb = self.time_embedding(time).to(dtype=cond_emb.dtype)
        noise_feature = torch.cat([time_emb, cond_emb], dim=-1)
        logvar = torch.tanh(self.mlp_logvar(noise_feature))
        logvar_min = torch.log(
            torch.tensor(self.min_std**2, device=cond_emb.device, dtype=cond_emb.dtype)
        )
        logvar_max = torch.log(
            torch.tensor(self.max_std**2, device=cond_emb.device, dtype=cond_emb.dtype)
        )
        logvar = logvar_min + (logvar_max - logvar_min) * (logvar + 1.0) / 2.0
        return torch.exp(0.5 * logvar)


class ActionTimeConditionedNoiseHead(TimeConditionedNoiseHead):
    """ReinFlow-style bounded-variance MLP additionally conditioned on X_t.

    Keep the legacy head's time embedding, activation and log-variance mapping.
    Only append the current normalized action chunk to the observation features;
    No temporal backbone or base schedule is used.
    """

    def __init__(
        self,
        cond_dim: int,
        action_horizon: int,
        action_dim: int,
        time_dim: int = 16,
        hidden_dims=(128, 128),
        min_std: float = 0.01,
        max_std: float = 0.05,
        activation: str = "tanh",
    ):
        super().__init__(
            cond_dim=cond_dim + action_horizon * action_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
            time_dim=time_dim,
            hidden_dims=hidden_dims,
            min_std=min_std,
            max_std=max_std,
            activation=activation,
        )
        self.input_cond_dim = int(cond_dim)

    def forward(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if cond_emb.ndim != 2 or cond_emb.shape[1] != self.input_cond_dim:
            raise ValueError(
                f"Expected cond_emb shape [B,{self.input_cond_dim}], got {tuple(cond_emb.shape)}"
            )
        expected_shape = (cond_emb.shape[0], self.action_horizon, self.action_dim)
        if tuple(x.shape) != expected_shape:
            raise ValueError(f"Expected x shape {expected_shape}, got {tuple(x.shape)}")
        action_feature = x.reshape(cond_emb.shape[0], -1).to(
            device=cond_emb.device, dtype=cond_emb.dtype,
        )
        return super().forward(torch.cat([cond_emb, action_feature], dim=-1), time)


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "mish":
        return nn.Mish()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation={name}")


def _zero_init(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPNoiseBackbone(nn.Module):
    def __init__(self, model_dim: int, hidden_dims=(128, 128), activation: str = "mish"):
        super().__init__()
        layers = []
        last_dim = int(model_dim)
        for dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, int(dim)), _make_activation(activation)])
            last_dim = int(dim)
        layers.append(nn.Linear(last_dim, int(model_dim)))
        layers.append(_make_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, activation: str = "mish"):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError(f"tcn_kernel_size must be odd for same padding, got {kernel_size}")
        self.conv = nn.Conv1d(dim, dim, int(kernel_size), padding=int(kernel_size) // 2)
        self.norm = nn.GroupNorm(1, dim)
        self.act = _make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class TCNNoiseBackbone(nn.Module):
    def __init__(
        self,
        model_dim: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        activation: str = "mish",
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TemporalConvBlock(model_dim, kernel_size=kernel_size, activation=activation)
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = z.transpose(1, 2)
        y = self.blocks(y)
        return y.transpose(1, 2)


class TransformerNoiseBackbone(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if int(model_dim) % int(num_heads) != 0:
            raise ValueError(
                f"noise_model_dim ({model_dim}) must be divisible by transformer_num_heads ({num_heads})"
            )
        transformer_activation = "relu" if str(activation).lower() == "relu" else "gelu"
        layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation=transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)


class ActionConditionedResidualNoiseHead(nn.Module):
    """Predicts an unbounded residual in log standard deviation per element."""

    def __init__(
        self,
        cond_dim: int,
        action_horizon: int,
        action_dim: int,
        time_dim: int = 16,
        model_dim: int = 128,
        action_embed_dim: int = 64,
        hidden_dims=(128, 128),
        activation: str = "mish",
        action_conditioned_noise: bool = True,
        noise_output_mode: str = "full",
        noise_backbone: str = "mlp",
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        transformer_num_layers: int = 1,
        transformer_num_heads: int = 4,
        transformer_ffn_dim: int = 256,
        transformer_dropout: float = 0.0,
        residual_scale: float = 1.0,
        zero_init_output: bool = True,
        use_horizon_pos_emb: bool = True,
        noise_conditioning_mode: str = "default",
    ):
        super().__init__()
        if noise_output_mode not in ("full", "chunk_shared", "dim_shared"):
            raise ValueError(f"Unsupported noise_output_mode={noise_output_mode}")
        if noise_backbone not in ("mlp", "tcn", "transformer"):
            raise ValueError(f"Unsupported noise_backbone={noise_backbone}")

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.time_dim = int(time_dim)
        self.model_dim = int(model_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.action_conditioned_noise = bool(action_conditioned_noise)
        self.noise_output_mode = noise_output_mode
        self.noise_backbone = noise_backbone
        self.residual_scale = float(residual_scale)
        self.use_horizon_pos_emb = bool(use_horizon_pos_emb)
        self.noise_conditioning_mode = str(noise_conditioning_mode)
        if self.noise_conditioning_mode != "default":
            raise ValueError("RAFE supports only noise_conditioning_mode=default")

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_proj = nn.Linear(time_dim, self.model_dim)
        self.cond_proj = nn.Linear(cond_dim, self.model_dim)

        if self.action_conditioned_noise:
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, self.action_embed_dim),
                _make_activation(activation),
                nn.Linear(self.action_embed_dim, self.model_dim),
            )
        else:
            self.action_proj = None

        if self.use_horizon_pos_emb:
            self.horizon_pos_emb = nn.Parameter(torch.zeros(1, self.action_horizon, self.model_dim))
        else:
            self.register_parameter("horizon_pos_emb", None)

        self.token_norm = nn.LayerNorm(self.model_dim)

        if noise_backbone == "mlp":
            self.backbone = MLPNoiseBackbone(self.model_dim, hidden_dims=hidden_dims, activation=activation)
        elif noise_backbone == "tcn":
            self.backbone = TCNNoiseBackbone(
                self.model_dim,
                kernel_size=tcn_kernel_size,
                num_layers=tcn_num_layers,
                activation=activation,
            )
        else:
            self.backbone = TransformerNoiseBackbone(
                self.model_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ffn_dim=transformer_ffn_dim,
                dropout=transformer_dropout,
                activation=activation,
            )

        output_dim = action_dim
        if noise_output_mode == "dim_shared":
            output_dim = 1
        self.out_proj = nn.Linear(self.model_dim, output_dim)
        if zero_init_output:
            _zero_init(self.out_proj)

    def _build_tokens(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, action_dim = x.shape
        if horizon != self.action_horizon or action_dim != self.action_dim:
            raise ValueError(
                f"Expected x shape [B,{self.action_horizon},{self.action_dim}], got {tuple(x.shape)}"
            )

        cond_emb = cond_emb.to(device=x.device, dtype=x.dtype)
        time = time.reshape(batch_size).to(device=x.device, dtype=x.dtype)
        time_emb = self.time_embedding(time).to(dtype=x.dtype)

        cond_token = self.cond_proj(cond_emb)[:, None, :].expand(batch_size, horizon, self.model_dim)
        time_token = self.time_proj(time_emb)[:, None, :].expand(batch_size, horizon, self.model_dim)
        z = cond_token + time_token
        if self.action_conditioned_noise:
            z = z + self.action_proj(x)
        if self.horizon_pos_emb is not None:
            z = z + self.horizon_pos_emb.to(device=x.device, dtype=x.dtype)
        return self.token_norm(z)

    def forward_raw(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, action_dim = x.shape
        z = self._build_tokens(cond_emb, time, x)
        y = self.backbone(z)
        if self.noise_output_mode == "chunk_shared":
            y = y.mean(dim=1, keepdim=True)

        raw_delta = self.out_proj(y)
        if self.noise_output_mode == "chunk_shared":
            raw_delta = raw_delta.expand(batch_size, horizon, action_dim)
        elif self.noise_output_mode == "dim_shared":
            raw_delta = raw_delta.expand(batch_size, horizon, action_dim)
        return raw_delta

    def predict_delta(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raw_delta = self.forward_raw(cond_emb, time, x)
        return self.residual_scale * raw_delta.float()

    def forward(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
        return_raw: bool = False,
    ) -> torch.Tensor:
        if return_raw:
            return self.forward_raw(cond_emb, time, x)
        return self.predict_delta(cond_emb, time, x)


class PPOManiFlowPointcloud(nn.Module):
    """PPO adapter around a normal ManiFlow point-cloud actor.

    The trainable actor remains ``base_actor``. ``actor_old`` is a frozen
    pretrained reference used only for optional BC anchoring.
    """

    def __init__(
        self,
        base_actor: ManiFlowTransformerPointcloudPolicy,
        inference_steps: Optional[int] = None,
        ft_denoising_steps: Optional[int] = None,
        min_sampling_denoising_std: float = 0.005,
        min_logprob_denoising_std: float = 0.01,
        max_logprob_denoising_std: float = 0.05,
        randn_clip_value: float = 2.0,
        denoised_clip_value: float = 1.5,
        final_action_clip_value: Optional[float] = 1.0,
        clip_ploss_coef: float = 0.2,
        clip_vloss_coef: Optional[float] = None,
        logprob_min: float = -20.0,
        logprob_max: float = 20.0,
        normalize_denoising_horizon: bool = True,
        normalize_act_space_dimension: bool = True,
        account_for_initial_stochasticity: bool = True,
        noise_hidden_dims=(128, 128),
        noise_activation: str = "tanh",
        residual_noise_activation: Optional[str] = None,
        noise_time_dim: int = 16,
        noise_head_type: str = "fixed",
        noise_backbone: str = "mlp",
        base_sigma_schedule: str = "constant",
        base_sigma: Optional[float] = None,
        base_sigma_min: Optional[float] = None,
        base_sigma_max: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        residual_scale: float = 1.0,
        action_conditioned_noise: bool = True,
        noise_output_mode: str = "full",
        noise_model_dim: int = 128,
        noise_action_embed_dim: int = 64,
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        transformer_num_layers: int = 1,
        transformer_num_heads: int = 4,
        transformer_ffn_dim: int = 256,
        transformer_dropout: float = 0.0,
        zero_init_noise_output: bool = True,
        use_horizon_pos_emb: bool = True,
        noise_conditioning_mode: str = "default",
        noise_parameterization: str = "log_residual",
        actor_old_device: str = "cpu",
        freeze_obs_encoder: bool = True,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.inference_steps = int(base_actor.num_inference_steps if inference_steps is None else inference_steps)
        self.ft_denoising_steps = int(self.inference_steps if ft_denoising_steps is None else ft_denoising_steps)
        if self.inference_steps < 1 or not 0 <= self.ft_denoising_steps <= self.inference_steps:
            raise ValueError("Require inference_steps >= 1 and 0 <= ft_denoising_steps <= inference_steps")
        self.learn_explore_noise_from = max(0, self.inference_steps - self.ft_denoising_steps)

        self.action_dim = base_actor.action_dim
        self.horizon = base_actor.horizon
        self.n_action_steps = base_actor.n_action_steps
        self.n_obs_steps = base_actor.n_obs_steps
        self.act_dim_total = self.horizon * self.action_dim

        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.max_logprob_denoising_std = max_logprob_denoising_std
        self.randn_clip_value = randn_clip_value
        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_vloss_coef = clip_vloss_coef
        self.logprob_min = logprob_min
        self.logprob_max = logprob_max
        self.normalize_denoising_horizon = normalize_denoising_horizon
        self.normalize_act_space_dimension = normalize_act_space_dimension
        self.account_for_initial_stochasticity = account_for_initial_stochasticity
        self.freeze_obs_encoder = freeze_obs_encoder
        self.noise_head_type = str(noise_head_type)
        self.noise_backbone = str(noise_backbone)
        self.base_sigma_schedule = str(base_sigma_schedule)
        self.base_sigma = float(base_sigma if base_sigma is not None else min_logprob_denoising_std)
        self.base_sigma_min = float(
            base_sigma_min if base_sigma_min is not None else min_logprob_denoising_std
        )
        self.base_sigma_max = float(
            base_sigma_max if base_sigma_max is not None else max_logprob_denoising_std
        )
        self.sigma_min = float(sigma_min if sigma_min is not None else min_logprob_denoising_std)
        self.sigma_max = float(sigma_max if sigma_max is not None else max_logprob_denoising_std)
        self.residual_scale = float(residual_scale)
        self.action_conditioned_noise = bool(action_conditioned_noise)
        self.noise_output_mode = str(noise_output_mode)
        self.noise_conditioning_mode = str(noise_conditioning_mode)
        self.noise_parameterization = str(noise_parameterization)
        if self.noise_parameterization != "log_residual":
            raise ValueError("RAFE requires noise_parameterization=log_residual; bounded-noise checkpoints cannot be reinterpreted")
        self.last_logprob_noise_stats = {}
        self.last_action_noise_stats = {}

        if self.noise_head_type not in ("fixed", "reinflow_xt", "residual_schedule"):
            raise ValueError(f"Unsupported noise_head_type={self.noise_head_type}")
        if self.base_sigma_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(f"Unsupported base_sigma_schedule={self.base_sigma_schedule}")
        scales = (self.sigma_min, self.sigma_max, self.base_sigma, self.base_sigma_min,
                  self.base_sigma_max, self.min_sampling_denoising_std)
        if any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("Noise standard deviations and base schedule values must be finite and strictly positive")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual_scale is a finite nonnegative linear gain; use 1 for learned noise or 0 for fixed sigma")
        if self.sigma_min > self.sigma_max:
            raise ValueError(
                f"Invalid sigma clamp bounds: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}"
            )
        if self.min_sampling_denoising_std > torch.finfo(torch.float32).max or self.sigma_max > torch.finfo(torch.float32).max:
            raise ValueError("Noise bounds must fit float32")
        if self.base_sigma_min > self.base_sigma_max:
            raise ValueError("base_sigma_min must not exceed base_sigma_max")
        if self.noise_conditioning_mode != "default":
            raise ValueError("RAFE supports only noise_conditioning_mode=default")

        if self.noise_head_type in ("fixed", "reinflow_xt"):
            noise_head_class = (
                ActionTimeConditionedNoiseHead
                if self.noise_head_type == "reinflow_xt"
                else TimeConditionedNoiseHead
            )
            self.noise_head = noise_head_class(
                cond_dim=base_actor.obs_feature_dim,
                action_horizon=self.horizon,
                action_dim=self.action_dim,
                time_dim=noise_time_dim,
                hidden_dims=noise_hidden_dims,
                min_std=min_logprob_denoising_std,
                max_std=max_logprob_denoising_std,
                activation=noise_activation,
            )
        else:
            self.noise_head = ActionConditionedResidualNoiseHead(
                cond_dim=base_actor.obs_feature_dim,
                action_horizon=self.horizon,
                action_dim=self.action_dim,
                time_dim=noise_time_dim,
                model_dim=noise_model_dim,
                action_embed_dim=noise_action_embed_dim,
                hidden_dims=noise_hidden_dims,
                activation=residual_noise_activation or "mish",
                action_conditioned_noise=action_conditioned_noise,
                noise_output_mode=noise_output_mode,
                noise_backbone=self.noise_backbone,
                tcn_kernel_size=tcn_kernel_size,
                tcn_num_layers=tcn_num_layers,
                transformer_num_layers=transformer_num_layers,
                transformer_num_heads=transformer_num_heads,
                transformer_ffn_dim=transformer_ffn_dim,
                transformer_dropout=transformer_dropout,
                residual_scale=residual_scale,
                zero_init_output=zero_init_noise_output,
                use_horizon_pos_emb=use_horizon_pos_emb,
                noise_conditioning_mode=self.noise_conditioning_mode,
            )

        actor_old = copy.deepcopy(base_actor).to(actor_old_device)
        actor_old.eval()
        for param in actor_old.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "actor_old", actor_old)

        if self.freeze_obs_encoder:
            for param in self.base_actor.obs_encoder.parameters():
                param.requires_grad_(False)

    def get_extra_state(self):
        """Persist noise semantics because tensor shapes do not identify them."""
        return {"version": 1, "noise_parameterization": self.noise_parameterization,
                "residual_scale": self.residual_scale}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError(
                "Incompatible RAFE noise checkpoint metadata: expected log_residual "
                "parameterization and the same residual_scale linear gain"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.actor_old.eval()
        if self.freeze_obs_encoder:
            self.base_actor.obs_encoder.eval()
        return self

    @property
    def device(self):
        return self.base_actor.device

    @property
    def dtype(self):
        return self.base_actor.dtype

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        for param in self.base_actor.parameters():
            if param.requires_grad:
                yield param
        yield from self.noise_head.parameters()

    def _to_actor_device(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return dict_apply(
            obs_dict,
            lambda x: x.to(device=self.device, dtype=self.dtype) if torch.is_tensor(x) else x,
        )

    def _encode_obs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actor: Optional[ManiFlowTransformerPointcloudPolicy] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actor = actor or self.base_actor
        device = actor.device
        dtype = actor.dtype
        obs_dict = dict_apply(
            obs_dict,
            lambda x: x.to(device=device, dtype=dtype) if torch.is_tensor(x) else x,
        )
        nobs = actor.normalizer.normalize(obs_dict)
        if not actor.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        to = actor.n_obs_steps
        obs_in = dict_apply(
            nobs,
            lambda x: x[:, :to].reshape(-1, *x.shape[2:]).to(device=device, dtype=dtype),
        )
        nobs_features = actor.obs_encoder(obs_in)
        vis_cond = nobs_features.reshape(batch_size, -1, actor.obs_feature_dim)
        cond_emb = vis_cond.mean(dim=1)
        return vis_cond, cond_emb

    def _target_t(self, t: torch.Tensor, dt: float, actor: Optional[ManiFlowTransformerPointcloudPolicy] = None):
        actor = actor or self.base_actor
        if actor.sample_target_t_mode == "absolute":
            return t + dt
        return torch.ones_like(t) * dt

    def _velocity(
        self,
        actor: ManiFlowTransformerPointcloudPolicy,
        x: torch.Tensor,
        t: torch.Tensor,
        vis_cond: torch.Tensor,
    ) -> torch.Tensor:
        dt = 1.0 / self.inference_steps
        return actor.model(
            sample=x,
            timestep=t,
            target_t=self._target_t(t, dt, actor),
            vis_cond=vis_cond,
        )

    def compute_base_sigma(
        self,
        time: torch.Tensor,
        x: torch.Tensor,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Return denoising-step base sigma with shape [B, H, action_dim].

        Current ManiFlow rollout starts from Gaussian action noise at step=0 and
        advances step=0..K-1 toward the final action. Therefore progress=0 is
        the early/noisy stage and progress=1 is the late/final-action stage.
        """
        if self.base_sigma_schedule == "constant":
            sigma = self.base_sigma
        else:
            if self.inference_steps <= 1:
                progress = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype)
            elif step is not None:
                progress_value = max(0.0, min(1.0, float(step) / float(self.inference_steps - 1)))
                progress = torch.full((x.shape[0], 1, 1), progress_value, device=x.device, dtype=x.dtype)
            else:
                max_time = float(self.inference_steps - 1) / float(self.inference_steps)
                progress = time.reshape(x.shape[0], 1, 1).to(device=x.device, dtype=x.dtype) / max_time
                progress = progress.clamp(0.0, 1.0)
            if self.base_sigma_schedule == "linear":
                sigma = self.base_sigma_max * (1.0 - progress) + self.base_sigma_min * progress
            else:
                sigma = self.base_sigma_min + 0.5 * (self.base_sigma_max - self.base_sigma_min) * (
                    1.0 + torch.cos(torch.tensor(math.pi, device=x.device, dtype=x.dtype) * progress)
                )
            return sigma.expand_as(x)
        return torch.full_like(x, float(sigma))

    def _base_sigma_tensor(self, step: int, x: torch.Tensor) -> torch.Tensor:
        time = torch.full((x.shape[0],), step / float(self.inference_steps), device=x.device, dtype=x.dtype)
        return self.compute_base_sigma(time, x, step=step)

    def _noise_stats(
        self,
        sigma: torch.Tensor,
        base_sigma: torch.Tensor,
        delta: torch.Tensor,
        clamp_ratio_min: Optional[torch.Tensor] = None,
        clamp_ratio_max: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if clamp_ratio_min is None:
            clamp_ratio_min = torch.zeros((), device=sigma.device, dtype=sigma.dtype)
        if clamp_ratio_max is None:
            clamp_ratio_max = torch.zeros((), device=sigma.device, dtype=sigma.dtype)
        return {
            "sigma_mean": sigma.mean(),
            "sigma_min_observed": sigma.amin(),
            "sigma_max_observed": sigma.amax(),
            "base_sigma_mean": base_sigma.mean(),
            "delta_mean": delta.mean(),
            "delta_abs_mean": delta.abs().mean(),
            "delta_min": delta.amin(),
            "delta_max": delta.amax(),
            "clamp_ratio_min": clamp_ratio_min,
            "clamp_ratio_max": clamp_ratio_max,
        }

    def _aggregate_noise_stats(self, stats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(stats) == 0:
            zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return {
                "sigma_mean": zero,
                "sigma_min_observed": zero,
                "sigma_max_observed": zero,
                "base_sigma_mean": zero,
                "delta_mean": zero,
                "delta_abs_mean": zero,
                "delta_min": zero,
                "delta_max": zero,
                "clamp_ratio_min": zero,
                "clamp_ratio_max": zero,
            }

        out = {}
        for key in stats[0]:
            values = torch.stack([item[key] for item in stats])
            if key in ("sigma_min_observed", "delta_min"):
                out[key] = values.min()
            elif key in ("sigma_max_observed", "delta_max"):
                out[key] = values.max()
            else:
                out[key] = values.mean()
        return out

    def compute_residual_sigma(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
        step: Optional[int] = None,
        learn_exploration_noise: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Schedule, log-space addition, clamp and exp stay float32 even under
        # autocast. Clamp before exp so a large raw residual cannot overflow it.
        base_sigma = self.compute_base_sigma(time, x.float(), step=step)
        delta = self.noise_head.predict_delta(cond_emb, time, x).float()
        if not torch.isfinite(base_sigma).all() or not (base_sigma > 0).all():
            raise FloatingPointError("RAFE base sigma must be finite and strictly positive")
        if not torch.isfinite(delta).all():
            raise FloatingPointError("RAFE raw log-sigma residual is NaN or Inf")
        log_sigma = base_sigma.log() + delta
        lower = max(self.sigma_min, self.min_sampling_denoising_std)
        upper = max(self.sigma_max, self.min_sampling_denoising_std)
        log_lower, log_upper = math.log(lower), math.log(upper)
        clamp_ratio_min = (log_sigma <= log_lower).float().mean()
        clamp_ratio_max = (log_sigma >= log_upper).float().mean()
        sigma = log_sigma.clamp(min=log_lower, max=log_upper).exp()
        if not torch.isfinite(sigma).all() or not (sigma > 0).all():
            raise FloatingPointError("RAFE log-space sigma is not representable as a finite positive float32 value")

        if not learn_exploration_noise:
            sigma = sigma.detach()
            base_sigma = base_sigma.detach()
            delta = delta.detach()
            clamp_ratio_min = clamp_ratio_min.detach()
            clamp_ratio_max = clamp_ratio_max.detach()
        return sigma, {
            "sigma": sigma,
            "base_sigma": base_sigma,
            "delta": delta,
            "clamp_ratio_min": clamp_ratio_min,
            "clamp_ratio_max": clamp_ratio_max,
        }

    def _noise_std(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        x: torch.Tensor,
        step: int,
        learn_exploration_noise: bool,
        return_stats: bool = False,
    ):
        if step < self.learn_explore_noise_from:
            base_sigma = torch.full_like(x, self.min_logprob_denoising_std)
            delta = torch.zeros_like(x)
            std = base_sigma.clamp(min=self.min_sampling_denoising_std)
            clamp_ratio_min = torch.zeros((), device=x.device, dtype=x.dtype)
            clamp_ratio_max = torch.zeros((), device=x.device, dtype=x.dtype)
        elif self.noise_head_type in ("fixed", "reinflow_xt"):
            if self.noise_head_type == "reinflow_xt":
                std = self.noise_head(cond_emb, time, x)
            else:
                std = self.noise_head(cond_emb, time)
            std = std.reshape(cond_emb.shape[0], self.horizon, self.action_dim)
            base_sigma = std
            delta = torch.zeros_like(std)
            clamp_ratio_min = torch.zeros((), device=x.device, dtype=x.dtype)
            clamp_ratio_max = torch.zeros((), device=x.device, dtype=x.dtype)
            std = std.clamp(min=self.min_sampling_denoising_std)
        else:
            std, sigma_info = self.compute_residual_sigma(
                cond_emb,
                time,
                x,
                step=step,
                learn_exploration_noise=learn_exploration_noise,
            )
            base_sigma = sigma_info["base_sigma"]
            delta = sigma_info["delta"]
            clamp_ratio_min = sigma_info["clamp_ratio_min"]
            clamp_ratio_max = sigma_info["clamp_ratio_max"]

        if not learn_exploration_noise:
            std = std.detach()
            base_sigma = base_sigma.detach()
            delta = delta.detach()
            clamp_ratio_min = clamp_ratio_min.detach()
            clamp_ratio_max = clamp_ratio_max.detach()
        if return_stats:
            return std, self._noise_stats(
                std,
                base_sigma,
                delta,
                clamp_ratio_min=clamp_ratio_min,
                clamp_ratio_max=clamp_ratio_max,
            )
        return std

    def gaussian_transition(
        self,
        x: torch.Tensor,
        step: int,
        vis_cond: torch.Tensor,
        cond_emb: torch.Tensor,
        learn_exploration_noise: bool = False,
        detach_mean: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """The single Gaussian transition used by rollout, PPO and expert NLL.

        The mean uses the current online actor, its target-time convention and
        denoised clipping. The noise branch uses only information available to
        rollout. ``detach_mean`` freezes velocity, while disabling noise learning
        detaches sigma without severing the PPO mean's gradient.
        """
        if not 0 <= int(step) < self.inference_steps:
            raise ValueError(f"Flow step {step} outside [0,{self.inference_steps})")
        dt = 1.0 / self.inference_steps
        t = torch.full((x.shape[0],), step * dt, device=x.device, dtype=x.dtype)

        def transition_mean():
            velocity = self._velocity(self.base_actor, x, t, vis_cond)
            mean = x + velocity * dt
            if self.denoised_clip_value is not None:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            return mean

        if detach_mean:
            with torch.no_grad():
                mean = transition_mean()
        else:
            mean = transition_mean()

        std, stats = self._noise_std(
            cond_emb, t, x, int(step), learn_exploration_noise=learn_exploration_noise,
            return_stats=True,
        )
        return mean, std, stats

    def compute_expert_interpolation_noise_loss(
        self,
        batch: Dict[str, torch.Tensor],
        sample_all_steps: bool = True,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fit only TCN sigma to expert interpolation transitions via Gaussian NLL.

        ``batch['action']`` is the entire raw expert chunk, normalized using the
        pretrained actor normalizer. Expert next states appear only as labels;
        observation, time and the current interpolated state condition sigma.
        """
        if (self.noise_head_type != "residual_schedule" or self.noise_backbone != "tcn"
                or self.noise_output_mode != "full" or not self.action_conditioned_noise):
            raise ValueError("Expert interpolation requires an action-conditioned TCN residual_schedule noise head with full output")
        steps = list(range(self.learn_explore_noise_from, self.inference_steps))
        if not steps:
            raise ValueError("Expert interpolation has no learnable Flow steps; ft_denoising_steps must be positive")
        if not sample_all_steps:
            choice = int(torch.randint(len(steps), (1,), generator=generator, device=self.device).item())
            steps = [steps[choice]]
        raw_action = batch["action"].to(device=self.device, dtype=self.dtype)
        expected_shape = (raw_action.shape[0], self.horizon, self.action_dim)
        if tuple(raw_action.shape) != expected_shape:
            raise ValueError(f"Expert action must be the full raw chunk {expected_shape}, got {tuple(raw_action.shape)}")
        if any(value.shape[0] != raw_action.shape[0] or value.shape[1] < self.n_obs_steps
               for value in batch["obs"].values()):
            raise ValueError("Expert observations must match action batch size and contain n_obs_steps")

        # Eval prevents observation/actor buffers or dropout from changing during
        # the noise-only update. Restore every module's original training state.
        module_modes = [(module, module.training) for module in self.base_actor.modules()]
        self.base_actor.eval()
        try:
            with torch.no_grad():
                action = self.base_actor.normalizer["action"].normalize(raw_action).detach()
                vis_cond, cond_emb = self._encode_obs(batch["obs"])
                vis_cond, cond_emb = vis_cond.detach(), cond_emb.detach()
                x0 = (torch.randn(action.shape, device=action.device, dtype=action.dtype, generator=generator)
                      if initial_noise is None else initial_noise.to(device=action.device, dtype=action.dtype).detach())
                if x0.shape != action.shape:
                    raise ValueError("Expert initial noise must match the complete action chunk")
                if not torch.isfinite(action).all() or not torch.isfinite(x0).all():
                    raise FloatingPointError("Expert normalized action or initial noise is NaN or Inf")

            losses, sigmas, residuals, saturation, below, above = [], [], [], [], [], []
            metrics = {}
            dt = 1.0 / self.inference_steps
            for step in steps:
                t = torch.full((action.shape[0],), step * dt, device=action.device, dtype=action.dtype)
                xt = compute_expert_interpolation_state(x0, action, t)
                target = compute_expert_interpolation_state(x0, action, t + dt)
                mean, sigma, stats = self.gaussian_transition(
                    xt, step, vis_cond, cond_emb, learn_exploration_noise=True,
                    detach_mean=True,
                )
                losses.append(diagonal_gaussian_nll(target, mean, sigma).mean())
                residual = target.float() - mean.detach().float()
                sigmas.append(sigma.detach().float())
                residuals.append(residual.detach())
                lower = max(self.sigma_min, self.min_sampling_denoising_std)
                upper = max(self.sigma_max, self.min_sampling_denoising_std)
                if self.residual_scale == 0:
                    # The matched fixed-linear baseline has no adjustable range.
                    fixed = self.compute_base_sigma(t, xt.float(), step=step).clamp(lower, upper)
                    possible_min = possible_max = fixed
                else:
                    possible_min = torch.full_like(sigma, lower)
                    possible_max = torch.full_like(sigma, upper)
                below.append((residual.abs() < possible_min).float().mean())
                above.append((residual.abs() > possible_max).float().mean())
                saturation.append((stats["clamp_ratio_min"] + stats["clamp_ratio_max"]).clamp(max=1).detach())
                metrics[f"noise_sigma_mean_by_step/{step}"] = sigma.detach().float().mean()
                metrics[f"noise_residual_mse_by_step/{step}"] = residual.detach().square().mean()
                metrics[f"noise_sigma_representable_min_by_step/{step}"] = possible_min.detach().mean()
                metrics[f"noise_sigma_representable_max_by_step/{step}"] = possible_max.detach().mean()
            loss = torch.stack(losses).mean()
            sigma_values, residual_values = torch.stack(sigmas), torch.stack(residuals)
            lower = max(self.sigma_min, self.min_sampling_denoising_std)
            upper = max(self.sigma_max, self.min_sampling_denoising_std)
            metrics.update({
                "noise_nll_loss": loss.detach(),
                "noise_sigma_mean": sigma_values.mean(),
                "noise_sigma_min": sigma_values.amin(),
                "noise_sigma_max": sigma_values.amax(),
                "noise_sigma_at_lower_bound_ratio": (sigma_values <= lower * (1.0 + 1e-5)).float().mean(),
                "noise_sigma_at_upper_bound_ratio": (sigma_values >= upper * (1.0 - 1e-5)).float().mean(),
                "noise_residual_mse": residual_values.square().mean(),
                "noise_residual_abs_mean": residual_values.abs().mean(),
                "noise_residual_below_representable_ratio": torch.stack(below).mean(),
                "noise_residual_above_representable_ratio": torch.stack(above).mean(),
                "noise_sigma_clamp_saturation_ratio": torch.stack(saturation).mean(),
                "noise_supervised_flow_steps": torch.tensor(float(len(steps)), device=self.device),
            })
            return loss, metrics
        finally:
            for module, training in module_modes:
                module.training = training

    def _finalize_step_noise_stats(
        self,
        stats: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        aggregated = self._aggregate_noise_stats(stats)
        for step, item in enumerate(stats):
            if "sigma_mean" in item:
                aggregated[f"sigma_step_{step}"] = item["sigma_mean"]
        return aggregated

    def get_last_noise_stats(self) -> Dict[str, torch.Tensor]:
        return self.last_logprob_noise_stats

    def _select_action_window(self, action_pred: torch.Tensor) -> torch.Tensor:
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]

    @torch.no_grad()
    def get_actions(
        self,
        obs_dict: Dict[str, torch.Tensor],
        eval_mode: bool = False,
        save_chains: bool = True,
        ret_logprob: bool = True,
    ):
        obs_dict = self._to_actor_device(obs_dict)
        vis_cond, cond_emb = self._encode_obs(obs_dict)
        batch_size = cond_emb.shape[0]

        xt = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if save_chains:
            chains = torch.empty(
                batch_size,
                self.inference_steps + 1,
                self.horizon,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            chains[:, 0] = xt
        logprob = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        logprob_steps = 0
        noise_stats = []
        if ret_logprob and self.account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(xt), torch.ones_like(xt))
            logprob = logprob + init_dist.log_prob(xt).sum(dim=(-2, -1))
            logprob_steps += 1

        for step in range(self.inference_steps):
            mean, std, step_noise_stats = self.gaussian_transition(
                xt, step, vis_cond, cond_emb, learn_exploration_noise=False,
            )
            noise_stats.append(step_noise_stats)
            dist = Normal(mean, std)
            if eval_mode:
                xt = mean
            else:
                xt = dist.sample()
            if ret_logprob:
                logprob = logprob + dist.log_prob(xt).sum(dim=(-2, -1))
                logprob_steps += 1
            if save_chains:
                chains[:, step + 1] = xt

        self.last_action_noise_stats = self._finalize_step_noise_stats(noise_stats)
        if ret_logprob:
            if self.normalize_denoising_horizon and logprob_steps > 0:
                logprob = logprob / logprob_steps
            if self.normalize_act_space_dimension:
                logprob = logprob / self.act_dim_total

        action_xt = xt
        if self.final_action_clip_value is not None:
            action_xt = action_xt.clamp(-self.final_action_clip_value, self.final_action_clip_value)
        action_pred = self.base_actor.normalizer["action"].unnormalize(action_xt)
        action = self._select_action_window(action_pred)
        if save_chains and ret_logprob:
            return action, chains, logprob
        if save_chains:
            return action, chains
        return action, logprob

    def get_logprobs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        chains: torch.Tensor,
        get_entropy: bool = False,
        learn_exploration_noise: bool = True,
    ):
        obs_dict = self._to_actor_device(obs_dict)
        chains = chains.to(device=self.device, dtype=self.dtype)
        vis_cond, cond_emb = self._encode_obs(obs_dict)
        batch_size = chains.shape[0]

        logprob = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        entropy = torch.zeros_like(logprob)
        logprob_steps = 0
        noise_stats = []

        if self.account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]))
            logprob = logprob + init_dist.log_prob(chains[:, 0]).sum(dim=(-2, -1))
            if get_entropy:
                entropy = entropy + init_dist.entropy().sum(dim=(-2, -1))
            logprob_steps += 1

        for step in range(self.inference_steps):
            xt = chains[:, step]
            xnext = chains[:, step + 1]
            mean, std, step_noise_stats = self.gaussian_transition(
                xt, step, vis_cond, cond_emb, learn_exploration_noise=learn_exploration_noise,
            )
            noise_stats.append(step_noise_stats)
            dist = Normal(mean, std)
            logprob = logprob + dist.log_prob(xnext).sum(dim=(-2, -1))
            if get_entropy:
                entropy = entropy + dist.entropy().sum(dim=(-2, -1))
            logprob_steps += 1

        if self.normalize_denoising_horizon and logprob_steps > 0:
            logprob = logprob / logprob_steps
            entropy = entropy / logprob_steps
        if self.normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            entropy = entropy / self.act_dim_total

        self.last_logprob_noise_stats = self._finalize_step_noise_stats(noise_stats)
        std_mean = self.last_logprob_noise_stats["sigma_mean"]
        if get_entropy:
            return logprob, entropy, std_mean
        return logprob, std_mean

    def _deterministic_normalized_action(
        self,
        actor: ManiFlowTransformerPointcloudPolicy,
        obs_dict: Dict[str, torch.Tensor],
        initial_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        actor_device = actor.device
        obs_on_device = dict_apply(
            obs_dict,
            lambda x: x.to(device=actor_device, dtype=actor.dtype) if torch.is_tensor(x) else x,
        )
        vis_cond, _ = self._encode_obs(obs_on_device, actor=actor)
        batch_size = vis_cond.shape[0]
        if initial_x is None:
            x = torch.zeros(
                batch_size,
                actor.horizon,
                actor.action_dim,
                device=actor_device,
                dtype=actor.dtype,
            )
        else:
            x = initial_x.to(device=actor_device, dtype=actor.dtype)
        dt = 1.0 / self.inference_steps
        for step in range(self.inference_steps):
            t = torch.full((batch_size,), step * dt, device=actor_device, dtype=actor.dtype)
            velocity = self._velocity(actor, x, t, vis_cond)
            x = x + velocity * dt
            if self.final_action_clip_value is not None and step == self.inference_steps - 1:
                x = x.clamp(-self.final_action_clip_value, self.final_action_clip_value)
        return x

    def bc_anchor_loss(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).shape[0]
        initial_x = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        current = self._deterministic_normalized_action(self.base_actor, obs_dict, initial_x=initial_x)
        with torch.no_grad():
            old = self._deterministic_normalized_action(self.actor_old, obs_dict, initial_x=initial_x).to(
                current.device
            )
        return F.mse_loss(current, old)

    def loss(
        self,
        obs_dict: Dict[str, torch.Tensor],
        chains: torch.Tensor,
        returns: torch.Tensor,
        oldvalues: torch.Tensor,
        advantages: torch.Tensor,
        oldlogprobs: torch.Tensor,
        critic: nn.Module,
        use_bc_loss: bool = False,
        normalize_advantages: bool = True,
        learn_exploration_noise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs_dict,
            chains,
            get_entropy=True,
            learn_exploration_noise=learn_exploration_noise,
        )
        noise_stats = self.get_last_noise_stats()
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.to(newlogprobs.device).clamp(min=self.logprob_min, max=self.logprob_max)
        returns = returns.to(newlogprobs.device)
        oldvalues = oldvalues.to(newlogprobs.device)
        advantages = advantages.to(newlogprobs.device)

        if normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        logratio = newlogprobs - oldlogprobs
        ratio = torch.exp(logratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.clip_ploss_coef,
            1.0 + self.clip_ploss_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = critic(obs_dict).view(-1)
        if self.clip_vloss_coef is None:
            value_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        else:
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            value_loss = 0.5 * torch.max(
                (newvalues - returns) ** 2,
                (v_clipped - returns) ** 2,
            ).mean()

        entropy_loss = -entropy.mean()
        bc_loss = self.bc_anchor_loss(obs_dict) if use_bc_loss else torch.zeros_like(pg_loss)

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean()

        out = {
            "pg_loss": pg_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "bc_loss": bc_loss,
            "approx_kl": approx_kl.detach(),
            "clipfrac": clipfrac.detach(),
            "ratio": ratio.mean().detach(),
            "noise_std": noise_std.detach(),
            "value_mean": newvalues.mean().detach(),
        }
        for key, value in noise_stats.items():
            out[key] = value.detach()
        return out
