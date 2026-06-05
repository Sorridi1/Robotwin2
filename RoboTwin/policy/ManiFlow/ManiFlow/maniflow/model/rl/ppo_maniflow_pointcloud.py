from typing import Dict, Iterable, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from maniflow.common.pytorch_util import dict_apply
from maniflow.model.diffusion.positional_embedding import SinusoidalPosEmb
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy


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
        noise_time_dim: int = 16,
        actor_old_device: str = "cpu",
        freeze_obs_encoder: bool = True,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.inference_steps = int(inference_steps or base_actor.num_inference_steps)
        self.ft_denoising_steps = int(ft_denoising_steps or self.inference_steps)
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

        self.noise_head = TimeConditionedNoiseHead(
            cond_dim=base_actor.obs_feature_dim,
            action_horizon=self.horizon,
            action_dim=self.action_dim,
            time_dim=noise_time_dim,
            hidden_dims=noise_hidden_dims,
            min_std=min_logprob_denoising_std,
            max_std=max_logprob_denoising_std,
            activation=noise_activation,
        )

        actor_old = copy.deepcopy(base_actor).to(actor_old_device)
        actor_old.eval()
        for param in actor_old.parameters():
            param.requires_grad_(False)
        object.__setattr__(self, "actor_old", actor_old)

        if self.freeze_obs_encoder:
            for param in self.base_actor.obs_encoder.parameters():
                param.requires_grad_(False)

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

    def _noise_std(
        self,
        cond_emb: torch.Tensor,
        time: torch.Tensor,
        step: int,
        learn_exploration_noise: bool,
    ) -> torch.Tensor:
        if step < self.learn_explore_noise_from:
            std = torch.full(
                (cond_emb.shape[0], self.act_dim_total),
                self.min_logprob_denoising_std,
                device=cond_emb.device,
                dtype=cond_emb.dtype,
            )
        else:
            std = self.noise_head(cond_emb, time)
        return std if learn_exploration_noise else std.detach()

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
        dt = 1.0 / self.inference_steps

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
        if ret_logprob and self.account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(xt), torch.ones_like(xt))
            logprob = logprob + init_dist.log_prob(xt).sum(dim=(-2, -1))
            logprob_steps += 1

        for step in range(self.inference_steps):
            t = torch.full((batch_size,), step * dt, device=self.device, dtype=self.dtype)
            vel = self._velocity(self.base_actor, xt, t, vis_cond)
            mean = xt + vel * dt
            if self.denoised_clip_value is not None:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            std = self._noise_std(cond_emb, t, step, learn_exploration_noise=False)
            std = std.reshape(batch_size, self.horizon, self.action_dim)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(mean, std)
            if eval_mode:
                xt = mean
            else:
                xt = dist.sample()
                xt = xt.clamp(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale,
                )
            if step == self.inference_steps - 1 and self.final_action_clip_value is not None:
                xt = xt.clamp(-self.final_action_clip_value, self.final_action_clip_value)
            if ret_logprob:
                logprob = logprob + dist.log_prob(xt).sum(dim=(-2, -1))
                logprob_steps += 1
            if save_chains:
                chains[:, step + 1] = xt

        if ret_logprob:
            if self.normalize_denoising_horizon and logprob_steps > 0:
                logprob = logprob / logprob_steps
            if self.normalize_act_space_dimension:
                logprob = logprob / self.act_dim_total

        action_pred = self.base_actor.normalizer["action"].unnormalize(xt)
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
        dt = 1.0 / self.inference_steps

        logprob = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        entropy = torch.zeros_like(logprob)
        logprob_steps = 0
        std_means = []

        if self.account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]))
            logprob = logprob + init_dist.log_prob(chains[:, 0]).sum(dim=(-2, -1))
            if get_entropy:
                entropy = entropy + init_dist.entropy().sum(dim=(-2, -1))
            logprob_steps += 1

        for step in range(self.inference_steps):
            xt = chains[:, step]
            xnext = chains[:, step + 1]
            t = torch.full((batch_size,), step * dt, device=self.device, dtype=self.dtype)
            vel = self._velocity(self.base_actor, xt, t, vis_cond)
            mean = xt + vel * dt
            if self.denoised_clip_value is not None:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)
            std = self._noise_std(cond_emb, t, step, learn_exploration_noise)
            std_means.append(std.mean())
            std = std.reshape(batch_size, self.horizon, self.action_dim)
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

        std_mean = torch.stack(std_means).mean() if len(std_means) > 0 else torch.tensor(0.0, device=self.device)
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
            x = x + self._velocity(actor, x, t, vis_cond) * dt
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
    ) -> Dict[str, torch.Tensor]:
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs_dict,
            chains,
            get_entropy=True,
            learn_exploration_noise=True,
        )
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

        return {
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
