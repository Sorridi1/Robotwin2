"""ManiFlow ODE sampling and per-MC conditional-flow-matching losses."""

import torch
from torch import nn

from .loss import modified_huber


class FPOActor(nn.Module):
    def __init__(self, policy, seed=0, inference_steps=None, final_action_clip=None,
                 intermediate_clip=None, huber_delta=0.5):
        super().__init__()
        self.policy = policy
        if getattr(policy, "language_conditioned", False):
            raise ValueError("This adapter supports point-cloud/proprioception checkpoints without language conditioning")
        if getattr(policy, "plan_output_mode", "flow") != "flow":
            raise ValueError("FPO++ requires a flow policy, not a plan-only diagnostic checkpoint")
        self.inference_steps = int(inference_steps or policy.num_inference_steps)
        if self.inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        for value in (final_action_clip, intermediate_clip):
            if value is not None and value <= 0:
                raise ValueError("Action clip bounds must be positive or null")
        self.final_action_clip = final_action_clip
        self.intermediate_clip = intermediate_clip
        self.huber_delta = float(huber_delta)
        policy.normalizer.requires_grad_(False)
        policy.obs_encoder.requires_grad_(False)
        if getattr(policy, "plan_predictor", None) is not None:
            policy.plan_predictor.requires_grad_(False)
        self.rng = torch.Generator(device=self.device).manual_seed(int(seed))
        self.eval()  # Deterministic dropout/normalization, with gradients in CFM.

    @property
    def device(self):
        return next(self.policy.model.parameters()).device

    @property
    def window(self):
        start = int(self.policy.n_obs_steps) - 1
        end = min(start + int(self.policy.n_action_steps), int(self.policy.horizon))
        if start < 0 or end <= start:
            raise ValueError("Checkpoint has an empty action window")
        return start, end

    def trainable_parameters(self):
        return (p for p in self.policy.model.parameters() if p.requires_grad)

    @torch.no_grad()
    def encode(self, observations):
        obs = {key: value.to(device=self.device, dtype=torch.float32) for key, value in observations.items()}
        normalized = self.policy.normalizer.normalize(obs)
        if not self.policy.use_pc_color:
            normalized["point_cloud"] = normalized["point_cloud"][..., :3]
        batch = normalized["agent_pos"].shape[0]
        count = int(self.policy.n_obs_steps)
        inputs = {key: value[:, :count].reshape(-1, *value.shape[2:]) for key, value in normalized.items()}
        features = self.policy.obs_encoder(inputs).reshape(batch, -1, self.policy.obs_feature_dim)
        plan = None
        if getattr(self.policy, "action_plan_enabled", False):
            plan = self.policy.predict_action_plan(features, detach_visual=True).detach()
        return features.detach(), plan

    def velocity(self, x, time, features, plan=None, dt=0.0):
        target = time + dt if self.policy.sample_target_t_mode == "absolute" else torch.full_like(time, dt)
        return self.policy.model(sample=x, timestep=time, target_t=target,
                                 vis_cond=features, plan_cond=plan)

    @torch.no_grad()
    def sample(self, features, plan=None, zero_latent=False):
        shape = (features.shape[0], self.policy.horizon, self.policy.action_dim)
        x = torch.zeros(shape, device=self.device) if zero_latent else torch.randn(
            shape, device=self.device, generator=self.rng)
        dt = 1.0 / self.inference_steps
        for step in range(self.inference_steps):
            time = torch.full((shape[0],), step * dt, device=self.device)
            x = x + self.velocity(x, time, features, plan, dt=dt) * dt
            if self.intermediate_clip is not None:
                x = x.clamp(-self.intermediate_clip, self.intermediate_clip)
        before_clip = x
        if self.final_action_clip is not None:
            x = x.clamp(-self.final_action_clip, self.final_action_clip)
        clip_fraction = (x != before_clip).float().mean().item()
        action_pred = self.policy.normalizer["action"].unnormalize(x)
        start, end = self.window
        return action_pred[:, start:end], x.detach(), clip_fraction

    def draw_cfm(self, actions, samples):
        if samples < 1:
            raise ValueError("MC samples must be positive")
        batch, horizon, dimension = actions.shape
        time = torch.rand((batch, samples, 1, 1), device=self.device, generator=self.rng)
        noise = torch.randn((batch, samples, horizon, dimension), device=self.device, generator=self.rng)
        return time, noise

    def cfm_losses(self, features, plan, actions, time, noise):
        """Return [B,N,H]. Features/plan/target/random variables stay fixed."""
        batch, samples, horizon, dimension = noise.shape
        if actions.shape != (batch, horizon, dimension) or time.shape != (batch, samples, 1, 1):
            raise ValueError("CFM inputs do not have matching B,N,H,D dimensions")
        target = actions.detach()[:, None]
        noise, time = noise.detach(), time.detach()
        xt = (1-time) * noise + time * target  # ManiFlow: noise -> data.
        velocity_target = target - noise
        repeated = features.detach().repeat_interleave(samples, dim=0)
        repeated_plan = plan.detach().repeat_interleave(samples, dim=0) if plan is not None else None
        # Instantaneous flow: dt=0, unlike the finite-dt ODE sampler above.
        velocity = self.velocity(xt.reshape(-1, horizon, dimension), time.reshape(-1),
                                 repeated, repeated_plan, dt=0.0)
        error = velocity.float().reshape(batch, samples, horizon, dimension) - velocity_target.float()
        return modified_huber(error, self.huber_delta).mean(-1)


def execution_mask(actor, executed):
    start, end = actor.window
    executed = torch.as_tensor(executed, device=actor.device, dtype=torch.long)
    if (executed < 0).any() or (executed > end-start).any():
        raise ValueError("Reported executed actions exceed the sampled window")
    positions = torch.arange(actor.policy.horizon, device=actor.device)[None]
    return ((positions >= start) & (positions < start + executed[:, None])).float()
