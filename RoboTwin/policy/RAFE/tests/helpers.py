"""Small numerical models; production PPO, noise head, buffer and loader stay real."""
import torch
from torch import nn

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.model.rl.ppo_maniflow_pointcloud import PPOManiFlowPointcloud


class TinyObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 4)

    def forward(self, obs):
        features = torch.cat((obs["point_cloud"].mean(dim=1), obs["agent_pos"]), dim=-1)
        return self.projection(features)


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(2, 2)
        self.visual = nn.Linear(4, 2)
        self.calls = []

    def forward(self, sample, timestep, target_t, vis_cond):
        self.calls.append((timestep.detach().clone(), target_t.detach().clone()))
        return (self.action(sample) + self.visual(vis_cond.mean(dim=1))[:, None]
                + 0.1 * timestep[:, None, None] + 0.2 * target_t[:, None, None])


class TinyBaseActor(nn.Module):
    horizon = 4
    action_dim = 2
    n_obs_steps = 2
    n_action_steps = 2
    num_inference_steps = 3
    obs_feature_dim = 4
    use_pc_color = False
    sample_target_t_mode = "relative"

    def __init__(self):
        super().__init__()
        self.model = TinyFlow()
        self.obs_encoder = TinyObsEncoder()
        self.normalizer = LinearNormalizer()
        # Nonidentity action scaling catches accidentally using raw action units.
        self.normalizer.fit({
            "action": torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
            "point_cloud": torch.tensor([[-1.0] * 3, [1.0] * 3]),
            "agent_pos": torch.tensor([[-1.0] * 2, [1.0] * 2]),
        })
        self.normalizer.requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)

    def set_normalizer(self, normalizer):
        pass

    def forward(self, obs):
        return self.head(obs["agent_pos"].mean(dim=1)).squeeze(-1)


def make_actor(**overrides):
    options = dict(
        inference_steps=3, ft_denoising_steps=2,
        noise_head_type="residual_schedule", noise_backbone="tcn",
        noise_conditioning_mode="default", noise_output_mode="full",
        action_conditioned_noise=True, base_sigma_schedule="linear",
        base_sigma_min=0.15, base_sigma_max=0.3, sigma_min=0.01, sigma_max=0.8,
        min_sampling_denoising_std=0.02, min_logprob_denoising_std=0.03,
        residual_scale=1.0, noise_model_dim=8, noise_action_embed_dim=4,
        noise_time_dim=4, tcn_num_layers=1, noise_hidden_dims=(8,),
        denoised_clip_value=1.0, final_action_clip_value=0.8,
        account_for_initial_stochasticity=True, actor_old_device="cpu",
        freeze_obs_encoder=True, logprob_min=-10000, logprob_max=10000,
    )
    options.update(overrides)
    return PPOManiFlowPointcloud(TinyBaseActor(), **options)


def expert_batch(batch_size=3):
    return {
        "obs": {
            "point_cloud": torch.randn(batch_size, 4, 5, 3),
            "agent_pos": torch.randn(batch_size, 4, 2),
        },
        "action": torch.randn(batch_size, 4, 2),
    }


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def parameters_equal(module, previous):
    return all(torch.equal(p.detach(), saved) for p, saved in zip(module.parameters(), previous))
