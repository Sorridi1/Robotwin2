import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from maniflow.model.action_plan import (
    FixedCoarseActionOperator,
    FutureActionPlanPredictor,
)
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.rl.ppo_maniflow_pointcloud import (
    PPOManiFlowPointcloud,
    compute_reference_flow_state,
    compute_reference_residual,
)
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.workspace.train_maniflow_robotwin2_workspace import (
    TrainManiFlowRoboTwinWorkspace,
)


def _small_ditx(plan_dim=None):
    return DiTX(
        input_dim=2,
        output_dim=2,
        horizon=4,
        n_obs_steps=1,
        cond_dim=3,
        visual_cond_len=3,
        diffusion_timestep_embed_dim=8,
        diffusion_target_t_embed_dim=8,
        n_layer=1,
        n_head=2,
        n_emb=8,
        plan_dim=plan_dim,
    )


def test_fixed_coarse_operator_shape_identity_lowpass_and_preserve_dims():
    actions = torch.tensor(
        [[[0.0, 10.0], [3.0, 20.0], [6.0, 30.0], [9.0, 40.0]]]
    )
    identity = FixedCoarseActionOperator(mode="identity", kernel_size=3)
    torch.testing.assert_close(identity(actions), actions)

    lowpass = FixedCoarseActionOperator(
        mode="lowpass", kernel_size=3, preserve_dims=[1]
    )
    coarse = lowpass(actions)
    expected_first_dim = torch.tensor([[1.0, 3.0, 6.0, 8.0]])
    assert coarse.shape == actions.shape
    torch.testing.assert_close(coarse[..., 0], expected_first_dim)
    torch.testing.assert_close(coarse[..., 1], actions[..., 1])
    assert len(list(lowpass.parameters())) == 0


def test_reference_flow_math_and_shapes():
    x0 = torch.randn(2, 4, 3, dtype=torch.float64)
    action = torch.randn_like(x0)
    plan = torch.randn_like(x0)

    residual_t0 = compute_reference_residual(
        x0, x0, plan, torch.zeros(x0.shape[0], dtype=x0.dtype)
    )
    assert residual_t0.shape == x0.shape
    torch.testing.assert_close(residual_t0, torch.zeros_like(x0), atol=0.0, rtol=0.0)

    time = torch.tensor([0.25, 0.75], dtype=x0.dtype)
    xt = (1.0 - time[:, None, None]) * x0 + time[:, None, None] * action
    reference = compute_reference_flow_state(x0, plan, time)
    residual = compute_reference_residual(xt, x0, plan, time)
    assert reference.shape == residual.shape == x0.shape
    torch.testing.assert_close(residual, time[:, None, None] * (action - plan))


def test_ditx_plan_projection_is_zero_initialized_and_behavior_preserving():
    torch.manual_seed(0)
    model = _small_ditx(plan_dim=2).eval()
    nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=0.1)
    sample = torch.randn(2, 4, 2)
    vis_cond = torch.randn(2, 3, 3)
    time = torch.tensor([0.1, 0.6])
    target_t = torch.tensor([0.2, 0.2])
    plan1 = torch.randn_like(sample)
    plan2 = torch.randn_like(sample)

    assert torch.count_nonzero(model.plan_proj.weight) == 0
    assert torch.count_nonzero(model.plan_proj.bias) == 0
    out1 = model(sample, time, target_t, vis_cond, plan_cond=plan1)
    out2 = model(sample, time, target_t, vis_cond, plan_cond=plan2)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


def test_plan_gradient_isolation_for_plan_and_flow_losses():
    torch.manual_seed(1)
    predictor = FutureActionPlanPredictor(visual_dim=3, horizon=4, action_dim=2, hidden_dims=(8,))
    vis_cond = torch.randn(2, 3, 3, requires_grad=True)
    target = torch.randn(2, 4, 2)
    plan_pred = predictor(vis_cond.detach())
    torch.nn.functional.l1_loss(plan_pred, target).backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in predictor.parameters())
    assert vis_cond.grad is None

    predictor.zero_grad(set_to_none=True)
    model = _small_ditx(plan_dim=2).eval()
    # Make the existing zero-initialized output head non-degenerate for this gradient check.
    nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=0.1)
    sample = torch.randn(2, 4, 2)
    flow_out = model(
        sample,
        torch.tensor([0.2, 0.7]),
        torch.tensor([0.1, 0.1]),
        vis_cond.detach(),
        plan_cond=predictor(vis_cond.detach()).detach(),
    )
    flow_out.square().mean().backward()
    assert all(param.grad is None for param in predictor.parameters())
    assert model.plan_proj.weight.grad is not None
    assert model.plan_proj.weight.grad.abs().sum() > 0


def test_legacy_checkpoint_allows_only_new_plan_keys_to_be_missing():
    policy = ManiFlowTransformerPointcloudPolicy.__new__(ManiFlowTransformerPointcloudPolicy)
    nn.Module.__init__(policy)
    policy.action_plan_enabled = True
    policy.model = _small_ditx(plan_dim=2)
    policy.plan_predictor = FutureActionPlanPredictor(3, 4, 2, hidden_dims=(8,))
    legacy_state = {
        key: value
        for key, value in policy.state_dict().items()
        if not key.startswith(("plan_predictor.", "model.plan_proj."))
    }
    incompatible = policy.load_state_dict(legacy_state, strict=True)
    assert set(incompatible.missing_keys) == {
        "model.plan_proj.weight",
        "model.plan_proj.bias",
        *{f"plan_predictor.{key}" for key in policy.plan_predictor.state_dict()},
    }

    invalid_state = copy.deepcopy(legacy_state)
    invalid_state.pop("model.input_emb.weight")
    with pytest.raises(RuntimeError, match="beyond newly introduced"):
        policy.load_state_dict(invalid_state, strict=True)


class _IdentityField:
    def unnormalize(self, value):
        return value


class _IdentityNormalizer:
    def normalize(self, value):
        return value

    def __getitem__(self, key):
        assert key == "action"
        return _IdentityField()


class _TinyObsEncoder(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.proj = nn.Linear(obs_dim, obs_dim)

    def forward(self, obs):
        return self.proj(obs["agent_pos"])


class _TinyVelocity(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.plan_proj = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, sample, timestep, target_t, vis_cond, plan_cond=None):
        del timestep, target_t, vis_cond
        plan_term = 0.0 if plan_cond is None else self.plan_proj(plan_cond)
        return self.scale * torch.tanh(plan_term - sample)


class _TinyPlanActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 4
        self.action_dim = 2
        self.n_action_steps = 2
        self.n_obs_steps = 2
        self.num_inference_steps = 3
        self.obs_feature_dim = 3
        self.use_pc_color = True
        self.sample_target_t_mode = "relative"
        self.action_plan_enabled = True
        self.obs_encoder = _TinyObsEncoder(self.obs_feature_dim)
        self.plan_predictor = FutureActionPlanPredictor(3, 4, 2, hidden_dims=(8,))
        self.model = _TinyVelocity(self.action_dim)
        self.normalizer = _IdentityNormalizer()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def predict_action_plan(self, vis_cond, detach_visual=True):
        if detach_visual:
            vis_cond = vis_cond.detach()
        return self.plan_predictor(vis_cond)


def test_reference_flow_rollout_logprob_recompute_is_finite_and_consistent():
    torch.manual_seed(4)
    actor = _TinyPlanActor()
    ppo = PPOManiFlowPointcloud(
        actor,
        inference_steps=3,
        ft_denoising_steps=3,
        noise_head_type="residual_schedule",
        noise_conditioning_mode="reference_flow",
        noise_model_dim=8,
        noise_action_embed_dim=4,
        noise_hidden_dims=(8,),
        base_sigma=0.02,
        sigma_min=0.005,
        sigma_max=0.05,
        actor_old_device="cpu",
        freeze_obs_encoder=True,
    ).eval()
    obs = {
        "point_cloud": torch.randn(2, 2, 5, 3),
        "agent_pos": torch.randn(2, 2, 3),
    }
    action, chains, rollout_logprob = ppo.get_actions(obs)
    recomputed_logprob, _ = ppo.get_logprobs(obs, chains)

    assert action.shape == (2, 2, 2)
    assert chains.shape == (2, 4, 4, 2)
    assert rollout_logprob.shape == recomputed_logprob.shape == (2,)
    assert torch.isfinite(recomputed_logprob).all()
    torch.testing.assert_close(rollout_logprob, recomputed_logprob, atol=1e-6, rtol=1e-6)
    assert all(not param.requires_grad for param in actor.plan_predictor.parameters())
    assert "reference_residual_step_0" in ppo.last_logprob_noise_stats
    torch.testing.assert_close(
        ppo.last_logprob_noise_stats["reference_residual_step_0"],
        torch.zeros(()),
        atol=1e-7,
        rtol=0.0,
    )


def test_action_plan_disabled_full_policy_and_default_rl_smoke():
    shape_meta = OmegaConf.create({
        "obs": {
            "point_cloud": {"shape": [8, 3]},
            "agent_pos": {"shape": [3]},
        },
        "action": {"shape": [2]},
    })
    pointcloud_cfg = OmegaConf.create(
        {
            "in_channels": 3,
            "out_channels": 8,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "normal_channel": False,
            "num_points": 8,
            "pointwise": True,
        }
    )
    policy = ManiFlowTransformerPointcloudPolicy(
        shape_meta=shape_meta,
        horizon=4,
        n_action_steps=2,
        n_obs_steps=2,
        num_inference_steps=2,
        visual_cond_len=8,
        n_layer=1,
        n_head=2,
        n_emb=16,
        encoder_output_dim=8,
        use_pc_color=False,
        pointcloud_encoder_cfg=pointcloud_cfg,
        action_plan={"enabled": False},
    )
    policy.normalizer.fit(
        {
            "point_cloud": torch.randn(16, 8, 3),
            "agent_pos": torch.randn(16, 3),
            "action": torch.randn(16, 2),
        },
        last_n_dims=1,
    )
    batch = {
        "obs": {
            "point_cloud": torch.randn(4, 2, 8, 3),
            "agent_pos": torch.randn(4, 2, 3),
        },
        "action": torch.randn(4, 4, 2),
    }
    ema_policy = copy.deepcopy(policy).eval()
    loss, loss_dict = policy.compute_loss(batch, ema_model=ema_policy)
    assert torch.isfinite(loss)
    assert "loss_plan" not in loss_dict
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (4, 2, 2)
    assert result["action_pred"].shape == (4, 4, 2)
    assert "plan_pred" not in result

    ppo = PPOManiFlowPointcloud(
        policy,
        inference_steps=2,
        ft_denoising_steps=2,
        noise_head_type="residual_schedule",
        noise_conditioning_mode="default",
        noise_model_dim=8,
        noise_action_embed_dim=4,
        noise_hidden_dims=(8,),
        actor_old_device="cpu",
    ).eval()
    action, chains, logprob = ppo.get_actions(batch["obs"])
    assert action.shape == (4, 2, 2)
    assert chains.shape == (4, 3, 4, 2)
    assert logprob.shape == (4,)
    assert torch.isfinite(logprob).all()


def test_evaluation_checkpoint_load_can_skip_incompatible_optimizer():
    workspace = TrainManiFlowRoboTwinWorkspace.__new__(TrainManiFlowRoboTwinWorkspace)
    workspace.model = nn.Linear(2, 2)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters())

    old_model = nn.Linear(2, 2)
    old_optimizer = torch.optim.AdamW([old_model.weight])
    payload = {
        "state_dicts": {
            "model": old_model.state_dict(),
            "optimizer": old_optimizer.state_dict(),
        },
        "pickles": {},
    }

    with pytest.raises(ValueError, match="parameter group"):
        workspace.load_payload(payload)
    workspace.load_payload(payload, exclude_keys=("optimizer",))
    torch.testing.assert_close(workspace.model.weight, old_model.weight)
