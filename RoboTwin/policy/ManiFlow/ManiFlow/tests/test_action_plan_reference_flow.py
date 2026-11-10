import copy
import io
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from maniflow.common.ppo_optimizer import build_actor_optimizer, load_actor_optimizer_state
from maniflow.model.action_plan import (
    FixedCoarseActionOperator,
    FutureActionPlanPredictor,
)
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.rl.ppo_maniflow_pointcloud import (
    ActionTimeConditionedNoiseHead,
    ActionConditionedResidualNoiseHead,
    PPOManiFlowPointcloud,
    TimeConditionedNoiseHead,
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
    def __init__(self, horizon=4, action_dim=2):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = 2
        self.n_obs_steps = 2
        self.num_inference_steps = 3
        self.obs_feature_dim = 3
        self.use_pc_color = True
        self.sample_target_t_mode = "relative"
        self.action_plan_enabled = True
        self.obs_encoder = _TinyObsEncoder(self.obs_feature_dim)
        self.plan_predictor = FutureActionPlanPredictor(3, horizon, action_dim, hidden_dims=(8,))
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


@pytest.mark.parametrize("mode", ["reference_flow", "reference_residual_only"])
def test_reference_flow_rollout_logprob_recompute_is_finite_and_consistent(mode):
    torch.manual_seed(4)
    actor = _TinyPlanActor()
    ppo = PPOManiFlowPointcloud(
        actor,
        inference_steps=3,
        ft_denoising_steps=3,
        noise_head_type="residual_schedule",
        noise_conditioning_mode=mode,
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


@pytest.mark.parametrize("plan_enabled,mode", [
    (False, "default"),
    (True, "default"),
    (True, "plan"),
    (True, "reference_flow"),
    (True, "reference_residual_only"),
    (False, "reinflow_xt"),
    (True, "reinflow_xt"),
])
def test_full_policy_and_rl_smoke(plan_enabled, mode):
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
        action_plan={"enabled": plan_enabled, "predictor_hidden_dims": (8,)},
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
    assert ("loss_plan" in loss_dict) == plan_enabled
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (4, 2, 2)
    assert result["action_pred"].shape == (4, 4, 2)
    assert ("plan_pred" in result) == plan_enabled

    ppo = PPOManiFlowPointcloud(
        policy,
        inference_steps=2,
        ft_denoising_steps=2,
        noise_head_type="reinflow_xt" if mode == "reinflow_xt" else "residual_schedule",
        noise_conditioning_mode="default" if mode == "reinflow_xt" else mode,
        noise_backbone="tcn",
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
    recomputed, _ = ppo.get_logprobs(batch["obs"], chains)
    torch.testing.assert_close(logprob, recomputed, atol=1e-6, rtol=1e-6)
    if plan_enabled:
        assert policy.model.plan_proj is not None
        assert all(not p.requires_grad for p in policy.plan_predictor.parameters())
    if mode == "reference_residual_only":
        # Exercise gradients through the real DiT-X with a non-degenerate
        # output, as provided by the pretrained actor during RL.
        nn.init.normal_(policy.model.final_layer.ffn_final.fc2.weight, std=0.02)
        nn.init.normal_(ppo.noise_head.out_proj.weight, std=0.02)
        ppo.train()
        _, chains, _ = ppo.get_actions(batch["obs"])
        recomputed, _ = ppo.get_logprobs(batch["obs"], chains)
        (-recomputed.mean()).backward()
        for projection in (policy.model.plan_proj, ppo.noise_head.reference_residual_proj):
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in policy.plan_predictor.parameters())


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


def _residual_only_ppo(horizon=4, action_dim=2, **overrides):
    options = dict(
        inference_steps=3,
        ft_denoising_steps=3,
        noise_head_type="residual_schedule",
        noise_conditioning_mode="reference_residual_only",
        noise_backbone="tcn",
        noise_model_dim=8,
        noise_action_embed_dim=4,
        base_sigma_schedule="linear",
        base_sigma_min=0.01,
        base_sigma_max=0.04,
        sigma_min=0.005,
        sigma_max=0.05,
        actor_old_device="cpu",
        freeze_obs_encoder=True,
    )
    options.update(overrides)
    return PPOManiFlowPointcloud(_TinyPlanActor(horizon, action_dim), **options)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("time_value", [0.0, 0.25, 1.0, None])
def test_residual_only_reference_math_at_scalar_and_per_sample_times(batch_size, time_value):
    x0 = torch.randn(batch_size, 16, 14, dtype=torch.float64)
    xk = torch.randn_like(x0)
    plan = torch.randn_like(x0)
    time = torch.linspace(0, 1, batch_size) if time_value is None else time_value
    t = torch.as_tensor(time, dtype=x0.dtype)
    if t.ndim == 1:
        t = t[:, None, None]
    snapshots = [value.clone() for value in (xk, x0, plan)]
    residual = compute_reference_residual(xk, x0, plan, time)
    assert residual.shape == (batch_size, 16, 14)
    torch.testing.assert_close(residual, xk - (1 - t) * x0 - t * plan)
    torch.testing.assert_close(
        compute_reference_residual(x0, x0, plan, 0.0),
        torch.zeros_like(x0), atol=0, rtol=0,
    )
    for value, snapshot in zip((xk, x0, plan), snapshots):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["default", "plan", "reference_flow", "reference_residual_only"])
def test_noise_tokens_use_exactly_the_selected_conditioning(mode):
    torch.manual_seed(20)
    head = ActionConditionedResidualNoiseHead(
        cond_dim=3, action_horizon=4, action_dim=2, model_dim=8,
        noise_backbone="tcn", noise_conditioning_mode=mode,
    )
    cond = torch.randn(2, 3)
    time = torch.tensor([0.0, 0.5])
    x = torch.randn(2, 4, 2)
    plan = torch.randn_like(x)
    residual = torch.randn_like(x)
    has_plan = mode in ("plan", "reference_flow")
    has_residual = mode in ("reference_flow", "reference_residual_only")
    assert (head.plan_proj is not None) == has_plan
    assert (head.reference_residual_proj is not None) == has_residual
    expected = head.cond_proj(cond)[:, None] + head.time_proj(head.time_embedding(time))[:, None]
    expected = expected + head.action_proj(x)
    if has_plan:
        expected = expected + head.plan_proj(plan)
    if has_residual:
        expected = expected + head.reference_residual_proj(residual)
    expected = head.token_norm(expected + head.horizon_pos_emb)
    actual = head._build_tokens(cond, time, x, plan_cond=plan, reference_residual=residual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if mode == "reference_residual_only":
        without_plan = head._build_tokens(cond, time, x, reference_residual=residual)
        # An unused plan must not trigger shape validation either.
        malformed_plan = head._build_tokens(
            cond, time, x, plan_cond=torch.empty(1), reference_residual=residual,
        )
        torch.testing.assert_close(actual, without_plan, atol=0, rtol=0)
        torch.testing.assert_close(actual, malformed_plan, atol=0, rtol=0)
        with pytest.raises(ValueError, match="reference_residual shape"):
            head._build_tokens(cond, time, x)


def test_residual_only_parameter_counts_and_strict_checkpoint_compatibility():
    modes = ("default", "plan", "reference_flow", "reference_residual_only")
    # YAML DP3 feature width: 128 point-cloud channels + 64 state channels.
    heads = {
        mode: ActionConditionedResidualNoiseHead(
            cond_dim=192, action_horizon=16, action_dim=14, model_dim=128,
            noise_backbone="tcn", noise_conditioning_mode=mode,
        )
        for mode in modes
    }
    counts = {mode: sum(p.numel() for p in head.parameters() if p.requires_grad)
              for mode, head in heads.items()}
    print(f"TCN noise-head trainable parameters: {counts}")
    assert counts["reference_flow"] - counts["reference_residual_only"] == 1920
    assert counts["plan"] == counts["reference_residual_only"] == counts["default"] + 1920
    residual_only = heads["reference_residual_only"]
    assert residual_only.plan_proj is None
    assert residual_only.reference_residual_proj is not None
    assert not any(key.startswith("plan_proj.") for key in residual_only.state_dict())
    for mode, head in heads.items():
        result = head.load_state_dict(copy.deepcopy(head.state_dict()), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
        if mode != "reference_residual_only":
            with pytest.raises(RuntimeError, match="plan_proj|reference_residual_proj"):
                residual_only.load_state_dict(head.state_dict(), strict=True)
            with pytest.raises(RuntimeError, match="plan_proj|reference_residual_proj"):
                head.load_state_dict(residual_only.state_dict(), strict=True)


@pytest.mark.parametrize("output_mode", ["full", "chunk_shared", "dim_shared"])
@pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
def test_residual_only_sigma_shapes_schedule_and_bounds(output_mode, schedule):
    ppo = _residual_only_ppo(
        horizon=16, action_dim=14, noise_model_dim=128,
        noise_output_mode=output_mode, base_sigma_schedule=schedule,
    )
    x0 = torch.randn(2, 16, 14)
    xk = torch.randn_like(x0)
    plan = torch.randn_like(x0)
    cond = torch.randn(2, 3)
    time = torch.full((2,), 1 / 3)
    residual = compute_reference_residual(xk, x0, plan, time)
    sigma, info = ppo.compute_residual_sigma(cond, time, xk, step=1, reference_residual=residual)
    assert sigma.shape == residual.shape == (2, 16, 14)
    # Preserve zero-init: the initial learned schedule equals the base schedule.
    torch.testing.assert_close(info["delta"], torch.zeros_like(xk), atol=0, rtol=0)
    torch.testing.assert_close(sigma, ppo.compute_base_sigma(time, xk, step=1), atol=0, rtol=0)
    nn.init.normal_(ppo.noise_head.out_proj.weight, std=0.05)
    sigma, info = ppo.compute_residual_sigma(cond, time, xk, step=1, reference_residual=residual)
    expected_delta = ppo.residual_scale * torch.tanh(ppo.noise_head.forward_raw(
        cond, time, xk, reference_residual=residual,
    ))
    torch.testing.assert_close(info["delta"], expected_delta, atol=0, rtol=0)
    expected_sigma = (info["base_sigma"] * expected_delta.exp()).clamp(ppo.sigma_min, ppo.sigma_max)
    expected_sigma = expected_sigma.clamp(min=ppo.min_sampling_denoising_std)
    torch.testing.assert_close(sigma, expected_sigma, atol=0, rtol=0)
    assert torch.isfinite(sigma).all()
    assert sigma.min() >= ppo.sigma_min and sigma.max() <= ppo.sigma_max


@pytest.mark.parametrize("learn_noise", [False, True])
@pytest.mark.parametrize("ft_steps", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("output_mode", ["full", "chunk_shared", "dim_shared"])
def test_residual_only_rollout_recompute_conditions_and_logprobs(
    monkeypatch, learn_noise, ft_steps, normalize, output_mode,
):
    torch.manual_seed(31)
    ppo = _residual_only_ppo(
        ft_denoising_steps=ft_steps,
        noise_output_mode=output_mode,
        normalize_denoising_horizon=normalize,
        normalize_act_space_dimension=normalize,
        account_for_initial_stochasticity=normalize,
    ).train()
    # A nonzero output makes this exercise residual conditioning numerically.
    nn.init.normal_(ppo.noise_head.out_proj.weight, std=0.05)
    obs = {"point_cloud": torch.randn(2, 2, 5, 3), "agent_pos": torch.randn(2, 2, 3)}
    with torch.no_grad():
        vis_cond, _ = ppo._encode_obs(obs)
        plan = ppo._predict_plan(ppo.base_actor, vis_cond)
    calls = []
    velocity_calls = []
    noise_std = ppo._noise_std
    velocity = ppo._velocity

    def record_noise(cond, time, x, step, *args, **kwargs):
        assert "plan_cond" not in kwargs
        calls.append((x.detach().clone(), time.clone(), kwargs["reference_residual"].detach().clone()))
        return noise_std(cond, time, x, step, *args, **kwargs)

    def record_velocity(actor, x, time, vis, plan_cond=None):
        torch.testing.assert_close(plan_cond, plan, atol=0, rtol=0)
        velocity_calls.append(True)
        return velocity(actor, x, time, vis, plan_cond=plan_cond)

    monkeypatch.setattr(ppo, "_noise_std", record_noise)
    monkeypatch.setattr(ppo, "_velocity", record_velocity)
    rng_before_rollout = torch.random.get_rng_state()
    action, chains, rollout_logprob = ppo.get_actions(
        obs, eval_mode=False, save_chains=True, ret_logprob=True,
    )
    initial_generator = torch.Generator().set_state(rng_before_rollout)
    expected_x0 = torch.randn(chains[:, 0].shape, generator=initial_generator)
    torch.testing.assert_close(chains[:, 0], expected_x0, atol=0, rtol=0)
    chains_before = chains.clone()
    rng_before_recompute = torch.random.get_rng_state()
    recomputed, entropy, _ = ppo.get_logprobs(
        obs, chains, get_entropy=True, learn_exploration_noise=learn_noise,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before_recompute)
    torch.testing.assert_close(chains, chains_before, atol=0, rtol=0)
    assert action.shape == (2, 2, 2)
    assert len(calls) == len(velocity_calls) == 2 * ppo.inference_steps
    for step in range(ppo.inference_steps):
        xt, time, residual = calls[step]
        torch.testing.assert_close(xt, chains[:, step], atol=0, rtol=0)
        torch.testing.assert_close(residual, compute_reference_residual(xt, chains[:, 0], plan, time))
        for rollout_value, recomputed_value in zip(calls[step], calls[step + ppo.inference_steps]):
            torch.testing.assert_close(rollout_value, recomputed_value, atol=0, rtol=0)
    torch.testing.assert_close(calls[0][2], torch.zeros_like(chains[:, 0]), atol=0, rtol=0)
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(rollout_logprob, recomputed, atol=1e-6, rtol=1e-6)
    for key, value in ppo.last_action_noise_stats.items():
        torch.testing.assert_close(value, ppo.last_logprob_noise_stats[key], atol=1e-7, rtol=1e-6)


def test_residual_only_gradients_after_zero_init_and_with_detached_noise():
    torch.manual_seed(41)
    ppo = _residual_only_ppo().train()
    obs = {"point_cloud": torch.randn(2, 2, 5, 3), "agent_pos": torch.randn(2, 2, 3)}
    frozen_plan = copy.deepcopy(ppo.base_actor.plan_predictor.state_dict())
    _, chains, _ = ppo.get_actions(obs)
    optimizer = torch.optim.SGD(ppo.actor_parameters(), lr=1e-3)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        logprob, _ = ppo.get_logprobs(obs, chains, learn_exploration_noise=True)
        (-logprob.mean()).backward()
        head = ppo.noise_head
        for parameter in head.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert head.out_proj.weight.grad.abs().sum() > 0
        residual_grad = head.reference_residual_proj.weight.grad
        # A zero output head blocks upstream nonzero gradients only on update 0.
        assert bool(residual_grad.abs().sum() > 0) == (update > 0)
        assert ppo.base_actor.model.plan_proj.weight.grad.abs().sum() > 0
        assert ppo.base_actor.model.scale.grad.abs() > 0
        assert not ppo.base_actor.plan_predictor.training
        assert all(not p.requires_grad and p.grad is None for p in ppo.base_actor.plan_predictor.parameters())
        assert all(p.grad is None for p in ppo.base_actor.obs_encoder.parameters())
        optimizer.step()
    for key, value in frozen_plan.items():
        torch.testing.assert_close(value, ppo.base_actor.plan_predictor.state_dict()[key], atol=0, rtol=0)
    optimizer.zero_grad(set_to_none=True)
    detached_logprob, _ = ppo.get_logprobs(obs, chains, learn_exploration_noise=False)
    (-detached_logprob.mean()).backward()
    assert all(p.grad is None for p in ppo.noise_head.parameters())
    assert ppo.base_actor.model.plan_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("options,error", [
    ({"noise_head_type": "fixed"}, "requires noise_head_type=residual_schedule"),
    ({"noise_conditioning_mode": "unknown"}, "Unsupported noise_conditioning_mode"),
])
def test_residual_only_rejects_invalid_configuration(options, error):
    with pytest.raises(ValueError, match=error):
        _residual_only_ppo(**options)


def test_residual_only_requires_pretrained_actor_reference_predictor():
    actor = _TinyPlanActor()
    actor.action_plan_enabled = False
    with pytest.raises(ValueError, match="requires action_plan.enabled=true"):
        PPOManiFlowPointcloud(
            actor, noise_head_type="residual_schedule",
            noise_conditioning_mode="reference_residual_only", actor_old_device="cpu",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("activation", ["tanh", "mish", "relu"])
def test_reinflow_xt_keeps_legacy_variance_mapping_and_appends_the_full_action_chunk(dtype, activation):
    torch.manual_seed(51)
    head = ActionTimeConditionedNoiseHead(
        cond_dim=3, action_horizon=16, action_dim=14, time_dim=8,
        hidden_dims=(8, 8), min_std=0.01, max_std=0.04, activation=activation,
    ).to(dtype=dtype)
    cond = torch.randn(2, 3, dtype=dtype)
    time = torch.tensor([0.0, 0.5], dtype=dtype)
    # Noncontiguous X_t must retain the same flattening order on rollout/recompute.
    x = torch.randn(2, 14, 16, dtype=dtype).transpose(1, 2)
    snapshots = [value.clone() for value in (cond, time, x)]
    inputs = []
    handle = head.mlp_logvar.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    sigma = head(cond, time, x)
    handle.remove()
    features = torch.cat([head.time_embedding(time), cond, x.reshape(2, -1)], dim=-1)
    torch.testing.assert_close(inputs[0], features, atol=0, rtol=0)
    assert head.mlp_logvar[0].in_features == 8 + 3 + 16 * 14
    raw = head.mlp_logvar(features)
    low = torch.tensor(0.01**2, dtype=dtype).log()
    high = torch.tensor(0.04**2, dtype=dtype).log()
    expected = (0.5 * (low + (high - low) * (raw.tanh() + 1) / 2)).exp()
    torch.testing.assert_close(sigma, expected, atol=0, rtol=0)
    assert sigma.shape == (2, 16 * 14)
    assert torch.isfinite(sigma).all() and sigma.min() >= 0.01 and sigma.max() <= 0.04
    for value, snapshot in zip((cond, time, x), snapshots):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


def test_reinflow_xt_numerically_depends_on_xt_and_preserves_its_gradient():
    head = ActionTimeConditionedNoiseHead(
        cond_dim=3, action_horizon=4, action_dim=2, time_dim=8,
        hidden_dims=(), min_std=0.01, max_std=0.04,
    )
    with torch.no_grad():
        head.mlp_logvar[0].weight.zero_()
        head.mlp_logvar[0].bias.zero_()
        head.mlp_logvar[0].weight[0, 8 + 3] = 1.0
    cond = torch.zeros(2, 3)
    time = torch.zeros(2)
    x = torch.zeros(2, 4, 2, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 0, 0] = 1.0
    sigma = head(cond, time, x)
    changed_sigma = head(cond, time, changed)
    assert torch.all(changed_sigma[:, 0] > sigma[:, 0])
    gradient, = torch.autograd.grad(sigma[:, 0].sum(), (x,))
    assert torch.isfinite(gradient).all() and torch.all(gradient[:, 0, 0] > 0)


@pytest.mark.parametrize("shape", [(1, 4, 2), (2, 3, 2), (2, 4, 3), (2, 8)])
def test_reinflow_xt_rejects_wrong_action_shapes(shape):
    head = ActionTimeConditionedNoiseHead(3, 4, 2, hidden_dims=(8,))
    with pytest.raises(ValueError, match="Expected x shape"):
        head(torch.randn(2, 3), torch.zeros(2), torch.randn(*shape))


def test_reinflow_xt_rejects_wrong_observation_feature_shape():
    head = ActionTimeConditionedNoiseHead(3, 4, 2, hidden_dims=(8,))
    with pytest.raises(ValueError, match="Expected cond_emb shape"):
        head(torch.randn(2, 4), torch.zeros(2), torch.randn(2, 4, 2))


def _reinflow_xt_ppo(plan_enabled=True, **overrides):
    actor = _TinyPlanActor()
    if not plan_enabled:
        actor.action_plan_enabled = False
        actor.plan_predictor = None
    options = dict(
        inference_steps=3, ft_denoising_steps=3,
        noise_head_type="reinflow_xt", noise_conditioning_mode="default",
        noise_hidden_dims=(8, 8), noise_time_dim=8,
        min_logprob_denoising_std=0.01, max_logprob_denoising_std=0.04,
        actor_old_device="cpu", freeze_obs_encoder=True,
    )
    options.update(overrides)
    return PPOManiFlowPointcloud(actor, **options)


@pytest.mark.parametrize("plan_enabled", [False, True])
@pytest.mark.parametrize("learn_noise", [False, True])
@pytest.mark.parametrize("ft_steps", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_reinflow_xt_rollout_and_logprob_recompute_use_the_same_current_xt(
    plan_enabled, learn_noise, ft_steps, normalize,
):
    torch.manual_seed(52)
    ppo = _reinflow_xt_ppo(
        plan_enabled=plan_enabled, ft_denoising_steps=ft_steps,
        normalize_denoising_horizon=normalize, normalize_act_space_dimension=normalize,
        account_for_initial_stochasticity=normalize,
    ).train()
    calls = []
    handle = ppo.noise_head.register_forward_pre_hook(
        lambda module, args: calls.append(tuple(value.detach().clone() for value in args))
    )
    obs = {"point_cloud": torch.randn(2, 2, 5, 3), "agent_pos": torch.randn(2, 2, 3)}
    action, chains, logprob = ppo.get_actions(obs)
    chain_snapshot = chains.clone()
    rng_snapshot = torch.random.get_rng_state()
    recomputed, entropy, _ = ppo.get_logprobs(
        obs, chains, get_entropy=True, learn_exploration_noise=learn_noise,
    )
    handle.remove()
    assert action.shape == (2, 2, 2) and chains.shape == (2, 4, 4, 2)
    assert len(calls) == 2 * ft_steps
    for index, step in enumerate(range(3 - ft_steps, 3)):
        rollout_call = calls[index]
        recompute_call = calls[index + ft_steps]
        assert len(rollout_call) == 3  # observation, time, X_t; no plan/reference input.
        torch.testing.assert_close(rollout_call[2], chains[:, step], atol=0, rtol=0)
        torch.testing.assert_close(rollout_call[1], torch.full((2,), step / 3), atol=0, rtol=0)
        for rollout_value, recompute_value in zip(rollout_call, recompute_call):
            torch.testing.assert_close(rollout_value, recompute_value, atol=0, rtol=0)
    torch.testing.assert_close(chains, chain_snapshot, atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_snapshot)
    torch.testing.assert_close(logprob, recomputed, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(entropy).all()
    for key, value in ppo.last_action_noise_stats.items():
        torch.testing.assert_close(value, ppo.last_logprob_noise_stats[key], atol=1e-7, rtol=1e-6)
    (-recomputed.mean()).backward()
    head_gradients = [p.grad for p in ppo.noise_head.parameters()]
    if learn_noise:
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in head_gradients)
        assert ppo.noise_head.mlp_logvar[-1].weight.grad.abs().sum() > 0
    else:
        assert all(gradient is None for gradient in head_gradients)
    assert ppo.base_actor.model.scale.grad.abs() > 0
    assert all(p.grad is None for p in ppo.base_actor.obs_encoder.parameters())
    if plan_enabled:
        assert not ppo.base_actor.plan_predictor.training
        assert all(not p.requires_grad and p.grad is None for p in ppo.base_actor.plan_predictor.parameters())


def test_reinflow_xt_does_not_use_residual_schedule_or_backbone_settings():
    ppo = _reinflow_xt_ppo(
        noise_backbone="tcn", action_conditioned_noise=False,
        base_sigma_schedule="linear", base_sigma_min=0.001, base_sigma_max=0.9,
        sigma_min=0.08, sigma_max=0.09, residual_scale=0.0,
    )
    assert isinstance(ppo.noise_head, ActionTimeConditionedNoiseHead)
    assert not any(isinstance(module, nn.Conv1d) for module in ppo.noise_head.modules())
    cond = torch.randn(2, 3)
    time = torch.full((2,), 0.5)
    x = torch.randn(2, 4, 2)
    expected = ppo.noise_head(cond, time, x).reshape_as(x)
    actual, stats = ppo._noise_std(cond, time, x, 1, True, return_stats=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.min() >= 0.01 and actual.max() <= 0.04
    assert actual.max() < ppo.sigma_min  # Residual-only clamp does not apply.
    assert stats["delta_abs_mean"] == 0
    assert stats["clamp_ratio_min"] == stats["clamp_ratio_max"] == 0


@pytest.mark.parametrize("mode", ["plan", "reference_flow", "reference_residual_only"])
def test_reinflow_xt_requires_default_noise_conditioning(mode):
    with pytest.raises(ValueError, match="requires noise_head_type=residual_schedule"):
        _reinflow_xt_ppo(noise_conditioning_mode=mode)


def test_reinflow_xt_strict_state_roundtrip_and_incompatible_legacy_noise_weights():
    source = _reinflow_xt_ppo()
    target = _reinflow_xt_ppo()
    incompatible = target.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    assert incompatible.missing_keys == incompatible.unexpected_keys == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], atol=0, rtol=0)
    legacy = _reinflow_xt_ppo(noise_head_type="fixed")
    with pytest.raises(RuntimeError, match="size mismatch for noise_head.mlp_logvar.0.weight"):
        target.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="size mismatch for noise_head.mlp_logvar.0.weight"):
        legacy.load_state_dict(source.state_dict(), strict=True)


def test_legacy_fixed_noise_head_still_ignores_xt_and_preserves_its_state_layout():
    ppo = _reinflow_xt_ppo(noise_head_type="fixed", noise_backbone="tcn")
    assert type(ppo.noise_head) is TimeConditionedNoiseHead
    head = ppo.noise_head
    assert head.mlp_logvar[0].in_features == 3 + 8
    reference = TimeConditionedNoiseHead(3, 4, 2, time_dim=8, hidden_dims=(8, 8), max_std=0.04)
    incompatible = reference.load_state_dict(copy.deepcopy(head.state_dict()), strict=True)
    assert incompatible.missing_keys == incompatible.unexpected_keys == []
    cond = torch.randn(2, 3)
    time = torch.zeros(2)
    x = torch.randn(2, 4, 2)
    sigma1 = ppo._noise_std(cond, time, x, 0, True)
    sigma2 = ppo._noise_std(cond, time, x * 100, 0, True)
    torch.testing.assert_close(sigma1, sigma2, atol=0, rtol=0)
    torch.testing.assert_close(sigma1.flatten(1), reference(cond, time), atol=0, rtol=0)


def test_reinflow_xt_ppo_update_optimizer_and_resume_roundtrip():
    torch.manual_seed(53)
    source = _reinflow_xt_ppo()
    config = OmegaConf.create({
        "actor_lr": 1e-4, "noise_lr": 3e-4,
        "actor_weight_decay": 0.0, "betas": [0.9, 0.95],
    })
    optimizer = build_actor_optimizer(source, config)
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == {
        id(p) for p in source.noise_head.parameters()
    }

    class TinyCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Linear(3, 1)

        def forward(self, observations):
            return self.value(observations["agent_pos"].mean(dim=1))

    obs = {"point_cloud": torch.randn(2, 2, 5, 3), "agent_pos": torch.randn(2, 2, 3)}
    _, chains, oldlogprobs = source.get_actions(obs)
    before = {key: value.clone() for key, value in source.noise_head.state_dict().items()}
    loss = source.loss(
        obs, chains, returns=torch.tensor([1.0, -0.5]), oldvalues=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]), oldlogprobs=oldlogprobs,
        critic=TinyCritic(),
    )
    (loss["pg_loss"] + 0.5 * loss["value_loss"]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in source.noise_head.parameters())
    action_columns = source.noise_head.mlp_logvar[0].weight.grad[:, 8 + 3:]
    assert action_columns.abs().sum() > 0
    optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in source.noise_head.state_dict().items())

    payload = io.BytesIO()
    torch.save({"ppo_actor": source.state_dict(), "actor_optimizer": optimizer.state_dict()}, payload)
    payload.seek(0)
    saved = torch.load(payload, map_location="cpu", weights_only=True)
    target = _reinflow_xt_ppo()
    target.load_state_dict(saved["ppo_actor"], strict=True)
    restored_optimizer = build_actor_optimizer(target, config)
    assert not load_actor_optimizer_state(restored_optimizer, saved["actor_optimizer"])
    for source_parameter, target_parameter in zip(source.actor_parameters(), target.actor_parameters()):
        torch.testing.assert_close(source_parameter, target_parameter, atol=0, rtol=0)
        for key, value in optimizer.state[source_parameter].items():
            torch.testing.assert_close(value, restored_optimizer.state[target_parameter][key], atol=0, rtol=0)
    expected, _ = source.get_logprobs(obs, chains)
    actual, _ = target.get_logprobs(obs, chains)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_reinflow_xt_can_be_selected_through_hydra_without_changing_other_experiments():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "maniflow/config"
    default = OmegaConf.load(config_dir / "reinflow_rl_pointcloud_robotwin2.yaml")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="reinflow_rl_pointcloud_robotwin2", overrides=[
            "rl.actor.noise_head_type=reinflow_xt", "rl.actor.noise_conditioning_mode=default",
            "rl.actor.actor_old_device=cpu", "policy.action_plan.target_mode=identity",
        ])
    assert config.rl.actor.noise_backbone == default.rl.actor.noise_backbone
    assert config.rl.actor.base_sigma_schedule == default.rl.actor.base_sigma_schedule
    ppo = PPOManiFlowPointcloud(
        _TinyPlanActor(), **OmegaConf.to_container(config.rl.actor, resolve=True),
    )
    assert isinstance(ppo.noise_head, ActionTimeConditionedNoiseHead)
    assert ppo.noise_head.mlp_logvar[0].in_features == 3 + config.rl.actor.noise_time_dim + 4 * 2
    std = ppo._noise_std(torch.randn(2, 3), torch.zeros(2), torch.randn(2, 4, 2), 0, True)
    assert std.shape == (2, 4, 2) and torch.isfinite(std).all()
