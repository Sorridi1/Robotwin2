"""A-D: interpolation, likelihood math, isolation, and shared transition semantics."""
import math

import pytest
import torch
from torch.distributions import Normal

from helpers import make_actor, expert_batch, TinyCritic, clone_parameters, parameters_equal
from maniflow.model.rl.ppo_maniflow_pointcloud import (
    compute_expert_interpolation_state, diagonal_gaussian_nll,
)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 4, 2), (2, 16, 14)])
def test_interpolation_endpoints_and_adjacent_states(shape):
    x0, action = torch.randn(shape), torch.randn(shape)
    torch.testing.assert_close(compute_expert_interpolation_state(x0, action, 0.0), x0)
    torch.testing.assert_close(compute_expert_interpolation_state(x0, action, 1.0), action)
    # One X0 and one complete action chunk define the entire path.
    for step in range(5):
        xt = compute_expert_interpolation_state(x0, action, step / 5)
        xn = compute_expert_interpolation_state(x0, action, (step + 1) / 5)
        torch.testing.assert_close(xn - xt, (action - x0) / 5, atol=3e-7, rtol=1e-5)


def test_gaussian_nll_analytic_optimum_and_mean_stop_gradient():
    mean = torch.tensor([0.2], requires_grad=True)
    target = torch.tensor([0.8])
    log_sigma = torch.tensor([math.log(0.6)], requires_grad=True)
    sigma = log_sigma.exp()
    loss = diagonal_gaussian_nll(target, mean, sigma).mean()
    expected = Normal(mean.detach(), sigma).log_prob(target).neg().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert mean.grad is None
    assert abs(float(log_sigma.grad)) < 2e-6
    for std, sign in [(0.2, -1), (1.2, 1)]:
        raw = torch.tensor([math.log(std)], requires_grad=True)
        diagonal_gaussian_nll(target, mean, raw.exp()).mean().backward()
        assert float(raw.grad) * sign > 0


@pytest.mark.parametrize("bound", [0.02, 0.4, 0.8])
def test_gaussian_nll_finite_at_bounds_and_clamped_optimum(bound):
    mean, target = torch.tensor([0.2]), torch.tensor([0.8])
    raw = torch.tensor([math.log(bound)], requires_grad=True)
    sigma = raw.exp().clamp(0.02, 0.4)
    loss = diagonal_gaussian_nll(target, mean, sigma).mean()
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, -Normal(mean, sigma).log_prob(target).mean())
    if bound > 0.4:
        loss.backward()
        assert raw.grad == 0  # clamp saturation is observable, not hidden by a penalty.
    grid = torch.linspace(0.02, 0.4, 100)
    values = diagonal_gaussian_nll(target.expand_as(grid), mean.expand_as(grid), grid)
    assert int(values.argmin()) == len(grid) - 1  # |residual|=.6 exceeds upper bound.


def test_ppo_and_nll_gradients_are_disjoint():
    actor = make_actor()
    batch = expert_batch()
    _, chains, old_logprob = actor.get_actions(batch["obs"])
    critic = TinyCritic()
    terms = actor.loss(
        batch["obs"], chains, torch.tensor([1.0, -0.4, 0.8]),
        torch.zeros(3), torch.tensor([1.0, -0.8, 0.5]), old_logprob,
        critic=critic, normalize_advantages=False, learn_exploration_noise=False,
    )
    (terms["pg_loss"] + terms["value_loss"]).backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in actor.base_actor.model.parameters())
    assert all(p.grad is None for p in actor.noise_head.parameters())
    assert all(p.grad is None for p in actor.base_actor.obs_encoder.parameters())
    assert all(not p.requires_grad for p in actor.base_actor.obs_encoder.parameters())
    assert all(p.grad is None for p in actor.base_actor.normalizer.parameters())

    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    before_actor, before_critic = clone_parameters(actor.base_actor), clone_parameters(critic)
    before_noise = clone_parameters(actor.noise_head)
    loss, metrics = actor.compute_expert_interpolation_noise_loss(batch)
    loss.backward()
    assert all(p.grad is None for p in actor.base_actor.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in actor.noise_head.parameters())
    optimizer = torch.optim.Adam(actor.noise_head.parameters(), lr=1e-3)
    optimizer.step()
    assert parameters_equal(actor.base_actor, before_actor)
    assert parameters_equal(critic, before_critic)
    assert not parameters_equal(actor.noise_head, before_noise)
    assert torch.isfinite(loss)
    assert "noise_residual_mse" in metrics


@pytest.mark.parametrize("target_mode", ["relative", "absolute"])
@pytest.mark.parametrize("clip", [None, 0.1])
def test_sampling_logprob_and_expert_nll_share_transition(monkeypatch, target_mode, clip):
    actor = make_actor(denoised_clip_value=clip)
    with torch.no_grad():
        actor.noise_head.out_proj.bias.copy_(torch.tensor([0.4, -0.8]))
    actor.base_actor.sample_target_t_mode = target_mode
    batch = expert_batch()
    recorded = []
    original = actor.gaussian_transition

    def record(x, step, vis_cond, cond_emb, **kwargs):
        result = original(x, step, vis_cond, cond_emb, **kwargs)
        recorded.append((step, x.detach().clone(), result[0].detach().clone(), result[1].detach().clone()))
        return result

    monkeypatch.setattr(actor, "gaussian_transition", record)
    _, chains, sampled_logprob = actor.get_actions(batch["obs"])
    sampled = list(recorded)
    recorded.clear()
    computed_logprob, _ = actor.get_logprobs(batch["obs"], chains, learn_exploration_noise=False)
    torch.testing.assert_close(computed_logprob, sampled_logprob, atol=1e-5, rtol=1e-5)
    for rollout, reevaluated in zip(sampled, recorded):
        assert rollout[0] == reevaluated[0]
        for left, right in zip(rollout[1:], reevaluated[1:]):
            torch.testing.assert_close(left, right)

    recorded.clear()
    x0 = torch.randn_like(batch["action"])
    loss, _ = actor.compute_expert_interpolation_noise_loss(batch, initial_noise=x0)
    assert [item[0] for item in recorded] == [1, 2]  # fixed-noise step zero excluded.
    normalized_action = actor.base_actor.normalizer["action"].normalize(batch["action"])
    vis_cond, cond_emb = actor._encode_obs(batch["obs"])
    nll_values = []
    for step, expert_x, expert_mean, expert_sigma in recorded:
        time = torch.full((len(x0),), step / actor.inference_steps)
        expected_x = compute_expert_interpolation_state(x0, normalized_action, time)
        target = compute_expert_interpolation_state(x0, normalized_action, (step + 1) / actor.inference_steps)
        torch.testing.assert_close(expert_x, expected_x)
        mean, sigma, _ = original(expert_x, step, vis_cond, cond_emb, learn_exploration_noise=False)
        torch.testing.assert_close(mean, expert_mean)
        torch.testing.assert_close(sigma, expert_sigma)
        velocity = actor._velocity(actor.base_actor, expert_x, time, vis_cond)
        expected_mean = expert_x + velocity / actor.inference_steps
        if clip is not None:
            expected_mean = expected_mean.clamp(-clip, clip)
        torch.testing.assert_close(mean, expected_mean)
        nll_values.append(-Normal(mean.detach(), sigma).log_prob(target).mean())
        recorded_t, target_t = actor.base_actor.model.calls[-1]
        expected_target = time + 1 / 3 if target_mode == "absolute" else torch.full_like(time, 1 / 3)
        torch.testing.assert_close(recorded_t, time)
        torch.testing.assert_close(target_t, expected_target)
    torch.testing.assert_close(loss, torch.stack(nll_values).mean())


def test_sampling_floor_and_sigma_parameterization():
    actor = make_actor(min_sampling_denoising_std=0.22)
    batch = expert_batch()
    vis, cond = actor._encode_obs(batch["obs"])
    x = torch.randn_like(batch["action"])
    for step in [1, 2]:
        _, sigma, _ = actor.gaussian_transition(x, step, vis, cond)
        time = torch.full((len(x),), step / 3)
        base = actor.compute_base_sigma(time, x, step)
        delta = actor.noise_head.predict_delta(cond, time, x)
        expected = (base.float().log() + delta.float()).clamp(
            math.log(max(actor.sigma_min, 0.22)),
            math.log(max(actor.sigma_max, 0.22)),
        ).exp()
        torch.testing.assert_close(sigma, expected)
        assert sigma.shape == x.shape
        assert bool((sigma > 0).all())


def test_sample_one_step_only_uses_learnable_steps(monkeypatch):
    actor, batch = make_actor(), expert_batch()
    steps = []
    original = actor.gaussian_transition

    def record(x, step, vis_cond, cond_emb, **kwargs):
        steps.append(step)
        return original(x, step, vis_cond, cond_emb, **kwargs)

    monkeypatch.setattr(actor, "gaussian_transition", record)
    for _ in range(8):
        actor.compute_expert_interpolation_noise_loss(batch, sample_all_steps=False)
    assert len(steps) == 8
    assert set(steps) <= {1, 2}


def test_nll_detaches_observation_encoder_even_if_actor_config_unfreezes_it():
    actor = make_actor(freeze_obs_encoder=False)
    assert all(p.requires_grad for p in actor.base_actor.obs_encoder.parameters())
    loss, _ = actor.compute_expert_interpolation_noise_loss(expert_batch())
    loss.backward()
    assert all(p.grad is None for p in actor.base_actor.parameters())
    assert any(p.grad is not None for p in actor.noise_head.parameters())


def test_hard_sigma_clamp_reports_saturation_and_finite_loss():
    actor = make_actor(sigma_max=0.1)
    loss, metrics = actor.compute_expert_interpolation_noise_loss(expert_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["noise_sigma_at_upper_bound_ratio"] == 1
    assert metrics["noise_sigma_clamp_saturation_ratio"] == 1
    assert metrics["noise_residual_above_representable_ratio"] > 0
    assert all(p.grad is None or not bool(p.grad.abs().sum()) for p in actor.noise_head.parameters())


def test_zero_finetuned_steps_and_truncated_actions_are_rejected():
    batch = expert_batch()
    with pytest.raises(ValueError, match="no learnable"):
        make_actor(ft_denoising_steps=0).compute_expert_interpolation_noise_loss(batch)
    batch["action"] = batch["action"][:, :2]
    with pytest.raises(ValueError, match="full raw chunk"):
        make_actor().compute_expert_interpolation_noise_loss(batch)
