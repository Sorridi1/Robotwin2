"""Unbounded log residuals retain only explicit positive sigma bounds."""
import inspect
import math

import pytest
import torch

from helpers import make_actor, expert_batch
from maniflow.model.rl.ppo_maniflow_pointcloud import ActionConditionedResidualNoiseHead, PPOManiFlowPointcloud


def constant_raw_output(actor, value):
    with torch.no_grad():
        actor.noise_head.out_proj.weight.zero_()
        actor.noise_head.out_proj.bias.fill_(value)


def inputs(actor, batch_size=3):
    batch = expert_batch(batch_size)
    vis, cond = actor._encode_obs(batch["obs"])
    x = torch.randn_like(batch["action"]).to(device=actor.device, dtype=actor.dtype)
    t = torch.full((batch_size,), 2 / 3, device=actor.device, dtype=actor.dtype)
    return batch, vis, cond, x, t


def test_default_log_residual_gain_is_one():
    assert inspect.signature(ActionConditionedResidualNoiseHead.__init__).parameters["residual_scale"].default == 1.0
    assert inspect.signature(PPOManiFlowPointcloud.__init__).parameters["residual_scale"].default == 1.0
    actor = make_actor()
    assert actor.noise_parameterization == "log_residual"


@pytest.mark.parametrize("raw", [-1.5, 1.5])
@pytest.mark.parametrize("gain", [0.5, 1.0])
def test_raw_residual_is_linear_reaches_beyond_old_bound_and_has_unsaturated_gradient(raw, gain):
    actor = make_actor(residual_scale=gain)
    # Keep the effective shift beyond the former +/-gain interval for both gains.
    unscaled = raw / gain
    constant_raw_output(actor, unscaled)
    _, _, cond, x, t = inputs(actor)
    raw_output = actor.noise_head.forward_raw(cond, t, x)
    delta = actor.noise_head.predict_delta(cond, t, x)
    torch.testing.assert_close(delta, raw_output.float() * gain)
    sigma, info = actor.compute_residual_sigma(cond, t, x, step=2)
    base = actor.compute_base_sigma(t, x, step=2)
    expected = (base.float().log() + raw).exp()
    torch.testing.assert_close(sigma, expected)
    if raw > 0:
        assert bool((sigma > base * math.exp(gain)).all())
    else:
        assert bool((sigma < base * math.exp(-gain)).all())
    assert info["clamp_ratio_min"] == 0 and info["clamp_ratio_max"] == 0
    sigma.log().sum().backward()
    expected_grad = torch.full_like(actor.noise_head.out_proj.bias, len(x) * actor.horizon * gain)
    torch.testing.assert_close(actor.noise_head.out_proj.bias.grad, expected_grad)


@pytest.mark.parametrize("raw,expected_boundary", [(-1e4, "lower"), (1e4, "upper")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_log_clamp_precedes_exponential_for_extreme_outputs(raw, expected_boundary, dtype):
    actor = make_actor()
    constant_raw_output(actor, raw)
    batch, _, cond, x, t = inputs(actor)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        assert actor.noise_head.forward_raw(cond, t, x).dtype == dtype
        sigma, info = actor.compute_residual_sigma(cond, t, x, step=2)
    assert sigma.dtype == torch.float32
    assert bool(torch.isfinite(sigma).all()) and bool((sigma > 0).all())
    expected = max(actor.sigma_min, actor.min_sampling_denoising_std) if expected_boundary == "lower" else max(actor.sigma_max, actor.min_sampling_denoising_std)
    torch.testing.assert_close(sigma, torch.full_like(sigma, expected))
    assert info["clamp_ratio_min" if expected_boundary == "lower" else "clamp_ratio_max"] == 1
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        loss, metrics = actor.compute_expert_interpolation_noise_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.noise_head.parameters())
    assert all(p.grad is None for p in actor.base_actor.parameters())
    assert metrics["noise_sigma_clamp_saturation_ratio"] == 1


def test_global_sigma_bounds_are_representable_and_no_tanh_saturation_metrics_remain():
    actor = make_actor()
    batch, _, cond, x, t = inputs(actor)
    _, _, transition_stats = actor.gaussian_transition(x, 2, *actor._encode_obs(batch["obs"]))
    loss, metrics = actor.compute_expert_interpolation_noise_loss(batch)
    assert torch.isfinite(loss)
    assert not any("tanh" in name for name in transition_stats)
    assert not any("tanh" in name for name in metrics)
    for step in [1, 2]:
        lower = metrics[f"noise_sigma_representable_min_by_step/{step}"]
        upper = metrics[f"noise_sigma_representable_max_by_step/{step}"]
        assert float(lower) == pytest.approx(max(actor.sigma_min, actor.min_sampling_denoising_std))
        assert float(upper) == pytest.approx(max(actor.sigma_max, actor.min_sampling_denoising_std))


def test_zero_gain_retains_fixed_schedule_even_with_large_raw_outputs():
    actor = make_actor(residual_scale=0)
    batch, vis, cond, x, t = inputs(actor)
    for raw in [-1e4, 1e4]:
        constant_raw_output(actor, raw)
        _, sigma, stats = actor.gaussian_transition(x, 2, vis, cond, learn_exploration_noise=True)
        base = actor.compute_base_sigma(t, x, step=2)
        expected = base.float().clamp(actor.sigma_min, actor.sigma_max).clamp_min(actor.min_sampling_denoising_std)
        torch.testing.assert_close(sigma, expected)
        assert not any("tanh" in name for name in stats)
        actor.zero_grad(set_to_none=True)
        sigma.log().mean().backward()
        assert all(p.grad is None or not bool(p.grad.abs().sum()) for p in actor.noise_head.parameters())


def test_wrapper_checkpoint_marks_noise_parameterization_and_rejects_old_or_changed_semantics():
    import copy
    actor = make_actor()
    state = copy.deepcopy(actor.state_dict())
    assert state["_extra_state"] == {"version": 1, "noise_parameterization": "log_residual", "residual_scale": 1.0}
    result = make_actor().load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    missing = copy.deepcopy(state)
    missing.pop("_extra_state")
    with pytest.raises(RuntimeError, match="_extra_state"):
        make_actor().load_state_dict(missing, strict=True)
    for changes in [{"version": 0}, {"noise_parameterization": "bounded_tanh"}, {"residual_scale": 0.5}]:
        mismatched = copy.deepcopy(state)
        mismatched["_extra_state"].update(changes)
        with pytest.raises((ValueError, RuntimeError), match="parameterization|sigma|residual|Incompatible|version"):
            make_actor().load_state_dict(mismatched, strict=True)


def test_unsupported_noise_parameterization_is_rejected():
    with pytest.raises(ValueError, match="parameterization|log_residual"):
        make_actor(noise_parameterization="bounded_tanh")
