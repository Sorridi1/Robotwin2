"""RAFE contains ordinary Flow + TCN sigma, without action-reference modules."""
import copy
import inspect
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.rl.ppo_maniflow_pointcloud import PPOManiFlowPointcloud, ActionConditionedResidualNoiseHead
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy


def tiny_policy_kwargs():
    return dict(
        shape_meta=OmegaConf.create({
            "obs": {"point_cloud": {"shape": [8, 3]}, "agent_pos": {"shape": [3]}},
            "action": {"shape": [2]},
        }),
        horizon=4, n_action_steps=2, n_obs_steps=2, num_inference_steps=2,
        visual_cond_len=8, n_layer=1, n_head=2, n_emb=16,
        encoder_output_dim=8, use_pc_color=False,
        pointcloud_encoder_cfg=OmegaConf.create({
            "in_channels": 3, "out_channels": 8, "use_layernorm": True,
            "final_norm": "layernorm", "normal_channel": False,
            "num_points": 8, "pointwise": True,
        }),
    )


def make_real_policy(**overrides):
    kwargs = tiny_policy_kwargs()
    kwargs.update(overrides)
    policy = ManiFlowTransformerPointcloudPolicy(**kwargs)
    policy.normalizer.fit({
        "point_cloud": torch.randn(16, 8, 3),
        "agent_pos": torch.randn(16, 3),
        "action": torch.randn(16, 2),
    }, last_n_dims=1)
    policy.normalizer.requires_grad_(False)
    return policy


def test_real_policy_and_noise_head_have_no_action_reference_modules_or_interfaces():
    policy = make_real_policy()
    adapter = PPOManiFlowPointcloud(
        policy, noise_head_type="residual_schedule", noise_backbone="tcn",
        noise_model_dim=8, noise_action_embed_dim=4, noise_time_dim=4,
        noise_hidden_dims=(8,), actor_old_device="cpu",
    )
    forbidden_attrs = {"action_plan_enabled", "plan_predictor", "plan_proj", "coarse_action_operator",
                       "predict_action_plan", "get_plan_condition", "build_coarse_action_target"}
    for module in (policy, policy.model, adapter, adapter.noise_head):
        assert not forbidden_attrs.intersection(vars(module))
        assert all(not hasattr(module, name) for name in forbidden_attrs)
        assert not any("plan" in name or "reference_residual" in name for name, _ in module.named_modules())
        assert not any("plan" in key or "reference_residual" in key for key in module.state_dict())
    for function in (ManiFlowTransformerPointcloudPolicy.__init__, DiTX.__init__, DiTX.forward,
                     PPOManiFlowPointcloud.gaussian_transition, PPOManiFlowPointcloud._velocity,
                     ActionConditionedResidualNoiseHead.forward, ActionConditionedResidualNoiseHead._build_tokens):
        parameters = inspect.signature(function).parameters
        assert not {"action_plan", "plan_cond", "plan_dim", "reference_residual"}.intersection(parameters)


def test_full_real_policy_supervised_and_ppo_nll_smoke_without_reference():
    policy = make_real_policy()
    batch = {
        "obs": {"point_cloud": torch.randn(4, 2, 8, 3), "agent_pos": torch.randn(4, 2, 3)},
        "action": torch.randn(4, 4, 2),
    }
    loss, metrics = policy.compute_loss(batch, ema_model=copy.deepcopy(policy).eval())
    assert torch.isfinite(loss)
    assert "loss_plan" not in metrics
    predictions = policy.eval().predict_action(batch["obs"])
    assert set(predictions) == {"action", "action_pred"}
    assert predictions["action"].shape == (4, 2, 2)
    assert predictions["action_pred"].shape == (4, 4, 2)
    adapter = PPOManiFlowPointcloud(
        policy, inference_steps=2, ft_denoising_steps=2, noise_head_type="residual_schedule",
        noise_backbone="tcn", noise_model_dim=8, noise_action_embed_dim=4,
        noise_time_dim=4, noise_hidden_dims=(8,), actor_old_device="cpu",
    ).eval()
    _, chain, original_logprob = adapter.get_actions(batch["obs"])
    new_logprob, _ = adapter.get_logprobs(batch["obs"], chain, learn_exploration_noise=False)
    torch.testing.assert_close(new_logprob, original_logprob)
    nll, _ = adapter.compute_expert_interpolation_noise_loss(batch)
    nll.backward()
    assert all(p.grad is None for p in policy.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in adapter.noise_head.parameters())


def test_original_maniflow_no_reference_checkpoint_strictly_loads_and_predicts_identically(tmp_path):
    """Generate weights with original ManiFlow in a clean process to avoid import aliasing."""
    original_package = Path(__file__).resolve().parents[2] / "ManiFlow" / "ManiFlow"
    script = '''
import sys
import torch
from omegaconf import OmegaConf
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy

torch.manual_seed(409)
kwargs = dict(
 shape_meta=OmegaConf.create({"obs":{"point_cloud":{"shape":[8,3]},"agent_pos":{"shape":[3]}},"action":{"shape":[2]}}),
 horizon=4,n_action_steps=2,n_obs_steps=2,num_inference_steps=2,
 visual_cond_len=8,n_layer=1,n_head=2,n_emb=16,encoder_output_dim=8,use_pc_color=False,
 pointcloud_encoder_cfg=OmegaConf.create({"in_channels":3,"out_channels":8,"use_layernorm":True,"final_norm":"layernorm","normal_channel":False,"num_points":8,"pointwise":True}),
 action_plan={"enabled":False})
policy=ManiFlowTransformerPointcloudPolicy(**kwargs).eval()
policy.normalizer.fit({"point_cloud":torch.randn(16,8,3),"agent_pos":torch.randn(16,3),"action":torch.randn(16,2)},last_n_dims=1)
obs={"point_cloud":torch.randn(2,2,8,3),"agent_pos":torch.randn(2,2,3)}
torch.manual_seed(918)
with torch.no_grad():
 prediction=policy.predict_action(obs)
torch.save({"state":policy.state_dict(),"obs":obs,"prediction":prediction},sys.argv[1])
'''
    path = tmp_path / "original_maniflow.ckpt"
    environment = dict(os.environ, PYTHONPATH=str(original_package))
    completed = subprocess.run([sys.executable, "-c", script, str(path)], cwd=tmp_path,
                               env=environment, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = make_real_policy().eval()
    incompatible = policy.load_state_dict(payload["state"], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    torch.manual_seed(918)
    with torch.no_grad():
        predicted = policy.predict_action(payload["obs"])
    for key in payload["prediction"]:
        torch.testing.assert_close(predicted[key], payload["prediction"][key], rtol=0, atol=0)


def test_old_disabled_configuration_metadata_is_discarded_without_creating_modules():
    policy = make_real_policy(action_plan={"enabled": False, "predictor_hidden_dims": [8]})
    assert not hasattr(policy, "plan_predictor")
    assert not hasattr(policy, "action_plan_enabled")
    normal = make_real_policy()
    result = normal.load_state_dict(policy.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys


@pytest.mark.parametrize("metadata", [{"enabled": True}, {"predictor_hidden_dims": [8]}, True])
def test_enabled_or_ambiguous_legacy_reference_configuration_is_rejected(metadata):
    with pytest.raises((TypeError, ValueError), match="plan|reference|Reference"):
        make_real_policy(action_plan=metadata)


def test_legacy_reference_checkpoint_and_injection_are_explicitly_rejected():
    policy = make_real_policy().eval()
    state = policy.state_dict()
    state["plan_predictor.weight"] = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="plan_predictor"):
        policy.load_state_dict(state, strict=True)
    sample = torch.randn(2, 4, 2)
    with pytest.raises(TypeError, match="plan_cond|reference|Reference"):
        policy.model(sample, torch.zeros(2), torch.ones(2), torch.randn(2, 16, policy.obs_feature_dim),
                     plan_cond=torch.randn_like(sample))
