"""E: actual expert Zarr loader, alternating optimizers and exact resume smoke test."""
import copy
from pathlib import Path

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import zarr

from helpers import TinyBaseActor, TinyCritic, clone_parameters, parameters_equal
from maniflow.common.expert_data import ExpertDataIterator, capture_rng_state
from maniflow.common.ppo_buffer_pointcloud import PPOBufferPointcloud
from maniflow.common.ppo_optimizer import build_actor_optimizer, load_actor_optimizer_state
from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace


@pytest.fixture
def tiny_config(tmp_path):
    policy_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(policy_root / "ManiFlow/maniflow/config/reinflow_rl_pointcloud_robotwin2.yaml")
    cfg.task = OmegaConf.load(policy_root / "ManiFlow/maniflow/config/task/demo_task.yaml")
    cfg.task_name, cfg.setting = "smoke", "smoke"
    cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps = 4, 2, 2
    cfg.shape_meta = {
        "action": {"shape": [2], "horizon": 4},
        "obs": {"point_cloud": {"shape": [5, 3], "type": "point_cloud"},
                "agent_pos": {"shape": [2], "type": "low_dim"}},
    }
    cfg.policy.pop("action_plan", None)
    cfg.policy.use_pc_color = False
    cfg.training.device = "cpu"
    cfg.training.seed = 17
    cfg.rl.actor.update({
        "inference_steps": 3, "ft_denoising_steps": 2,
        "noise_conditioning_mode": "default", "noise_model_dim": 8,
        "noise_action_embed_dim": 4, "noise_time_dim": 4,
        "noise_hidden_dims": [8], "tcn_num_layers": 1,
        "actor_old_device": "cpu", "base_sigma_min": 0.15,
        "base_sigma_max": 0.3, "sigma_min": 0.01, "sigma_max": 0.8,
        "min_sampling_denoising_std": 0.02,
        "denoised_clip_value": 1.0, "logprob_min": -10000, "logprob_max": 10000,
    })
    cfg.rl.ppo.update({
        "batch_size": 2, "grad_accum_steps": 2, "update_epochs": 2,
        "critic_warmup_iterations": 0, "actor_lr": 1e-3,
        "critic_lr": 1e-3, "use_amp": False, "target_kl": None,
        "reward_scale_running": False,
    })
    cfg.rl.bc_anchor.enabled = True
    cfg.rl.eval.enabled = False
    cfg.rl.noise_guidance = {
        "enabled": True, "mode": "expert_interpolation", "dataset_path": str(tmp_path / "expert.zarr"),
        "batch_size": 2, "update_interval": 1, "gradient_steps": 1,
        "learning_rate": 1e-3, "initial_warmup_steps": 0, "sample_all_steps": True,
        "log_diagnostics": True, "max_grad_norm": 1.0, "weight_decay": 0.0, "num_workers": 0,
    }
    cfg.task.dataset.val_ratio = 0.0
    cfg.task.dataset.pointcloud_color_aug_cfg.aug_color = False
    group = zarr.open_group(cfg.rl.noise_guidance.dataset_path, mode="w")
    data = group.create_group("data")
    generator = np.random.default_rng(83)
    data.array("point_cloud", generator.normal(size=(12, 5, 3)).astype(np.float32))
    data.array("state", generator.normal(size=(12, 2)).astype(np.float32))
    data.array("action", generator.normal(size=(12, 2)).astype(np.float32))
    group.create_group("meta").array("episode_ends", np.array([6, 12], dtype=np.int64))
    checkpoint = tmp_path / "pretrained.ckpt"
    torch.save({"state_dicts": {"ema_model": TinyBaseActor().state_dict()}}, checkpoint, pickle_module=dill)
    cfg.rl.pretrained_checkpoint_path = str(checkpoint)
    return cfg


@pytest.fixture
def workspace_factory(monkeypatch, tiny_config, tmp_path):
    original = hydra.utils.instantiate

    def instantiate(config, *args, **kwargs):
        target = config.get("_target_", "")
        if target.endswith("ManiFlowTransformerPointcloudPolicy"):
            return TinyBaseActor()
        if target.endswith("PointcloudCritic"):
            return TinyCritic()
        return original(config, *args, **kwargs)

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)

    def factory(name="run", cfg=None):
        return TrainReinFlowRLRoboTwinWorkspace(copy.deepcopy(tiny_config if cfg is None else cfg),
                                               output_dir=str(tmp_path / name))
    return factory


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            assert_nested_equal(one, two)
    else:
        assert left == right


def toy_rollout(workspace):
    """Two short vectorized steps, using real chains/logprobs and PPO buffer."""
    buffer = PPOBufferPointcloud(
        n_steps=2, n_envs=2, n_obs_steps=2, point_cloud_shape=(5, 3), agent_pos_shape=(2,),
        denoising_steps=3, horizon_steps=4, action_dim=2, device="cpu",
    )
    obs = {
        "point_cloud": np.ones((2, 2, 5, 3), dtype=np.float32) * 0.2,
        "agent_pos": np.array([[[0.1, 0.2], [0.2, 0.3]], [[-0.2, 0.1], [-0.1, 0.2]]], dtype=np.float32),
    }
    for step in range(2):
        tensors = {key: torch.from_numpy(value) for key, value in obs.items()}
        action, chain, logprob = workspace.ppo_actor.get_actions(tensors)
        with torch.no_grad():
            value = workspace.critic(tensors)
        buffer.add(step=step, obs=obs, chains=chain.numpy(),
                   reward=np.array([1.0, -0.1], dtype=np.float32) - (action.numpy() ** 2).mean(axis=(1, 2)),
                   terminated=np.array([step == 1, step == 1]), truncated=np.zeros(2, bool),
                   value=value.numpy(), logprob=logprob.numpy(), success=np.array([True, False]))
        obs = {key: value + 0.05 for key, value in obs.items()}
    workspace.current_obs = obs
    return buffer


def test_actual_dataset_loader_resume_and_independent_rng(tiny_config):
    outer = capture_rng_state()
    first = ExpertDataIterator(tiny_config)
    assert_nested_equal(outer, capture_rng_state())
    batch = next(first)
    assert tuple(batch["action"].shape) == (2, 4, 2)
    assert tuple(batch["obs"]["point_cloud"].shape) == (2, 4, 5, 3)
    assert_nested_equal(outer, capture_rng_state())
    state = copy.deepcopy(first.state_dict())
    expected = [next(first) for _ in range(10)]  # cross epoch boundary as well.
    restored = ExpertDataIterator(tiny_config)
    restored.load_state_dict(state)
    for expected_batch in expected:
        assert_nested_equal(next(restored), expected_batch)
    cfg = copy.deepcopy(tiny_config)
    cfg.rl.noise_guidance.batch_size = 1
    with pytest.raises(ValueError, match="differs"):
        ExpertDataIterator(cfg).load_state_dict(state)


def test_missing_dataset_fails_without_skipping_nll(tiny_config):
    tiny_config.rl.noise_guidance.dataset_path += ".missing"
    with pytest.raises(FileNotFoundError, match="NLL updates cannot be skipped"):
        ExpertDataIterator(tiny_config)


def test_full_noise_ppo_save_resume_cycle(workspace_factory):
    workspace = workspace_factory()
    before_actor = clone_parameters(workspace.model)
    before_noise = clone_parameters(workspace.ppo_actor.noise_head)
    metrics = workspace.noise_nll_update()
    assert metrics["noise_update_steps"] == 1
    assert metrics["noise_update_grad_norm"] > 0
    assert parameters_equal(workspace.model, before_actor)
    assert not parameters_equal(workspace.ppo_actor.noise_head, before_noise)

    before_noise = clone_parameters(workspace.ppo_actor.noise_head)
    before_flow = clone_parameters(workspace.model.model)
    buffer = toy_rollout(workspace)
    result = workspace.ppo_update(buffer)
    assert result["loss/actor_update_enabled"] == 1
    assert result["loss/optimizer_steps"] == 2  # 2 epochs, gradient accumulation of 2 minibatches.
    assert parameters_equal(workspace.ppo_actor.noise_head, before_noise)
    assert not parameters_equal(workspace.model.model, before_flow)
    workspace.itr, workspace.global_step = 3, 8
    workspace.noise_warmup_complete = True
    path = workspace.save_resume_checkpoint()
    exported = workspace.export_actor_checkpoint()
    actor_payload = torch.load(exported, pickle_module=dill, map_location="cpu")
    assert set(actor_payload["state_dicts"]) == {"model", "ema_model"}
    assert not any("noise_head" in key for key in actor_payload["state_dicts"]["model"])
    # Continue once, restore, then run the same update and compare weights/moments exactly.
    expected_rng = capture_rng_state()
    expected_metrics = workspace.noise_nll_update()
    expected_weights = copy.deepcopy(workspace.ppo_actor.state_dict())
    expected_noise_opt = copy.deepcopy(workspace.noise_optimizer.state_dict())
    expected_next_obs = {key: torch.from_numpy(value) for key, value in workspace.current_obs.items()}
    expected_actions = workspace.ppo_actor.get_actions(expected_next_obs)
    restored = workspace_factory("restored")
    restored.load_resume_checkpoint(path)
    assert restored.itr == 3 and restored.global_step == 8
    assert restored.noise_update_steps == 1
    assert_nested_equal(expected_rng, capture_rng_state())
    actual_metrics = restored.noise_nll_update()
    for key in expected_metrics:
        if key != "noise_update_time":
            assert actual_metrics[key] == expected_metrics[key]
    assert_nested_equal(restored.ppo_actor.state_dict(), expected_weights)
    assert_nested_equal(restored.noise_optimizer.state_dict(), expected_noise_opt)
    actual_actions = restored.ppo_actor.get_actions(expected_next_obs)
    assert_nested_equal(actual_actions, expected_actions)
    assert_nested_equal(restored.actor_optimizer.state_dict(), workspace.actor_optimizer.state_dict())
    assert_nested_equal(restored.critic_optimizer.state_dict(), workspace.critic_optimizer.state_dict())
    assert_nested_equal(restored.critic.state_dict(), workspace.critic.state_dict())


def test_resume_rejects_missing_independent_optimizer(workspace_factory):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    payload = torch.load(path, pickle_module=dill, map_location="cpu")
    payload.pop("noise_optimizer")
    torch.save(payload, path, pickle_module=dill)
    restored = workspace_factory("bad_resume")
    with pytest.raises((KeyError, ValueError), match="noise|[Ii]ncompatible|Incomplete"):
        restored.load_resume_checkpoint(path)


def test_no_reference_pretrained_structure_checked(workspace_factory, tiny_config):
    payload = torch.load(tiny_config.rl.pretrained_checkpoint_path, pickle_module=dill, map_location="cpu")
    payload["state_dicts"]["ema_model"]["plan_predictor.weight"] = torch.zeros(2)
    torch.save(payload, tiny_config.rl.pretrained_checkpoint_path, pickle_module=dill)
    with pytest.raises((RuntimeError, ValueError), match="reference|Reference|plan|checkpoint"):
        workspace_factory(cfg=tiny_config)


def test_legacy_joint_optimizer_state_migration_preserves_moments():
    from helpers import make_actor
    actor = make_actor()
    config = {"actor_lr": 1e-3, "noise_lr": 2e-3, "actor_weight_decay": 0, "betas": [0.9, 0.95]}
    legacy = torch.optim.AdamW(list(actor.actor_parameters()), lr=0.01)
    for parameter in actor.actor_parameters():
        parameter.grad = torch.ones_like(parameter)
    legacy.step()
    saved = legacy.state_dict()
    joint = build_actor_optimizer(actor, config)
    assert load_actor_optimizer_state(joint, saved) is True
    assert [group["lr"] for group in joint.param_groups] == [1e-3, 2e-3]
    assert_nested_equal(joint.state_dict()["state"], saved["state"])
    isolated = build_actor_optimizer(actor, config, include_noise=False)
    with pytest.raises(ValueError, match="actor-only"):
        load_actor_optimizer_state(isolated, saved)


def test_loader_resumes_pointcloud_color_augmentation_rng(tiny_config):
    tiny_config.policy.use_pc_color = True
    tiny_config.shape_meta.obs.point_cloud.shape = [5, 6]
    tiny_config.task.dataset.pointcloud_color_aug_cfg.aug_color = True
    tiny_config.task.dataset.pointcloud_color_aug_cfg.prob = 1.0
    group = zarr.open_group(tiny_config.rl.noise_guidance.dataset_path, mode="a")
    xyz = group["data/point_cloud"][:]
    rgb = np.random.default_rng(44).uniform(size=xyz.shape).astype(np.float32)
    group["data"].array("point_cloud", np.concatenate([xyz, rgb], axis=-1), overwrite=True)
    original = ExpertDataIterator(tiny_config)
    next(original)
    state = copy.deepcopy(original.state_dict())
    expected = [next(original) for _ in range(10)]
    restored = ExpertDataIterator(tiny_config)
    restored.load_state_dict(state)
    for expected_batch in expected:
        assert_nested_equal(next(restored), expected_batch)


def test_sampled_step_multi_update_aggregates_sparse_metrics(workspace_factory, tiny_config):
    tiny_config.rl.noise_guidance.sample_all_steps = False
    tiny_config.rl.noise_guidance.gradient_steps = 8
    workspace = workspace_factory(cfg=tiny_config)
    metrics = workspace.noise_nll_update()
    assert metrics["noise_update_steps"] == 8
    assert metrics["noise_supervised_flow_steps"] == 1
    # This seed visits both eligible steps; per-step metric keys vary by update.
    assert {key for key in metrics if key.startswith("noise_sigma_mean_by_step/")} == {
        "noise_sigma_mean_by_step/1", "noise_sigma_mean_by_step/2",
    }
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("key,value", [
    ("rl.actor.inference_steps", 4),
    ("rl.actor.residual_scale", 0.8),
    ("policy.sample_target_t_mode", "absolute"),
    ("policy.num_inference_steps", 5),
])
def test_resume_rejects_changed_transition_semantics(workspace_factory, tiny_config, key, value):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    changed = copy.deepcopy(tiny_config)
    OmegaConf.update(changed, key, value)
    restored = workspace_factory("changed_transition", cfg=changed)
    with pytest.raises(ValueError, match="Incompatible RAFE (noise )?resume"):
        restored.load_resume_checkpoint(path)


def test_resume_rejects_changed_bc_reference_path(workspace_factory, tiny_config, tmp_path):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    changed = copy.deepcopy(tiny_config)
    alternate = tmp_path / "alternate_pretrained.ckpt"
    alternate.write_bytes(Path(changed.rl.pretrained_checkpoint_path).read_bytes())
    changed.rl.pretrained_checkpoint_path = str(alternate)
    restored = workspace_factory("changed_reference", cfg=changed)
    with pytest.raises(ValueError, match="BC anchor changed"):
        restored.load_resume_checkpoint(path)


def test_resume_rejects_modified_bc_reference_file(workspace_factory, tiny_config):
    import os
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    pretrained = Path(tiny_config.rl.pretrained_checkpoint_path)
    metadata = pretrained.stat()
    os.utime(pretrained, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000))
    restored = workspace_factory("changed_reference_file")
    with pytest.raises(ValueError, match="size/mtime changed"):
        restored.load_resume_checkpoint(path)


def test_comparison_configs_compose_and_fixed_linear_sigma_does_not_update():
    from helpers import make_actor, expert_batch
    config_dir = str(Path(__file__).resolve().parents[1] / "ManiFlow/maniflow/config")
    names = ["expert_interp", "ppo_sigma", "fixed_sigma"]
    configs = {}
    for name in names:
        with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            configs[name] = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2_" + name,
                                         overrides=["task_name=smoke", "setting=smoke"])
    for cfg in configs.values():
        assert "action_plan" not in cfg.policy
        assert cfg.rl.actor.noise_conditioning_mode == "default"
        assert cfg.rl.actor.noise_head_type == "residual_schedule"
        assert cfg.rl.actor.noise_backbone == "tcn"
        assert cfg.rl.actor.noise_output_mode == "full"
        assert cfg.rl.actor.noise_parameterization == "log_residual"
    assert configs["expert_interp"].rl.noise_guidance.enabled is True
    assert configs["expert_interp"].rl.actor.residual_scale == 1.0
    assert configs["ppo_sigma"].rl.actor.residual_scale == 1.0
    assert configs["ppo_sigma"].rl.noise_guidance.enabled is False
    fixed_cfg = configs["fixed_sigma"]
    assert fixed_cfg.rl.actor.residual_scale == 0
    assert fixed_cfg.rl.ppo.noise_lr == 0
    actor, batch = make_actor(residual_scale=0), expert_batch()
    before = clone_parameters(actor.noise_head)
    vis, cond = actor._encode_obs(batch["obs"])
    x = torch.randn_like(batch["action"])
    for step in [1, 2]:
        _, sigma, _ = actor.gaussian_transition(x, step, vis, cond, learn_exploration_noise=True)
        t = torch.full((len(x),), step / 3)
        expected = actor.compute_base_sigma(t, x, step).clamp(actor.sigma_min, actor.sigma_max)
        torch.testing.assert_close(sigma, expected)
    _, chain, _ = actor.get_actions(batch["obs"])
    optimizer = build_actor_optimizer(actor, fixed_cfg.rl.ppo)
    logprob, _ = actor.get_logprobs(batch["obs"], chain, learn_exploration_noise=True)
    (-logprob.mean()).backward()
    optimizer.step()
    assert parameters_equal(actor.noise_head, before)


def test_resume_accepts_old_disabled_metadata_and_rejects_enabled_metadata(workspace_factory):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    payload = torch.load(path, pickle_module=dill, map_location="cpu")
    payload["cfg"].policy.action_plan = {"enabled": False, "predictor_hidden_dims": [8]}
    torch.save(payload, path, pickle_module=dill)
    restored = workspace_factory("disabled_metadata_resume")
    restored.load_resume_checkpoint(path)
    assert "action_plan" not in restored.cfg.policy
    assert_nested_equal(restored.ppo_actor.state_dict(), workspace.ppo_actor.state_dict())
    payload["cfg"].policy.action_plan.enabled = True
    torch.save(payload, path, pickle_module=dill)
    with pytest.raises(ValueError, match="plan|reference|Reference"):
        restored.load_resume_checkpoint(path)


def test_workspace_discards_disabled_metadata_before_model_instantiation(workspace_factory, tiny_config):
    tiny_config.policy.action_plan = {"enabled": False, "predictor_hidden_dims": [8]}
    workspace = workspace_factory(cfg=tiny_config)
    assert "action_plan" not in workspace.cfg.policy
    assert not hasattr(workspace.model, "plan_predictor")


def test_resume_rejects_legacy_reference_state_tensors(workspace_factory):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    payload = torch.load(path, pickle_module=dill, map_location="cpu")
    payload["ppo_actor"]["base_actor.plan_predictor.weight"] = torch.zeros(2, 2)
    torch.save(payload, path, pickle_module=dill)
    restored = workspace_factory("legacy_state_resume")
    with pytest.raises((ValueError, RuntimeError), match="plan_predictor|reference|Reference"):
        restored.load_resume_checkpoint(path)


def test_resume_rejects_previous_sigma_version_and_missing_parameterization(workspace_factory):
    workspace = workspace_factory()
    path = workspace.save_resume_checkpoint()
    original = torch.load(path, pickle_module=dill, map_location="cpu")
    assert original["rafe_resume_version"] == 2
    assert original["cfg"].rl.actor.noise_parameterization == "log_residual"
    for version in [None, 1]:
        previous = copy.deepcopy(original)
        if version is None:
            previous.pop("rafe_resume_version")
        else:
            previous["rafe_resume_version"] = version
        torch.save(previous, path, pickle_module=dill)
        with pytest.raises(ValueError, match="Incompatible|parameterization|version"):
            workspace.load_resume_checkpoint(path)
    previous = copy.deepcopy(original)
    previous["cfg"].rl.actor.pop("noise_parameterization")
    torch.save(previous, path, pickle_module=dill)
    with pytest.raises(ValueError, match="Incompatible|parameterization|configuration changed"):
        workspace.load_resume_checkpoint(path)


def test_ppo_sigma_resume_requires_versioned_log_residual_semantics(workspace_factory, tiny_config):
    tiny_config.rl.noise_guidance.enabled = False
    tiny_config.rl.noise_guidance.mode = "none"
    workspace = workspace_factory("ppo_sigma", cfg=tiny_config)
    path = workspace.save_resume_checkpoint()
    original = torch.load(path, pickle_module=dill, map_location="cpu")
    assert original["rafe_resume_version"] == 2
    restored = workspace_factory("ppo_sigma_restored", cfg=tiny_config)
    restored.load_resume_checkpoint(path)
    assert_nested_equal(restored.ppo_actor.state_dict(), workspace.ppo_actor.state_dict())
    for remove_marker in [False, True]:
        previous = copy.deepcopy(original)
        if remove_marker:
            previous["ppo_actor"].pop("_extra_state")
        else:
            previous.pop("rafe_resume_version")
        torch.save(previous, path, pickle_module=dill)
        with pytest.raises((ValueError, RuntimeError), match="Incompatible|parameterization|version|_extra_state"):
            restored.load_resume_checkpoint(path)


def test_optional_dgn_tuning_preset_keeps_default_experiment_unchanged():
    config_dir = str(Path(__file__).resolve().parents[1] / "ManiFlow/maniflow/config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        default = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2_expert_interp")
        tuned = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2_expert_interp_dgn_tuning")
    assert default.rl.noise_guidance.learning_rate == 3e-6
    assert default.rl.noise_guidance.gradient_steps == 4
    assert tuned.rl.noise_guidance.learning_rate == 1e-4
    assert tuned.rl.noise_guidance.gradient_steps == 4
    assert default.rl.actor.noise_parameterization == tuned.rl.actor.noise_parameterization == "log_residual"
    assert default.rl.actor.residual_scale == tuned.rl.actor.residual_scale == 1.0
    assert default.rl.ppo.actor_lr == tuned.rl.ppo.actor_lr
    assert default.rl.ppo.num_iterations == tuned.rl.ppo.num_iterations
    assert default.rl.rollout.n_steps == tuned.rl.rollout.n_steps
    assert default.policy.num_inference_steps == tuned.policy.num_inference_steps
    assert "action_plan" not in tuned.policy
