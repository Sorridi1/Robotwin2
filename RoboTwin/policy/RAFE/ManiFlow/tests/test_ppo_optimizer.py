import copy

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from maniflow.common.ppo_optimizer import build_actor_optimizer, load_actor_optimizer_state


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_actor = nn.Linear(3, 2)
        self.base_actor.obs_encoder = nn.Linear(3, 3).requires_grad_(False)
        self.noise_head = nn.Linear(3, 2)
        self.actor_old = nn.Linear(3, 2).requires_grad_(False)

    def get_extra_state(self):
        return {"version": 1, "noise_parameterization": "log_residual", "residual_scale": 1.0}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Incompatible test noise parameterization")

    def actor_parameters(self):
        yield from (p for p in self.base_actor.parameters() if p.requires_grad)
        yield from self.noise_head.parameters()


def config(**overrides):
    return OmegaConf.create({
        "actor_lr": 3e-6, "noise_lr": 3e-5, "actor_weight_decay": 0.0,
        "betas": [0.9, 0.95], **overrides,
    })


def legacy_optimizer(actor):
    return torch.optim.AdamW(actor.actor_parameters(), lr=3e-6, weight_decay=0.0, betas=(0.9, 0.95))


def update(actor, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum((index + 1) * p.square().sum() for index, p in enumerate(actor.actor_parameters()))
    loss.backward()
    optimizer.step()


def assert_moments_equal(source, target, source_actor, target_actor):
    for old, new in zip(source_actor.actor_parameters(), target_actor.actor_parameters()):
        assert source.state[old].keys() == target.state[new].keys()
        for key, value in source.state[old].items():
            torch.testing.assert_close(value, target.state[new][key], rtol=0, atol=0)


def test_parameter_groups_cover_exactly_legacy_trainable_sequence_without_frozen_modules():
    actor = TinyActor()
    optimizer = build_actor_optimizer(actor, config())
    groups = optimizer.param_groups
    assert [(group["name"], group["lr"]) for group in groups] == [("actor", 3e-6), ("noise", 3e-5)]
    parameters = [p for group in groups for p in group["params"]]
    assert [id(p) for p in parameters] == [id(p) for p in actor.actor_parameters()]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert all(p.requires_grad for p in parameters)


@pytest.mark.parametrize("setting", ["missing", None])
def test_missing_or_null_noise_lr_keeps_legacy_rate(setting):
    cfg = config(noise_lr=setting)
    if setting == "missing":
        del cfg["noise_lr"]
    optimizer = build_actor_optimizer(TinyActor(), cfg)
    assert [group["lr"] for group in optimizer.param_groups] == [3e-6, 3e-6]


@pytest.mark.parametrize("key", ["actor_lr", "noise_lr"])
@pytest.mark.parametrize("value", [-1e-5, float("nan"), float("inf")])
def test_invalid_learning_rates_fail_before_training(key, value):
    with pytest.raises(ValueError, match=key):
        build_actor_optimizer(TinyActor(), config(**{key: value}))


def test_noise_can_update_ten_times_faster_without_changing_actor_or_frozen_weights():
    actor = TinyActor()
    with torch.no_grad():
        actor.base_actor.weight.fill_(1.0)
        actor.noise_head.weight.fill_(1.0)
    frozen = {name: parameter.clone() for name, parameter in actor.named_parameters() if not parameter.requires_grad}
    optimizer = build_actor_optimizer(actor, config(actor_lr=1e-4, noise_lr=1e-3))
    loss = actor.base_actor.weight.sum() + actor.noise_head.weight.sum()
    loss.backward()
    optimizer.step()
    actor_change = (1.0 - actor.base_actor.weight).mean()
    noise_change = (1.0 - actor.noise_head.weight).mean()
    assert float(noise_change / actor_change) == pytest.approx(10.0, rel=1e-3)
    for name, parameter in actor.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)


def test_legacy_migration_preserves_adam_moments_and_next_update_exactly():
    old_actor = TinyActor()
    old = legacy_optimizer(old_actor)
    for _ in range(3):
        update(old_actor, old)
    new_actor = copy.deepcopy(old_actor)
    new = build_actor_optimizer(new_actor, config(noise_lr=None))
    saved = copy.deepcopy(old.state_dict())
    assert load_actor_optimizer_state(new, saved)
    assert len(saved["param_groups"]) == 1  # Migration does not mutate the checkpoint payload.
    assert_moments_equal(old, new, old_actor, new_actor)
    update(old_actor, old)
    update(new_actor, new)
    assert_moments_equal(old, new, old_actor, new_actor)
    for left, right in zip(old_actor.parameters(), new_actor.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("legacy", [True, False])
def test_resume_keeps_configured_rates_not_checkpoint_rates(legacy):
    actor = TinyActor()
    old = legacy_optimizer(actor) if legacy else build_actor_optimizer(actor, config())
    update(actor, old)
    old.param_groups[0]["initial_lr"] = old.param_groups[0]["lr"]
    new_actor = copy.deepcopy(actor)
    new = build_actor_optimizer(new_actor, config(actor_lr=2e-6, noise_lr=5e-5))
    assert load_actor_optimizer_state(new, copy.deepcopy(old.state_dict())) is legacy
    assert [group["lr"] for group in new.param_groups] == [2e-6, 5e-5]
    for group in new.param_groups:
        if "initial_lr" in group:
            assert group["initial_lr"] == group["lr"]
    assert_moments_equal(old, new, actor, new_actor)


def test_warmup_checkpoint_without_actor_adam_state_can_resume():
    actor = TinyActor()
    old = legacy_optimizer(actor)
    new = build_actor_optimizer(actor, config())
    assert load_actor_optimizer_state(new, old.state_dict())
    assert not new.state
    update(actor, new)
    assert all(float(new.state[p]["step"]) == 1 for p in actor.actor_parameters())


@pytest.mark.parametrize("legacy", [True, False])
def test_changed_parameter_count_is_not_silently_ignored(legacy):
    actor = TinyActor()
    old = legacy_optimizer(actor) if legacy else build_actor_optimizer(actor, config())
    saved = copy.deepcopy(old.state_dict())
    saved["param_groups"][0]["params"].pop()
    with pytest.raises(ValueError, match="parameter count"):
        load_actor_optimizer_state(build_actor_optimizer(actor, config()), saved)


def test_changed_moment_shape_is_not_silently_ignored():
    actor = TinyActor()
    old = legacy_optimizer(actor)
    update(actor, old)
    saved = copy.deepcopy(old.state_dict())
    first = saved["param_groups"][0]["params"][0]
    saved["state"][first]["exp_avg"] = torch.zeros(99)
    with pytest.raises(ValueError, match="shape differs"):
        load_actor_optimizer_state(build_actor_optimizer(actor, config()), saved)


def test_reordered_modern_parameter_groups_are_rejected():
    actor = TinyActor()
    saved = build_actor_optimizer(actor, config()).state_dict()
    saved["param_groups"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        load_actor_optimizer_state(build_actor_optimizer(actor, config()), saved)


@pytest.mark.parametrize("legacy", [True, False])
def test_workspace_resume_restores_weights_counters_and_optimizer_without_losing_noise_lr(tmp_path, legacy):
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace

    def workspace():
        result = TrainReinFlowRLRoboTwinWorkspace.__new__(TrainReinFlowRLRoboTwinWorkspace)
        result.cfg = OmegaConf.create({"rl": {"ppo": config(),
            "actor": {"noise_parameterization": "log_residual"}}})
        result._output_dir = str(tmp_path)
        result.ppo_actor = TinyActor()
        result.critic = nn.Linear(3, 1)
        result.actor_optimizer = build_actor_optimizer(result.ppo_actor, result.cfg.rl.ppo)
        result.critic_optimizer = torch.optim.AdamW(result.critic.parameters())
        result.scaler = torch.cuda.amp.GradScaler(enabled=False)
        result.itr, result.global_step = 20, 72159
        result.consecutive_zero_success_iters = 0
        result.best_eval_score, result.best_eval_itr = 0.87, 20
        return result

    old = workspace()
    if legacy:
        old.actor_optimizer = legacy_optimizer(old.ppo_actor)
    update(old.ppo_actor, old.actor_optimizer)
    path = old.save_resume_checkpoint()
    new = workspace()
    new.itr, new.global_step = 0, 0
    new.load_resume_checkpoint(path)
    assert new.itr == 20 and new.global_step == 72159
    assert new.best_eval_score == 0.87 and new.best_eval_itr == 20
    assert [group["lr"] for group in new.actor_optimizer.param_groups] == [3e-6, 3e-5]
    assert_moments_equal(old.actor_optimizer, new.actor_optimizer, old.ppo_actor, new.ppo_actor)
    for left, right in zip(old.ppo_actor.parameters(), new.ppo_actor.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
