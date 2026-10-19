import io

import numpy as np
import pytest

from maniflow.common.rl_progress import RLIterationProgress


def test_rollout_keeps_cumulative_success_and_refreshes_every_batch():
    output = io.StringIO()
    with RLIterationProgress(3, 4, global_step=100, mininterval=1000, stream=output) as progress:
        assert "itr=3 reset" in output.getvalue()
        progress.start_rollout()
        progress.advance_rollout(1, [True, False], [True, False], [False, True], 17, 117, [1, 16])
        assert progress.bar.n == 1
        assert progress.bar.postfix == "succ=1/2 step=117(+17) env_step=1..16"
        previous = len(output.getvalue())
        progress.advance_rollout(2, [False, False], [False, False], [False, False], 32, 149, [16, 16])
        assert len(output.getvalue()) > previous  # Not delayed by adaptive miniters/mininterval.
        assert progress.bar.n == 2
        assert progress.bar.postfix == "succ=1/2 step=149(+32) env_step=16"
        assert progress.bar.leave is True
    assert output.getvalue().endswith("\n")
    assert "succ=1/2" in output.getvalue().split("\r")[-1]


def test_one_bar_survives_update_phase_and_early_kl_stop():
    output = io.StringIO()
    with RLIterationProgress(5, 64, stream=output) as progress:
        bar = progress.bar
        progress.start_rollout()
        progress.advance_rollout(64, [True], [True], [False], 4, 4096, [4])
        progress.start_update(minibatches=128, optimizer_steps=32)
        assert progress.bar is bar
        assert bar.n == 0 and bar.total == 128
        assert bar.postfix == "succ=1/1 step=4096 opt=0/32"
        progress.advance_update(3, 1, stopped_by_kl=True)
        assert bar.n == 3  # An early stop must not falsely report all minibatches completed.
        assert "KL-stop" in bar.desc
        assert "opt=1/32" in bar.postfix


def test_success_and_steps_fit_a_normal_80_column_terminal(monkeypatch):
    import os
    import maniflow.common.rl_progress as module

    monkeypatch.setattr(module.shutil, "get_terminal_size", lambda: os.terminal_size((80, 24)))
    output = io.StringIO()
    with RLIterationProgress(99, 64, stream=output) as progress:
        progress.bar.dynamic_ncols = False
        progress.bar.ncols = 80
        progress.start_rollout()
        progress.advance_rollout(64, [True] * 4, [True] * 4, [False] * 4,
                                 64, 1000000, [16, 32, 48, 450], reward=0.5)
        visible = str(progress.bar)
        assert "succ=4/4" in visible
        assert "step=1000000(+64)" in visible
        assert "env_step=16..450" in visible
        assert "reward=" not in visible  # Omit optional fields, don't leave cut-off labels.
        assert "| succ=" in visible
        assert "|" in visible


def test_exception_closes_bar_without_swallowing_error():
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="worker failed"):
        with RLIterationProgress(0, 1, stream=output) as progress:
            raise RuntimeError("worker failed")
    assert progress.bar.disable


def _ppo_workspace(kl):
    """Exercise the real PPO loop with tiny CPU networks, no simulator/GPU."""
    import torch
    from omegaconf import OmegaConf
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, obs):
            return self.linear(obs["agent_pos"])

    class Actor(torch.nn.Module):
        clip_vloss_coef = None

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.loss_calls = 0

        def actor_parameters(self):
            return self.parameters()

        def loss(self, obs, chains, returns, values, advantages, logprobs, critic, **kwargs):
            self.loss_calls += 1
            zero = self.weight * 0
            return {
                "pg_loss": self.weight * advantages.mean(),
                "entropy_loss": zero,
                "bc_loss": zero,
                "value_loss": 0.5 * ((critic(obs).view(-1) - returns) ** 2).mean(),
                "approx_kl": zero + kl,
            }

    class Buffer:
        n_steps = 3
        n_envs = 2

        def update(self, *args):
            pass

        def make_dataset(self):
            obs = {"agent_pos": torch.arange(12).float().reshape(6, 2) / 12}
            returns = torch.arange(6).float() / 6
            return obs, torch.zeros(6, 1), returns, torch.zeros(6), returns, torch.zeros(6)

        def get_explained_var(self, *args):
            return 0.0

    torch.manual_seed(12)
    workspace = TrainReinFlowRLRoboTwinWorkspace.__new__(TrainReinFlowRLRoboTwinWorkspace)
    workspace.cfg = OmegaConf.create({"rl": {
        "ppo": {"batch_size": 2, "grad_accum_steps": 2, "update_epochs": 2,
                "target_kl": 0.03, "critic_warmup_iterations": 0, "max_grad_norm": 1.0,
                "use_amp": False, "ent_coef": 0.0, "vf_coef": 0.5},
        "bc_anchor": {"enabled": False, "coeff": 0.0},
    }})
    workspace.device = torch.device("cpu")
    workspace.itr = 0
    workspace.ppo_actor = Actor()
    workspace.critic = Critic()
    workspace.actor_optimizer = torch.optim.SGD(workspace.ppo_actor.parameters(), lr=0.01)
    workspace.critic_optimizer = torch.optim.SGD(workspace.critic.parameters(), lr=0.01)
    workspace.scaler = torch.cuda.amp.GradScaler(enabled=False)
    workspace.current_obs = {"point_cloud": np.zeros((1, 3)), "agent_pos": np.zeros((2,))}
    return workspace, Buffer()


@pytest.mark.parametrize("update_actor,kl,expected_batches,expected_updates", [
    (False, 0.0, 6, 4),  # Critic-only warmup still displays updates.
    (True, 0.0, 6, 4),
    (True, 0.2, 1, 1),  # Target-KL early stop.
])
def test_progress_matches_real_ppo_updates_without_changing_training(
    update_actor, kl, expected_batches, expected_updates,
):
    import torch

    baseline, buffer = _ppo_workspace(kl)
    expected = baseline.ppo_update(buffer, update_actor=update_actor)
    workspace, buffer = _ppo_workspace(kl)
    with RLIterationProgress(0, 3, stream=io.StringIO()) as progress:
        actual = workspace.ppo_update(buffer, update_actor=update_actor, progress=progress)
        assert progress.bar.n == expected_batches
        assert progress.bar.total == 6
        assert progress.optimizer_steps == expected_updates
        assert progress.total_optimizer_steps == 4
    assert actual == expected
    assert actual["loss/optimizer_steps"] == expected_updates
    for observed, reference in zip(workspace.critic.parameters(), baseline.critic.parameters()):
        torch.testing.assert_close(observed, reference)
    torch.testing.assert_close(workspace.ppo_actor.weight, baseline.ppo_actor.weight)


@pytest.mark.parametrize("update_actor,kl,expected_updates", [
    (False, 0.0, 0), (True, 0.0, 4), (True, 0.2, 1),
])
def test_two_optimizer_groups_follow_warmup_and_kl_gating(update_actor, kl, expected_updates):
    import torch
    from maniflow.common.ppo_optimizer import build_actor_optimizer

    workspace, buffer = _ppo_workspace(kl)
    actor = workspace.ppo_actor
    actor.base_actor = torch.nn.ParameterList([actor.weight])
    actor.noise_head = torch.nn.Linear(1, 1)
    base_loss = actor.loss

    def loss(*args, **kwargs):
        result = base_loss(*args, **kwargs)
        result["pg_loss"] = result["pg_loss"] + 0.1 * (
            actor.noise_head.weight.sum() + actor.noise_head.bias.sum()
        )
        return result

    actor.loss = loss
    ppo_config = workspace.cfg.rl.ppo
    ppo_config.actor_lr, ppo_config.noise_lr = 3e-6, 3e-5
    ppo_config.actor_weight_decay, ppo_config.betas = 0.0, [0.9, 0.95]
    workspace.actor_optimizer = build_actor_optimizer(actor, ppo_config)
    original = {name: p.clone() for name, p in actor.named_parameters()}
    workspace.ppo_update(buffer, update_actor=update_actor)
    optimizer = workspace.actor_optimizer
    if expected_updates:
        assert all(float(optimizer.state[p]["step"]) == expected_updates for p in actor.parameters())
        assert not torch.equal(actor.noise_head.weight, original["noise_head.weight"])
    else:
        assert not optimizer.state
        for name, p in actor.named_parameters():
            torch.testing.assert_close(p, original[name], rtol=0, atol=0)
    assert [group["lr"] for group in optimizer.param_groups] == [3e-6, 3e-5]
