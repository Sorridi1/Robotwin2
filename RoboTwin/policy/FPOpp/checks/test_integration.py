import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
import torch

from fpopp.actor import FPOActor, execution_mask
from fpopp.checkpoint import atomic_save, load_payload, load_policy, policy_configuration
from fpopp.runtime import POLICY_ROOT
from fpopp.workspace import Workspace, stack_observations
from fpopp.loss import surrogate_loss
from fixtures import FakePool, tiny_checkpoint


def configuration(tmp_path, checkpoint):
    cfg = OmegaConf.load(POLICY_ROOT / "configs/fpopp_robotwin2.yaml")
    cfg.checkpoint = str(checkpoint)
    cfg.device = "cpu"
    cfg.output_dir = str(tmp_path / "run")
    cfg.rollout.n_envs, cfg.rollout.n_steps = 2, 2
    cfg.fpo.mc_samples, cfg.fpo.mc_batch_size = 3, 2
    cfg.ppo.batch_size, cfg.ppo.grad_accum_steps = 3, 2
    cfg.ppo.critic_warmup_iterations, cfg.ppo.update_epochs = 0, 2
    cfg.ppo.reward_scale_running, cfg.ppo.use_amp = False, False
    cfg.max_iterations = 1
    return cfg


def test_checkpoint_roundtrip_strict_plan_and_normalizer(tmp_path):
    payload = tiny_checkpoint()
    path = tmp_path / "input.ckpt"
    atomic_save(payload, path)
    policy, config, key = load_policy(load_payload(path))
    assert key == "ema_model" and policy.action_plan_enabled
    for name, tensor in payload["state_dicts"][key].items():
        assert torch.equal(tensor, policy.state_dict()[name]), name
    bad = copy.deepcopy(payload)
    del bad["state_dicts"]["ema_model"]["plan_predictor.mlp.0.weight"]
    with pytest.raises(RuntimeError):
        load_policy(bad)
    no_normalizer = copy.deepcopy(payload)
    for state in no_normalizer["state_dicts"].values():
        for name in list(state):
            if name.startswith("normalizer."):
                del state[name]
    with pytest.raises(ValueError, match="normalizer"):
        load_policy(no_normalizer)


def test_native_sampler_instantaneous_cfm_and_window(tmp_path):
    payload = tiny_checkpoint()
    policy, _, _ = load_policy(payload)
    actor = FPOActor(policy, seed=7)
    obs = stack_observations(FakePool().reset(), "cpu")
    features, plan = actor.encode(obs)
    rng = actor.rng.get_state()
    action, targets, _ = actor.sample(features, plan)
    actor.rng.set_state(rng)
    latent = torch.randn(targets.shape, generator=actor.rng)
    native = policy.sample_ode(x0=latent, N=actor.inference_steps, vis_cond=features, plan_cond=plan)[-1]
    assert torch.allclose(targets, native)
    assert torch.allclose(action, policy.normalizer["action"].unnormalize(native)[:, 1:4])
    mask = execution_mask(actor, [1, 3])
    assert mask.tolist() == [[0, 1, 0, 0], [0, 1, 1, 1]]
    time, noise = actor.draw_cfm(targets, 3)
    old = actor.cfm_losses(features, plan, targets, time, noise).detach()
    assert torch.equal(old, actor.cfm_losses(features, plan, targets, time, noise))
    seen = []
    handle = policy.model.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs["target_t"]), with_kwargs=True)
    actor.cfm_losses(features, plan, targets, time, noise)
    handle.remove()
    assert seen[0].eq(0).all()
    policy.n_obs_steps, policy.horizon, policy.n_action_steps = 2, 16, 16
    assert actor.window == (1, 16)


def test_full_collect_update_save_resume_and_deployment(tmp_path):
    from deploy_policy import get_model, reset_model, eval as evaluate
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(), base)
    cfg = configuration(tmp_path, base)
    work = Workspace(cfg)
    before_encoder = copy.deepcopy(work.actor.policy.obs_encoder.state_dict())
    before_plan = copy.deepcopy(work.actor.policy.plan_predictor.state_dict())
    before_flow = copy.deepcopy(work.actor.policy.model.state_dict())
    pool = FakePool()
    work.run(env_factory=lambda: pool)
    assert pool.closed and work.action_steps == 8 and work.decision_steps == 4
    for name, tensor in before_encoder.items():
        assert torch.equal(tensor, work.actor.policy.obs_encoder.state_dict()[name])
    for name, tensor in before_plan.items():
        assert torch.equal(tensor, work.actor.policy.plan_predictor.state_dict()[name])
    assert any(not torch.equal(value, work.actor.policy.model.state_dict()[key]) for key, value in before_flow.items())
    path = Path(cfg.output_dir) / "checkpoints/latest.ckpt"
    payload = load_payload(path)
    cfg.resume = str(path)
    resumed = Workspace(cfg)
    assert resumed.action_steps == work.action_steps and resumed.iteration == 1
    assert torch.equal(work.actor.rng.get_state(), resumed.actor.rng.get_state())
    assert resumed.next_seeds == work.next_seeds
    assert resumed.actor_optimizer.state_dict()["state"]
    assert torch.equal(torch.get_rng_state(), payload["trainer"]["rng"]["torch"])
    args = {"checkpoint_path": str(path), "device": "cpu", "eval_output_dir": str(tmp_path / "evaluation")}
    model = get_model(args)
    observation = {"joint_action": {"vector": [0., 0.]}, "pointcloud": torch.ones(8, 6).numpy()*.1}
    class Task:
        eval_success, step_lim, take_action_cnt = False, 10, 0
        def take_action(self, action, action_type):
            assert action_type == "qpos"
            self.take_action_cnt += 1
            self.eval_success = True
        def get_obs(self):
            return observation
    reset_model(model)
    task = Task()
    evaluate(task, model, observation)
    assert task.take_action_cnt == 1
    reset_model(model)
    assert len(model.history) == 0


def test_cli_dry_run_and_resume_guard(tmp_path):
    from train import parse_config
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(plan=False), base)
    cfg, args = parse_config(["--checkpoint", str(base), "--device", "cpu", "--dry-run", "fpo.mc_samples=2",
                              "ppo.actor_eps=1e-5"])
    assert args.dry_run and cfg.fpo.mc_samples == 2
    assert cfg.ppo.actor_eps == 1e-5
    with pytest.raises(SystemExit):
        parse_config(["--checkpoint", str(base), "validation.interval=1"])
    with pytest.raises(Exception):
        parse_config(["--checkpoint", str(base), "fpo.typo=3"])


def test_mc_microbatch_matches_full_objective_gradient():
    policy, _, _ = load_policy(tiny_checkpoint())
    actor = FPOActor(policy, seed=4)
    features, plan = actor.encode(stack_observations(FakePool().reset(), "cpu"))
    _, actions, _ = actor.sample(features, plan)
    time, noise = actor.draw_cfm(actions, 3)
    old = actor.cfm_losses(features, plan, actions, time, noise).detach()
    mask, advantage = execution_mask(actor, [2, 3]), torch.tensor([1., -1.])
    # Perturb the frozen old estimate so ratios are not trivially identical.
    old = old + torch.tensor([.01, -.01, .02])[None, :, None]
    current = actor.cfm_losses(features, plan, actions, time, noise)
    loss, _ = surrogate_loss(old, current, mask, advantage, old_loss_cap=None)
    loss.backward()
    parameters = list(actor.trainable_parameters())
    full = [p.grad.clone() if p.grad is not None else None for p in parameters]
    actor.zero_grad(set_to_none=True)
    for index in range(3):
        current = actor.cfm_losses(features, plan, actions, time[:, index:index+1], noise[:, index:index+1])
        loss, _ = surrogate_loss(old[:, index:index+1], current, mask, advantage, old_loss_cap=None)
        (loss/3).backward()
    for expected, parameter in zip(full, parameters):
        if expected is not None:
            assert torch.allclose(expected, parameter.grad, atol=1e-6, rtol=1e-4)


def test_periodic_validation_complete_report_and_best_guard(tmp_path, monkeypatch):
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(), base)
    cfg = configuration(tmp_path, base)
    cfg.validation.seed_file = str(tmp_path / "separate_validation_bank.json")
    cfg.validation.episodes = 2
    work = Workspace(cfg)
    work.iteration = 1
    report = {"status": "complete", "completed_episodes": 2, "successes": 1,
              "episodes": [{"seed": 9000001}, {"seed": 9000002}]}
    def public_evaluator(command, **kwargs):
        selected = command[command.index("--eval_scene_seed_file")+1]
        assert selected == str(Path(cfg.validation.seed_file).resolve())
        root = Path(command[command.index("--eval_output_dir")+1])
        path = root / "eval_results/epoch_1/scene_rl/_scene_selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
    monkeypatch.setattr("fpopp.workspace.subprocess.run", public_evaluator)
    assert work.validate() == .5
    best = Path(cfg.output_dir) / "checkpoints/best.ckpt"
    best_time = best.stat().st_mtime_ns
    assert best.is_file() and work.best_score == .5
    assert not (best.parent / "validation_candidate.ckpt").exists()
    report["status"] = "aborted"
    with pytest.raises(RuntimeError, match="did not complete"):
        work.validate()
    assert best.stat().st_mtime_ns == best_time


def test_evaluation_results_are_reserved_per_mode_without_overwriting(tmp_path):
    from fpopp.results import reserve_results
    marker = reserve_results(tmp_path, 20, "rl")
    marker.write_text('{"checkpoint": "first"}')
    report = marker.parent / "_scene_selection.json"
    report.write_text('{"status": "complete"}')
    with pytest.raises(FileExistsError, match="--eval_output_dir"):
        reserve_results(tmp_path, 20, "rl")
    assert marker.read_text() == '{"checkpoint": "first"}'
    assert report.read_text() == '{"status": "complete"}'
    assert reserve_results(tmp_path, 20, "expert_fail").parent.name == "scene_expert_fail"
    assert reserve_results(tmp_path, 40, "rl").parent.parent.name == "epoch_40"
    orphan = tmp_path / "eval_results/epoch_60/scene_rl"
    orphan.mkdir(parents=True)
    (orphan / "episode0.mp4").write_bytes(b"old video")
    with pytest.raises(FileExistsError):
        reserve_results(tmp_path, 60, "rl")
    assert (orphan / "episode0.mp4").read_bytes() == b"old video"


def test_final_validation_runs_when_budget_ends_before_interval(tmp_path, monkeypatch):
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(plan=False), base)
    cfg = configuration(tmp_path, base)
    cfg.validation.interval = 20
    cfg.validation.run_final = True
    cfg.validation.seed_file = str(tmp_path / "validation.json")
    work = Workspace(cfg)
    calls = []
    def validate():
        calls.append(work.iteration)
        return .5
    monkeypatch.setattr(work, "validate", validate)
    work.run(env_factory=FakePool)
    assert calls == [1] and work.iteration == 1


def test_failed_validation_preserves_completed_training_checkpoint(tmp_path, monkeypatch):
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(plan=False), base)
    cfg = configuration(tmp_path, base)
    cfg.validation.interval = 1
    cfg.validation.seed_file = str(tmp_path / "validation.json")
    work = Workspace(cfg)
    def failed_evaluation():
        raise RuntimeError("simulator allocation failed")
    monkeypatch.setattr(work, "validate", failed_evaluation)
    with pytest.raises(RuntimeError, match="simulator allocation failed"):
        work.run(env_factory=FakePool)
    saved = load_payload(Path(cfg.output_dir) / "checkpoints/latest.ckpt")
    assert saved["iteration"] == 1 and saved["action_steps"] == 8
    assert saved["trainer"]["actor_optimizer"]["state"]


@pytest.mark.parametrize("configured_eps", [None, 1e-5])
def test_actor_epsilon_training_and_resume_including_legacy_config(tmp_path, configured_eps):
    base = tmp_path / "bc.ckpt"
    atomic_save(tiny_checkpoint(plan=False), base)
    cfg = configuration(tmp_path, base)
    if configured_eps is None:
        del cfg.ppo.actor_eps  # Configuration saved before this option existed.
    else:
        cfg.ppo.actor_eps = configured_eps
    expected = 1e-8 if configured_eps is None else configured_eps
    work = Workspace(cfg)
    work.run(env_factory=FakePool)
    assert work.actor_optimizer.param_groups[0]["eps"] == expected
    path = Path(cfg.output_dir) / "checkpoints/latest.ckpt"
    payload = load_payload(path)
    assert payload["trainer"]["actor_optimizer"]["param_groups"][0]["eps"] == expected
    resumed_cfg = OmegaConf.create(payload["cfg"])
    resumed_cfg.resume = str(path)
    resumed = Workspace(resumed_cfg)
    assert resumed.actor_optimizer.param_groups[0]["eps"] == expected
    assert resumed.actor_optimizer.state_dict()["state"]
