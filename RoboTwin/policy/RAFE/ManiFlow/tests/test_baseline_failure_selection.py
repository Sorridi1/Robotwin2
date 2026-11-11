from collections import deque
import copy
import json
import random
from types import SimpleNamespace

import dill
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from test_eval_scene_selection import Task, Validator, Unstable, args, evaluator
from script.eval_baseline import FixedBaselineProbe
from script.eval_scene_selection import EvalSceneSelector, build_scene_cache, load_scene_manifest
from script.eval_seed_bank import SharedEvalSeedBank
from maniflow.common.robotwin_seed_cache import ExpertSeedCache
from maniflow.env.robotwin.scene_validator import SceneValidationResult


BASELINE_ID = {"checkpoint": {"path": "/fixed/model500.ckpt", "size_bytes": 100, "mtime_ns": 1},
               "protocol_sha256": "test_baseline"}


class Probe:
    def __init__(self, passed=False, error=None):
        self.passed, self.error = passed, error
        self.calls = []

    def __call__(self, task, info):
        self.calls.append(task.seed)
        if self.error is not None:
            raise self.error
        task.state = "baseline_end"
        return {"passed": self.passed, "action_steps": 100, "step_limit": 100}


def selector(args, probe, validator=None, cache=None, identity=None):
    return EvalSceneSelector("baseline_fail", args, Unstable, validator or Validator(), cache,
                             probe, identity or BASELINE_ID)


def test_failure_rebuilds_same_seed_and_never_calls_expert(args, tmp_path):
    args["eval_video_save_dir"] = tmp_path
    task, probe = Task(), Probe()
    selected_by = selector(args, probe)
    selected = selected_by.prepare(task, 10, 0)
    assert task.setups == [10, 10] and task.plays == [] and probe.calls == [10]
    assert selected_by.validator.calls == [10, 10]
    assert task.state == "initial" and task.eval_video_path == tmp_path
    assert selected.record["baseline_result"]["passed"] is False
    assert selected.record["expert_result"] is None
    selected_by.close(task)


def test_success_is_rejected_and_cached_without_becoming_failure(args, tmp_path):
    cache = ExpertSeedCache(tmp_path, "baseline")
    selected_by, task, probe = selector(args, Probe(True), cache=cache), Task(), Probe(True)
    selected_by.baseline_probe = probe
    assert selected_by.prepare(task, 10, 0) is None
    assert cache.lookup(10)[0] is False
    assert cache.lookup(10)[1]["reason"] == "baseline_success"
    assert task.setups == [10] and task.plays == []
    assert not selected_by.active
    assert selected_by.prepare(task, 10, 0) is None
    assert probe.calls == [10] and task.setups == [10]


def test_cheap_invalid_scene_never_runs_baseline(args):
    probe = Probe()
    selected_by = selector(args, probe, Validator(invalid=[10]))
    assert selected_by.prepare(Task(), 10, 0) is None
    assert probe.calls == [] and selected_by.stats["baseline_checks"] == 0


@pytest.mark.parametrize("error", [RuntimeError("inference error"), Unstable("simulation error")])
def test_probe_exceptions_are_not_recorded_as_failures(args, tmp_path, error):
    cache = ExpertSeedCache(tmp_path, "baseline")
    selected_by = selector(args, Probe(error=error), cache=cache)
    with pytest.raises(RuntimeError, match="not caching"):
        selected_by.prepare(Task(), 10, 0)
    assert cache.lookup(10) is None
    assert selected_by.records == [] and not selected_by.active


def test_failed_probe_must_pass_cheap_check_again_after_rebuild(args):
    class RejectRebuild(Validator):
        def validate(self, task):
            result = super().validate(task)
            return result if len(self.calls) == 1 else SceneValidationResult(False, "initial_success")
    selected_by = selector(args, Probe(), RejectRebuild())
    assert selected_by.prepare(Task(), 10, 0) is None
    assert selected_by.records == [] and not selected_by.active


def test_cached_failure_keeps_one_scene_and_skips_probe(args, tmp_path):
    cache = ExpertSeedCache(tmp_path, "baseline")
    task, first = Task(), selector(args, Probe(), cache=cache)
    first.prepare(task, 10, 0)
    first.close(task)
    probe, fresh = Probe(error=AssertionError("must not run baseline")), Task()
    second = selector(args, probe, cache=cache)
    assert second.prepare(fresh, 10, 0).record["baseline_result"]["passed"] is False
    assert probe.calls == [] and fresh.setups == [10] and second.validator.calls == [10]
    second.close(fresh)


def test_baseline_cache_and_bank_cannot_mix_another_checkpoint(args, tmp_path):
    other_id = {**BASELINE_ID, "checkpoint": {"path": "/another.ckpt"}}
    cache_a = build_scene_cache(args, "baseline_fail", Validator(), tmp_path, baseline_identity=BASELINE_ID)
    cache_b = build_scene_cache(args, "baseline_fail", Validator(), tmp_path, baseline_identity=other_id)
    assert cache_a.directory != cache_b.directory
    path = tmp_path / "shared.json"
    shared = SharedEvalSeedBank(path, args, "baseline_fail", 10, baseline_identity=BASELINE_ID)
    task, first = Task(), selector(args, Probe())
    shared.prepare(task, first, 0, 0, max_candidates=2)
    first.close(task)
    saved = path.read_text()
    contents = json.loads(saved)["cohorts"][shared.key]
    assert contents["baseline_identity"] == BASELINE_ID
    assert contents["scenes"][0]["baseline_result"]["passed"] is False
    fresh, second_probe = Task(), Probe(error=AssertionError("baseline must not rerun"))
    second = selector(args, second_probe)
    shared.prepare(fresh, second, 0, 0, max_candidates=2)
    assert fresh.setups == [10] and second_probe.calls == [] and fresh.plays == []
    second.close(fresh)
    assert path.read_text() == saved
    with pytest.raises(ValueError, match="Refusing to mix or overwrite"):
        SharedEvalSeedBank(path, args, "baseline_fail", 10, baseline_identity=other_id).prepare(
            Task(), selector(args, Probe(), identity=other_id), 0, 0, max_candidates=2)


def test_legacy_manifest_requires_same_baseline_identity(args, tmp_path):
    selected_by, task = selector(args, Probe()), Task()
    selected_by.prepare(task, 10, 0).record["success"] = True
    report = tmp_path / "report.json"
    selected_by.save_report(report, args["task_name"], args["task_config"], 1, 10, status="complete")
    assert load_scene_manifest(report, selected_by, args["task_name"], args["task_config"], 1) == [10]
    other = selector(args, Probe(), identity={"another_baseline": True})
    with pytest.raises(ValueError, match="baseline identity"):
        load_scene_manifest(report, other, args["task_name"], args["task_config"], 1)
    selected_by.close(task)


class TinyPolicy(torch.nn.Module):
    def __init__(self, num_inference_steps=10):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))
        self.num_inference_steps = num_inference_steps
        self.inputs = []
        self.resets = 0

    def predict_action(self, batch):
        self.inputs.append(batch)
        return {"action": torch.zeros((1, 2, 14))}

    def reset(self):
        self.resets += 1


def checkpoint(tmp_path, use_ema=True):
    path = tmp_path / "model500.ckpt"
    cfg = OmegaConf.create({"task_name": "place_empty_cup", "n_obs_steps": 2,
        "training": {"use_ema": use_ema}, "policy": {
            "_target_": "test_baseline_failure_selection.TinyPolicy", "num_inference_steps": 10}})
    payload = {"cfg": cfg, "state_dicts": {"model": {"weight": torch.tensor([1.])},
        "ema_model": {"weight": torch.tensor([2.])}, "optimizer": {"must_not_be_restored": True}},
        "pickles": {"epoch": dill.dumps(500)}}
    torch.save(payload, path, pickle_module=dill)
    return path


@pytest.mark.parametrize("use_ema", [False, True])
def test_loader_uses_checkpoint_cfg_and_exact_weights_not_modified_archive(tmp_path, use_ema):
    path = checkpoint(tmp_path, use_ema)
    archive = tmp_path / ".hydra/config.yaml"
    archive.parent.mkdir()
    archive.write_text("policy: {num_inference_steps: 2, n_layer: 3}\n")
    probe = FixedBaselineProbe(path, lambda *args: None, "place_empty_cup", device="cpu")
    torch.manual_seed(123)
    torch_before = torch.get_rng_state().clone()
    numpy_before, random_before = np.random.get_state(), random.getstate()
    probe._load()
    assert probe.policy.num_inference_steps == 10
    assert probe.policy.weight.item() == (2. if use_ema else 1.)
    assert not probe.policy.training and not probe.policy.weight.requires_grad
    assert torch.equal(torch_before, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
    assert random_before == random.getstate()


class RolloutTask:
    def __init__(self, success_at=None, stuck=False):
        self.take_action_cnt = 0
        self.step_lim = 5
        self.eval_success = False
        self.success_at, self.stuck = success_at, stuck
        self.obs_calls = 0

    def get_obs(self):
        self.obs_calls += 1
        return {"observation": {"head_camera": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}},
                "pointcloud": np.zeros((4, 6), dtype=np.float32),
                "joint_action": {"vector": np.full(14, self.take_action_cnt, dtype=np.float32)}}

    def take_action(self, action):
        if not self.stuck:
            self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at


@pytest.mark.parametrize("success_at,expected_steps", [(None, 5), (3, 3)])
def test_probe_uses_full_episode_limit_and_resets_history(tmp_path, success_at, expected_steps):
    probe = FixedBaselineProbe(checkpoint(tmp_path), lambda *args: None, "place_empty_cup", device="cpu")
    task = RolloutTask(success_at=success_at)
    result = probe(task, {})
    assert result == {"passed": success_at is not None, "action_steps": expected_steps, "step_limit": 5}
    assert len(probe.obs) == 0 and probe.policy.resets == 2
    initial = probe.policy.inputs[0]["agent_pos"]
    assert initial.shape == (1, 2, 14) and torch.equal(initial[:, 0], initial[:, 1])
    assert task.obs_calls == expected_steps + (expected_steps + 1) // 2


def test_checkpoint_mutation_and_stalled_inference_are_not_failures(tmp_path):
    path = checkpoint(tmp_path)
    probe = FixedBaselineProbe(path, lambda *args: None, "place_empty_cup", device="cpu")
    with pytest.raises(RuntimeError, match="no action progress"):
        probe(RolloutTask(stuck=True), {})
    assert len(probe.obs) == 0
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        probe(RolloutTask(), {})


def test_main_wires_fixed_baseline_then_reuses_its_public_failure_set(evaluator, args, tmp_path, monkeypatch):
    import script.eval_seed_bank as bank_module
    monkeypatch.setattr(bank_module, "_scene_identity", lambda *args: "test_scene")
    probes = []
    class FakeFixedProbe(Probe):
        checkpoint = "/fixed/baseline500.ckpt"
        identity = BASELINE_ID
        def __init__(self, *args, **kwargs):
            super().__init__()
            probes.append(self)
    monkeypatch.setattr(evaluator, "FixedBaselineProbe", FakeFixedProbe)
    monkeypatch.setattr(evaluator, "load_rl_scene_validation", lambda *args: (Validator(), "/rules"))
    monkeypatch.setattr(evaluator, "scene_validator_from_config", lambda *args: Validator())
    monkeypatch.setattr(evaluator, "build_scene_cache", lambda *args: None)
    monkeypatch.setattr(evaluator, "class_decorator", lambda *args: Task())
    monkeypatch.setattr(evaluator, "get_embodiment_config", lambda *args:
                        {"arm_joints_name": [["joint"], ["joint"]]})
    monkeypatch.setattr(evaluator, "get_camera_config", lambda *args: {"w": 640, "h": 480})
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *args, **kwargs:
                        SimpleNamespace(stdin=SimpleNamespace(close=lambda: None), wait=lambda: None))
    original_decorator = evaluator.eval_function_decorator
    for number in range(2):
        model = SimpleNamespace(run_dir=str(tmp_path / f"target{number}"), epoch=20)
        monkeypatch.setattr(evaluator, "eval_function_decorator", lambda policy, name:
                            (lambda args: model) if name == "get_model" else original_decorator(policy, name))
        evaluator.main({"task_name": "place_empty_cup", "task_config": "demo_messy_ur5",
            "ckpt_setting": "demo_messy_ur5", "policy_name": "fake", "instruction_type": "unseen",
            "seed": 10, "eval_num_episodes": 1, "eval_scene_selection": "baseline_fail",
            "eval_scene_seed_file": str(tmp_path / "shared.json")})
    assert probes[0].calls == [1100000] and probes[1].calls == []
    reports = [json.loads((tmp_path / f"target{number}/eval_results/epoch_20/scene_baseline_fail/_scene_selection.json").read_text())
               for number in range(2)]
    assert all(report["baseline_identity"] == BASELINE_ID for report in reports)
    assert reports[1]["selected_seeds"] == [1100000]
    assert reports[1]["episodes"][0]["success"] is True
    assert reports[1]["episodes"][0]["baseline_result"]["passed"] is False


def test_baseline_filename_with_equals_is_a_valid_hydra_override():
    from hydra import compose, initialize_config_dir
    from script.eval_scene_selection import MANIFLOW_ROOT
    path = "/shared/epoch=0500-val_loss=0.3.ckpt"
    with initialize_config_dir(config_dir=str(MANIFLOW_ROOT / "maniflow/config"), version_base=None):
        cfg = compose(config_name="reinflow_rl_pointcloud_robotwin2", overrides=[
            "rl.eval.scene_selection=baseline_fail", f'rl.eval.baseline_checkpoint="{path}"',
        ])
    assert cfg.rl.eval.baseline_checkpoint == path and cfg.rl.eval.scene_selection == "baseline_fail"
