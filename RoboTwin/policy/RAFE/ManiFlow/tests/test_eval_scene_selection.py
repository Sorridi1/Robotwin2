import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROBOTWIN_ROOT))

from script.eval_scene_selection import (
    EvalSceneSelector, build_scene_cache, evaluation_result_dir,
    load_rl_scene_validation, load_scene_manifest,
)
from maniflow.common.robotwin_seed_cache import ExpertSeedCache
from maniflow.env.robotwin.scene_validator import SceneValidationResult


class Unstable(Exception):
    pass


class Task:
    def __init__(self, outcomes=None, setup_error=None):
        self.outcomes = outcomes or {}
        self.setup_error = setup_error
        self.setups = []
        self.closes = []
        self.plays = []
        self.render_freq = 0

    def setup_demo(self, seed, **options):
        self.seed = seed
        self.setups.append(seed)
        self.render_freq = options["render_freq"]
        self.eval_video_path = options.get("eval_video_save_dir")
        if self.setup_error:
            raise self.setup_error
        self.state = "initial"
        self.plan_success = True
        self.info = {"info": {"object": str(seed)}}
        self.take_action_cnt = 0
        self.step_lim = 1
        self.eval_success = False

    def get_scene_metadata(self):
        assert self.state == "initial"
        return self.info

    def play_once(self):
        self.plays.append(self.seed)
        self.state = "expert_end"
        outcome = self.outcomes.get(self.seed, "success")
        if isinstance(outcome, Exception):
            raise outcome
        self.plan_success = outcome != "plan_failure"
        return self.info

    def check_success(self):
        return self.state == "expert_end" and self.outcomes.get(self.seed, "success") == "success"

    def close_env(self, clear_cache=False):
        self.closes.append((self.seed, clear_cache))
        self.state = "closed"

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_obs(self):
        assert self.state != "expert_end"
        return {"seed": self.seed}

    def _set_eval_video_ffmpeg(self, process):
        self.ffmpeg = process

    def _del_eval_video_ffmpeg(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()


class Validator:
    config = {"check_finite_state": True}

    def __init__(self, invalid=()):
        self.invalid = set(invalid)
        self.calls = []

    def validate(self, task):
        assert task.state == "initial"
        self.calls.append(task.seed)
        valid = task.seed not in self.invalid
        return SceneValidationResult(valid, "accepted" if valid else "severe_penetration",
                                     {"depth": np.float64(.001)})


@pytest.fixture
def args():
    return {"task_name": "place_empty_cup", "task_config": "demo_messy_ur5",
            "policy_name": "fake", "ckpt_setting": "demo_messy_ur5",
            "render_freq": 0, "clear_cache_freq": 5, "eval_mode": True}


def selector(mode, args, validator=None, cache=None):
    return EvalSceneSelector(mode, args, Unstable, validator or Validator(), cache)


def test_legacy_expert_filters_failure_and_rebuilds_success_before_policy(args):
    task = Task({10: "plan_failure"})
    selection = selector("expert", args)
    assert selection.prepare(task, 10, 0) is None
    selected = selection.prepare(task, 11, 0)
    assert task.setups == [10, 11, 11]
    assert task.plays == [10, 11]
    assert selection.validator.calls == []
    assert selected.record["expert_result"]["passed"] is True
    assert task.state == "initial"
    selection.close(task)


def test_rl_keeps_the_same_valid_initial_scene_without_expert(args):
    task = Task({10: "plan_failure"})
    selection = selector("rl", args)
    selected = selection.prepare(task, 10, 0)
    assert selected.episode_info == {"info": {"object": "10"}}
    assert task.setups == [10] and task.plays == [] and task.closes == []
    assert selection.validator.calls == [10]
    assert selected.record["expert_result"] is None
    selection.close(task)


def test_invalid_scene_is_rejected_before_any_expert_probe(args):
    task = Task()
    selection = selector("expert_fail", args, Validator(invalid=[10]))
    assert selection.prepare(task, 10, 0) is None
    assert task.plays == [] and task.setups == [10]
    assert not selection.active and task.state == "closed"


@pytest.mark.parametrize("failure", ["plan_failure", "task_failure"])
def test_expert_fail_accepts_normal_failures_and_rechecks_same_seed(args, failure):
    task = Task({10: failure})
    selection = selector("expert_fail", args)
    selected = selection.prepare(task, 10, 0)
    assert task.setups == [10, 10] and task.plays == [10]
    assert selection.validator.calls == [10, 10]
    assert selected.record["expert_result"]["passed"] is False
    assert task.state == "initial"
    selection.close(task)


def test_expert_success_is_excluded_from_expert_fail(args):
    task = Task()
    selection = selector("expert_fail", args)
    assert selection.prepare(task, 10, 0) is None
    assert task.setups == [10] and task.plays == [10]
    assert selection.stats["rejections"] == {"expert_success": 1}
    assert selection.records == []


def test_expert_exception_is_never_a_hard_scene_or_cached_failure(args, tmp_path):
    task = Task({10: RuntimeError("planner resource error")})
    cache = ExpertSeedCache(tmp_path, "cohort")
    selection = selector("expert_fail", args, cache=cache)
    assert selection.prepare(task, 10, 0) is None
    assert cache.lookup(10) is None
    assert selection.stats["rejections"] == {"expert_exception:RuntimeError": 1}
    assert not selection.active


@pytest.mark.parametrize("mode", ["rl", "expert_fail"])
def test_unknown_setup_errors_propagate_without_caching(args, tmp_path, mode):
    cache = ExpertSeedCache(tmp_path, "cohort")
    selection = selector(mode, args, cache=cache)
    task = Task(setup_error=RuntimeError("missing robot asset"))
    with pytest.raises(RuntimeError, match="missing robot asset"):
        selection.prepare(task, 10, 0)
    assert cache.lookup(10) is None
    assert not selection.active


def test_unstable_initial_scene_is_rejected(args):
    selection = selector("rl", args)
    assert selection.prepare(Task(setup_error=Unstable()), 10, 0) is None
    assert selection.stats["rejections"] == {"unstable_setup": 1}


def test_rebuilt_expert_failure_must_still_pass_cheap_validation(args):
    class RejectRebuild(Validator):
        def validate(self, task):
            result = super().validate(task)
            return result if len(self.calls) == 1 else SceneValidationResult(False, "initial_success")

    task = Task({10: "plan_failure"})
    selection = selector("expert_fail", args, RejectRebuild())
    assert selection.prepare(task, 10, 0) is None
    assert task.setups == [10, 10]
    assert selection.records == [] and not selection.active


def test_cached_expert_failures_skip_expert_but_recheck_fresh_scene(args, tmp_path):
    cache = ExpertSeedCache(tmp_path, "cohort")
    first = selector("expert_fail", args, cache=cache)
    first_task = Task({10: "plan_failure"})
    assert first.prepare(first_task, 10, 0)
    first.close(first_task)
    second = selector("expert_fail", args, cache=cache)
    task = Task()  # Expert would succeed if called; the recorded failure is reused.
    assert second.prepare(task, 10, 0).record["expert_result"]["passed"] is False
    assert task.setups == [10] and task.plays == []
    assert second.validator.calls == [10]
    second.close(task)


def test_cached_positive_never_bypasses_fresh_validity_and_negative_skips_setup(args, tmp_path):
    cache = ExpertSeedCache(tmp_path, "cohort")
    first = selector("rl", args, cache=cache)
    task = Task()
    assert first.prepare(task, 10, 0)
    first.close(task)
    second = selector("rl", args, Validator(invalid=[10]), cache)
    assert second.prepare(task, 10, 0) is None
    assert cache.lookup(10)[0] is False
    third = selector("rl", args, cache=cache)
    fresh = Task()
    assert third.prepare(fresh, 10, 0) is None
    assert fresh.setups == []


def test_cache_is_shared_across_models_but_isolates_selection_and_thresholds(args, tmp_path):
    a = build_scene_cache(args, "rl", Validator(), tmp_path)
    other_model = {**args, "policy_name": "another_model", "eval_video_save_dir": "/unused/output"}
    b = build_scene_cache(other_model, "rl", Validator(), tmp_path)
    c = build_scene_cache(args, "expert_fail", Validator(), tmp_path)
    validator = Validator()
    validator.config = {"check_finite_state": False}
    d = build_scene_cache(args, "rl", validator, tmp_path)
    assert a.directory == b.directory
    assert len({a.directory, c.directory, d.directory}) == 3


def write_config(path, tolerance=.05, mode="cheap"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"rl": {"env": {"scene_validation": {
        "mode": mode, "joint_limit_tolerance": tolerance,
    }}}}))


def test_validation_uses_archived_rules_and_allows_shared_baseline_config(tmp_path):
    archived = tmp_path / ".hydra/config.yaml"
    write_config(archived, .031)
    validator, source = load_rl_scene_validation(tmp_path)
    assert validator.config["joint_limit_tolerance"] == .031
    assert source == str(archived)
    shared = tmp_path / "shared.yaml"
    write_config(shared, .017)
    validator, source = load_rl_scene_validation(tmp_path, shared)
    assert validator.config["joint_limit_tolerance"] == .017
    write_config(shared, mode="expert")
    with pytest.raises(ValueError, match="mode=cheap"):
        load_rl_scene_validation(tmp_path, shared)


def test_baseline_without_rl_config_uses_documented_repository_fallback(tmp_path):
    archived = tmp_path / ".hydra/config.yaml"
    archived.parent.mkdir()
    archived.write_text("policy: {}\n")
    validator, source = load_rl_scene_validation(tmp_path)
    assert validator.config["mode"] == "cheap"
    assert source.endswith("reinflow_rl_pointcloud_robotwin2.yaml")
    with pytest.raises(ValueError, match="no rl.env.scene_validation"):
        load_rl_scene_validation(tmp_path, archived)


def test_report_can_replay_same_cohort_and_refuses_other_rules(args, tmp_path):
    selection = selector("rl", args)
    selected = selection.prepare(Task(), 10, 0)
    selected.record["success"] = False
    report = tmp_path / "report.json"
    selection.save_report(report, "place_empty_cup", "demo_messy_ur5", 1, 10, status="complete")
    data = json.loads(report.read_text())
    assert data["selected_seeds"] == [10] and data["successes"] == 0
    assert load_scene_manifest(report, selection, "place_empty_cup", "demo_messy_ur5", 1) == [10]
    with pytest.raises(ValueError, match="scene_selection"):
        load_scene_manifest(report, selector("expert_fail", args), "place_empty_cup", "demo_messy_ur5", 1)
    with pytest.raises(ValueError, match="enough distinct"):
        load_scene_manifest(report, selection, "place_empty_cup", "demo_messy_ur5", 2)
    selection.validator.config = {"changed": True}
    with pytest.raises(ValueError, match="validation_config"):
        load_scene_manifest(report, selection, "place_empty_cup", "demo_messy_ur5", 1)


def test_cohort_result_paths_keep_old_results_separate(tmp_path):
    assert evaluation_result_dir(tmp_path, "expert") == tmp_path
    assert evaluation_result_dir(tmp_path, "rl") == tmp_path / "scene_rl"
    assert evaluation_result_dir(tmp_path, "expert_fail") == tmp_path / "scene_expert_fail"
    with pytest.raises(ValueError, match="eval_scene_selection"):
        evaluation_result_dir(tmp_path, "policy_success_only")


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.chdir(ROBOTWIN_ROOT)
    module = importlib.import_module("script.eval_policy")
    monkeypatch.setattr(module, "generate_episode_descriptions",
                        lambda *args: [{"unseen": ["move the object"]}])
    def evaluate(task, model, obs):
        task.take_action_cnt += 1
        task.eval_success = task.seed % 2 == 0
    monkeypatch.setattr(module, "eval_function_decorator",
                        lambda policy, name: evaluate if name == "eval" else lambda model: None)
    return module


def test_real_eval_loop_counts_policy_failures_and_keeps_selected_seeds(evaluator, args, tmp_path):
    selection = selector("rl", args, Validator(invalid=[10]))
    task = Task()
    report = tmp_path / "report.json"
    end, successes = evaluator.eval_policy("place_empty_cup", task, args, object(), 10,
        test_num=2, instruction_type="unseen", selector=selection, report_path=report)
    assert (end, successes) == (13, 1)
    assert task.setups == [10, 11, 12] and task.plays == []
    data = json.loads(report.read_text())
    assert data["status"] == "complete" and data["completed_episodes"] == 2
    assert data["selected_seeds"] == [11, 12]
    assert [record["success"] for record in data["episodes"]] == [False, True]
    assert not selection.active


def test_candidate_budget_aborts_instead_of_reporting_partial_rate(evaluator, args, tmp_path):
    task = Task()
    selection = selector("expert_fail", args)
    report = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="No partial success rate"):
        evaluator.eval_policy("place_empty_cup", task, args, object(), 10, test_num=1,
            instruction_type="unseen", selector=selection, report_path=report, max_seed_tries=3)
    data = json.loads(report.read_text())
    assert data["status"] == "aborted" and data["completed_episodes"] == 0
    assert data["stats"]["attempts"] == 3
    assert task.setups == [10, 11, 12] and not selection.active


def test_manifest_rejection_never_silently_substitutes_another_seed(evaluator, args):
    selection = selector("rl", args, Validator(invalid=[30]))
    task = Task()
    with pytest.raises(RuntimeError, match="refusing to replace"):
        evaluator.eval_policy("place_empty_cup", task, args, object(), 10, test_num=1,
            instruction_type="unseen", selector=selection, manifest_seeds=[30])
    assert task.setups == [30]


def test_video_and_scene_cleanup_on_policy_exception(evaluator, args, monkeypatch, tmp_path):
    args["eval_video_save_dir"] = tmp_path
    selection = selector("rl", args)
    task = Task()
    events = []
    def fail(*unused):
        raise RuntimeError("policy error")
    monkeypatch.setattr(evaluator, "eval_function_decorator",
                        lambda policy, name: fail if name == "eval" else lambda model: None)
    process = SimpleNamespace(stdin=SimpleNamespace(close=lambda: events.append("close")),
                              wait=lambda: events.append("wait"))
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *args, **kwargs: process)
    report = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="policy error"):
        evaluator.eval_policy("place_empty_cup", task, args, object(), 10, test_num=1,
            instruction_type="unseen", selector=selection, report_path=report, video_size="640x480")
    assert events == ["close", "wait"] and not selection.active
    data = json.loads(report.read_text())
    assert data["status"] == "aborted" and data["episodes"][0]["success"] is None


@pytest.mark.parametrize("mode", ["expert", "rl", "expert_fail"])
def test_main_propagates_video_output_and_keeps_other_cohort_results(
    evaluator, monkeypatch, tmp_path, mode,
):
    task = Task({1100000: "plan_failure"} if mode == "expert_fail" else {})
    model = SimpleNamespace(run_dir=str(tmp_path), epoch=20)
    original_decorator = evaluator.eval_function_decorator
    monkeypatch.setattr(evaluator, "eval_function_decorator", lambda policy, name:
                        (lambda usr_args: model) if name == "get_model" else original_decorator(policy, name))
    monkeypatch.setattr(evaluator, "class_decorator", lambda name: task)
    monkeypatch.setattr(evaluator, "get_embodiment_config", lambda *args:
                        {"arm_joints_name": [["joint"], ["joint"]]})
    monkeypatch.setattr(evaluator, "get_camera_config", lambda *args: {"w": 640, "h": 480})
    monkeypatch.setattr(evaluator, "load_rl_scene_validation", lambda *args: (Validator(), "/rules.yaml"))
    monkeypatch.setattr(evaluator, "build_scene_cache", lambda *args: None)
    import script.eval_seed_bank as bank_module
    monkeypatch.setattr(bank_module, "_scene_identity", lambda *args: "test_scene_identity")
    videos = []
    def video_process(command, **kwargs):
        videos.append(command[-1])
        return SimpleNamespace(stdin=SimpleNamespace(close=lambda: None), wait=lambda: None)
    monkeypatch.setattr(evaluator.subprocess, "Popen", video_process)
    original_result = tmp_path / "eval_results/epoch_20/_result.txt"
    original_result.parent.mkdir(parents=True)
    original_result.write_text("0.84")
    evaluator.main({"task_name": "place_empty_cup", "task_config": "demo_messy_ur5",
                    "ckpt_setting": "demo_messy_ur5", "policy_name": "fake",
                    "instruction_type": "unseen", "seed": 10,
                    "eval_num_episodes": 1, "eval_scene_selection": mode,
                    "eval_scene_seed_file": str(tmp_path / "shared_seeds.json")})
    result_dir = evaluation_result_dir(original_result.parent, mode)
    assert videos == [str(result_dir / "episode0.mp4")]
    assert (result_dir / "_result.txt").read_text().endswith("1.0")
    report = json.loads((result_dir / "_scene_selection.json").read_text())
    assert report["status"] == "complete" and report["selected_seeds"] == [1100000]
    if mode != "expert":
        assert original_result.read_text() == "0.84"


def test_resume_refuses_to_mix_saved_best_scores_from_other_cohorts(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace
    import maniflow.workspace.train_reinflow_rl_robotwin2_workspace as module

    workspace = TrainReinFlowRLRoboTwinWorkspace.__new__(TrainReinFlowRLRoboTwinWorkspace)
    workspace.cfg = OmegaConf.create({"rl": {"eval": {"scene_selection": "expert_fail"}}})
    path = tmp_path / "checkpoint.ckpt"
    path.write_text("placeholder")
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs:
                        {"cfg": {"rl": {"eval": {}}}, "best_eval_score": .84})
    with pytest.raises(ValueError, match="Cannot compare saved best score"):
        workspace.load_resume_checkpoint(path)


@pytest.mark.parametrize("mode", ["expert", "rl", "expert_fail"])
def test_periodic_eval_forwards_selection_and_reads_only_its_own_results(monkeypatch, tmp_path, mode):
    from omegaconf import OmegaConf
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace
    import maniflow.workspace.train_reinflow_rl_robotwin2_workspace as module

    workspace = TrainReinFlowRLRoboTwinWorkspace.__new__(TrainReinFlowRLRoboTwinWorkspace)
    workspace._output_dir = str(tmp_path)
    workspace.itr = 19
    workspace.best_eval_score = float("-inf")
    workspace.best_eval_itr = -1
    workspace.cfg = OmegaConf.create({"task_name": "place_empty_cup", "task_config": "demo_messy_ur5",
        "expert_data_num": 50, "training": {"seed": 0}, "rl": {"eval": {
        "enabled": True, "scene_selection": mode, "num_episodes": 2,
        "max_seed_tries": 30, "addition_info": "test", "scene_validation_config": "/shared/config.yaml",
    }}})
    exported = []
    def export(tag, itr):
        path = tmp_path / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(exist_ok=True)
        path.write_text("test actor")
        exported.append(tag)
        return str(path)
    workspace.export_actor_checkpoint = export
    original = tmp_path / "eval_results/epoch_20/_result.txt"
    original.parent.mkdir(parents=True)
    original.write_text("0.84")
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        result = evaluation_result_dir(original.parent, mode) / "_result.txt"
        result.parent.mkdir(exist_ok=True)
        result.write_text("Scene selection: " + mode + "\nEpisodes: 2\n0.5")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    metrics = workspace.run_eval_and_update_best(20)
    command = commands[0]
    assert command[command.index("--eval_scene_selection") + 1] == mode
    assert command[command.index("--eval_num_episodes") + 1] == "2"
    assert command[command.index("--eval_scene_validation_config") + 1] == "/shared/config.yaml"
    assert command[command.index("--eval_scene_seed_file") + 1].endswith("/outputs/eval_scene_seeds.json")
    assert command[command.index("--eval_baseline_checkpoint") + 1].endswith("epoch=0500-val_loss=0.373935.ckpt")
    assert metrics["eval/success_rate"] == .5
    assert exported == ["eval_candidate", "best"]
    if mode != "expert":
        assert original.read_text() == "0.84"


@pytest.mark.parametrize("script", ["train_eval_robotwin2.sh", "scripts/train_rl_policy.sh"])
def test_shell_eval_only_forwards_scene_options_without_training(tmp_path, script):
    fake_python = tmp_path / "python"
    arguments = tmp_path / "arguments.json"
    fake_python.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                           "open(os.environ['TEST_ARGUMENTS'], 'w').write(json.dumps(sys.argv[1:]))\n")
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "PYTHON_BIN": str(fake_python), "TEST_ARGUMENTS": str(arguments),
           "TRAIN": "false", "EVAL": "true", "EVAL_SCENE_SELECTION": "expert_fail",
           "EVAL_NUM_EPISODES": "2", "EVAL_MAX_SEED_TRIES": "30", "EVAL_SEED": "10",
           "EVAL_SCENE_VALIDATION_CONFIG": "/shared/config.yaml",
           "EVAL_SCENE_SEED_FILE": "/shared/common_seeds.json",
           "EVAL_BASELINE_CHECKPOINT": "/shared/epoch=0500-val_loss=0.3.ckpt",
           "EVAL_SCENE_MANIFEST": "/shared/seeds.json", "EVAL_CKPT_TAG": "best"}
    policy_dir = ROBOTWIN_ROOT / "policy/ManiFlow"
    if script.startswith("scripts/"):
        positional = ["reinflow_rl_pointcloud_robotwin2", "place_empty_cup", "1016", "50",
                      "test", "0", "1", "/unused/pretrained.ckpt", "demo_messy_ur5"]
    else:
        positional = ["place_empty_cup", "reinflow_rl_pointcloud_robotwin2", "demo_messy_ur5",
                      "50", "test", "0", "1"]
    result = subprocess.run(["bash", str(policy_dir / script), *positional], cwd=ROBOTWIN_ROOT,
                            env=env, capture_output=True, text=True, check=True)
    command = json.loads(arguments.read_text())
    assert command[0] == "script/eval_policy.py"
    assert command[command.index("--eval_scene_selection") + 1] == "expert_fail"
    assert command[command.index("--eval_scene_manifest") + 1] == "/shared/seeds.json"
    assert command[command.index("--eval_scene_seed_file") + 1] == "/shared/common_seeds.json"
    assert command[command.index("--eval_baseline_checkpoint") + 1] == "/shared/epoch=0500-val_loss=0.3.ckpt"
    assert command[command.index("--seed") + 1] == "10"
    assert "Training is disabled" in result.stdout or "RL training disabled" in result.stdout


def test_training_script_forwards_scene_options_to_periodic_eval(tmp_path):
    fake_python = tmp_path / "python"
    arguments = tmp_path / "arguments.json"
    fake_python.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                           "open(os.environ['TEST_ARGUMENTS'], 'w').write(json.dumps(sys.argv[1:]))\n")
    fake_python.chmod(0o755)
    env = {**os.environ, "PYTHON_BIN": str(fake_python), "TEST_ARGUMENTS": str(arguments),
           "TRAIN": "true", "EVAL": "false", "EVAL_SCENE_SELECTION": "rl",
           "EVAL_NUM_EPISODES": "2", "EVAL_MAX_SEED_TRIES": "30", "EVAL_SEED": "10",
           "EVAL_SCENE_CACHE_MODE": "off"}
    env["EVAL_SCENE_SEED_FILE"] = "/shared/common_seeds.json"
    env["EVAL_BASELINE_CHECKPOINT"] = "/shared/epoch=0500-val_loss=0.3.ckpt"
    script = ROBOTWIN_ROOT / "policy/ManiFlow/scripts/train_rl_policy.sh"
    subprocess.run(["bash", str(script), "reinflow_rl_pointcloud_robotwin2", "place_empty_cup",
                    "1016", "50", "test", "0", "1", "/unused/pretrained.ckpt", "demo_messy_ur5"],
                   cwd=ROBOTWIN_ROOT, env=env, capture_output=True, text=True, check=True)
    command = json.loads(arguments.read_text())
    assert command[0] == "train_reinflow_rl_robotwin2_workspace.py"
    assert "rl.eval.scene_selection=rl" in command
    assert "rl.eval.scene_cache_mode=off" in command
    assert "rl.eval.scene_seed_file=/shared/common_seeds.json" in command
    assert 'rl.eval.baseline_checkpoint="/shared/epoch=0500-val_loss=0.3.ckpt"' in command
    assert "rl.eval.num_episodes=2" in command and "rl.eval.seed=10" in command
