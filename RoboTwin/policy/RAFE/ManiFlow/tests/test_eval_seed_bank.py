from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from test_eval_scene_selection import Task, Validator, Unstable, args, evaluator, selector
from script.eval_seed_bank import DEFAULT_SHARED_SEED_FILE, SharedEvalSeedBank, _scene_identity


def bank(tmp_path, args, mode="rl", start=10):
    return SharedEvalSeedBank(tmp_path / "common_seeds.json", args, mode, start)


def cohort(shared):
    return json.loads(shared.path.read_text())["cohorts"][shared.key]


def test_default_file_is_shared_not_a_training_directory():
    assert DEFAULT_SHARED_SEED_FILE == "/media/Elements1/ljj/ManiFlow/outputs/eval_scene_seeds.json"


def test_first_run_appends_before_policy_and_other_models_replay_same_prefix(tmp_path, args):
    shared = bank(tmp_path, args)
    task = Task()
    first = selector("rl", args, Validator(invalid=[10]))
    selected = shared.prepare(task, first, 0, 0, max_candidates=3)
    assert selected.record["seed"] == 11 and task.setups == [10, 11]
    saved = shared.path.read_text()
    assert [entry["seed"] for entry in cohort(shared)["scenes"]] == [11]
    assert "success" not in saved
    selected.record["success"] = False  # This model's failure must never affect the common bank.
    first.close(task)
    other_args = {**args, "policy_name": "different_model", "ckpt_setting": "different_checkpoint"}
    other = bank(tmp_path, other_args, start=999)
    fresh = Task()
    selection = selector("rl", other_args)
    replay = other.prepare(fresh, selection, 0, 0, max_candidates=3)
    assert replay.record["seed"] == 11 and replay.record["success"] is None
    assert fresh.setups == [11] and fresh.plays == []
    assert selection.stats["shared_bank_hits"] == 1
    assert other.path.read_text() == saved
    selection.close(fresh)


def test_growth_only_appends_and_ignores_later_initial_seed(tmp_path, args):
    first = bank(tmp_path, args)
    selection, task = selector("rl", args), Task()
    first.prepare(task, selection, 0, 0, max_candidates=2)
    selection.close(task)
    prefix = copy.deepcopy(cohort(first)["scenes"])
    later = bank(tmp_path, args, start=500)
    found, validation = later.validation_config()
    assert found and validation == selection.validator.config and later.start_seed == 10
    later.prepare(task, selection, 1, 1, max_candidates=2)
    data = cohort(later)
    assert data["scenes"][:1] == prefix
    assert [entry["seed"] for entry in data["scenes"]] == [10, 11]
    assert data["next_seed"] == 12
    selection.close(task)


@pytest.mark.parametrize("mode", ["expert", "expert_fail"])
def test_recorded_expert_labels_are_reused_without_running_expert_again(tmp_path, args, mode):
    shared = bank(tmp_path, args, mode)
    task = Task({10: "plan_failure"} if mode == "expert_fail" else {})
    first = selector(mode, args)
    shared.prepare(task, first, 0, 0, max_candidates=2)
    assert task.plays == [10] and task.setups == [10, 10]
    first.close(task)
    fresh, second = Task(), selector(mode, args)
    selected = shared.prepare(fresh, second, 0, 0, max_candidates=2)
    assert fresh.plays == [] and fresh.setups == [10]
    assert selected.record["expert_result"]["passed"] is (mode == "expert")
    if mode == "expert_fail":
        assert second.validator.calls == [10]
    second.close(fresh)


def test_tasks_and_selection_modes_share_one_file_but_have_distinct_groups(tmp_path, args):
    for mode in ("rl", "expert", "expert_fail"):
        shared = bank(tmp_path, args, mode)
        task = Task({10: "plan_failure"} if mode == "expert_fail" else {})
        selection = selector(mode, args)
        shared.prepare(task, selection, 0, 0, max_candidates=2)
        selection.close(task)
    other_args = {**args, "task_name": "another_task"}
    other = bank(tmp_path, other_args)
    task, selection = Task(), selector("rl", other_args)
    other.prepare(task, selection, 0, 0, max_candidates=2)
    selection.close(task)
    data = json.loads(other.path.read_text())
    assert len(data["cohorts"]) == 4
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_invalid_replay_never_drops_or_replaces_seed(tmp_path, args):
    shared = bank(tmp_path, args)
    first, task = selector("rl", args), Task()
    shared.prepare(task, first, 0, 0, max_candidates=2)
    first.close(task)
    saved = shared.path.read_text()
    later, fresh = selector("rl", args, Validator(invalid=[10])), Task()
    with pytest.raises(RuntimeError, match="refusing to replace"):
        shared.prepare(fresh, later, 0, 0, max_candidates=2)
    assert fresh.setups == [10] and not later.active
    assert shared.path.read_text() == saved


def test_setup_error_does_not_advance_cursor_or_mark_seed_invalid(tmp_path, args):
    shared = bank(tmp_path, args)
    selection = selector("rl", args)
    with pytest.raises(RuntimeError, match="missing asset"):
        shared.prepare(Task(setup_error=RuntimeError("missing asset")), selection, 0, 0, max_candidates=2)
    assert cohort(shared)["scenes"] == [] and cohort(shared)["next_seed"] == 10
    assert not selection.active


def test_exhausted_search_is_resumed_from_saved_cursor(tmp_path, args):
    shared, first = bank(tmp_path, args, "expert_fail"), selector("expert_fail", args)
    task = Task({12: "plan_failure"})
    with pytest.raises(RuntimeError, match="No partial success rate"):
        shared.prepare(task, first, 0, 0, max_candidates=2)
    assert cohort(shared)["next_seed"] == 12 and cohort(shared)["scenes"] == []
    later = selector("expert_fail", args)
    selected = shared.prepare(task, later, 0, 0, max_candidates=2)
    assert selected.record["seed"] == 12 and task.plays == [10, 11, 12]
    later.close(task)


def test_concurrent_models_do_not_overwrite_or_duplicate_first_seed(tmp_path, args):
    barrier = threading.Barrier(2)
    class SlowTask(Task):
        def play_once(self):
            time.sleep(.03)
            return super().play_once()
    def prepare():
        shared = bank(tmp_path, args, "expert_fail")
        task = SlowTask({10: "plan_failure"})
        selection = selector("expert_fail", args)
        barrier.wait(timeout=10)
        selected = shared.prepare(task, selection, 0, 0, max_candidates=2)
        selection.close(task)
        return selected.record["seed"], len(task.plays)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare(), range(2)))
    assert [seed for seed, _ in results] == [10, 10]
    assert sum(count for _, count in results) == 1
    assert len(cohort(bank(tmp_path, args, "expert_fail"))["scenes"]) == 1


def test_replay_does_not_wait_for_an_unrelated_append_lock(tmp_path, args):
    shared, selection, task = bank(tmp_path, args), selector("rl", args), Task()
    shared.prepare(task, selection, 0, 0, max_candidates=2)
    selection.close(task)
    def replay():
        other, fresh = selector("rl", args), Task()
        selected = bank(tmp_path, args).prepare(fresh, other, 0, 0, max_candidates=2)
        other.close(fresh)
        return selected.record["seed"]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with shared._locked():
            assert executor.submit(replay).result(timeout=3) == 10


def test_validation_or_physical_changes_cannot_silently_replace_common_cohort(tmp_path, args):
    shared, first, task = bank(tmp_path, args), selector("rl", args), Task()
    shared.prepare(task, first, 0, 0, max_candidates=2)
    first.close(task)
    saved = shared.path.read_text()
    changed = Validator()
    changed.config = {"different_threshold": True}
    with pytest.raises(ValueError, match="Refusing to mix or overwrite"):
        bank(tmp_path, args).prepare(Task(), selector("rl", args, changed), 0, 0, max_candidates=2)
    changed_args = {**args, "domain_randomization": {"cluttered_table": False}}
    with pytest.raises(ValueError, match="Refusing to mix or overwrite"):
        bank(tmp_path, changed_args).prepare(Task(), selector("rl", changed_args), 0, 0, max_candidates=2)
    assert shared.path.read_text() == saved


def test_identity_ignores_model_and_logging_but_not_physics(args):
    identity = _scene_identity(args, "rl", Validator.config)
    changed = {**args, "policy_name": "another", "save_path": "/other",
               "eval_video_log": False, "eval_video_save_dir": "/other/video", "render_freq": 10}
    assert _scene_identity(changed, "rl", Validator.config) == identity
    changed["action_type"] = "different"
    assert _scene_identity(changed, "rl", Validator.config) != identity


@pytest.mark.parametrize("content", ["{broken json", '{"version": 2, "cohorts": {}}'])
def test_corrupt_common_file_is_not_overwritten(tmp_path, args, content):
    shared = bank(tmp_path, args)
    shared.path.write_text(content)
    with pytest.raises(ValueError):
        shared.prepare(Task(), selector("rl", args), 0, 0, max_candidates=2)
    assert shared.path.read_text() == content


def test_failed_atomic_append_keeps_previous_file_and_closes_scene(tmp_path, args, monkeypatch):
    import script.eval_seed_bank as module
    shared, selection, task = bank(tmp_path, args), selector("rl", args), Task()
    shared.prepare(task, selection, 0, 0, max_candidates=2)
    selection.close(task)
    saved = shared.path.read_text()
    def fail(*args):
        raise OSError("disk write failed")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="disk write failed"):
        shared.prepare(task, selection, 1, 1, max_candidates=2)
    assert shared.path.read_text() == saved and not selection.active
    assert not list(tmp_path.glob(".common_seeds.json-*"))


def test_policy_failure_or_exception_never_changes_bank_membership(evaluator, args, tmp_path, monkeypatch):
    shared = bank(tmp_path, args)
    def fail(*args):
        raise RuntimeError("policy crashed")
    monkeypatch.setattr(evaluator, "eval_function_decorator",
                        lambda policy, name: fail if name == "eval" else lambda model: None)
    selection = selector("rl", args)
    with pytest.raises(RuntimeError, match="policy crashed"):
        evaluator.eval_policy("place_empty_cup", Task(), args, object(), 10, test_num=1,
            instruction_type="unseen", selector=selection, seed_bank=shared,
            report_path=tmp_path / "report.json")
    data = cohort(shared)
    assert [entry["seed"] for entry in data["scenes"]] == [10]
    assert "success" not in shared.path.read_text()
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "aborted"


def test_two_models_use_shared_file_without_a_manifest_or_model_archive(evaluator, args, tmp_path, monkeypatch):
    import script.eval_seed_bank as module
    monkeypatch.setattr(module, "_scene_identity", lambda *args: "test_physics")
    calls = []
    def load_validation(*args):
        calls.append("archive")
        return Validator(), "/first_model/config.yaml"
    monkeypatch.setattr(evaluator, "load_rl_scene_validation", load_validation)
    monkeypatch.setattr(evaluator, "scene_validator_from_config", lambda config: Validator())
    monkeypatch.setattr(evaluator, "build_scene_cache", lambda *args: None)
    monkeypatch.setattr(evaluator, "class_decorator", lambda name: Task())
    monkeypatch.setattr(evaluator, "get_embodiment_config", lambda *args:
                        {"arm_joints_name": [["joint"], ["joint"]]})
    monkeypatch.setattr(evaluator, "get_camera_config", lambda *args: {"w": 640, "h": 480})
    monkeypatch.setattr(evaluator.subprocess, "Popen", lambda *args, **kwargs:
                        SimpleNamespace(stdin=SimpleNamespace(close=lambda: None), wait=lambda: None))
    original_decorator = evaluator.eval_function_decorator
    shared_path = tmp_path / "common.json"
    for number in range(2):
        model = SimpleNamespace(run_dir=str(tmp_path / f"model{number}"), epoch=20)
        monkeypatch.setattr(evaluator, "eval_function_decorator", lambda policy, name:
                            (lambda usr_args: model) if name == "get_model" else original_decorator(policy, name))
        evaluator.main({"task_name": "place_empty_cup", "task_config": "demo_messy_ur5",
                        "ckpt_setting": "demo_messy_ur5", "policy_name": "fake",
                        "instruction_type": "unseen", "seed": 10 + number,
                        "eval_num_episodes": 1, "eval_scene_selection": "rl",
                        "eval_scene_seed_file": str(shared_path)})
    assert calls == ["archive"]
    reports = [json.loads((tmp_path / f"model{number}/eval_results/epoch_20/scene_rl/_scene_selection.json").read_text())
               for number in range(2)]
    assert reports[0]["selected_seeds"] == reports[1]["selected_seeds"] == [1100000]
    assert reports[1]["shared_seed_file"] == str(shared_path)
    assert reports[1]["stats"]["expert_checks"] == 0
