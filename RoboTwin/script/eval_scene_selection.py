"""Policy-independent evaluation cohorts, sharing the online RL scene checks."""

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback

import yaml


ROBOTWIN_ROOT = Path(__file__).resolve().parents[1]
MANIFLOW_ROOT = ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow"
SCENE_SELECTION_MODES = ("expert", "rl", "expert_fail", "baseline_fail")


def check_selection_mode(mode):
    if mode not in SCENE_SELECTION_MODES:
        raise ValueError(f"eval_scene_selection must be one of {SCENE_SELECTION_MODES}, got {mode!r}")
    return mode


def evaluation_result_dir(epoch_dir, mode):
    """Keep the legacy expert path; isolate different cohorts without timestamps."""
    check_selection_mode(mode)
    path = Path(epoch_dir)
    return path if mode == "expert" else path / f"scene_{mode}"


def _maniflow_imports():
    # Do not import the ManiFlow policy/model (or require it for legacy evaluation).
    if str(MANIFLOW_ROOT) not in sys.path:
        sys.path.append(str(MANIFLOW_ROOT))
    from maniflow.common.robotwin_seed_cache import ExpertSeedCache, seed_cache_fingerprint
    from maniflow.env.robotwin.scene_validator import CheapSceneValidator
    return CheapSceneValidator, ExpertSeedCache, seed_cache_fingerprint


def scene_validator_from_config(config):
    if not isinstance(config, dict) or config.get("mode", "cheap") != "cheap":
        raise ValueError("Shared RL evaluation rules must use scene_validation.mode=cheap")
    validator_class, _, _ = _maniflow_imports()
    return validator_class(config)


def load_rl_scene_validation(run_dir=None, config_path=None):
    """Prefer the archived RL rules; an explicit path lets baselines use identical rules."""
    explicit = config_path not in (None, "", "null", "None")
    candidates = ([Path(config_path).expanduser()] if explicit else [])
    if not explicit and run_dir is not None:
        candidates.append(Path(run_dir) / ".hydra/config.yaml")
    if not explicit:
        candidates.append(MANIFLOW_ROOT / "maniflow/config/reinflow_rl_pointcloud_robotwin2.yaml")
    for path in candidates:
        if not path.is_file():
            if explicit:
                raise FileNotFoundError(f"Scene validation config does not exist: {path}")
            continue
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        env = (config or {}).get("rl", {}).get("env", {})
        if "scene_validation" not in env:
            if explicit:
                raise ValueError(f"{path} has no rl.env.scene_validation configuration")
            continue
        validation = dict(env["scene_validation"])
        mode = validation.get("mode", "expert" if env.get("expert_check", True) else "none")
        if mode != "cheap":
            raise ValueError(f"RL-aligned evaluation requires scene_validation.mode=cheap, got {mode!r} in {path}")
        validator_class, _, _ = _maniflow_imports()
        validator = validator_class(validation)
        return validator, str(path.resolve())
    raise ValueError("Could not find the online RL scene validation configuration")


def build_scene_cache(args, mode, validator, directory, cache_mode="readwrite", baseline_identity=None):
    """Separate evaluation/selection modes and thresholds from all training caches."""
    if mode == "expert":
        return None
    _, cache_class, fingerprint_fn = _maniflow_imports()
    if cache_mode == "off":
        return cache_class(None, "disabled", mode="off")
    ignored = {"ckpt_setting", "policy_name", "save_path", "eval_video_save_dir",
               "eval_video_log", "render_freq", "clear_cache_freq", "log_action_steps"}
    identity = {key: value for key, value in args.items() if key not in ignored}
    identity.update(eval_scene_selection=mode, scene_validation=validator.config)
    if mode == "baseline_fail":
        if baseline_identity is None:
            raise ValueError("baseline_fail cache requires a fixed baseline identity")
        identity["baseline_identity"] = baseline_identity
    sources = list((ROBOTWIN_ROOT / "envs").rglob("*.py"))
    sources.extend([
        Path(__file__), MANIFLOW_ROOT / "maniflow/env/robotwin/scene_validator.py",
        ROBOTWIN_ROOT / "task_config" / f"{args['task_config']}.yml",
        ROBOTWIN_ROOT / "task_config/_camera_config.yml",
        ROBOTWIN_ROOT / "task_config/_embodiment_config.yml",
    ])
    for side in ("left", "right"):
        if args.get(f"{side}_robot_file"):
            sources.append(Path(args[f"{side}_robot_file"]) / "config.yml")
    fingerprint = fingerprint_fn(identity, sources, namespace="eval-scene-selection-v1")
    return cache_class(directory, fingerprint, mode=cache_mode)


@dataclass
class SelectedScene:
    episode_info: dict
    record: dict


class EvalSceneSelector:
    """Selection may use a fixed baseline, never the model being evaluated."""

    def __init__(self, mode, args, unstable_error, validator=None, cache=None,
                 baseline_probe=None, baseline_identity=None):
        self.mode = check_selection_mode(mode)
        self.args = copy.deepcopy(args)
        self.unstable_error = unstable_error
        self.validator = validator
        if mode != "expert" and validator is None:
            raise ValueError(f"{mode} selection requires the shared RL cheap validator")
        self.cache = cache
        self.baseline_probe = baseline_probe
        self.baseline_identity = copy.deepcopy(baseline_identity)
        if mode == "baseline_fail" and (not callable(baseline_probe) or baseline_identity is None):
            raise ValueError("baseline_fail requires a callable probe and a fixed baseline identity")
        self.active = False
        self.close_count = 0
        self.records = []
        self.stats = {"attempts": 0, "setup_count": 0, "expert_checks": 0, "baseline_checks": 0,
                      "cache_hits": 0, "cache_misses": 0, "rejections": {},
                      "shared_bank_hits": 0, "shared_bank_appends": 0}
        self.shared_seed_file = None

    def _reject(self, seed, reason):
        counts = self.stats["rejections"]
        counts[reason] = counts.get(reason, 0) + 1
        print(f"[Eval Scene] mode={self.mode} seed={seed} rejected: {reason}", flush=True)

    def _store(self, seed, valid, reason, expert_result=None, baseline_result=None):
        if self.cache is not None:
            metadata = {"reason": reason, "expert_result": expert_result}
            if self.mode == "baseline_fail":
                metadata["baseline_result"] = baseline_result
            self.cache.store(seed, valid, metadata)

    def _setup(self, task, seed, episode_index, probe=False):
        options = copy.deepcopy(self.args)
        options["eval_mode"] = True
        if probe:
            options["render_freq"] = 0
            if self.mode == "baseline_fail":
                options["eval_video_save_dir"] = None
                options["log_action_steps"] = False
        self.stats["setup_count"] += 1
        self.active = True  # Also close partially initialized scenes on exceptions.
        task.setup_demo(now_ep_num=episode_index, seed=int(seed), is_test=True, **options)

    def close(self, task):
        if self.active:
            self.active = False
            self.close_count += 1
            frequency = max(1, int(self.args.get("clear_cache_freq", 5)))
            task.close_env(clear_cache=self.close_count % frequency == 0)

    @staticmethod
    def _metadata(task):
        hook = getattr(task, "get_scene_metadata", None)
        return copy.deepcopy(hook() if callable(hook) else getattr(task, "info", {"info": {}}))

    def _expert(self, task, seed):
        self.stats["expert_checks"] += 1
        render_freq = task.render_freq
        task.render_freq = 0
        try:
            episode_info = task.play_once()
            plan_success = bool(task.plan_success)
            task_success = bool(task.check_success()) if plan_success else None
            result = {"passed": bool(plan_success and task_success),
                      "plan_success": plan_success, "task_success": task_success}
            return result, copy.deepcopy(episode_info)
        except Exception as exc:
            # A planner/resource/programming exception is not evidence of a hard scene.
            self._reject(seed, f"expert_exception:{type(exc).__name__}")
            traceback.print_exc()
            return None, None
        finally:
            task.render_freq = render_freq

    def prepare(self, task, seed, episode_index):
        """Return an accepted, initialized scene; never leave the expert's end state."""
        self.stats["attempts"] += 1
        selected = None
        try:
            cached = None
            if self.cache is not None:
                cached = self.cache.lookup(seed)
                self.stats["cache_hits" if cached is not None else "cache_misses"] += 1
                if cached is not None and not cached[0]:
                    self._reject(seed, cached[1]["reason"])
                    return None

            self._setup(task, seed, episode_index,
                        probe=self.mode == "expert" or (self.mode == "baseline_fail" and cached is None))
            baseline_result = None
            if self.mode == "expert":
                expert_result, episode_info = self._expert(task, seed)
                self.close(task)
                if expert_result is None:
                    return None
                if not expert_result["passed"]:
                    self._reject(seed, "expert_failure")
                    return None
                self._setup(task, seed, episode_index)
                metrics = None
            else:
                episode_info = self._metadata(task)
                result = self.validator.validate(task)
                if not result.valid:
                    self._store(seed, False, result.reason)
                    self._reject(seed, result.reason)
                    return None
                metrics = result.metrics
                expert_result = None
                if self.mode == "expert_fail":
                    if cached is not None:
                        expert_result = cached[1]["expert_result"]
                        if not isinstance(expert_result, dict) or expert_result.get("passed") is not False:
                            raise ValueError("Invalid cached expert-failure selection record")
                    else:
                        expert_result, _ = self._expert(task, seed)
                        self.close(task)
                        if expert_result is None:
                            return None
                        if expert_result["passed"]:
                            self._store(seed, False, "expert_success", expert_result)
                            self._reject(seed, "expert_success")
                            return None
                        # The probe changed the scene: rebuild and recheck the SAME seed.
                        self._setup(task, seed, episode_index)
                        result = self.validator.validate(task)
                        if not result.valid:
                            self._reject(seed, f"rebuilt_{result.reason}")
                            return None
                        episode_info, metrics = self._metadata(task), result.metrics
                elif self.mode == "baseline_fail":
                    if cached is not None:
                        baseline_result = cached[1]["baseline_result"]
                    else:
                        self.stats["baseline_checks"] += 1
                        # Exceptions propagate: a failed execution is not a normal baseline failure.
                        try:
                            baseline_result = self.baseline_probe(task, copy.deepcopy(episode_info))
                        except Exception as exc:
                            raise RuntimeError(
                                f"Fixed baseline rollout failed with {type(exc).__name__}: {exc}; "
                                "not caching this as a baseline-failure scene"
                            ) from exc
                        if not isinstance(baseline_result, dict) or type(baseline_result.get("passed")) is not bool:
                            raise ValueError("Baseline probe must return a boolean passed result")
                        self.close(task)
                        if baseline_result["passed"]:
                            self._store(seed, False, "baseline_success", baseline_result=baseline_result)
                            self._reject(seed, "baseline_success")
                            return None
                        self._setup(task, seed, episode_index)
                        result = self.validator.validate(task)
                        if not result.valid:
                            self._reject(seed, f"rebuilt_{result.reason}")
                            return None
                        episode_info, metrics = self._metadata(task), result.metrics
                    if not isinstance(baseline_result, dict) or baseline_result.get("passed") is not False:
                        raise ValueError("Invalid cached baseline-failure result")
                self._store(seed, True, "accepted", expert_result, baseline_result)
            record = {"seed": int(seed), "cheap_metrics": metrics,
                      "expert_result": expert_result, "cache_hit": cached is not None,
                      "scene_info": episode_info.get("info", {}), "success": None}
            if self.mode == "baseline_fail":
                record["baseline_result"] = copy.deepcopy(baseline_result)
            self.records.append(record)
            selected = SelectedScene(episode_info, record)
            print(f"[Eval Scene] mode={self.mode} accepted seed={seed} "
                  f"({len(self.records)} selected / {self.stats['attempts']} candidates)", flush=True)
            return selected
        except self.unstable_error:
            self._reject(seed, "unstable_setup")
            if self.mode != "expert":
                self._store(seed, False, "unstable_setup")
            return None
        except Exception as exc:
            self._reject(seed, f"selection_exception:{type(exc).__name__}")
            if self.mode == "expert":
                # Preserve the original evaluator's retry-on-error expert behavior.
                traceback.print_exc()
                return None
            raise
        finally:
            if selected is None:
                self.close(task)

    def prepare_from_bank(self, task, entry, episode_index):
        """Replay a certified scene; fresh validity is checked, expert labels are reused."""
        seed = entry["seed"]
        self.stats["attempts"] += 1
        selected = None
        try:
            expert_result = entry["expert_result"]
            if self.mode in ("expert", "expert_fail"):
                expected = self.mode == "expert"
                if not isinstance(expert_result, dict) or expert_result.get("passed") is not expected:
                    raise ValueError(f"Shared seed {seed} has an invalid expert result for {self.mode}")
            baseline_result = entry.get("baseline_result")
            if self.mode == "baseline_fail":
                if not isinstance(baseline_result, dict) or baseline_result.get("passed") is not False:
                    raise ValueError(f"Shared seed {seed} has an invalid baseline-failure result")
            self._setup(task, seed, episode_index)
            metrics = None
            if self.mode != "expert":
                result = self.validator.validate(task)
                if not result.valid:
                    self._reject(seed, result.reason)
                    raise RuntimeError(
                        f"Shared seed {seed} failed fresh validation: {result.reason}; "
                        "refusing to replace a fixed evaluation scene"
                    )
                metrics = result.metrics
            episode_info = (copy.deepcopy(entry["episode_info"]) if self.mode == "expert"
                            else self._metadata(task))
            record = {"seed": seed, "cheap_metrics": metrics,
                      "expert_result": copy.deepcopy(expert_result), "cache_hit": False,
                      "scene_info": episode_info.get("info", {}), "success": None,
                      "seed_source": "shared_bank"}
            if self.mode == "baseline_fail":
                record["baseline_result"] = copy.deepcopy(baseline_result)
            self.records.append(record)
            self.stats["shared_bank_hits"] += 1
            selected = SelectedScene(episode_info, record)
            print(f"[Eval Scene] mode={self.mode} replay shared seed={seed}", flush=True)
            return selected
        except self.unstable_error as exc:
            self._reject(seed, "unstable_setup")
            raise RuntimeError(f"Shared seed {seed} became unstable; refusing to replace it") from exc
        finally:
            if selected is None:
                self.close(task)

    def save_report(self, path, task_name, task_config, requested, start_seed,
                    validation_source=None, status="running"):
        if path is None:
            return
        completed = [record for record in self.records if record["success"] is not None]
        payload = {
            "status": status, "scene_selection": self.mode,
            "task_name": task_name, "task_config": task_config,
            "requested_episodes": requested, "completed_episodes": len(completed),
            "start_seed": start_seed, "selected_seeds": [record["seed"] for record in self.records],
            "successes": sum(record["success"] for record in completed),
            "validation_source": validation_source,
            "validation_config": self.validator.config if self.validator is not None else None,
            "shared_seed_file": self.shared_seed_file,
            "baseline_identity": self.baseline_identity,
            "stats": self.stats, "episodes": self.records,
        }
        def encode(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            raise TypeError(f"Cannot encode evaluation report value: {type(value).__name__}")
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=encode) + "\n",
                              encoding="utf-8")


def load_scene_manifest(path, selector, task_name, task_config, count):
    """Replay a completed cohort; a rejected seed must abort, never be replaced."""
    if path in (None, "", "null", "None"):
        return None
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = {"status": "complete", "scene_selection": selector.mode,
                "task_name": task_name, "task_config": task_config,
                "validation_config": selector.validator.config if selector.validator is not None else None}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Scene manifest {key} does not match the requested evaluation")
    if selector.mode == "baseline_fail" and manifest.get("baseline_identity") != selector.baseline_identity:
        raise ValueError("Scene manifest baseline identity does not match the fixed baseline")
    seeds = manifest.get("selected_seeds", [])
    if (len(seeds) < count or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Scene manifest needs enough distinct nonnegative integer seeds")
    return seeds[:count]
