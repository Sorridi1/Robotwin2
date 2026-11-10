"""One append-only evaluation seed bank shared by all models and training runs."""

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from script.eval_scene_selection import MANIFLOW_ROOT, ROBOTWIN_ROOT, check_selection_mode


DEFAULT_SHARED_SEED_FILE = "/media/Elements1/ljj/ManiFlow/outputs/eval_scene_seeds.json"


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize shared scene value: {type(value).__name__}")


def _scene_identity(args, mode, validation_config):
    # Model names, output paths and runtime logging cannot change cohort membership.
    ignored = {"ckpt_setting", "policy_name", "save_path", "eval_video_save_dir",
               "eval_video_log", "render_freq", "clear_cache_freq", "log_action_steps",
               "collect_data", "save_data", "save_freq", "episode_num", "use_seed",
               "language_num", "eval_mode"}
    identity = {key: value for key, value in args.items() if key not in ignored}
    digest = hashlib.sha256()
    digest.update(json.dumps([identity, mode, validation_config], sort_keys=True,
                             default=_json_value).encode())
    # Task config is already represented semantically above, excluding output/log flags.
    sources = list((ROBOTWIN_ROOT / "envs").rglob("*.py"))
    sources.extend([ROBOTWIN_ROOT / "task_config/_camera_config.yml",
                    ROBOTWIN_ROOT / "task_config/_embodiment_config.yml"])
    if mode != "expert":
        sources.append(MANIFLOW_ROOT / "maniflow/env/robotwin/scene_validator.py")
    for side in ("left", "right"):
        if args.get(f"{side}_robot_file"):
            sources.append(Path(args[f"{side}_robot_file"]) / "config.yml")
    for path in sorted({source.resolve() for source in sources}):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SharedEvalSeedBank:
    """Freeze a common ordered cohort; growth never changes its existing prefix."""

    def __init__(self, path, args, mode, start_seed, baseline_identity=None):
        self.path = Path(path).expanduser().resolve()
        self.args = copy.deepcopy(args)
        self.mode = check_selection_mode(mode)
        self.baseline_identity = copy.deepcopy(baseline_identity)
        if mode == "baseline_fail" and baseline_identity is None:
            raise ValueError("baseline_fail shared bank requires a fixed baseline identity")
        self.key = "/".join((args["task_name"], args["task_config"], mode))
        self.start_seed = int(start_seed)
        if self.start_seed < 0:
            raise ValueError("Shared seed bank start_seed must be nonnegative")
        self._identity = None
        self._validation_config = None

    def _read(self):
        try:
            with self.path.open(encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            return {"version": 1, "cohorts": {}}
        if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("cohorts"), dict):
            raise ValueError(f"Invalid shared evaluation seed bank: {self.path}; not overwriting it")
        return state

    def _cohort(self, state):
        cohort = state["cohorts"].get(self.key)
        if cohort is None:
            return None
        if not isinstance(cohort, dict) or not isinstance(cohort.get("scenes"), list):
            raise ValueError(f"Invalid shared cohort {self.key}; not overwriting it")
        expected = {"task_name": self.args["task_name"], "task_config": self.args["task_config"],
                    "scene_selection": self.mode}
        if any(cohort.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Shared cohort metadata mismatch: {self.key}")
        seeds = [entry.get("seed") if isinstance(entry, dict) else None for entry in cohort["scenes"]]
        if (any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)
                or type(cohort.get("start_seed")) is not int
                or cohort["start_seed"] < 0
                or type(cohort.get("next_seed")) is not int
                or cohort["next_seed"] < cohort["start_seed"]
                or (seeds and cohort["next_seed"] <= max(seeds))):
            raise ValueError(f"Invalid seeds/cursor in shared cohort {self.key}; not overwriting it")
        return cohort

    def validation_config(self):
        """Use the bank's frozen rules, not a later model's training directory."""
        cohort = self._cohort(self._read())
        if cohort is None:
            return False, None
        self.start_seed = cohort["start_seed"]
        return True, copy.deepcopy(cohort["validation_config"])

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[Eval Seed Bank] waiting for writer: {self.path}", flush=True)
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, state):
        # Atomic replacement prevents readers observing truncated JSON after interruption.
        content = json.dumps(state, ensure_ascii=False, indent=2, default=_json_value) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _check_identity(self, cohort, validation):
        if (cohort["validation_config"] != validation or cohort["scene_identity"] != self._identity
                or (self.mode == "baseline_fail" and cohort.get("baseline_identity") != self.baseline_identity)):
            raise ValueError(
                f"Shared cohort {self.key} has different validation/environment settings. "
                "Refusing to mix or overwrite its seeds; keep the original settings or "
                "explicitly choose another eval_scene_seed_file for a new benchmark."
            )

    def prepare(self, task, selector, index, episode_index, max_candidates):
        if selector.mode != self.mode:
            raise ValueError("Shared bank and scene selector modes must match")
        if self.mode == "baseline_fail" and selector.baseline_identity != self.baseline_identity:
            raise ValueError("Shared bank and selector must use the same fixed baseline")
        selector.shared_seed_file = str(self.path)
        validation = (selector.validator.config if self.mode != "expert" else None)
        if self._identity is None or self._validation_config != validation:
            self._identity = _scene_identity(self.args, self.mode, validation)
            if self.mode == "baseline_fail":
                self._identity = hashlib.sha256((self._identity + json.dumps(
                    self.baseline_identity, sort_keys=True)).encode()).hexdigest()
            self._validation_config = copy.deepcopy(validation)
        # Atomic reads need no lock: already-fixed scenes can be replayed while a
        # different process is slowly searching for additional scenes.
        existing = self._cohort(self._read())
        if existing is not None:
            self._check_identity(existing, validation)
            if 0 <= index < len(existing["scenes"]):
                entry = copy.deepcopy(existing["scenes"][index])
                print(f"[Eval Seed Bank] replay {self.key}[{index}] seed={entry['seed']} "
                      f"from {self.path}", flush=True)
                return selector.prepare_from_bank(task, entry, episode_index)
        entry = None
        with self._locked():
            state = self._read()
            cohort = self._cohort(state)
            if cohort is None:
                cohort = {"task_name": self.args["task_name"], "task_config": self.args["task_config"],
                          "scene_selection": self.mode, "validation_config": copy.deepcopy(validation),
                          "scene_identity": self._identity, "start_seed": self.start_seed,
                          "next_seed": self.start_seed, "scenes": []}
                if self.mode == "baseline_fail":
                    cohort["baseline_identity"] = copy.deepcopy(self.baseline_identity)
                state["cohorts"][self.key] = cohort
                self._write(state)
            self._check_identity(cohort, validation)
            if index < 0 or index > len(cohort["scenes"]):
                raise ValueError("Shared cohort must be consumed in order")
            if index < len(cohort["scenes"]):
                entry = copy.deepcopy(cohort["scenes"][index])
            else:
                for _ in range(max_candidates):
                    candidate = cohort["next_seed"]
                    selected = selector.prepare(task, candidate, episode_index)
                    cohort["next_seed"] = candidate + 1
                    if selected is not None:
                        entry = {"seed": candidate, "expert_result": copy.deepcopy(selected.record["expert_result"]),
                                 "episode_info": copy.deepcopy(selected.episode_info)}
                        if self.mode == "baseline_fail":
                            entry["baseline_result"] = copy.deepcopy(selected.record["baseline_result"])
                        cohort["scenes"].append(entry)
                        try:
                            # Persist membership BEFORE the evaluated policy acts.
                            self._write(state)
                        except BaseException:
                            selector.close(task)
                            raise
                        selector.stats["shared_bank_appends"] += 1
                        selected.record["seed_source"] = "shared_bank_append"
                        print(f"[Eval Seed Bank] appended seed={candidate} to {self.key}; "
                              f"size={len(cohort['scenes'])}", flush=True)
                        return selected
                    self._write(state)
                raise RuntimeError(
                    f"Not enough eligible scenes in shared cohort {self.key} after "
                    f"{max_candidates} more candidates. No partial success rate is reported."
                )
        # Replaying an existing entry requires no writer lock or expert execution.
        print(f"[Eval Seed Bank] {self.path} group={self.key} index={index} seed={entry['seed']}", flush=True)
        return selector.prepare_from_bank(task, entry, episode_index)
