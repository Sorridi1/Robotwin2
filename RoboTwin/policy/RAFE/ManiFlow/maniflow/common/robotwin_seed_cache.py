"""Process-safe seed validation records; callers isolate modes in the fingerprint."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def seed_cache_fingerprint(task_args, source_paths, namespace="v1"):
    # Experiment names and output locations do not change scene feasibility.
    ignored = {"ckpt_setting", "policy_name", "save_path"}
    identity = {key: value for key, value in task_args.items() if key not in ignored}
    digest = hashlib.sha256()
    digest.update(json.dumps([namespace, identity], sort_keys=True, default=str).encode())
    for path in sorted({Path(path).resolve() for path in source_paths}):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class ExpertSeedCache:
    def __init__(self, directory, fingerprint, mode="readwrite"):
        if mode not in ("off", "readwrite", "readonly"):
            raise ValueError(f"Unsupported expert seed cache mode: {mode}")
        if mode != "off" and not directory:
            raise ValueError("expert_seed_cache_dir is required when caching is enabled")
        self.mode = mode
        self.directory = Path(directory).expanduser() / fingerprint if directory else None

    def _read(self, seed):
        path = self.directory / f"{int(seed)}.json"
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if entry.get("seed") != int(seed) or type(entry.get("valid")) is not bool:
            raise ValueError(f"Invalid expert seed cache entry: {path}")
        return entry["valid"], entry["episode_info"]

    def lookup(self, seed):
        if self.mode == "off":
            return None
        cached = self._read(seed)
        if cached is None and self.mode == "readonly":
            raise FileNotFoundError(
                f"Seed {seed} is not prevalidated for this validation configuration: "
                f"{self.directory}. Run scripts/prevalidate_rl_seeds.py first "
                "or use expert_seed_cache_mode=readwrite."
            )
        return cached

    def _write(self, seed, valid, info):
        entry = {"seed": int(seed), "valid": bool(valid), "episode_info": info}
        content = json.dumps(entry, ensure_ascii=False)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{seed}-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.directory / f"{int(seed)}.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def store(self, seed, valid, info):
        """Refresh a record after checking a freshly initialized scene."""
        if self.mode == "readwrite":
            with self._locked(seed):
                self._write(seed, valid, info)

    @contextmanager
    def _locked(self, seed):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"{int(seed)}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def get_or_validate(self, seed, validate):
        """Return (valid, episode_info, cache_hit); exceptions are never cached."""
        seed = int(seed)
        if self.mode == "off":
            valid, info = validate()
            return bool(valid), info, False
        cached = self.lookup(seed)
        if cached is not None:
            return *cached, True
        with self._locked(seed):
            cached = self._read(seed)
            if cached is not None:
                return *cached, True
            valid, info = validate()
            self._write(seed, valid, info)
            return bool(valid), info, False
