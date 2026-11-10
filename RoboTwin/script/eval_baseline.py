"""A lazily loaded, fixed ManiFlow checkpoint used only to define a failure cohort."""

from collections import deque
import hashlib
from pathlib import Path
import random
import sys

import numpy as np

from script.eval_scene_selection import MANIFLOW_ROOT


DEFAULT_BASELINE_CHECKPOINT = (
    "/media/Elements1/ljj/ManiFlow/outputs/"
    "place_empty_cup-maniflow_pointcloud_policy_robotwin2-0428_seed0/"
    "checkpoints/epoch=0500-val_loss=0.373935.ckpt"
)


class FixedBaselineProbe:
    """Never trains or reads a later-edited .hydra/config.yaml or optimizer state."""

    def __init__(self, checkpoint, instruction_setter, task_name, device="cuda:0"):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.instruction_setter = instruction_setter
        self.task_name = task_name
        self.device = device
        self.policy = None
        self.cfg = None
        self.obs = None
        self.identity = {
            "checkpoint": self._stamp(), "configuration": "checkpoint.cfg",
            "weights": "ema_model_if_training_use_ema_else_model",
            "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _stamp(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Fixed baseline checkpoint does not exist: {self.checkpoint}")
        stat = self.checkpoint.stat()
        return {"path": str(self.checkpoint), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _check_checkpoint(self):
        if self._stamp() != self.identity["checkpoint"]:
            raise RuntimeError("The fixed baseline checkpoint changed during evaluation; refusing to mix labels")

    def _load(self):
        if self.policy is not None:
            return
        self._check_checkpoint()
        if str(MANIFLOW_ROOT) not in sys.path:
            sys.path.append(str(MANIFLOW_ROOT))
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(str(self.checkpoint), map_location="cpu", mmap=True,
                             pickle_module=dill, weights_only=False)
        cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=False))
        if cfg.get("task_name") not in (None, self.task_name):
            raise ValueError(f"Baseline task {cfg.task_name!r} does not match {self.task_name!r}")
        key = "ema_model" if bool(cfg.training.get("use_ema", False)) else "model"
        state = payload["state_dicts"][key]
        # Lazy module initialization must not alter the scene-seeded sampling RNG.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                policy = hydra.utils.instantiate(cfg.policy)
                incompatible = policy.load_state_dict(state, strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(f"Fixed baseline checkpoint must match exactly: {incompatible}")
                policy.requires_grad_(False)
                policy.eval()
                policy.to(device)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self._check_checkpoint()
        self.cfg, self.policy = cfg, policy
        self.obs = deque(maxlen=int(cfg.n_obs_steps) + 1)
        epoch = dill.loads(payload["pickles"]["epoch"])
        print(f"[Baseline] loaded {self.checkpoint}; epoch={epoch}, weights={key}, "
              f"inference_steps={policy.num_inference_steps}, "
              f"action_plan={getattr(policy, 'action_plan_enabled', False)}", flush=True)

    @staticmethod
    def _encode(observation):
        # Same observation preprocessing as ManiFlow/deploy_policy.py.
        return {"head_cam": np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255,
                "agent_pos": observation["joint_action"]["vector"],
                "point_cloud": observation["pointcloud"]}

    def _actions(self):
        import torch
        observations = list(self.obs)
        n_obs = int(self.cfg.n_obs_steps)
        batch = {}
        for key in observations[-1]:
            recent = observations[-n_obs:]
            padded = [recent[0]] * (n_obs - len(recent)) + recent
            stacked = np.stack([item[key] for item in padded])
            batch[key] = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        batch["task_name"] = [self.task_name]
        with torch.inference_mode():
            actions = self.policy.predict_action(batch)["action"].squeeze(0).cpu().numpy()
        if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
            raise RuntimeError("Baseline returned an empty, malformed or nonfinite action chunk")
        return actions

    def __call__(self, task, episode_info):
        self._check_checkpoint()
        self._load()
        self.instruction_setter(task, episode_info)
        self.obs.clear()
        self.policy.reset()
        start = task.take_action_cnt
        try:
            while task.take_action_cnt < task.step_lim and not task.eval_success:
                # Match the outer eval_policy -> deploy_policy.eval observation
                # call even when history is populated; get_obs may consume RNG.
                observation = task.get_obs()
                if not self.obs:
                    self.obs.append(self._encode(observation))
                before = task.take_action_cnt
                for action in self._actions():
                    if task.take_action_cnt >= task.step_lim or task.eval_success:
                        break
                    task.take_action(action)
                    self.obs.append(self._encode(task.get_obs()))
                if task.take_action_cnt == before and not task.eval_success:
                    raise RuntimeError("Baseline rollout made no action progress")
            passed = bool(task.eval_success)
            if not passed and task.take_action_cnt < task.step_lim:
                raise RuntimeError("Baseline stopped before success or the task step limit")
            result = {"passed": passed, "action_steps": int(task.take_action_cnt - start),
                      "step_limit": int(task.step_lim)}
            print(f"[Baseline] {'success' if passed else 'failure'} "
                  f"steps={result['action_steps']}/{result['step_limit']}", flush=True)
            return result
        finally:
            self.obs.clear()
            self.policy.reset()
