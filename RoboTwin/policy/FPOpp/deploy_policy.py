"""Adapter for the existing RoboTwin2 script/eval_policy.py entry point."""

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpopp.runtime import POLICY_ROOT, configure_paths

configure_paths()

import numpy as np
from omegaconf import OmegaConf
import torch

from fpopp.actor import FPOActor
from fpopp.checkpoint import FORMAT, load_payload, load_policy
from fpopp.results import reserve_results


class EvaluationModel:
    def __init__(self, arguments):
        path = Path(arguments["checkpoint_path"]).expanduser().resolve()
        payload = load_payload(path)
        policy, policy_cfg, _ = load_policy(payload, device=arguments.get("device", "cuda:0"))
        actor_config = payload["cfg"]["actor"] if payload.get("format") == FORMAT else {}
        self.base_sampling_seed = int(arguments.get("sampling_seed", 0))
        self.actor = FPOActor(policy, seed=self.base_sampling_seed, **dict(actor_config)).eval()
        self.epoch = int(payload.get("iteration", 0))
        self.run_dir = arguments.get("eval_output_dir")
        if not self.run_dir:
            self.run_dir = (path.parent.parent if payload.get("format") == FORMAT else POLICY_ROOT / "output" /
                            f"bc_eval_{datetime.now():%Y%m%d_%H%M%S}")
        self.run_dir = str(Path(self.run_dir).expanduser().resolve())
        self.zero_latent = bool(arguments.get("zero_latent", False))
        self.history = deque(maxlen=int(policy.n_obs_steps))
        self.episode_index = -1
        seeds = set(payload.get("training_seeds", []))
        bank_path = arguments.get("eval_scene_seed_file")
        if bank_path and Path(bank_path).expanduser().is_file():
            state = json.loads(Path(bank_path).expanduser().read_text())
            key = "/".join((arguments["task_name"], arguments["task_config"], arguments.get("eval_scene_selection", "rl")))
            cohort = state.get("cohorts", {}).get(key, {})
            selected = {int(scene["seed"]) for scene in cohort.get("scenes", [])[:int(arguments.get("eval_num_episodes", 100))]}
            if selected.intersection(seeds):
                raise ValueError("Evaluation bank overlaps actual training seeds; use an independent shared bank")
        protocol_path = reserve_results(self.run_dir, self.epoch, arguments.get("eval_scene_selection", "rl"))
        # Preserve validator provenance when an explicit output subdirectory is used.
        archive = Path(self.run_dir) / ".hydra"
        archive.mkdir(parents=True, exist_ok=True)
        config_path = archive / "config.yaml"
        if not config_path.exists():
            if payload.get("format") == FORMAT:
                cfg = OmegaConf.create(payload["cfg"])
            else:
                cfg = OmegaConf.load(POLICY_ROOT / "configs/fpopp_robotwin2.yaml")
                cfg = OmegaConf.create({"rl": {"env": OmegaConf.to_container(cfg.env, resolve=True)}})
            OmegaConf.save(cfg, config_path)
        protocol = {"checkpoint": str(path), "policy_cfg": OmegaConf.to_container(policy_cfg, resolve=True),
                    "actor": actor_config, "sampling_seed": self.base_sampling_seed, "zero_latent": self.zero_latent,
                    "action_window": self.actor.window, "training_seeds": sorted(seeds)}
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

    def observe(self, observation):
        self.history.append({"agent_pos": np.array(observation["joint_action"]["vector"], dtype=np.float32, copy=True),
                             "point_cloud": np.array(observation["pointcloud"], dtype=np.float32, copy=True)})
        while len(self.history) < self.history.maxlen:
            self.history.appendleft(self.history[0])

    @torch.no_grad()
    def actions(self):
        observations = {key: torch.as_tensor(np.stack([item[key] for item in self.history])[None],
                                              device=self.actor.device) for key in self.history[0]}
        features, plan = self.actor.encode(observations)
        actions, _, _ = self.actor.sample(features, plan, zero_latent=self.zero_latent)
        return actions[0].cpu().numpy()


def get_model(usr_args):
    if not usr_args.get("checkpoint_path"):
        raise ValueError("FPOpp requires --checkpoint_path PATH")
    return EvaluationModel(usr_args)


def eval(TASK_ENV, model, observation):
    if not model.history:
        model.observe(observation)
    for action in model.actions():
        if TASK_ENV.eval_success or (TASK_ENV.step_lim is not None and TASK_ENV.take_action_cnt >= TASK_ENV.step_lim):
            break
        TASK_ENV.take_action(action, action_type="qpos")
        model.observe(TASK_ENV.get_obs())


def reset_model(model):
    model.history.clear()
    model.episode_index += 1
    model.actor.rng.manual_seed(model.base_sampling_seed + model.episode_index)
