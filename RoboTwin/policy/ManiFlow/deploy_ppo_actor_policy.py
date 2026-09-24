import os
import pathlib
import random
import re
import sys
from collections import deque

import dill
import hydra
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from termcolor import cprint


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
maniflow_root = os.path.join(parent_directory, "ManiFlow")
sys.path.append(maniflow_root)

from maniflow.model.rl.ppo_maniflow_pointcloud import PPOManiFlowPointcloud


OmegaConf.register_new_resolver("eval", eval, replace=True)


def encode_obs(observation):
    return {
        "point_cloud": observation["pointcloud"].astype(np.float32),
        "agent_pos": observation["joint_action"]["vector"].astype(np.float32),
    }


def _safe_eval_mode(value):
    value = str(value).strip().lower()
    if value in ("deterministic", "eval", "true", "1", "mean"):
        return True, "deterministic"
    if value in ("stochastic", "sample", "false", "0", "train"):
        return False, "stochastic"
    raise ValueError(
        "ppo_eval_mode must be one of stochastic/sample/false or deterministic/eval/true, "
        f"got {value!r}"
    )


def _sanitize_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _resolve_run_dir(usr_args):
    task_name = usr_args["task_name"]
    alg_name = usr_args["alg_name"]
    addition_info = usr_args["addition_info"]
    seed = usr_args["training_seed"]
    exp_name = f"{task_name}-{alg_name}-{addition_info}"
    return os.path.join("/media/Elements1/ljj/ManiFlow/outputs", exp_name + f"_seed{seed}")


def _resolve_rl_ckpt_path(usr_args, run_dir):
    explicit_path = usr_args.get("ppo_ckpt_path", None)
    if explicit_path not in (None, "", "none", "None", "null"):
        return pathlib.Path(str(explicit_path)).expanduser()

    ckpt_tag = str(usr_args.get("ckpt_tag", "latest_rl"))
    ckpt_path = pathlib.Path(ckpt_tag).expanduser()
    if ckpt_path.is_absolute() or ckpt_path.parent != pathlib.Path("."):
        return ckpt_path if ckpt_path.suffix == ".ckpt" else ckpt_path.with_suffix(".ckpt")

    ckpt_name = ckpt_tag if ckpt_tag.endswith(".ckpt") else f"{ckpt_tag}.ckpt"
    return pathlib.Path(run_dir) / "checkpoints" / ckpt_name


def _compose_fallback_cfg(usr_args):
    config_path = "./ManiFlow/maniflow/config"
    config_name = f"{usr_args['config_name']}.yaml"
    with initialize(config_path=config_path, version_base="1.2"):
        cfg = compose(config_name=config_name)
    return cfg


def _prepare_cfg(cfg, usr_args, device):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.setting = usr_args["ckpt_setting"]
    cfg.training.seed = int(usr_args["training_seed"])
    cfg.training.device = str(device)
    if "rl" in cfg and "actor" in cfg.rl:
        cfg.rl.actor.actor_old_device = "cpu"
    OmegaConf.set_struct(cfg, True)
    return cfg


class PPOActorDeployModel:
    def __init__(self, usr_args):
        self.usr_args = usr_args
        self.run_dir = _resolve_run_dir(usr_args)
        self.ckpt_path = _resolve_rl_ckpt_path(usr_args, self.run_dir)
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"PPO RL checkpoint does not exist: {self.ckpt_path}")

        payload = torch.load(self.ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
        if "ppo_actor" not in payload:
            raise KeyError(f"{self.ckpt_path} does not contain payload['ppo_actor']")

        requested_device = str(usr_args.get("device", "cuda:0"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.eval_mode, self.sample_mode = _safe_eval_mode(usr_args.get("ppo_eval_mode", "stochastic"))

        cfg = payload.get("cfg", None)
        if cfg is None:
            cprint("[PPO Deploy] Checkpoint has no cfg; composing config from CLI arguments.", "yellow")
            cfg = _compose_fallback_cfg(usr_args)
        self.cfg = _prepare_cfg(cfg, usr_args, self.device)

        seed = int(usr_args.get("training_seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        base_actor = hydra.utils.instantiate(self.cfg.policy)
        self.ppo_actor = PPOManiFlowPointcloud(
            base_actor=base_actor,
            **OmegaConf.to_container(self.cfg.rl.actor, resolve=True),
        )
        missing, unexpected = self.ppo_actor.load_state_dict(payload["ppo_actor"], strict=False)
        if missing:
            cprint(f"[PPO Deploy] Missing PPO actor keys: {missing}", "yellow")
        if unexpected:
            cprint(f"[PPO Deploy] Unexpected PPO actor keys: {unexpected}", "yellow")

        self.ppo_actor.to(self.device)
        self.ppo_actor.eval()

        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.obs = deque(maxlen=self.n_obs_steps + 1)
        self.t = 0
        self.temporal_agg = False

        itr = payload.get("itr", payload.get("epoch", "unknown"))
        ckpt_tag = pathlib.Path(self.ckpt_path).stem
        self.epoch = f"{itr}_{_sanitize_tag(ckpt_tag)}_{self.sample_mode}"

        cprint(
            f"[PPO Deploy] Loaded {self.ckpt_path} on {self.device}; "
            f"mode={self.sample_mode}, result epoch tag={self.epoch}",
            "green",
        )

    def reset_obs(self):
        self.obs.clear()
        self.t = 0

    def update_obs(self, observation):
        self.obs.append(observation)

    @staticmethod
    def _stack_last_n_obs(all_obs, n_steps):
        if len(all_obs) == 0:
            raise RuntimeError("No observation is recorded; call update_obs first.")
        all_obs = list(all_obs)
        result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.asarray(all_obs[start_idx:])
        if n_steps > len(all_obs):
            result[:start_idx] = result[start_idx]
        return result

    def _get_stacked_obs(self):
        if len(self.obs) == 0:
            raise RuntimeError("No observation is recorded; call update_obs first.")
        return {
            key: self._stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)
            for key in self.obs[0].keys()
        }

    @torch.no_grad()
    def get_action(self):
        stacked_obs = self._get_stacked_obs()
        obs_dict = {
            key: torch.from_numpy(value).unsqueeze(0).to(device=self.device)
            for key, value in stacked_obs.items()
        }
        result = self.ppo_actor.get_actions(
            obs_dict,
            eval_mode=self.eval_mode,
            save_chains=False,
            ret_logprob=False,
        )
        action = result[0] if isinstance(result, tuple) else result
        self.t += 1
        return action.detach().cpu().numpy().squeeze(0)


def get_model(usr_args):
    return PPOActorDeployModel(usr_args)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    if len(model.obs) == 0:
        model.update_obs(obs)

    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
