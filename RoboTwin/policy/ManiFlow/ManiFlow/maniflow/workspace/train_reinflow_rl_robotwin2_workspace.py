if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from typing import Dict, Optional

import dill
import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from termcolor import cprint

MANIFLOW_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
ROBOTWIN_ROOT = str(pathlib.Path(MANIFLOW_ROOT).parent.parent)
sys.path.append(MANIFLOW_ROOT)
sys.path.append(os.path.join(MANIFLOW_ROOT, "ManiFlow"))
sys.path.append(os.path.join(MANIFLOW_ROOT, "ManiFlow", "maniflow"))
sys.path.append(ROBOTWIN_ROOT)

from maniflow.common.ppo_buffer_pointcloud import PPOBufferPointcloud
from maniflow.common.pytorch_util import dict_apply
from maniflow.model.rl.pointcloud_critic import PointcloudCritic
from maniflow.model.rl.ppo_maniflow_pointcloud import PPOManiFlowPointcloud
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_state_to_cpu(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in state_dict.items()}


def _to_frozen_param_dict(nested_dict):
    result = nn.ParameterDict()
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            result[key] = _to_frozen_param_dict(value)
        else:
            result[key] = nn.Parameter(value.detach().clone(), requires_grad=False)
    return result


def _restore_policy_normalizer_from_state(policy, state_dict):
    prefix = "normalizer.params_dict."
    grouped = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split(".")
        field = parts[0]
        group = grouped.setdefault(field, {})
        if parts[1] == "input_stats":
            group.setdefault("input_stats", {})[parts[2]] = value
        else:
            group[parts[1]] = value
    if grouped:
        policy.normalizer.params_dict = _to_frozen_param_dict(grouped)
        policy.normalizer.requires_grad_(False)


class TrainReinFlowRLRoboTwinWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.itr = 0
        self.global_step = 0
        self.epoch = 0
        self.current_obs = None
        self.consecutive_zero_success_iters = 0
        self.best_eval_score = float("-inf")
        self.best_eval_itr = -1

        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)
        self._load_pretrained_actor()

        self.ppo_actor = PPOManiFlowPointcloud(
            base_actor=self.model,
            **OmegaConf.to_container(cfg.rl.actor, resolve=True),
        )
        self.critic: PointcloudCritic = hydra.utils.instantiate(cfg.rl.critic)
        self.critic.set_normalizer(self.model.normalizer)

        device = torch.device(cfg.training.device)
        self.device = device
        self.ppo_actor.to(device)
        self.critic.to(device)
        self.ppo_actor.actor_old.to(cfg.rl.actor.actor_old_device)

        actor_params = list(self.ppo_actor.actor_parameters())
        self.actor_optimizer = torch.optim.AdamW(
            actor_params,
            lr=cfg.rl.ppo.actor_lr,
            weight_decay=cfg.rl.ppo.actor_weight_decay,
            betas=tuple(cfg.rl.ppo.betas),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=cfg.rl.ppo.critic_lr,
            weight_decay=cfg.rl.ppo.critic_weight_decay,
            betas=tuple(cfg.rl.ppo.betas),
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(cfg.rl.ppo.use_amp and device.type == "cuda")
        )

    @property
    def output_dir(self):
        if self._output_dir is not None:
            return self._output_dir
        return HydraConfig.get().runtime.output_dir

    def _load_pretrained_actor(self):
        ckpt_path = self.cfg.rl.pretrained_checkpoint_path
        if ckpt_path is None or str(ckpt_path).lower() in ("", "none", "null"):
            cprint("[RL] No pretrained checkpoint path provided; using randomly initialized actor.", "yellow")
            return
        ckpt_path = pathlib.Path(str(ckpt_path)).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pretrained ManiFlow checkpoint not found: {ckpt_path}")

        payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
        state_dicts = payload.get("state_dicts", {})
        preferred_key = self.cfg.rl.get("pretrained_state_key", "ema_model")
        if preferred_key in state_dicts:
            state_dict = state_dicts[preferred_key]
        elif "ema_model" in state_dicts:
            state_dict = state_dicts["ema_model"]
        elif "model" in state_dicts:
            state_dict = state_dicts["model"]
        else:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain state_dicts.model/ema_model")

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=bool(self.cfg.rl.get("strict_load_pretrained", True)),
        )
        _restore_policy_normalizer_from_state(self.model, state_dict)
        if "action" not in self.model.normalizer.params_dict:
            raise RuntimeError(
                f"Loaded checkpoint {ckpt_path} but action normalizer is missing. "
                "The checkpoint must contain normalizer.params_dict.action.* keys."
            )
        if len(missing) > 0 or len(unexpected) > 0:
            cprint(f"[RL] Loaded pretrained with missing={missing}, unexpected={unexpected}", "yellow")
        cprint(f"[RL] Loaded pretrained actor from {ckpt_path}", "green")

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray], device=None) -> Dict[str, torch.Tensor]:
        device = device or self.device
        obs_list = list(obs) if isinstance(obs, (list, tuple)) else [obs]
        return {
            "point_cloud": torch.from_numpy(
                np.stack([item["point_cloud"] for item in obs_list], axis=0)
            ).to(device=device).float(),
            "agent_pos": torch.from_numpy(
                np.stack([item["agent_pos"] for item in obs_list], axis=0)
            ).to(device=device).float(),
        }

    def _obs_to_buffer(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs_list = list(obs) if isinstance(obs, (list, tuple)) else [obs]
        return {
            "point_cloud": np.stack([item["point_cloud"] for item in obs_list], axis=0).astype(np.float32),
            "agent_pos": np.stack([item["agent_pos"] for item in obs_list], axis=0).astype(np.float32),
        }

    def _make_envs(self, start_seeds=None):
        n_envs = int(self.cfg.rl.rollout.n_envs)
        if n_envs < 1:
            raise ValueError(f"rl.rollout.n_envs must be >= 1, got {n_envs}")
        base_seed = int(self.cfg.training.seed)
        base_start_seed = 100000 * (1 + base_seed)
        env_seed_stride = int(self.cfg.rl.rollout.get("env_seed_stride", 1000))
        envs = []
        for env_idx in range(n_envs):
            envs.append(
                hydra.utils.instantiate(
                    self.cfg.rl.env,
                    seed=base_seed + env_idx,
                    start_seed=(
                        int(start_seeds[env_idx])
                        if start_seeds is not None
                        else base_start_seed + env_idx * env_seed_stride
                    ),
                )
            )
        cprint(
            f"[RL] Created {n_envs} RoboTwin envs with start_seed="
            f"{start_seeds if start_seeds is not None else base_start_seed}, "
            f"env_seed_stride={env_seed_stride}",
            "cyan",
        )
        return envs

    def _make_buffer(self):
        obs_meta = self.cfg.shape_meta.obs
        return PPOBufferPointcloud(
            n_steps=int(self.cfg.rl.rollout.n_steps),
            n_envs=int(self.cfg.rl.rollout.n_envs),
            n_obs_steps=int(self.cfg.n_obs_steps),
            point_cloud_shape=tuple(obs_meta.point_cloud.shape),
            agent_pos_shape=tuple(obs_meta.agent_pos.shape),
            denoising_steps=int(self.ppo_actor.inference_steps),
            horizon_steps=int(self.cfg.horizon),
            action_dim=int(self.model.action_dim),
            gamma=float(self.cfg.rl.ppo.gamma),
            gae_lambda=float(self.cfg.rl.ppo.gae_lambda),
            reward_scale_running=bool(self.cfg.rl.ppo.reward_scale_running),
            reward_scale_const=float(self.cfg.rl.ppo.reward_scale_const),
            bootstrap_on_truncated=bool(self.cfg.rl.ppo.bootstrap_on_truncated),
            device=str(self.cfg.rl.rollout.buffer_device),
        )

    def should_update_actor(self, rollout_log: Dict[str, float]):
        warmup_iters = int(self.cfg.rl.ppo.critic_warmup_iterations)
        success_rate = float(rollout_log.get("success_rate", 0.0))
        raw_reward_sum = float(rollout_log.get("rollout_raw_reward_sum", 0.0))
        actor_loss_scale = 1.0

        if success_rate <= 0.0:
            self.consecutive_zero_success_iters += 1
        else:
            self.consecutive_zero_success_iters = 0

        if self.itr < warmup_iters:
            return False, "critic_warmup", 1, 0.0

        min_success = float(self.cfg.rl.ppo.get("min_success_rate_for_actor_update", 0.0))
        if bool(self.cfg.rl.ppo.get("skip_actor_update_on_zero_success", True)) and success_rate <= min_success:
            return False, "zero_success_rollout", 2, 0.0

        min_raw_reward = float(self.cfg.rl.ppo.get("min_raw_reward_for_actor_update", 0.0))
        if bool(self.cfg.rl.ppo.get("skip_actor_update_on_zero_reward", False)) and raw_reward_sum <= min_raw_reward:
            return False, "zero_reward_rollout", 3, 0.0

        downweight_reasons = []
        downweight_code = 0
        if bool(self.cfg.rl.ppo.get("downweight_actor_update_on_zero_success", False)) and success_rate <= min_success:
            actor_loss_scale = min(
                actor_loss_scale,
                float(self.cfg.rl.ppo.get("zero_success_actor_loss_scale", 0.25)),
            )
            downweight_reasons.append("zero_success_downweighted")
            downweight_code = max(downweight_code, 4)

        if bool(self.cfg.rl.ppo.get("downweight_actor_update_on_zero_reward", False)) and raw_reward_sum <= min_raw_reward:
            actor_loss_scale = min(
                actor_loss_scale,
                float(self.cfg.rl.ppo.get("zero_reward_actor_loss_scale", 0.25)),
            )
            downweight_reasons.append("zero_reward_downweighted")
            downweight_code = max(downweight_code, 5)

        actor_loss_scale = max(0.0, min(1.0, actor_loss_scale))
        if downweight_reasons:
            return True, "+".join(downweight_reasons), downweight_code, actor_loss_scale

        return True, "enabled", 0, actor_loss_scale

    def collect_rollout(self, envs, buffer: PPOBufferPointcloud):
        if not isinstance(envs, (list, tuple)):
            envs = [envs]
        n_envs = len(envs)
        if n_envs != buffer.n_envs:
            raise ValueError(f"Expected {buffer.n_envs} envs, got {n_envs}")

        if self.current_obs is None or bool(self.cfg.rl.rollout.reset_each_iteration):
            self.current_obs = [env.reset() for env in envs]

        self.ppo_actor.eval()
        self.critic.eval()
        rollout_infos = []
        desc = f"RL rollout itr {self.itr} ({n_envs} envs)"
        iterator = range(buffer.n_steps)
        with tqdm.tqdm(iterator, desc=desc, leave=False, mininterval=self.cfg.training.tqdm_interval_sec) as pbar:
            for step in pbar:
                obs_for_buffer = self._obs_to_buffer(self.current_obs)
                obs_ts = self._obs_to_tensor(self.current_obs)
                with torch.no_grad():
                    value = self.critic(obs_ts).detach().cpu().numpy()
                    action, chains, logprob = self.ppo_actor.get_actions(
                        obs_ts,
                        eval_mode=False,
                        save_chains=True,
                        ret_logprob=True,
                    )
                if action.shape[0] != n_envs or chains.shape[0] != n_envs or logprob.shape[0] != n_envs:
                    raise RuntimeError(
                        "Batched actor output does not match n_envs: "
                        f"action={tuple(action.shape)}, chains={tuple(chains.shape)}, "
                        f"logprob={tuple(logprob.shape)}, n_envs={n_envs}"
                    )
                action_np = action.detach().cpu().numpy()

                next_obs_batch = []
                rewards = np.zeros(n_envs, dtype=np.float32)
                raw_rewards = np.zeros(n_envs, dtype=np.float32)
                terminateds = np.zeros(n_envs, dtype=bool)
                truncateds = np.zeros(n_envs, dtype=bool)
                successes = np.zeros(n_envs, dtype=bool)
                executed_steps = 0
                step_infos = []

                for env_idx, env in enumerate(envs):
                    next_obs, reward, terminated, truncated, info = env.step(action_np[env_idx])
                    raw_reward = float(np.sum(info.get("raw_rewards", [reward])))

                    rewards[env_idx] = float(reward)
                    raw_rewards[env_idx] = raw_reward
                    terminateds[env_idx] = bool(terminated)
                    truncateds[env_idx] = bool(truncated)
                    successes[env_idx] = bool(info.get("success", False))
                    executed_steps += int(info.get("executed_action_steps", self.cfg.n_action_steps))
                    step_infos.append(info)
                    rollout_infos.append(info)

                    if terminated or truncated:
                        next_obs = env.reset()
                    next_obs_batch.append(next_obs)

                buffer.add(
                    step=step,
                    obs=obs_for_buffer,
                    chains=chains.detach().cpu().numpy(),
                    reward=rewards,
                    terminated=terminateds,
                    truncated=truncateds,
                    value=value,
                    logprob=logprob.detach().cpu().numpy(),
                    success=successes,
                    raw_reward=raw_rewards,
                )
                self.global_step += executed_steps
                pbar.set_postfix(
                    reward=f"{float(np.mean(rewards)):.2f}",
                    success=int(np.sum(successes)),
                    env_step=max((info.get("take_action_cnt", 0) for info in step_infos), default=0),
                    refresh=False,
                )
                self.current_obs = next_obs_batch

        summary = buffer.summarize_episode_reward()
        summary["rollout_envs"] = n_envs
        summary["rollout_chunks"] = len(rollout_infos)
        summary["rollout_last_env_step"] = rollout_infos[-1].get("take_action_cnt", 0) if rollout_infos else 0
        summary["rollout_any_success_info"] = float(any(info.get("success", False) for info in rollout_infos))
        for key in ("forward", "success", "healthy", "ctrl", "contact", "time"):
            summary[f"reward_{key}_sum"] = float(
                np.sum([info.get(f"reward_{key}", 0.0) for info in rollout_infos])
            )
        return summary

    def ppo_update(
        self,
        buffer: PPOBufferPointcloud,
        update_actor: bool = True,
        actor_loss_scale: float = 1.0,
    ):
        buffer.update(self._obs_to_buffer(self.current_obs), self.critic)
        obs, chains, returns, values, advantages, logprobs = buffer.make_dataset()
        explained_var = buffer.get_explained_var(values, returns)
        total_steps = returns.shape[0]
        expected_steps = buffer.n_steps * buffer.n_envs
        if total_steps != expected_steps:
            raise RuntimeError(f"PPO dataset has {total_steps} steps, expected {expected_steps}")
        batch_size = int(self.cfg.rl.ppo.batch_size)
        update_epochs = int(self.cfg.rl.ppo.update_epochs)
        target_kl = self.cfg.rl.ppo.target_kl
        actor_update_enabled = bool(update_actor) and self.itr >= int(self.cfg.rl.ppo.critic_warmup_iterations)
        actor_loss_scale = max(0.0, min(1.0, float(actor_loss_scale)))

        self.ppo_actor.eval()
        self.critic.train()
        metrics = []
        stopped_by_kl = False
        for _ in range(update_epochs):
            indices = torch.randperm(total_steps)
            for start in range(0, total_steps, batch_size):
                mb_idx = indices[start : start + batch_size]
                mb_obs = {k: v[mb_idx].to(self.device, non_blocking=True) for k, v in obs.items()}
                mb_returns = returns[mb_idx].to(self.device, non_blocking=True)
                mb_values = values[mb_idx].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=bool(self.cfg.rl.ppo.use_amp and self.device.type == "cuda")):
                    if actor_update_enabled:
                        mb_chains = chains[mb_idx].to(self.device, non_blocking=True)
                        mb_adv = advantages[mb_idx].to(self.device, non_blocking=True)
                        mb_logprobs = logprobs[mb_idx].to(self.device, non_blocking=True)
                        loss_dict = self.ppo_actor.loss(
                            mb_obs,
                            mb_chains,
                            mb_returns,
                            mb_values,
                            mb_adv,
                            mb_logprobs,
                            critic=self.critic,
                            use_bc_loss=bool(self.cfg.rl.bc_anchor.enabled),
                        )
                        actor_rl_loss = loss_dict["pg_loss"] + self.cfg.rl.ppo.ent_coef * loss_dict["entropy_loss"]
                        bc_loss = self.cfg.rl.bc_anchor.coeff * loss_dict["bc_loss"]
                        loss = (
                            actor_loss_scale * actor_rl_loss
                            + self.cfg.rl.ppo.vf_coef * loss_dict["value_loss"]
                            + bc_loss
                        )
                    else:
                        newvalues = self.critic(mb_obs).view(-1)
                        if self.ppo_actor.clip_vloss_coef is None:
                            value_loss = 0.5 * ((newvalues - mb_returns) ** 2).mean()
                        else:
                            v_clipped = mb_values + torch.clamp(
                                newvalues - mb_values,
                                -self.ppo_actor.clip_vloss_coef,
                                self.ppo_actor.clip_vloss_coef,
                            )
                            value_loss = 0.5 * torch.max(
                                (newvalues - mb_returns) ** 2,
                                (v_clipped - mb_returns) ** 2,
                            ).mean()
                        zero = value_loss.detach() * 0.0
                        loss_dict = {
                            "pg_loss": zero,
                            "entropy_loss": zero,
                            "value_loss": value_loss,
                            "bc_loss": zero,
                            "approx_kl": zero,
                            "clipfrac": zero,
                            "ratio": zero + 1.0,
                            "noise_std": zero,
                        }
                        loss = self.cfg.rl.ppo.vf_coef * value_loss

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self.cfg.rl.ppo.max_grad_norm is not None:
                    if actor_update_enabled:
                        self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    if actor_update_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.ppo_actor.actor_parameters()),
                            float(self.cfg.rl.ppo.max_grad_norm),
                        )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        float(self.cfg.rl.ppo.max_grad_norm),
                    )

                if actor_update_enabled:
                    self.scaler.step(self.actor_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()

                item = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()}
                item["loss"] = float(loss.detach().cpu())
                item["actor_loss_scale"] = actor_loss_scale if actor_update_enabled else 0.0
                metrics.append(item)
                if actor_update_enabled and target_kl is not None and item["approx_kl"] > float(target_kl):
                    stopped_by_kl = True
                    break
            if stopped_by_kl:
                break

        if len(metrics) == 0:
            return {"explained_var": explained_var}
        mean_metrics = {f"loss/{k}": float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
        mean_metrics["loss/explained_var"] = explained_var
        mean_metrics["loss/stopped_by_kl"] = float(stopped_by_kl)
        mean_metrics["loss/actor_update_enabled"] = float(actor_update_enabled)
        return mean_metrics

    def save_resume_checkpoint(self, tag="latest_rl", itr=None):
        path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_itr = self.itr if itr is None else int(itr)
        payload = {
            "cfg": self.cfg,
            "itr": save_itr,
            "global_step": self.global_step,
            "ppo_actor": self.ppo_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "consecutive_zero_success_iters": self.consecutive_zero_success_iters,
            "best_eval_score": self.best_eval_score,
            "best_eval_itr": self.best_eval_itr,
        }
        torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path)

    def load_resume_checkpoint(self, path):
        payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
        self.itr = int(payload["itr"])
        self.global_step = int(payload.get("global_step", 0))
        self.ppo_actor.load_state_dict(payload["ppo_actor"], strict=False)
        self.critic.load_state_dict(payload["critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.consecutive_zero_success_iters = int(payload.get("consecutive_zero_success_iters", 0))
        self.best_eval_score = float(payload.get("best_eval_score", float("-inf")))
        self.best_eval_itr = int(payload.get("best_eval_itr", -1))
        cprint(f"[RL] Resumed RL checkpoint from {path}", "green")

    def export_actor_checkpoint(self, tag="latest"):
        path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)

        export_cfg = copy.deepcopy(self.cfg)
        OmegaConf.set_struct(export_cfg, False)
        export_cfg.training.use_ema = True
        OmegaConf.set_struct(export_cfg, True)

        actor_state = _copy_state_to_cpu(self.model.state_dict())
        payload = {
            "cfg": export_cfg,
            "state_dicts": {
                "model": actor_state,
                "ema_model": copy.deepcopy(actor_state),
            },
            "pickles": {
                "global_step": dill.dumps(self.global_step),
                "epoch": dill.dumps(self.itr),
                "_output_dir": dill.dumps(self.output_dir),
            },
        }
        torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path)

    def _eval_cfg_value(self, key, default=None):
        eval_cfg = self.cfg.rl.get("eval", {})
        if eval_cfg is None:
            return default
        return eval_cfg.get(key, default)

    def _close_envs_for_eval(self, envs):
        next_start_seeds = [int(getattr(env, "next_seed", 0)) for env in envs]
        for env in envs:
            env.close(clear_cache=True)
        self.current_obs = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return next_start_seeds

    def _parse_eval_score(self, result_path: pathlib.Path) -> float:
        text = result_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", text)
        if len(matches) == 0:
            raise RuntimeError(f"Could not parse eval score from {result_path}")
        return float(matches[-1])

    def _derive_addition_info(self, config_name: str) -> str:
        addition_info = self._eval_cfg_value("addition_info", None)
        if addition_info not in (None, "", "null", "None"):
            return str(addition_info)
        exp_name = str(self.cfg.get("exp_name", ""))
        prefix = f"{self.cfg.task_name}-{config_name}-"
        if exp_name.startswith(prefix):
            return exp_name[len(prefix):]
        return exp_name

    def run_eval_and_update_best(self, eval_iteration: int) -> Dict[str, float]:
        if not bool(self._eval_cfg_value("enabled", False)):
            return {}

        config_name = str(self._eval_cfg_value("config_name", "reinflow_rl_pointcloud_robotwin2"))
        alg_name = str(self._eval_cfg_value("alg_name", config_name))
        policy_name = str(self._eval_cfg_value("policy_name", "ManiFlow"))
        task_config = str(self._eval_cfg_value("task_config", self.cfg.task_config))
        ckpt_setting = str(self._eval_cfg_value("ckpt_setting", task_config))
        eval_seed = str(self._eval_cfg_value("seed", 0))
        candidate_tag_base = str(self._eval_cfg_value("candidate_ckpt_tag", "eval_candidate"))
        best_tag = str(self._eval_cfg_value("best_ckpt_tag", "best"))
        candidate_tag = candidate_tag_base
        addition_info = self._derive_addition_info(config_name)

        self.export_actor_checkpoint(candidate_tag)

        log_dir = pathlib.Path(self.output_dir) / "eval_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = log_dir / f"{candidate_tag_base}_itr_{eval_iteration:04d}.log"
        result_path = pathlib.Path(self.output_dir) / "eval_results" / f"epoch_{self.itr}" / "_result.txt"
        if result_path.exists():
            result_path.unlink()

        cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            f"policy/{policy_name}/deploy_policy.yml",
            "--overrides",
            "--config_name",
            config_name,
            "--task_name",
            str(self.cfg.task_name),
            "--task_config",
            task_config,
            "--ckpt_setting",
            ckpt_setting,
            "--expert_data_num",
            str(self.cfg.expert_data_num),
            "--training_seed",
            str(self.cfg.training.seed),
            "--seed",
            eval_seed,
            "--policy_name",
            policy_name,
            "--addition_info",
            addition_info,
            "--alg_name",
            alg_name,
            "--ckpt_tag",
            candidate_tag,
        ]
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        cprint(
            f"[RL Eval] itr={self.itr} running eval with ckpt_tag={candidate_tag}; log={eval_log_path}",
            "yellow",
        )
        with eval_log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=ROBOTWIN_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Periodic eval failed with return code {proc.returncode}. See {eval_log_path}"
            )
        if not result_path.is_file():
            raise FileNotFoundError(f"Periodic eval did not write expected result file: {result_path}")

        score = self._parse_eval_score(result_path)
        is_best = score > self.best_eval_score
        if is_best:
            self.best_eval_score = score
            self.best_eval_itr = int(eval_iteration)
            self.export_actor_checkpoint(best_tag)
            cprint(
                f"[RL Eval] new best {score:.4f} at iteration {eval_iteration}; saved {best_tag}.ckpt",
                "green",
            )
        else:
            cprint(
                f"[RL Eval] score={score:.4f}, best={self.best_eval_score:.4f} "
                f"at iteration {self.best_eval_itr}",
                "yellow",
            )

        return {
            "eval/success_rate": score,
            "eval/is_best": float(is_best),
            "eval/best_success_rate": float(self.best_eval_score),
            "eval/best_iteration": float(self.best_eval_itr),
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.rl.resume_path:
            self.load_resume_checkpoint(cfg.rl.resume_path)
        elif bool(cfg.rl.resume):
            latest = pathlib.Path(self.output_dir) / "checkpoints" / "latest_rl.ckpt"
            if latest.is_file():
                self.load_resume_checkpoint(latest)

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        use_wandb = cfg.logging.mode not in ("disabled", "none", None)
        wandb_run = None
        if use_wandb:
            cfg.logging.name = str(cfg.logging.name)
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir})

        envs = self._make_envs()
        buffer = self._make_buffer()
        try:
            while self.itr < int(cfg.rl.ppo.num_iterations):
                t0 = time.time()
                num_iterations = int(cfg.rl.ppo.num_iterations)
                buffer.reset()
                rollout_log = self.collect_rollout(envs, buffer)
                update_actor, actor_skip_reason, actor_skip_code, actor_loss_scale = self.should_update_actor(rollout_log)
                step_log = {
                    "itr": self.itr,
                    "epoch": self.itr,
                    "global_step": self.global_step,
                    "lr/actor": self.actor_optimizer.param_groups[0]["lr"],
                    "lr/critic": self.critic_optimizer.param_groups[0]["lr"],
                    "rl/actor_update_enabled": float(update_actor),
                    "rl/actor_loss_scale": float(actor_loss_scale),
                    "rl/actor_update_skip_code": float(actor_skip_code),
                    "rl/consecutive_zero_success_iters": float(self.consecutive_zero_success_iters),
                }
                step_log.update({f"rollout/{k}": v for k, v in rollout_log.items()})
                update_log = self.ppo_update(
                    buffer,
                    update_actor=update_actor,
                    actor_loss_scale=actor_loss_scale,
                )
                step_log.update(update_log)
                step_log["time/itr_sec"] = time.time() - t0

                cprint(
                    f"[RL] itr={self.itr} step={self.global_step} "
                    f"succ={step_log.get('rollout/success_rate', 0):.3f} "
                    f"raw_reward={step_log.get('rollout/rollout_raw_reward_sum', 0):.3f} "
                    f"actor_update={int(update_actor)}({actor_skip_reason}, scale={actor_loss_scale:.2f}) "
                    f"loss={step_log.get('loss/loss', 0):.4f}",
                    "cyan",
                )

                if (
                    self.itr % int(cfg.rl.checkpoint_every) == 0
                    or self.itr == num_iterations - 1
                ):
                    self.save_resume_checkpoint("latest_rl", itr=self.itr + 1)
                    if bool(cfg.rl.export_actor_checkpoint):
                        self.export_actor_checkpoint("latest")

                eval_interval = int(self._eval_cfg_value("interval", 0))
                eval_enabled = bool(self._eval_cfg_value("enabled", False)) and eval_interval > 0
                eval_iteration = self.itr + 1
                should_eval = eval_enabled and (
                    eval_iteration % eval_interval == 0
                    or (
                        bool(self._eval_cfg_value("run_final", True))
                        and self.itr == num_iterations - 1
                    )
                )
                if should_eval:
                    next_start_seeds = self._close_envs_for_eval(envs)
                    envs = []
                    eval_log = self.run_eval_and_update_best(eval_iteration)
                    step_log.update(eval_log)
                    self.save_resume_checkpoint("latest_rl", itr=self.itr + 1)
                    if self.itr < num_iterations - 1:
                        envs = self._make_envs(start_seeds=next_start_seeds)

                if use_wandb:
                    wandb_run.log(step_log, step=self.itr)

                self.itr += 1
                self.epoch = self.itr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            cprint(
                f"[RL] Caught KeyboardInterrupt at itr={self.itr}, global_step={self.global_step}. "
                "Saving resume checkpoint before exit.",
                "yellow",
            )
            self.save_resume_checkpoint("latest_rl")
            if bool(cfg.rl.export_actor_checkpoint):
                self.export_actor_checkpoint("latest")
            raise
        finally:
            for env in envs:
                env.close(clear_cache=True)
            if use_wandb:
                wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
)
def main(cfg):
    workspace = TrainReinFlowRLRoboTwinWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
