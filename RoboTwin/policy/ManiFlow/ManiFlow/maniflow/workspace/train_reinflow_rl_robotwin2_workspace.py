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
        self.best_checkpoint_metric = None
        self.best_checkpoint_itr = -1
        self.consecutive_zero_success_iters = 0

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
        return {
            "point_cloud": torch.from_numpy(obs["point_cloud"]).unsqueeze(0).to(device=device).float(),
            "agent_pos": torch.from_numpy(obs["agent_pos"]).unsqueeze(0).to(device=device).float(),
        }

    def _obs_to_buffer(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            "point_cloud": obs["point_cloud"][None].astype(np.float32),
            "agent_pos": obs["agent_pos"][None].astype(np.float32),
        }

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

    def _get_best_checkpoint_cfg(self):
        if "best_checkpoint" in self.cfg.rl:
            return self.cfg.rl.best_checkpoint
        return {}

    def _is_better_checkpoint_metric(self, metric_value: float) -> bool:
        best_cfg = self._get_best_checkpoint_cfg()
        mode = str(best_cfg.get("mode", "max")).lower()
        min_delta = float(best_cfg.get("min_delta", 0.0))
        if self.best_checkpoint_metric is None:
            return True
        if mode == "min":
            return metric_value < float(self.best_checkpoint_metric) - min_delta
        return metric_value > float(self.best_checkpoint_metric) + min_delta

    def maybe_export_best_actor_checkpoint(self, step_log: Dict[str, float]) -> bool:
        best_cfg = self._get_best_checkpoint_cfg()
        if not bool(best_cfg.get("enabled", True)):
            return False
        if not bool(self.cfg.rl.export_actor_checkpoint):
            return False

        metric_key = str(best_cfg.get("metric", "rollout/success_rate"))
        if metric_key not in step_log:
            cprint(f"[RL] Best checkpoint metric {metric_key} is missing; skip best export.", "yellow")
            return False

        metric_value = float(step_log[metric_key])
        if not self._is_better_checkpoint_metric(metric_value):
            return False

        tag = str(best_cfg.get("tag", "best_success"))
        self.best_checkpoint_metric = metric_value
        self.best_checkpoint_itr = int(self.itr)
        best_path = self.export_actor_checkpoint(tag)

        if bool(best_cfg.get("update_latest_alias", True)):
            self.export_actor_checkpoint("latest")

        cprint(
            f"[RL] New best actor checkpoint at itr={self.itr}: "
            f"{metric_key}={metric_value:.4f}, saved {best_path}",
            "green",
        )
        return True

    def should_update_actor(self, rollout_log: Dict[str, float]):
        warmup_iters = int(self.cfg.rl.ppo.critic_warmup_iterations)
        success_rate = float(rollout_log.get("success_rate", 0.0))
        raw_reward_sum = float(rollout_log.get("rollout_raw_reward_sum", 0.0))

        if success_rate <= 0.0:
            self.consecutive_zero_success_iters += 1
        else:
            self.consecutive_zero_success_iters = 0

        if self.itr < warmup_iters:
            return False, "critic_warmup", 1

        min_success = float(self.cfg.rl.ppo.get("min_success_rate_for_actor_update", 0.0))
        if bool(self.cfg.rl.ppo.get("skip_actor_update_on_zero_success", True)) and success_rate <= min_success:
            return False, "zero_success_rollout", 2

        min_raw_reward = float(self.cfg.rl.ppo.get("min_raw_reward_for_actor_update", 0.0))
        if bool(self.cfg.rl.ppo.get("skip_actor_update_on_zero_reward", False)) and raw_reward_sum <= min_raw_reward:
            return False, "zero_reward_rollout", 3

        return True, "enabled", 0

    def collect_rollout(self, env, buffer: PPOBufferPointcloud):
        if self.current_obs is None or bool(self.cfg.rl.rollout.reset_each_iteration):
            self.current_obs = env.reset()

        self.ppo_actor.eval()
        self.critic.eval()
        rollout_infos = []
        desc = f"RL rollout itr {self.itr}"
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
                action_np = action.squeeze(0).detach().cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                raw_reward = float(np.sum(info.get("raw_rewards", [reward])))
                buffer.add(
                    step=step,
                    obs=obs_for_buffer,
                    chains=chains.detach().cpu().numpy(),
                    reward=np.asarray([reward], dtype=np.float32),
                    terminated=np.asarray([terminated], dtype=bool),
                    truncated=np.asarray([truncated], dtype=bool),
                    value=value,
                    logprob=logprob.detach().cpu().numpy(),
                    success=np.asarray([info.get("success", False)], dtype=bool),
                    raw_reward=np.asarray([raw_reward], dtype=np.float32),
                )
                self.global_step += int(info.get("executed_action_steps", self.cfg.n_action_steps))
                rollout_infos.append(info)
                pbar.set_postfix(
                    reward=f"{reward:.2f}",
                    success=int(info.get("success", False)),
                    env_step=info.get("take_action_cnt", 0),
                    refresh=False,
                )
                self.current_obs = next_obs
                if terminated or truncated:
                    self.current_obs = env.reset()

        summary = buffer.summarize_episode_reward()
        summary["rollout_chunks"] = len(rollout_infos)
        summary["rollout_last_env_step"] = rollout_infos[-1].get("take_action_cnt", 0) if rollout_infos else 0
        summary["rollout_any_success_info"] = float(any(info.get("success", False) for info in rollout_infos))
        return summary

    def ppo_update(self, buffer: PPOBufferPointcloud, update_actor: bool = True):
        buffer.update(self._obs_to_buffer(self.current_obs), self.critic)
        obs, chains, returns, values, advantages, logprobs = buffer.make_dataset()
        explained_var = buffer.get_explained_var(values, returns)
        total_steps = returns.shape[0]
        batch_size = int(self.cfg.rl.ppo.batch_size)
        update_epochs = int(self.cfg.rl.ppo.update_epochs)
        target_kl = self.cfg.rl.ppo.target_kl
        actor_update_enabled = bool(update_actor) and self.itr >= int(self.cfg.rl.ppo.critic_warmup_iterations)

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
                        loss = (
                            loss_dict["pg_loss"]
                            + self.cfg.rl.ppo.ent_coef * loss_dict["entropy_loss"]
                            + self.cfg.rl.ppo.vf_coef * loss_dict["value_loss"]
                            + self.cfg.rl.bc_anchor.coeff * loss_dict["bc_loss"]
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
            "best_checkpoint_metric": self.best_checkpoint_metric,
            "best_checkpoint_itr": self.best_checkpoint_itr,
            "consecutive_zero_success_iters": self.consecutive_zero_success_iters,
        }
        torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path)

    def load_resume_checkpoint(self, path, load_optimizers=True):
        payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
        self.itr = int(payload["itr"])
        self.global_step = int(payload.get("global_step", 0))
        self.ppo_actor.load_state_dict(payload["ppo_actor"], strict=False)
        self.critic.load_state_dict(payload["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if load_optimizers and "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.best_checkpoint_metric = payload.get("best_checkpoint_metric", None)
        self.best_checkpoint_itr = int(payload.get("best_checkpoint_itr", -1))
        self.consecutive_zero_success_iters = int(payload.get("consecutive_zero_success_iters", 0))
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

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if int(cfg.rl.rollout.n_envs) != 1:
            raise NotImplementedError("This minimal RoboTwin RL workspace currently supports n_envs=1.")

        loaded_resume_checkpoint = False
        export_resume_actor_only = bool(cfg.rl.get("export_resume_actor_only", False))
        if cfg.rl.resume_path:
            self.load_resume_checkpoint(cfg.rl.resume_path, load_optimizers=not export_resume_actor_only)
            loaded_resume_checkpoint = True
        elif bool(cfg.rl.resume):
            latest = pathlib.Path(self.output_dir) / "checkpoints" / "latest_rl.ckpt"
            if latest.is_file():
                self.load_resume_checkpoint(latest, load_optimizers=not export_resume_actor_only)
                loaded_resume_checkpoint = True

        if export_resume_actor_only:
            if not loaded_resume_checkpoint:
                raise RuntimeError(
                    "rl.export_resume_actor_only=true requires rl.resume_path "
                    "or rl.resume=true with checkpoints/latest_rl.ckpt present."
                )
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            tag = str(cfg.rl.get("export_resume_actor_tag", "final_rl_actor"))
            export_path = self.export_actor_checkpoint(tag)
            cprint(f"[RL] Exported deploy actor checkpoint from RL resume to {export_path}", "green")
            return

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

        env = hydra.utils.instantiate(cfg.rl.env)
        buffer = self._make_buffer()
        try:
            while self.itr < int(cfg.rl.ppo.num_iterations):
                t0 = time.time()
                buffer.reset()
                rollout_log = self.collect_rollout(env, buffer)
                update_actor, actor_skip_reason, actor_skip_code = self.should_update_actor(rollout_log)
                step_log = {
                    "itr": self.itr,
                    "epoch": self.itr,
                    "global_step": self.global_step,
                    "lr/actor": self.actor_optimizer.param_groups[0]["lr"],
                    "lr/critic": self.critic_optimizer.param_groups[0]["lr"],
                    "rl/actor_update_enabled": float(update_actor),
                    "rl/actor_update_skip_code": float(actor_skip_code),
                    "rl/consecutive_zero_success_iters": float(self.consecutive_zero_success_iters),
                }
                step_log.update({f"rollout/{k}": v for k, v in rollout_log.items()})
                saved_best = self.maybe_export_best_actor_checkpoint(step_log)
                step_log["rl/saved_best_actor_checkpoint"] = float(saved_best)
                step_log["rl/best_checkpoint_itr"] = float(self.best_checkpoint_itr)
                step_log["rl/best_checkpoint_metric"] = (
                    float(self.best_checkpoint_metric) if self.best_checkpoint_metric is not None else float("nan")
                )
                update_log = self.ppo_update(buffer, update_actor=update_actor)
                step_log.update(update_log)
                step_log["time/itr_sec"] = time.time() - t0

                cprint(
                    f"[RL] itr={self.itr} step={self.global_step} "
                    f"succ={step_log.get('rollout/success_rate', 0):.3f} "
                    f"raw_reward={step_log.get('rollout/rollout_raw_reward_sum', 0):.3f} "
                    f"actor_update={int(update_actor)}({actor_skip_reason}) "
                    f"loss={step_log.get('loss/loss', 0):.4f}",
                    "cyan",
                )

                if use_wandb:
                    wandb_run.log(step_log, step=self.itr)

                if (
                    self.itr % int(cfg.rl.checkpoint_every) == 0
                    or self.itr == int(cfg.rl.ppo.num_iterations) - 1
                ):
                    self.save_resume_checkpoint("latest_rl", itr=self.itr + 1)
                    if bool(cfg.rl.export_actor_checkpoint) and not bool(
                        self._get_best_checkpoint_cfg().get("enabled", True)
                    ):
                        self.export_actor_checkpoint("latest")

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
            if bool(cfg.rl.export_actor_checkpoint) and self.best_checkpoint_metric is None:
                self.export_actor_checkpoint("latest")
            raise
        finally:
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
