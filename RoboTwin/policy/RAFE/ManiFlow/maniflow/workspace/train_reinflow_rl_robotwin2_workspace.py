if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import json
from contextlib import nullcontext
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
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from termcolor import cprint

MANIFLOW_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
ROBOTWIN_ROOT = str(pathlib.Path(MANIFLOW_ROOT).parent.parent)
sys.path.append(MANIFLOW_ROOT)
sys.path.insert(0, os.path.join(MANIFLOW_ROOT, "ManiFlow"))
sys.path.append(os.path.join(MANIFLOW_ROOT, "ManiFlow", "maniflow"))
sys.path.append(ROBOTWIN_ROOT)

from maniflow.common.ppo_buffer_pointcloud import PPOBufferPointcloud
from maniflow.common.ppo_optimizer import build_actor_optimizer, load_actor_optimizer_state
from maniflow.common.expert_data import ExpertDataIterator, capture_rng_state, restore_rng_state
from maniflow.common.robotwin_env_pool import SerialEnvPool, SubprocessEnvPool
from maniflow.common.rl_progress import RLIterationProgress
from maniflow.common.pytorch_util import dict_apply
from maniflow.model.rl.pointcloud_critic import PointcloudCritic
from maniflow.model.rl.ppo_maniflow_pointcloud import PPOManiFlowPointcloud
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from script.eval_scene_selection import check_selection_mode, evaluation_result_dir
from script.eval_seed_bank import DEFAULT_SHARED_SEED_FILE
from script.eval_baseline import DEFAULT_BASELINE_CHECKPOINT

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_state_to_cpu(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in state_dict.items()}


def _plain_config(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def _canonical_policy_config(value):
    """Accept disabled legacy metadata without retaining a predictor setting."""
    config = _plain_config(value)
    legacy = config.pop("action_plan", None)
    if legacy is not None and (not isinstance(legacy, dict) or legacy.get("enabled", True) is not False):
        raise ValueError("RAFE does not support enabled Action Reference Predictor checkpoints/configurations")
    return config


def _reject_action_reference_weights(state_dict):
    legacy_modules = {"plan_predictor", "plan_proj", "coarse_action_operator"}
    incompatible = [key for key in state_dict if legacy_modules.intersection(key.split("."))]
    if incompatible:
        raise ValueError(f"RAFE cannot load Action Reference Predictor weights: {incompatible[:4]}")


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
    # Legacy workspace instances/snapshots have no expert-guidance attributes.
    expert_noise_enabled = False
    noise_optimizer = None

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        _canonical_policy_config(cfg.policy)
        if cfg.rl.actor.get("noise_parameterization") != "log_residual":
            raise ValueError("RAFE requires explicit rl.actor.noise_parameterization=log_residual; legacy bounded-noise configurations cannot be reinterpreted")
        if "action_plan" in cfg.policy:
            # Older no-reference RAFE configs carried a disabled, unused group.
            with open_dict(cfg.policy):
                del cfg.policy["action_plan"]
        self.cfg = cfg
        self._output_dir = output_dir
        self.itr = 0
        self.global_step = 0
        self.epoch = 0
        self.current_obs = None
        self.consecutive_zero_success_iters = 0
        self.best_eval_score = float("-inf")
        self.best_eval_itr = -1
        self.noise_guidance = cfg.rl.get("noise_guidance", {}) or {}
        self.expert_noise_enabled = bool(self.noise_guidance.get("enabled", False))
        self.noise_update_steps = 0
        self.noise_warmup_complete = False
        self.noise_lower_bound_streak = 0
        self.noise_unchanged_streak = 0
        self.noise_optimizer = None
        self.expert_data = None
        if self.expert_noise_enabled:
            self._validate_noise_guidance()

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

        self.actor_optimizer = build_actor_optimizer(self.ppo_actor, cfg.rl.ppo, include_noise=not self.expert_noise_enabled)
        if self.expert_noise_enabled:
            self.model.normalizer.requires_grad_(False)
            self.noise_optimizer = torch.optim.AdamW(
                self.ppo_actor.noise_head.parameters(),
                lr=float(self.noise_guidance.learning_rate),
                weight_decay=float(self.noise_guidance.get("weight_decay", 0.0)),
                betas=tuple(cfg.rl.ppo.betas),
            )
            self.noise_generator = torch.Generator(device=device).manual_seed(seed + 19109)
            self.expert_data = ExpertDataIterator(cfg)
            actor_ids = {id(p) for g in self.actor_optimizer.param_groups for p in g["params"]}
            noise_ids = {id(p) for g in self.noise_optimizer.param_groups for p in g["params"]}
            if actor_ids & noise_ids:
                raise RuntimeError("PPO and noise optimizers must have disjoint parameters")
        self._log_actor_learning_rates()
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
            if self.expert_noise_enabled:
                raise ValueError("RAFE requires an original ManiFlow pretrained checkpoint with its action normalizer")
            cprint("[RL] No pretrained checkpoint path provided; using randomly initialized actor.", "yellow")
            return
        ckpt_path = pathlib.Path(str(ckpt_path)).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pretrained ManiFlow checkpoint not found: {ckpt_path}")
        if self.expert_noise_enabled:
            stat = ckpt_path.stat()
            self.pretrained_reference_signature = {
                "path": str(ckpt_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "state_key": self.cfg.rl.get("pretrained_state_key", "ema_model"),
            }

        payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
        checkpoint_policy = _canonical_policy_config(payload.get("cfg", {}).get("policy", {}))
        if self.expert_noise_enabled:
            architecture_keys = (
                "block_type", "n_layer", "n_head", "n_emb", "visual_cond_len", "max_lang_cond_len",
                "qkv_bias", "qk_norm", "language_conditioned", "pre_norm_modality", "encoder_type",
                "encoder_output_dim", "pointnet_type", "pointcloud_encoder_cfg", "use_pc_color",
                "horizon", "n_obs_steps", "n_action_steps", "shape_meta", "obs_as_global_cond",
                "diffusion_timestep_embed_dim", "diffusion_target_t_embed_dim", "sample_target_t_mode",
            )
            for key in architecture_keys:
                if key in checkpoint_policy and _plain_config(checkpoint_policy[key]) != _plain_config(self.cfg.policy.get(key)):
                    raise ValueError(f"RAFE pretrained policy architecture/semantics mismatch at policy.{key}; use a matching original ManiFlow checkpoint/configuration")
        state_dicts = payload.get("state_dicts", {})
        preferred_key = self.cfg.rl.get("pretrained_state_key", "ema_model")
        if preferred_key in state_dicts:
            state_dict = state_dicts[preferred_key]
        elif self.expert_noise_enabled:
            raise KeyError(f"RAFE checkpoint missing requested state_dicts.{preferred_key}; select the intended pretrained state explicitly")
        elif "ema_model" in state_dicts:
            state_dict = state_dicts["ema_model"]
        elif "model" in state_dicts:
            state_dict = state_dicts["model"]
        else:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain state_dicts.model/ema_model")

        _reject_action_reference_weights(state_dict)
        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=True if self.expert_noise_enabled else bool(self.cfg.rl.get("strict_load_pretrained", True)),
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

    def _log_actor_learning_rates(self):
        rates = {group["name"]: group["lr"] for group in self.actor_optimizer.param_groups}
        noise_rate = self.noise_optimizer.param_groups[0]["lr"] if self.noise_optimizer is not None else rates['noise']
        cprint(
            f"[RL] AdamW learning rates: actor={rates['actor']:.2e}, noise={noise_rate:.2e}",
            "cyan",
        )

    def _validate_noise_guidance(self):
        guidance = self.noise_guidance
        if guidance.get("mode") != "expert_interpolation":
            raise ValueError("Unsupported enabled noise_guidance.mode; expected expert_interpolation")
        actor = self.cfg.rl.actor
        required = {"noise_conditioning_mode": "default", "noise_output_mode": "full",
                    "noise_head_type": "residual_schedule", "noise_backbone": "tcn",
                    "base_sigma_schedule": "linear", "freeze_obs_encoder": True,
                    "noise_parameterization": "log_residual"}
        for key, value in required.items():
            if actor.get(key) != value:
                raise ValueError(f"RAFE expert interpolation requires rl.actor.{key}={value!r}")
        for key in ("batch_size", "update_interval", "gradient_steps"):
            if int(guidance[key]) < 1:
                raise ValueError(f"noise_guidance.{key} must be positive")
        if int(guidance.get("initial_warmup_steps", 0)) < 0:
            raise ValueError("noise_guidance.initial_warmup_steps must be nonnegative")
        rate = float(guidance.learning_rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("noise_guidance.learning_rate must be finite and positive")
        if float(self.cfg.rl.ppo.ent_coef) != 0:
            raise ValueError("RAFE expert interpolation requires rl.ppo.ent_coef=0 (no additional entropy objective)")

    def noise_nll_update(self, gradient_steps=None):
        """Update only the exploration head, after the complete PPO batch."""
        if not self.expert_noise_enabled:
            return {}
        started = time.perf_counter()
        count = int(self.noise_guidance.gradient_steps if gradient_steps is None else gradient_steps)
        if count < 1:
            return {}
        self.ppo_actor.eval()
        # Clear stale PPO gradients so isolation is directly observable.
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        fixed = list(self.model.parameters()) + list(self.critic.parameters())
        versions = [p._version for p in fixed]
        metrics = []
        noise_params = list(self.ppo_actor.noise_head.parameters())
        for _ in range(count):
            batch = next(self.expert_data)
            self.noise_optimizer.zero_grad(set_to_none=True)
            # Gaussian likelihood arithmetic stays float32; PPO retains its AMP path.
            loss, info = self.ppo_actor.compute_expert_interpolation_noise_loss(
                batch, sample_all_steps=bool(self.noise_guidance.sample_all_steps),
                generator=self.noise_generator,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("RAFE expert Gaussian NLL is NaN/Inf; inspect residuals, sigma and configured bounds")
            loss.backward()
            if any(p.grad is not None for p in fixed):
                raise RuntimeError("Expert NLL leaked gradients into actor, observation encoder, normalizer or critic")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                noise_params, float(self.noise_guidance.get("max_grad_norm", 1.0)),
                error_if_nonfinite=True,
            )
            before = [p.detach().clone() for p in noise_params]
            self.noise_optimizer.step()
            changed = any(not torch.equal(p.detach(), old) for p, old in zip(noise_params, before))
            self.noise_unchanged_streak = 0 if changed else self.noise_unchanged_streak + 1
            self.noise_update_steps += 1
            item = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in info.items()}
            item.update(noise_update_grad_norm=float(grad_norm), noise_parameters_changed=float(changed))
            self.noise_lower_bound_streak = self.noise_lower_bound_streak + 1 if item.get("noise_sigma_at_lower_bound_ratio", 0) >= 0.999 else 0
            if self.noise_lower_bound_streak == 10 or self.noise_unchanged_streak == 10:
                cprint("[RAFE] Noise has remained at its lower bound or unchanged for 10 updates; inspect clamp saturation and representable residual range.", "yellow")
            if self.noise_update_steps == 1 and item.get("noise_residual_above_representable_ratio", 0) > 0.5:
                cprint("[RAFE] Most expert residuals exceed the representable sigma range. Inspect per-step residual/sigma logs and sigma bounds; residual_scale is a linear gain and does not set the log-residual range.", "yellow")
            metrics.append(item)
        if any(p._version != version for p, version in zip(fixed, versions)):
            raise RuntimeError("Expert noise optimizer modified actor/critic parameters")
        self.noise_optimizer.zero_grad(set_to_none=True)
        keys = set().union(*(item.keys() for item in metrics))
        result = {key: float(np.mean([item[key] for item in metrics if key in item])) for key in sorted(keys)}
        if not bool(self.noise_guidance.get("log_diagnostics", True)):
            result = {key: value for key, value in result.items() if "_by_step/" not in key}
        result.update(noise_update_steps=float(self.noise_update_steps), noise_update_time=time.perf_counter()-started,
                      noise_lower_bound_streak=float(self.noise_lower_bound_streak),
                      noise_unchanged_streak=float(self.noise_unchanged_streak))
        return result

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
        env_configs = []
        for env_idx in range(n_envs):
            env_config = OmegaConf.to_container(self.cfg.rl.env, resolve=True)
            env_config.update(
                seed=base_seed + env_idx,
                start_seed=(int(start_seeds[env_idx]) if start_seeds is not None
                            else base_start_seed + env_idx * env_seed_stride),
            )
            env_configs.append(env_config)
        backend = str(self.cfg.rl.rollout.get("backend", "serial"))
        if backend == "serial":
            envs = SerialEnvPool(env_configs)
        elif backend == "subprocess":
            observation_shapes = None
            if bool(self.cfg.rl.rollout.get("shared_memory", True)):
                observation_shapes = {
                    key: (int(self.cfg.n_obs_steps), *tuple(self.cfg.shape_meta.obs[key].shape))
                    for key in ("point_cloud", "agent_pos")
                }
            envs = SubprocessEnvPool(
                env_configs, observation_shapes=observation_shapes,
                worker_threads=int(self.cfg.rl.rollout.get("worker_threads", 1)),
                timeout_sec=float(self.cfg.rl.rollout.get("worker_timeout_sec", 1800)),
            )
        else:
            raise ValueError(f"Unsupported rollout backend: {backend}")
        cprint(
            f"[RL] Created {n_envs} RoboTwin envs ({backend}) with start_seed="
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
        low_success = success_rate < min_success if min_success > 0.0 else success_rate <= 0.0
        skip_low_success = bool(
            self.cfg.rl.ppo.get(
                "skip_actor_update_on_low_success",
                self.cfg.rl.ppo.get("skip_actor_update_on_zero_success", True),
            )
        )
        if skip_low_success and low_success:
            return False, "low_success_rollout", 2, 0.0

        min_raw_reward = float(self.cfg.rl.ppo.get("min_raw_reward_for_actor_update", 0.0))
        if bool(self.cfg.rl.ppo.get("skip_actor_update_on_zero_reward", False)) and raw_reward_sum <= min_raw_reward:
            return False, "zero_reward_rollout", 3, 0.0

        downweight_reasons = []
        downweight_code = 0
        downweight_low_success = bool(
            self.cfg.rl.ppo.get(
                "downweight_actor_update_on_low_success",
                self.cfg.rl.ppo.get("downweight_actor_update_on_zero_success", False),
            )
        )
        if downweight_low_success and low_success:
            actor_loss_scale = min(
                actor_loss_scale,
                float(
                    self.cfg.rl.ppo.get(
                        "low_success_actor_loss_scale",
                        self.cfg.rl.ppo.get("zero_success_actor_loss_scale", 0.25),
                    )
                ),
            )
            downweight_reasons.append("low_success_downweighted")
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

    def collect_rollout(self, envs, buffer: PPOBufferPointcloud, progress=None):
        n_envs = len(envs)
        if n_envs != buffer.n_envs:
            raise ValueError(f"Expected {buffer.n_envs} envs, got {n_envs}")

        initial_reset_sec = 0.0
        reset_infos = []
        self.ppo_actor.eval()
        self.critic.eval()
        rollout_infos = []
        sampling_noise_stats = []
        actor_sec = 0.0
        env_wall_sec = 0.0
        progress_context = nullcontext(progress) if progress is not None else RLIterationProgress(
            self.itr, buffer.n_steps, self.global_step, self.cfg.training.tqdm_interval_sec,
        )
        with progress_context as progress:
            if self.current_obs is None or bool(self.cfg.rl.rollout.reset_each_iteration):
                started = time.perf_counter()
                self.current_obs = envs.reset()
                initial_reset_sec = time.perf_counter() - started
                reset_infos.extend(envs.last_reset_infos)
            progress.start_rollout()
            for step in range(buffer.n_steps):
                actor_started = time.perf_counter()
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
                    sampling_noise_stats.append({
                        k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                        for k, v in self.ppo_actor.last_action_noise_stats.items()
                    })
                if action.shape[0] != n_envs or chains.shape[0] != n_envs or logprob.shape[0] != n_envs:
                    raise RuntimeError(
                        "Batched actor output does not match n_envs: "
                        f"action={tuple(action.shape)}, chains={tuple(chains.shape)}, "
                        f"logprob={tuple(logprob.shape)}, n_envs={n_envs}"
                    )
                action_np = action.detach().cpu().numpy()
                actor_sec += time.perf_counter() - actor_started

                next_obs_batch = []
                rewards = np.zeros(n_envs, dtype=np.float32)
                raw_rewards = np.zeros(n_envs, dtype=np.float32)
                terminateds = np.zeros(n_envs, dtype=bool)
                truncateds = np.zeros(n_envs, dtype=bool)
                successes = np.zeros(n_envs, dtype=bool)
                executed_steps = 0
                step_infos = []

                started = time.perf_counter()
                results = envs.step(action_np)
                env_wall_sec += time.perf_counter() - started
                for env_idx, (next_obs, reward, terminated, truncated, info) in enumerate(results):
                    raw_reward = float(np.sum(info.get("raw_rewards", [reward])))

                    rewards[env_idx] = float(reward)
                    raw_rewards[env_idx] = raw_reward
                    terminateds[env_idx] = bool(terminated)
                    truncateds[env_idx] = bool(truncated)
                    successes[env_idx] = bool(info.get("success", False))
                    executed_steps += int(info.get("executed_action_steps", self.cfg.n_action_steps))
                    step_infos.append(info)
                    rollout_infos.append(info)

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
                progress.advance_rollout(
                    completed=step + 1,
                    successes=successes,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    executed_steps=executed_steps,
                    global_step=self.global_step,
                    env_steps=[int(info.get("take_action_cnt", 0)) for info in step_infos],
                    reward=float(np.mean(rewards)),
                )
                self.current_obs = next_obs_batch

        summary = buffer.summarize_episode_reward()
        summary["rollout_envs"] = n_envs
        summary["rollout_chunks"] = len(rollout_infos)
        summary["rollout_last_env_step"] = rollout_infos[-1].get("take_action_cnt", 0) if rollout_infos else 0
        summary["rollout_any_success_info"] = float(any(info.get("success", False) for info in rollout_infos))
        summary["actor_inference_sec"] = actor_sec
        summary["env_wall_sec"] = env_wall_sec
        summary["initial_reset_wall_sec"] = initial_reset_sec
        for key in ("env_step_sec", "reset_sec", "expert_check_sec", "expert_cache_hits", "expert_cache_misses"):
            summary[f"worker_{key}_sum"] = float(sum(
                info.get(f"perf/{key}", 0.0) for info in [*reset_infos, *rollout_infos]
            ))
        for info in [*reset_infos, *rollout_infos]:
            validation = info.get("validation", {})
            for key in ("validation_attempts", "rejected_count", "validation_time_sec", "setup_demo_count", "play_once_count"):
                metric = f"scene_validation/{key}"
                summary[metric] = summary.get(metric, 0.0) + float(validation.get(key, 0.0))
            for reason, count in validation.get("rejection_reasons", {}).items():
                metric = f"scene_validation/rejected/{reason}"
                summary[metric] = summary.get(metric, 0.0) + float(count)
            mode = validation.get("validation_mode")
            if mode is not None:
                summary[f"scene_validation/mode_{mode}"] = 1.0
        if len(sampling_noise_stats) > 0:
            for key in sampling_noise_stats[0]:
                values = np.asarray([item[key] for item in sampling_noise_stats], dtype=np.float32)
                if key in ("sigma_min_observed", "delta_min"):
                    summary[f"sampling_{key}"] = float(np.min(values))
                elif key in ("sigma_max_observed", "delta_max"):
                    summary[f"sampling_{key}"] = float(np.max(values))
                else:
                    summary[f"sampling_{key}"] = float(np.mean(values))
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
        progress=None,
    ):
        buffer.update(self._obs_to_buffer(self.current_obs), self.critic)
        obs, chains, returns, values, advantages, logprobs = buffer.make_dataset()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        explained_var = buffer.get_explained_var(values, returns)
        total_steps = returns.shape[0]
        expected_steps = buffer.n_steps * buffer.n_envs
        if total_steps != expected_steps:
            raise RuntimeError(f"PPO dataset has {total_steps} steps, expected {expected_steps}")
        batch_size = int(self.cfg.rl.ppo.batch_size)
        grad_accum_steps = int(self.cfg.rl.ppo.get("grad_accum_steps", 1))
        if grad_accum_steps < 1:
            raise ValueError(f"rl.ppo.grad_accum_steps must be >= 1, got {grad_accum_steps}")
        update_epochs = int(self.cfg.rl.ppo.update_epochs)
        target_kl = self.cfg.rl.ppo.target_kl
        actor_update_enabled = bool(update_actor) and self.itr >= int(self.cfg.rl.ppo.critic_warmup_iterations)
        actor_loss_scale = max(0.0, min(1.0, float(actor_loss_scale)))
        actor_grad_params = list(self.ppo_actor.actor_parameters()) if actor_update_enabled else []
        minibatch_starts = list(range(0, total_steps, batch_size))
        if progress is not None:
            groups_per_epoch = (len(minibatch_starts) + grad_accum_steps - 1) // grad_accum_steps
            progress.start_update(
                minibatches=update_epochs * len(minibatch_starts),
                optimizer_steps=update_epochs * groups_per_epoch,
            )

        self.ppo_actor.eval()
        self.critic.train()
        if self.expert_noise_enabled:
            self.noise_optimizer.zero_grad(set_to_none=True)
            noise_versions = [p._version for p in self.ppo_actor.noise_head.parameters()]
        metrics = []
        stopped_by_kl = False
        optimizer_steps = 0
        for _ in range(update_epochs):
            indices = torch.randperm(total_steps)
            for group_start in range(0, len(minibatch_starts), grad_accum_steps):
                group_starts = minibatch_starts[group_start : group_start + grad_accum_steps]
                accum_divisor = float(len(group_starts))
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                for start in group_starts:
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
                                normalize_advantages=False,
                                learn_exploration_noise=not self.expert_noise_enabled,
                            )
                            actor_rl_loss = (
                                loss_dict["pg_loss"]
                                + self.cfg.rl.ppo.ent_coef * loss_dict["entropy_loss"]
                            )
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
                                "sigma_mean": zero,
                                "sigma_min_observed": zero,
                                "sigma_max_observed": zero,
                                "base_sigma_mean": zero,
                                "delta_mean": zero,
                                "delta_abs_mean": zero,
                                "delta_min": zero,
                                "delta_max": zero,
                                "clamp_ratio_min": zero,
                                "clamp_ratio_max": zero,
                                "value_mean": newvalues.mean().detach(),
                            }
                            loss = self.cfg.rl.ppo.vf_coef * value_loss

                    self.scaler.scale(loss / accum_divisor).backward()
                    if self.expert_noise_enabled and any(p.grad is not None for p in self.ppo_actor.noise_head.parameters()):
                        raise RuntimeError("PPO propagated gradients into the NLL-only noise head")

                    item = {
                        k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                        for k, v in loss_dict.items()
                    }
                    item["loss"] = float(loss.detach().cpu())
                    item["actor_loss_scale"] = actor_loss_scale if actor_update_enabled else 0.0
                    metrics.append(item)
                    if progress is not None:
                        progress.advance_update(len(metrics), optimizer_steps)
                    if actor_update_enabled and target_kl is not None and item["approx_kl"] > float(target_kl):
                        stopped_by_kl = True
                        break

                if self.cfg.rl.ppo.max_grad_norm is not None:
                    if actor_update_enabled:
                        self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    if actor_update_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            actor_grad_params,
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
                optimizer_steps += 1
                if progress is not None:
                    progress.advance_update(len(metrics), optimizer_steps, stopped_by_kl=stopped_by_kl)
                if stopped_by_kl:
                    break
            if stopped_by_kl:
                break

        if self.expert_noise_enabled and any(
            p._version != version for p, version in zip(self.ppo_actor.noise_head.parameters(), noise_versions)
        ):
            raise RuntimeError("PPO modified the NLL-only noise network during its batch update")

        if len(metrics) == 0:
            return {"explained_var": explained_var}
        mean_metrics = {f"loss/{k}": float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
        for key in (
            "sigma_mean",
            "sigma_min_observed",
            "sigma_max_observed",
            "base_sigma_mean",
            "delta_mean",
            "delta_abs_mean",
            "delta_min",
            "delta_max",
            "clamp_ratio_min",
            "clamp_ratio_max",
        ):
            if f"loss/{key}" in mean_metrics:
                mean_metrics[f"noise/{key}"] = mean_metrics[f"loss/{key}"]
        backbone_ids = {"fixed": -1.0, "mlp": 0.0, "tcn": 1.0, "transformer": 2.0}
        schedule_ids = {"constant": 0.0, "linear": 1.0, "cosine": 2.0}
        mean_metrics["noise/backbone_id"] = backbone_ids.get(
            getattr(self.ppo_actor, "noise_backbone", "fixed")
            if getattr(self.ppo_actor, "noise_head_type", "fixed") == "residual_schedule"
            else "fixed",
            -1.0,
        )
        mean_metrics["noise/base_schedule_id"] = schedule_ids.get(
            getattr(self.ppo_actor, "base_sigma_schedule", "constant"),
            -1.0,
        )
        mean_metrics["loss/explained_var"] = explained_var
        mean_metrics["loss/stopped_by_kl"] = float(stopped_by_kl)
        mean_metrics["loss/actor_update_enabled"] = float(actor_update_enabled)
        mean_metrics["loss/grad_accum_steps"] = float(grad_accum_steps)
        mean_metrics["loss/effective_batch_size"] = float(batch_size * grad_accum_steps)
        mean_metrics["loss/optimizer_steps"] = float(optimizer_steps)
        return mean_metrics

    def save_resume_checkpoint(self, tag="latest_rl", itr=None):
        path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_itr = self.itr if itr is None else int(itr)
        payload = {
            "rafe_resume_version": 2,
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
        if self.expert_noise_enabled:
            payload.update(
                pretrained_reference_signature=self.pretrained_reference_signature,
                noise_guidance=OmegaConf.to_container(self.noise_guidance, resolve=True),
                noise_optimizer=self.noise_optimizer.state_dict(),
                noise_update_steps=self.noise_update_steps,
                noise_warmup_complete=self.noise_warmup_complete,
                noise_lower_bound_streak=self.noise_lower_bound_streak,
                noise_unchanged_streak=self.noise_unchanged_streak,
                noise_generator=self.noise_generator.get_state(),
                expert_data=self.expert_data.state_dict(),
                rng_state=capture_rng_state(),
            )
        torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path)

    def load_resume_checkpoint(self, path):
        payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
        saved_cfg = payload.get("cfg") or {}
        saved_actor_cfg = (saved_cfg.get("rl") or {}).get("actor") or {}
        saved_noise_state = (payload.get("ppo_actor") or {}).get("_extra_state")
        if (payload.get("rafe_resume_version") != 2
                or saved_actor_cfg.get("noise_parameterization") != "log_residual"
                or saved_noise_state != self.ppo_actor.get_extra_state()):
            raise ValueError(
                "Incompatible RAFE resume checkpoint: noise version 2 with explicit log_residual "
                "configuration and matching noise metadata is required. Old bounded-noise weights "
                "must not be reinterpreted; initialize a new run from an original ManiFlow Actor checkpoint."
            )
        saved_policy = _canonical_policy_config(saved_cfg.get("policy", {}))
        _reject_action_reference_weights(payload.get("ppo_actor", {}))
        if self.expert_noise_enabled:
            required = {"noise_optimizer", "noise_guidance", "noise_update_steps", "noise_warmup_complete",
                        "noise_generator", "expert_data", "rng_state", "scaler", "actor_optimizer", "critic_optimizer"}
            if required - payload.keys():
                raise ValueError("Incompatible RAFE resume checkpoint: expert-interpolation optimizer/RNG/data state is required; use original ManiFlow checkpoint only for initialization")
            if payload["noise_guidance"] != OmegaConf.to_container(self.noise_guidance, resolve=True):
                raise ValueError("RAFE noise_guidance configuration differs from resume checkpoint")
            if saved_policy != _canonical_policy_config(self.cfg.policy):
                raise ValueError("Incompatible RAFE resume: policy architecture or Flow inference semantics changed")
            saved_actor = _plain_config(saved_cfg.get("rl", {}).get("actor", {}))
            current_actor = _plain_config(self.cfg.rl.actor)
            saved_actor.pop("actor_old_device", None)
            current_actor.pop("actor_old_device", None)
            if saved_actor != current_actor:
                raise ValueError("Incompatible RAFE resume: rl.actor noise/time-grid/transition configuration changed")
            # actor_old deliberately bypasses module registration and is rebuilt
            # from initialization. Preserve that fixed BC reference on resume.
            if bool(self.cfg.rl.bc_anchor.enabled):
                saved_rl = saved_cfg.get("rl", {})
                saved_path = saved_rl.get("pretrained_checkpoint_path")
                current_path = self.cfg.rl.pretrained_checkpoint_path
                same_path = pathlib.Path(str(saved_path)).expanduser().resolve() == pathlib.Path(str(current_path)).expanduser().resolve()
                if not same_path or saved_rl.get("pretrained_state_key", "ema_model") != self.cfg.rl.get("pretrained_state_key", "ema_model"):
                    raise ValueError("Incompatible RAFE resume: the pretrained checkpoint/state selecting the fixed BC anchor changed")
                if payload.get("pretrained_reference_signature") != self.pretrained_reference_signature:
                    raise ValueError("Incompatible RAFE resume: original BC-reference checkpoint size/mtime changed; keep the pretrained source immutable")
        elif bool(saved_cfg.get("rl", {}).get("noise_guidance", {}).get("enabled", False)):
            raise ValueError("Cannot resume expert-interpolation checkpoint with noise guidance disabled")
        saved_eval = payload.get("cfg", {}).get("rl", {}).get("eval", {}) or {}
        saved_selection = saved_eval.get("scene_selection", "expert")
        current_selection = self._eval_cfg_value("scene_selection", "expert")
        if (saved_selection != current_selection
                and np.isfinite(float(payload.get("best_eval_score", float("-inf"))))):
            raise ValueError(
                f"Cannot compare saved best score on {saved_selection!r} scenes with "
                f"{current_selection!r} scenes. Use standalone evaluation for the new cohort, "
                "or start a new training experiment instead of changing cohorts on resume."
            )
        self.itr = int(payload["itr"])
        self.global_step = int(payload.get("global_step", 0))
        self.ppo_actor.load_state_dict(payload["ppo_actor"], strict=True)
        self.critic.load_state_dict(payload["critic"])
        migrated = load_actor_optimizer_state(self.actor_optimizer, payload["actor_optimizer"])
        if migrated:
            cprint("[RL] Split legacy actor optimizer into actor/noise groups; Adam state preserved.", "yellow")
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.expert_noise_enabled:
            self.noise_optimizer.load_state_dict(payload["noise_optimizer"])
            self.noise_update_steps = int(payload["noise_update_steps"])
            self.noise_warmup_complete = bool(payload["noise_warmup_complete"])
            self.noise_lower_bound_streak = int(payload.get("noise_lower_bound_streak", 0))
            self.noise_unchanged_streak = int(payload.get("noise_unchanged_streak", 0))
            self.noise_generator.set_state(payload["noise_generator"].cpu())
            self.expert_data.load_state_dict(payload["expert_data"])
            restore_rng_state(payload["rng_state"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.consecutive_zero_success_iters = int(payload.get("consecutive_zero_success_iters", 0))
        self.best_eval_score = float(payload.get("best_eval_score", float("-inf")))
        self.best_eval_itr = int(payload.get("best_eval_itr", -1))
        cprint(f"[RL] Resumed RL checkpoint from {path}", "green")
        self._log_actor_learning_rates()

    def export_actor_checkpoint(self, tag="latest", itr=None):
        path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_itr = self.itr if itr is None else int(itr)

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
                "epoch": dill.dumps(save_itr),
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
        next_start_seeds = envs.get_next_seeds()
        envs.close(clear_cache=True)
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
        policy_name = str(self._eval_cfg_value("policy_name", "RAFE"))
        deploy_config_policy_name = str(
            self._eval_cfg_value("deploy_config_policy_name", policy_name.split(".", 1)[0])
        )
        task_config = str(self._eval_cfg_value("task_config", self.cfg.task_config))
        ckpt_setting = str(self._eval_cfg_value("ckpt_setting", task_config))
        eval_seed = str(self._eval_cfg_value("seed", 0))
        candidate_tag = str(self._eval_cfg_value("candidate_ckpt_tag", "eval_candidate"))
        best_tag = str(self._eval_cfg_value("best_ckpt_tag", "best"))
        addition_info = self._derive_addition_info(config_name)
        scene_selection = check_selection_mode(self._eval_cfg_value("scene_selection", "expert"))

        candidate_path = pathlib.Path(
            self.export_actor_checkpoint(candidate_tag, itr=eval_iteration)
        )

        log_dir = pathlib.Path(self.output_dir) / "eval_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        cohort_suffix = "" if scene_selection == "expert" else f"_{scene_selection}"
        eval_log_path = log_dir / f"{candidate_tag}{cohort_suffix}_itr_{eval_iteration:04d}.log"
        result_dir = evaluation_result_dir(
            pathlib.Path(self.output_dir) / "eval_results" / f"epoch_{eval_iteration}", scene_selection,
        )
        result_path = result_dir / "_result.txt"
        if result_path.exists():
            result_path.unlink()

        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        env["RAFE_RUN_DIR"] = str(pathlib.Path(self.output_dir).resolve())

        def run_eval_subprocess(cmd, log_path: pathlib.Path, label: str):
            cprint(
                f"[RL Eval] itr={self.itr} running {label}; log={log_path}",
                "yellow",
            )
            with log_path.open("w", encoding="utf-8") as log_file:
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
                    f"Periodic eval failed with return code {proc.returncode}. See {log_path}"
                )

        actor_cmd = [
            sys.executable,
            "script/eval_policy.py",
            "--config",
            f"policy/{deploy_config_policy_name}/deploy_policy.yml",
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
            "--eval_scene_selection",
            scene_selection,
            "--eval_num_episodes",
            str(self._eval_cfg_value("num_episodes", 100)),
            "--eval_max_seed_tries",
            str(self._eval_cfg_value("max_seed_tries", 10000)),
            "--eval_scene_cache_mode",
            str(self._eval_cfg_value("scene_cache_mode", "readwrite")),
            "--eval_scene_seed_file",
            str(self._eval_cfg_value("scene_seed_file", DEFAULT_SHARED_SEED_FILE)),
            "--eval_baseline_checkpoint",
            str(self._eval_cfg_value("baseline_checkpoint", DEFAULT_BASELINE_CHECKPOINT)),
        ]
        for cfg_key, cli_key in (("scene_validation_config", "eval_scene_validation_config"),
                                 ("scene_manifest", "eval_scene_manifest")):
            value = self._eval_cfg_value(cfg_key)
            if value not in (None, "", "null", "None"):
                actor_cmd.extend([f"--{cli_key}", str(value)])

        try:
            run_eval_subprocess(
                actor_cmd,
                eval_log_path,
                f"actor eval without noise_head, ckpt_tag={candidate_tag}, scenes={scene_selection}",
            )
            if not result_path.is_file():
                raise FileNotFoundError(f"Periodic actor eval did not write expected result file: {result_path}")

            score = self._parse_eval_score(result_path)
            is_best = score > self.best_eval_score
            if is_best:
                self.best_eval_score = score
                self.best_eval_itr = int(eval_iteration)
                self.export_actor_checkpoint(best_tag, itr=eval_iteration)
                cprint(
                    f"[RL Eval] new actor best {score:.4f} at iteration {eval_iteration}; "
                    f"saved {best_tag}.ckpt",
                    "green",
                )
            else:
                cprint(
                    f"[RL Eval] actor score={score:.4f}, best={self.best_eval_score:.4f} "
                    f"at iteration {self.best_eval_itr}",
                    "yellow",
                )
        finally:
            best_path = pathlib.Path(self.output_dir) / "checkpoints" / f"{best_tag}.ckpt"
            if candidate_path.is_file() and candidate_path.resolve() != best_path.resolve():
                candidate_path.unlink()
                cprint(f"[RL Eval] removed evaluated actor candidate {candidate_path}", "yellow")

        result = {
            "eval/actor_success_rate": score,
            "eval/exported_actor_success_rate": score,
            "eval/success_rate": score,
            "eval/is_best": float(is_best),
            "eval/best_success_rate": float(self.best_eval_score),
            "eval/best_iteration": float(self.best_eval_itr),
        }
        return {
            k: v for k, v in result.items()
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
        if self.expert_noise_enabled and not self.noise_warmup_complete:
            warmup_steps = int(self.noise_guidance.get("initial_warmup_steps", 0))
            if warmup_steps:
                warmup_log = self.noise_nll_update(warmup_steps)
                with (pathlib.Path(self.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"phase": "noise_warmup", "global_step": self.global_step, **warmup_log}) + "\n")
            self.noise_warmup_complete = True
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

        envs = None
        try:
            envs = self._make_envs()
            buffer = self._make_buffer()
            while self.itr < int(cfg.rl.ppo.num_iterations):
                t0 = time.time()
                num_iterations = int(cfg.rl.ppo.num_iterations)
                buffer.reset()
                with RLIterationProgress(
                    self.itr, buffer.n_steps, self.global_step, cfg.training.tqdm_interval_sec,
                ) as progress:
                    rollout_log = self.collect_rollout(envs, buffer, progress=progress)
                    rollout_sec = time.time() - t0
                    update_actor, actor_skip_reason, actor_skip_code, actor_loss_scale = self.should_update_actor(rollout_log)
                    step_log = {
                        "itr": self.itr,
                        "epoch": self.itr,
                        "global_step": self.global_step,
                        "lr/actor": self.actor_optimizer.param_groups[0]["lr"],
                        "lr/noise": (self.noise_optimizer.param_groups[0]["lr"] if self.expert_noise_enabled else self.actor_optimizer.param_groups[1]["lr"]),
                        "lr/critic": self.critic_optimizer.param_groups[0]["lr"],
                        "rl/actor_update_enabled": float(update_actor),
                        "rl/actor_loss_scale": float(actor_loss_scale),
                        "rl/actor_update_skip_code": float(actor_skip_code),
                        "rl/consecutive_zero_success_iters": float(self.consecutive_zero_success_iters),
                    }
                    step_log.update({f"rollout/{k}": v for k, v in rollout_log.items()})
                    update_started = time.perf_counter()
                    update_log = self.ppo_update(
                        buffer,
                        update_actor=update_actor,
                        actor_loss_scale=actor_loss_scale,
                        progress=progress,
                    )
                step_log.update(update_log)
                step_log["rollout/online_success_rate"] = rollout_log.get("success_rate", 0.0)
                step_log["time/rollout_sec"] = rollout_sec
                step_log["time/ppo_update_sec"] = time.perf_counter() - update_started
                if self.expert_noise_enabled and (self.itr + 1) % int(self.noise_guidance.update_interval) == 0:
                    step_log.update(self.noise_nll_update())
                step_log["time/itr_sec"] = time.time() - t0

                cprint(
                    f"[RL] itr={self.itr} step={self.global_step} "
                    f"succ={step_log.get('rollout/success_rate', float('nan')):.3f} "
                    f"raw_reward={step_log.get('rollout/rollout_raw_reward_sum', 0):.3f} "
                    f"actor_update={int(update_actor)}({actor_skip_reason}, scale={actor_loss_scale:.2f}) "
                    f"sigma={step_log.get('rollout/sampling_sigma_mean', 0):.6f} "
                    f"|delta|={step_log.get('rollout/sampling_delta_abs_mean', 0):.6f} "
                    f"loss={step_log.get('loss/loss', 0):.4f}",
                    "cyan",
                )

                if (
                    self.itr % int(cfg.rl.checkpoint_every) == 0
                    or self.itr == num_iterations - 1
                ):
                    self.save_resume_checkpoint("latest_rl", itr=self.itr + 1)
                    if bool(cfg.rl.get("export_actor_checkpoint", True)):
                        self.export_actor_checkpoint("latest", itr=self.itr + 1)

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
                    envs = None
                    eval_log = self.run_eval_and_update_best(eval_iteration)
                    step_log.update(eval_log)
                    self.save_resume_checkpoint("latest_rl", itr=self.itr + 1)
                    if self.itr < num_iterations - 1:
                        envs = self._make_envs(start_seeds=next_start_seeds)

                if use_wandb:
                    wandb_run.log(step_log, step=self.itr)
                with (pathlib.Path(self.output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(step_log) + "\n")

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
            raise
        finally:
            if envs is not None:
                envs.close(clear_cache=True)
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
