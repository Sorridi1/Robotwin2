"""On-policy chunk collection, FPO++ updates, checkpointing and validation."""

from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
from omegaconf import OmegaConf
import torch

from .actor import FPOActor, execution_mask
from .buffer import FPOBuffer, cpu_copy
from .checkpoint import FORMAT, atomic_save, load_payload, load_policy, restore_rng, rng_state
from .loss import surrogate_loss
from .runtime import POLICY_ROOT, ROBOTWIN_ROOT
from .results import result_directory


def stack_observations(observations, device):
    return {key: torch.as_tensor(np.stack([obs[key] for obs in observations]),
                                  device=device, dtype=torch.float32) for key in observations[0]}


def source_fingerprint(task_name=None):
    paths = list((POLICY_ROOT / "fpopp").glob("*.py")) + [
        POLICY_ROOT / "train.py", POLICY_ROOT / "deploy_policy.py",
        POLICY_ROOT / "configs/fpopp_robotwin2.yaml",
        ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py",
        ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow/maniflow/env/robotwin/robotwin_rl_env.py",
        ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow/maniflow/env/robotwin/scene_validator.py",
        ROBOTWIN_ROOT / "policy/ManiFlow/ManiFlow/maniflow/common/robotwin_env_pool.py",
        ROBOTWIN_ROOT / "envs/_base_task.py",
    ]
    if task_name:
        path = ROBOTWIN_ROOT / "envs" / f"{task_name}.py"
        if path.is_file():
            paths.append(path)
    return {str(path.relative_to(ROBOTWIN_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class Workspace:
    def __init__(self, cfg, policy=None, critic=None):
        """Policy/critic injection permits a full CPU integration check with tiny models."""
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process. Use the GPU runtime, or --device cpu for checks.")
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        self.iteration, self.action_steps, self.decision_steps = 0, 0, 0
        self.best_score = -math.inf
        self.next_seeds = None
        self.training_seeds = set()
        self.policy_cfg = None
        self.initial_checkpoint = None
        resume = load_payload(cfg.resume) if cfg.resume else None
        if resume is not None and resume.get("format") != FORMAT:
            raise ValueError("--resume requires a full FPO++ training checkpoint")
        if policy is None:
            payload = resume if resume is not None else load_payload(cfg.checkpoint)
            policy, self.policy_cfg, selected_key = load_policy(payload, "model" if resume else cfg.state_key, cfg.device)
            self.initial_checkpoint = (resume["initial_checkpoint"] if resume is not None else
                                       {"path": str(Path(cfg.checkpoint).expanduser().resolve()), "state_key": selected_key})
            if resume is None:
                stat = Path(cfg.checkpoint).expanduser().stat()
                self.initial_checkpoint.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            cfg.policy = self.policy_cfg
        policy = policy.to(self.device)
        cfg.n_obs_steps, cfg.horizon = int(policy.n_obs_steps), int(policy.horizon)
        cfg.n_action_steps = int(policy.n_action_steps)
        cfg.shape_meta = OmegaConf.create(policy.shape_meta) if hasattr(policy, "shape_meta") else cfg.policy.shape_meta
        self.actor = FPOActor(policy, seed=int(cfg.seed)+7183, **OmegaConf.to_container(cfg.actor, resolve=True))
        if critic is None:
            from maniflow.model.rl.pointcloud_critic import PointcloudCritic
            pcfg = self.policy_cfg
            critic = PointcloudCritic(
                shape_meta=pcfg.shape_meta, n_obs_steps=policy.n_obs_steps,
                encoder_output_dim=pcfg.encoder_output_dim, use_pc_color=pcfg.use_pc_color,
                pointnet_type=pcfg.pointnet_type, downsample_points=pcfg.downsample_points,
                pointcloud_encoder_cfg=copy.deepcopy(pcfg.pointcloud_encoder_cfg),
            )
            critic.set_normalizer(policy.normalizer)
        self.critic = critic.to(self.device).eval()
        self.actor_optimizer = torch.optim.AdamW(self.actor.trainable_parameters(), lr=float(cfg.ppo.actor_lr),
                                                betas=(0.9, 0.95), weight_decay=0,
                                                eps=float(cfg.ppo.get("actor_eps", 1e-8)))
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=float(cfg.ppo.critic_lr),
                                                 betas=(0.9, 0.95), weight_decay=1e-5)
        self.amp = bool(cfg.ppo.use_amp and self.device.type == "cuda")
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[str(cfg.ppo.amp_dtype)]
        if self.amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU lacks BF16 support; set ppo.use_amp=false or ppo.amp_dtype=float16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        from maniflow.common.ppo_buffer_pointcloud import RunningRewardScaler
        self.reward_scaler = (RunningRewardScaler(int(cfg.rollout.n_envs), float(cfg.ppo.gamma))
                              if cfg.ppo.reward_scale_running else None)
        cfg.env.update({"task_name": cfg.task_name, "task_config": cfg.task_config,
                        "n_obs_steps": cfg.n_obs_steps, "n_action_steps": cfg.n_action_steps,
                        "ckpt_setting": cfg.task_config})
        # The existing public evaluator reads rl.env.scene_validation from this archive.
        cfg.rl = {"env": OmegaConf.to_container(cfg.env, resolve=True)}
        self.output_dir = Path(cfg.output_dir).expanduser().resolve()
        if resume is not None:
            self._restore(resume)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype) if self.amp else nullcontext()

    def _make_envs(self):
        from maniflow.common.robotwin_env_pool import SerialEnvPool, SubprocessEnvPool
        count = int(self.cfg.rollout.n_envs)
        start = self.cfg.rollout.start_seed
        start = int(start) if start is not None else 100000 * (int(self.cfg.seed)+1)
        configs = []
        for index in range(count):
            config = OmegaConf.to_container(self.cfg.env, resolve=True)
            config.update(seed=int(self.cfg.seed)+index, start_seed=(self.next_seeds[index] if self.next_seeds
                          else start + index * int(self.cfg.rollout.env_seed_stride)))
            configs.append(config)
        if self.cfg.rollout.backend == "serial":
            return SerialEnvPool(configs)
        shapes = None
        if self.cfg.rollout.shared_memory:
            shapes = {key: (self.cfg.n_obs_steps, *meta.shape) for key, meta in self.cfg.shape_meta.obs.items()}
        return SubprocessEnvPool(configs, observation_shapes=shapes,
                                 worker_threads=int(self.cfg.rollout.worker_threads),
                                 timeout_sec=float(self.cfg.rollout.worker_timeout_sec))

    def _remember_resets(self, infos):
        for info in infos:
            seed = info.get("validation", {}).get("accepted_seed")
            if seed is not None:
                self.training_seeds.add(int(seed))

    def collect(self, envs):
        buffer = FPOBuffer()
        observations = envs.reset()
        self._remember_resets(envs.last_reset_infos)
        self.actor.eval()
        self.critic.eval()
        raw_reward, successes, episodes, clips = 0.0, 0, 0, []
        for _ in range(int(self.cfg.rollout.n_steps)):
            obs = stack_observations(observations, self.device)
            # Own the observations before workers overwrite their shared-memory slots.
            stored_obs = cpu_copy(obs)
            with torch.no_grad():
                features, plan = self.actor.encode(obs)
                actions, targets, clipping = self.actor.sample(features, plan)
                time_sample, noise = self.actor.draw_cfm(targets, int(self.cfg.fpo.mc_samples))
                old_losses = []
                for start in range(0, noise.shape[1], int(self.cfg.fpo.mc_batch_size)):
                    end = start + int(self.cfg.fpo.mc_batch_size)
                    with self._autocast():
                        old_losses.append(self.actor.cfm_losses(features, plan, targets, time_sample[:, start:end], noise[:, start:end]))
                old_loss = torch.cat(old_losses, dim=1)
                values = self.critic(obs)
            if not torch.isfinite(actions).all() or not torch.isfinite(old_loss).all():
                raise FloatingPointError("Nonfinite sampled action or old CFM loss")
            results = envs.step(actions.detach().cpu().numpy())
            infos = [result[4] for result in results]
            executed = [int(info["executed_action_steps"]) for info in infos]
            if any(n == 0 and not (result[2] or result[3]) for n, result in zip(executed, results)):
                raise RuntimeError("Environment executed zero actions without terminating")
            buffer.add(obs=stored_obs, features=features, plan=plan, actions=targets,
                       time=time_sample, noise=noise, old_loss=old_loss,
                       mask=execution_mask(self.actor, executed), value=values,
                       reward=torch.tensor([r[1] for r in results], dtype=torch.float32),
                       terminated=torch.tensor([r[2] for r in results]),
                       truncated=torch.tensor([r[3] for r in results]))
            self.action_steps += sum(executed)
            self.decision_steps += len(results)
            self._remember_resets(infos)
            raw_reward += sum(float(r[1]) for r in results)
            episodes += sum(bool(r[2] or r[3]) for r in results)
            successes += sum(bool(r[4].get("success", False)) and bool(r[2] or r[3]) for r in results)
            clips.append(clipping)
            observations = [r[0] for r in results]
            if len(buffer.records) == 1 or len(buffer.records) % 8 == 0:
                print(f"[FPO++] rollout {len(buffer.records)}/{self.cfg.rollout.n_steps}, "
                      f"actual actions={self.action_steps}, completed episodes={episodes}", flush=True)
            if self.action_steps >= int(self.cfg.total_action_steps):
                break
        with torch.no_grad():
            last_values = self.critic(stack_observations(observations, self.device))
        self.next_seeds = envs.get_next_seeds()
        data = buffer.finish(last_values, gamma=float(self.cfg.ppo.gamma), gae_lambda=float(self.cfg.ppo.gae_lambda),
                             reward_scaler=self.reward_scaler, reward_scale=float(self.cfg.ppo.reward_scale_const))
        return data, {"rollout/reward_sum": raw_reward, "rollout/episodes": episodes,
                      "rollout/success_rate": successes/max(episodes, 1), "rollout/action_clip_fraction": float(np.mean(clips))}

    def update(self, data):
        count = len(data["advantage"])
        batch = int(self.cfg.ppo.batch_size)
        accumulation = int(self.cfg.ppo.grad_accum_steps)
        samples = int(self.cfg.fpo.mc_samples)
        actor_enabled = self.iteration >= int(self.cfg.ppo.critic_warmup_iterations)
        metrics, metric_weight, optimizer_steps, overflow_steps = {}, 0.0, 0, 0
        self.actor.eval()
        self.critic.eval()
        for _ in range(int(self.cfg.ppo.update_epochs)):
            order = torch.randperm(count)
            batches = list(order.split(batch))
            for group_start in range(0, len(batches), accumulation):
                group = batches[group_start:group_start+accumulation]
                group_count = sum(len(indices) for indices in group)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                for indices in group:
                    mb = {key: ({field: value[indices].to(self.device) for field, value in item.items()}
                                if isinstance(item, dict) else item[indices].to(self.device)) for key, item in data.items()}
                    batch_weight = len(indices)/group_count
                    with self._autocast():
                        prediction = self.critic(mb["obs"])
                        critic_loss = 0.5 * (prediction.float()-mb["returns"]).square().mean()
                    self.scaler.scale(critic_loss * float(self.cfg.ppo.vf_coef) * batch_weight).backward()
                    if not torch.isfinite(critic_loss):
                        raise FloatingPointError("Nonfinite critic loss")
                    metrics["critic_loss"] = metrics.get("critic_loss", 0.0) + float(critic_loss.detach()) * len(indices)
                    metric_weight += len(indices)
                    if actor_enabled:
                        for start in range(0, samples, int(self.cfg.fpo.mc_batch_size)):
                            end = min(start+int(self.cfg.fpo.mc_batch_size), samples)
                            with self._autocast():
                                current = self.actor.cfm_losses(mb["features"], mb.get("plan"), mb["actions"],
                                                                mb["time"][:, start:end], mb["noise"][:, start:end])
                                loss, stats = surrogate_loss(mb["old_loss"][:, start:end], current, mb["mask"],
                                    mb["advantage"], clip_coef=float(self.cfg.fpo.clip_coef),
                                    old_loss_cap=self.cfg.fpo.old_loss_cap, logratio_cap=self.cfg.fpo.logratio_cap,
                                    average_chunk=bool(self.cfg.fpo.average_chunk))
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Nonfinite FPO++ surrogate")
                            # Backprop each MC slice separately to bound retained activations.
                            weight = (end-start)/samples
                            self.scaler.scale(loss * weight * batch_weight).backward()
                            for key, value in stats.items():
                                metrics[key] = metrics.get(key, 0.0) + float(value) * weight * len(indices)
                self.scaler.unscale_(self.critic_optimizer)
                strict_finite = not self.scaler.is_enabled()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.ppo.max_grad_norm),
                                                             error_if_nonfinite=strict_finite)
                actor_norm = torch.tensor(0.)
                if actor_enabled:
                    self.scaler.unscale_(self.actor_optimizer)
                    actor_norm = torch.nn.utils.clip_grad_norm_(list(self.actor.trainable_parameters()),
                        float(self.cfg.ppo.max_grad_norm), error_if_nonfinite=strict_finite)
                    self.scaler.step(self.actor_optimizer)
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
                overflow = not (torch.isfinite(actor_norm) and torch.isfinite(critic_norm))
                overflow_steps += int(overflow)
                optimizer_steps += int(not overflow)
        return {**{f"loss/{key}": value/max(metric_weight, 1) for key, value in metrics.items()},
                "update/actor_enabled": actor_enabled, "update/optimizer_steps": optimizer_steps,
                "update/overflow_groups": overflow_steps, "update/grad_scale": self.scaler.get_scale()}

    def save(self, name="latest", full=True):
        trainer = None
        if full:
            trainer = {"critic": cpu_copy(self.critic.state_dict()),
                       "actor_optimizer": self.actor_optimizer.state_dict(),
                       "critic_optimizer": self.critic_optimizer.state_dict(), "amp": self.scaler.state_dict(),
                       "rng": rng_state(self.actor), "next_seeds": self.next_seeds,
                       "training_seeds": sorted(self.training_seeds),
                       "reward_scaler": copy.deepcopy(self.reward_scaler.__dict__) if self.reward_scaler else None,
                       "best_score": self.best_score}
        payload = {"format": FORMAT, "cfg": OmegaConf.to_container(self.cfg, resolve=True),
                   "policy_cfg": OmegaConf.to_container(self.policy_cfg, resolve=True) if self.policy_cfg is not None else None,
                   "state_dicts": {"model": cpu_copy(self.actor.policy.state_dict())}, "trainer": trainer,
                   "iteration": self.iteration, "action_steps": self.action_steps, "decision_steps": self.decision_steps,
                   "initial_checkpoint": self.initial_checkpoint, "training_seeds": sorted(self.training_seeds),
                   "sources": source_fingerprint(self.cfg.task_name)}
        path = self.output_dir / "checkpoints" / f"{name}.ckpt"
        atomic_save(payload, path)
        return path

    def _restore(self, payload):
        state = payload.get("trainer")
        if state is None:
            raise ValueError("Actor export cannot resume training; use latest.ckpt")
        for key in ("actor", "fpo", "ppo", "rollout", "env", "validation", "task_name", "task_config", "seed"):
            current = OmegaConf.to_container(self.cfg, resolve=True)[key]
            if current != payload["cfg"][key]:
                raise ValueError(f"Cannot change {key} on resume; start a new fine-tuning run instead")
        self.critic.load_state_dict(state["critic"], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.scaler.load_state_dict(state["amp"])
        if (state["reward_scaler"] is None) != (self.reward_scaler is None):
            raise ValueError("Cannot change running reward scaling on resume")
        if self.reward_scaler:
            self.reward_scaler.__dict__.update(state["reward_scaler"])
        self.iteration, self.action_steps = int(payload["iteration"]), int(payload["action_steps"])
        self.decision_steps = int(payload["decision_steps"])
        self.next_seeds = state["next_seeds"]
        self.training_seeds = set(state["training_seeds"])
        self.best_score = state["best_score"]
        restore_rng(state["rng"], self.actor)

    def validate(self):
        """Run the common evaluator in a subprocess, temporarily offloading training state."""
        validation = self.cfg.validation
        if not validation.seed_file:
            raise ValueError("Periodic best selection requires validation.seed_file (a separate validation bank)")
        actor_path = self.save("validation_candidate", full=False)
        evaluation_root = self.output_dir
        destination = result_directory(evaluation_root, self.iteration, str(validation.scene_selection))
        if ((evaluation_root / f"validation_{self.iteration}.log").exists()
                or (destination.is_dir() and any(item.is_file() for item in destination.iterdir()))):
            evaluation_root = self.output_dir / "evaluation_retries" / f"epoch_{self.iteration}_{uuid.uuid4().hex[:12]}"
        modules = [self.actor, self.critic]
        optimizers = [self.actor_optimizer, self.critic_optimizer]

        def move_optimizers(device):
            for optimizer in optimizers:
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value) and key != "step":
                            state[key] = value.to(device)
        for module in modules:
            module.to("cpu")
        move_optimizers("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            command = ["bash", str(POLICY_ROOT / "eval.sh"), str(actor_path), self.cfg.task_name,
                       self.cfg.task_config, str(validation.seed), "visible",
                       "--eval_scene_seed_file", str(Path(validation.seed_file).expanduser().resolve()),
                       "--eval_num_episodes", str(validation.episodes),
                       "--eval_max_seed_tries", str(validation.max_seed_tries),
                       "--eval_scene_selection", str(validation.scene_selection), "--device", str(self.device),
                       "--eval_output_dir", str(evaluation_root)]
            import os
            environment = dict(os.environ, PYTHON_BIN=sys.executable)
            evaluation_root.mkdir(parents=True, exist_ok=True)
            with (evaluation_root / f"validation_{self.iteration}.log").open("x") as handle:
                subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
            report_path = result_directory(evaluation_root, self.iteration, str(validation.scene_selection)) / "_scene_selection.json"
            report = json.loads(report_path.read_text())
            if report["status"] != "complete" or report["completed_episodes"] != int(validation.episodes):
                raise RuntimeError("Validation did not complete all requested scenes")
            overlap = self.training_seeds.intersection(int(r["seed"]) for r in report["episodes"])
            if overlap:
                raise ValueError(f"Validation overlaps training seeds: {sorted(overlap)[:10]}")
            score = report["successes"]/report["completed_episodes"]
            if score > self.best_score:
                self.best_score = score
                shutil.copyfile(actor_path, self.output_dir / "checkpoints/best.ckpt")
            return score
        finally:
            for module in modules:
                module.to(self.device)
            move_optimizers(self.device)
            actor_path.unlink(missing_ok=True)

    def run(self, env_factory=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / ".hydra").mkdir(exist_ok=True)
        OmegaConf.save(self.cfg, self.output_dir / ".hydra/config.yaml")
        envs = None
        last_saved_iteration = None
        try:
            while self.iteration < int(self.cfg.max_iterations) and self.action_steps < int(self.cfg.total_action_steps):
                started = time.perf_counter()
                if envs is None:
                    envs = env_factory() if env_factory else self._make_envs()
                data, metrics = self.collect(envs)
                metrics.update(self.update(data))
                self.iteration += 1
                interval = int(self.cfg.validation.interval)
                finished = self.iteration >= int(self.cfg.max_iterations) or self.action_steps >= int(self.cfg.total_action_steps)
                should_evaluate = bool(interval and (self.iteration % interval == 0 or (finished and self.cfg.validation.run_final)))
                # Keep the completed update even if the subsequent evaluator fails.
                if self.iteration % int(self.cfg.checkpoint_every) == 0 or finished or should_evaluate:
                    self.save()
                    last_saved_iteration = self.iteration
                if should_evaluate:
                    envs.close()
                    envs = None
                    metrics["validation/success_rate"] = self.validate()
                    self.save()  # Persist the best score after successful validation.
                    last_saved_iteration = self.iteration
                metrics.update(iteration=self.iteration, executed_action_steps=self.action_steps,
                               decision_steps=self.decision_steps, iteration_seconds=time.perf_counter()-started)
                line = json.dumps(metrics, allow_nan=False)
                print(line, flush=True)
                with (self.output_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(line + "\n")
                (self.output_dir / "training_scene_seeds.json").write_text(json.dumps(sorted(self.training_seeds)))
            if last_saved_iteration != self.iteration:
                self.save()
        finally:
            if envs is not None:
                envs.close()
