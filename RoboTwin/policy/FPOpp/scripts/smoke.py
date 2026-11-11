#!/usr/bin/env python3
"""Small real-checkpoint FPO++ update; --env also runs real RoboTwin collection."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpopp.runtime import POLICY_ROOT, configure_paths

configure_paths()

from omegaconf import OmegaConf
import torch

from fpopp.actor import FPOActor, execution_mask
from fpopp.checkpoint import load_payload, load_policy
from fpopp.loss import surrogate_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env", action="store_true", help="Create real RoboTwin tasks, collect two chunks, update and checkpoint")
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", default="demo_messy_ur5")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="fpopp-smoke-"))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "report.json").exists() or (directory / "checkpoints/latest.ckpt").exists():
        parser.error("Use a fresh output directory to preserve earlier smoke artifacts")
    if args.env:
        from fpopp.workspace import Workspace
        cfg = OmegaConf.load(POLICY_ROOT / "configs/fpopp_robotwin2.yaml")
        cfg.checkpoint, cfg.device, cfg.output_dir = args.checkpoint, args.device, str(directory)
        cfg.task_name, cfg.task_config = args.task_name, args.task_config
        cfg.rollout.n_envs, cfg.rollout.n_steps, cfg.rollout.start_seed = 1, 2, 1700000
        cfg.fpo.mc_samples, cfg.fpo.mc_batch_size = 2, 1
        cfg.ppo.batch_size, cfg.ppo.grad_accum_steps, cfg.ppo.update_epochs = 1, 2, 1
        cfg.ppo.critic_warmup_iterations, cfg.max_iterations = 0, 1
        work = Workspace(cfg)
        work.run()
        report = {"kind": "real_robotwin_collect_update_checkpoint", "action_steps": work.action_steps,
                  "decision_steps": work.decision_steps, "training_seeds": sorted(work.training_seeds)}
    else:
        policy, config, key = load_policy(load_payload(args.checkpoint), device=args.device)
        actor = FPOActor(policy, seed=123)
        observations = {}
        for field in ("agent_pos", "point_cloud"):
            shape = (2, int(policy.n_obs_steps), *config.shape_meta.obs[field].shape)
            stats = policy.normalizer[field].get_input_stats()
            mean, std = stats["mean"].to(args.device), stats["std"].to(args.device)
            observations[field] = mean + std.clamp_min(.01) * torch.randn(shape, device=args.device)*.1
        features, plan = actor.encode(observations)
        action, target, _ = actor.sample(features, plan)
        time, noise = actor.draw_cfm(target, 2)
        with torch.no_grad():
            old = torch.cat([actor.cfm_losses(features, plan, target, time[:, i:i+1], noise[:, i:i+1])
                             for i in range(2)], dim=1)
        optimizer = torch.optim.AdamW(actor.trainable_parameters(), lr=1e-6)
        for index in range(2):
            current = actor.cfm_losses(features, plan, target, time[:, index:index+1], noise[:, index:index+1])
            loss, stats = surrogate_loss(old[:, index:index+1], current,
                execution_mask(actor, [action.shape[1]]*2), torch.tensor([1., -1.], device=args.device))
            (loss/2).backward()
        norm = torch.nn.utils.clip_grad_norm_(list(actor.trainable_parameters()), 1., error_if_nonfinite=True)
        if not torch.isfinite(action).all() or not torch.isfinite(norm) or float(norm) == 0:
            raise RuntimeError("Real checkpoint smoke did not produce finite actions and nonzero gradients")
        optimizer.step()
        report = {"kind": "real_checkpoint_actor_update_synthetic_observations", "checkpoint": args.checkpoint,
                  "state_key": key, "device": args.device, "action_shape": list(action.shape),
                  "action_window": actor.window, "plan_preserved": policy.action_plan_enabled,
                  "inference_steps": actor.inference_steps, "gradient_norm": float(norm),
                  "metrics": {key: float(value) for key, value in stats.items()}}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({**report, "output_dir": str(directory)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
