#!/usr/bin/env python3
"""Measure real reset modes and optionally exercise actor -> simulation -> PPO buffer."""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT / "ManiFlow"))

import hydra
import numpy as np
from omegaconf import OmegaConf


def benchmark(cfg, mode, count, start_seed):
    env_config = OmegaConf.to_container(cfg.rl.env, resolve=True)
    env_config["scene_validation"]["mode"] = mode
    env_config["expert_seed_cache_mode"] = "off"  # Measure validation, not cache hits.
    env = hydra.utils.instantiate(env_config, seed=0, start_seed=start_seed)
    records = []
    try:
        for offset in range(count):
            started = time.perf_counter()
            obs = env.reset(seed=start_seed + offset)
            wall = time.perf_counter() - started
            assert np.isfinite(obs["point_cloud"]).all()
            record = {"reset_wall_sec": wall, **copy.deepcopy(env.last_validation_info)}
            records.append(record)
            print(json.dumps({"mode": mode, "reset_index": offset, **record}), flush=True)
    finally:
        env.close(clear_cache=True)
    report = {
        "mode": mode, "reset_count": count,
        "total_reset_wall_sec": sum(item["reset_wall_sec"] for item in records),
        "mean_reset_wall_sec": float(np.mean([item["reset_wall_sec"] for item in records])),
        "setup_demo_count": sum(item["setup_demo_count"] for item in records),
        "play_once_count": sum(item["play_once_count"] for item in records),
        "records": records,
    }
    del env
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return report


def smoke(cfg, checkpoint):
    import torch
    from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace

    cfg = copy.deepcopy(cfg)
    cfg.rl.pretrained_checkpoint_path = str(checkpoint)
    cfg.rl.env.scene_validation.mode = "cheap"
    cfg.rl.env.expert_seed_cache_mode = "off"
    cfg.rl.rollout.backend = "subprocess"
    cfg.rl.rollout.n_envs = 2
    cfg.rl.rollout.n_steps = 2
    cfg.training.device = "cuda:0"
    cfg.rl.actor.actor_old_device = "cuda:0"
    workspace = TrainReinFlowRLRoboTwinWorkspace(cfg, output_dir="/tmp/maniflow-scene-validation-smoke")
    envs = workspace._make_envs()
    try:
        buffer = workspace._make_buffer()
        log = workspace.collect_rollout(envs, buffer)
        buffer.update(workspace._obs_to_buffer(workspace.current_obs), workspace.critic)
        obs, chains, returns, values, advantages, old_logprob = buffer.make_dataset()
        with torch.no_grad():
            recomputed, _ = workspace.ppo_actor.get_logprobs(obs, chains)
        torch.testing.assert_close(recomputed.cpu(), old_logprob.cpu(), atol=2e-4, rtol=2e-4)
        assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()
        return {"transitions": int(returns.numel()), "global_step": workspace.global_step,
                "logprob_recompute_consistent": True, "rollout_log": log}
    finally:
        envs.close(clear_cache=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-seed-tries", type=int, default=5, help="Bound this short check if every candidate is rejected")
    parser.add_argument("--modes", nargs="+", choices=("none", "cheap", "expert"), default=["expert", "cheap"])
    parser.add_argument("--smoke-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resets < 1 or args.max_seed_tries < 1:
        parser.error("resets and max-seed-tries must be positive")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(POLICY_ROOT / "ManiFlow/maniflow/config"), version_base="1.2"):
        cfg = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2", overrides=[
            "task_name=place_empty_cup", "setting=demo_messy_ur5", "training.seed=0",
            "logging.mode=disabled", "rl.eval.enabled=false",
            f"rl.env.scene_validation.max_seed_tries={args.max_seed_tries}",
        ])
    report = {"benchmarks": []}
    for mode in args.modes:
        report["benchmarks"].append(benchmark(cfg, mode, args.resets, args.start_seed))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.smoke_checkpoint:
        report["smoke"] = smoke(cfg, args.smoke_checkpoint)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SCENE_VALIDATION_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
