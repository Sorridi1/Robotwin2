#!/usr/bin/env python3
"""Prevalidate scene seeds with cheap checks or a legacy expert; no actor is loaded."""

import argparse
from pathlib import Path
import sys

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT / "ManiFlow"))

import hydra
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", default="demo_messy_ur5")
    parser.add_argument("--config-name", default="reinflow_rl_pointcloud_robotwin2")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=100, help="Number of candidate seeds, including rejected ones")
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--mode", choices=("cheap", "expert"), default="cheap")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides; use the same environment options as training")
    args = parser.parse_args()
    if args.count < 1 or args.env_index < 0:
        parser.error("count must be positive and env-index must be nonnegative")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    overrides = [
        f"task_name={args.task_name}", f"task_config={args.task_config}",
        f"setting={args.task_config}", f"training.seed={args.training_seed}",
        *args.overrides,
    ]
    if args.cache_dir:
        overrides.append(f"rl.env.expert_seed_cache_dir={args.cache_dir}")
    overrides.extend([f"rl.env.scene_validation.mode={args.mode}", "rl.env.expert_seed_cache_mode=readwrite"])
    with hydra.initialize_config_dir(config_dir=str(POLICY_ROOT / "ManiFlow/maniflow/config"), version_base="1.2"):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    start = args.start_seed
    if start is None:
        start = 100000 * (1 + args.training_seed) + args.env_index * int(cfg.rl.rollout.env_seed_stride)
    env = hydra.utils.instantiate(cfg.rl.env, seed=args.training_seed + args.env_index, start_seed=start)
    valid_count = 0
    try:
        print(f"Validation mode={args.mode}; seed cache: {env.expert_seed_cache.directory}", flush=True)
        for seed in range(start, start + args.count):
            valid, _ = env.validate_seed(seed)
            valid_count += int(valid)
            print(f"seed={seed} valid={valid} ({seed - start + 1}/{args.count})", flush=True)
        print(f"Completed: valid={valid_count}, rejected={args.count - valid_count}; {env.last_reset_info}")
    finally:
        # Validation already closes each scene, but force a final renderer/cache cleanup.
        env._close_active_env(clear_cache=True)


if __name__ == "__main__":
    main()
