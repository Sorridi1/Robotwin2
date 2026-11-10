#!/usr/bin/env python3
"""Small real-simulator smoke check; holds joints without loading/training a policy."""

import argparse
import json
from pathlib import Path
import sys
import time

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT / "ManiFlow"))

import hydra
import numpy as np
from omegaconf import OmegaConf

from maniflow.common.robotwin_env_pool import SerialEnvPool, SubprocessEnvPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("serial", "subprocess"), default="subprocess")
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--legacy-sensors", action="store_true")
    parser.add_argument("--expert-check", action="store_true")
    parser.add_argument("--validation-mode", choices=("none", "cheap", "expert"), default="cheap")
    parser.add_argument("--output", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.n_envs < 1 or args.steps < 1:
        parser.error("n-envs and steps must be positive")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(POLICY_ROOT / "ManiFlow/maniflow/config"), version_base="1.2"):
        cfg = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2", overrides=[
            "task_name=place_empty_cup", "setting=demo_messy_ur5", "training.seed=0",
            f"rl.env.scene_validation.mode={'expert' if args.expert_check else args.validation_mode}",
            "rl.env.expert_seed_cache_dir=/tmp/maniflow-simulation-check-seeds",
            *args.overrides,
        ])
    configs = []
    for index in range(args.n_envs):
        config = OmegaConf.to_container(cfg.rl.env, resolve=True)
        config.update(seed=index, start_seed=100000 + index * 1000)
        if args.legacy_sensors:
            config.update(head_pointcloud_only=False, render_on_observation_only=False)
        configs.append(config)
    shapes = {key: (int(cfg.n_obs_steps), *cfg.shape_meta.obs[key].shape)
              for key in ("point_cloud", "agent_pos")}
    started = time.perf_counter()
    pool = (SerialEnvPool(configs) if args.backend == "serial" else
            SubprocessEnvPool(configs, observation_shapes=shapes, timeout_sec=300))
    timings = {"construct_sec": time.perf_counter() - started}
    try:
        started = time.perf_counter()
        obs = pool.reset()
        timings["reset_sec"] = time.perf_counter() - started
        print(json.dumps({"phase": "reset", **timings}), flush=True)
        results_log = []
        for step in range(args.steps):
            # Two commands per chunk are sufficient to exercise TOPP, physics and cameras.
            actions = np.stack([np.repeat(value["agent_pos"][-1: ], 2, axis=0) for value in obs])
            started = time.perf_counter()
            results = pool.step(actions)
            wall_sec = time.perf_counter() - started
            obs = [value[0] for value in results]
            for value in obs:
                for key, shape in shapes.items():
                    assert value[key].shape == shape, (key, value[key].shape, shape)
                    assert np.isfinite(value[key]).all(), key
            record = {
                "step": step, "wall_sec": wall_sec,
                "rewards": [value[1] for value in results],
                "terminated": [value[2] for value in results],
                "truncated": [value[3] for value in results],
                "qpos": [value["agent_pos"][-1].tolist() for value in obs],
                "worker_env_sec_sum": sum(value[4]["perf/env_step_sec"] for value in results),
                "reset_sec_sum": sum(value[4]["perf/reset_sec"] for value in results),
            }
            results_log.append(record)
            print(json.dumps(record), flush=True)
        report = {"backend": args.backend, "legacy_sensors": args.legacy_sensors,
                  "n_envs": args.n_envs, "timings": timings, "steps": results_log,
                  "initial_reset_info": pool.last_reset_infos, "next_seeds": pool.get_next_seeds()}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        pool.close(clear_cache=True)
    print("SIMULATION_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
