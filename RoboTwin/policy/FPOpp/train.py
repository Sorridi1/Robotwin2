#!/usr/bin/env python3
"""Fine-tune a checkpoint-native ManiFlow model with FPO++ in the current environment."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from fpopp.runtime import POLICY_ROOT, configure_paths

configure_paths()

from omegaconf import OmegaConf
from fpopp.checkpoint import FORMAT, load_payload, policy_configuration


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", help="Input ManiFlow BC checkpoint")
    selection.add_argument("--resume", help="Full FPO++ latest.ckpt to resume")
    parser.add_argument("--config", type=Path, help="Algorithm YAML; defaults to configs/fpopp_robotwin2.yaml")
    parser.add_argument("--task-name")
    parser.add_argument("--task-config")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Inspect saved architecture/config without creating a model or scene")
    parser.add_argument("overrides", nargs="*", help="Existing config keys as key=value")
    args = parser.parse_args(argv)
    if args.resume:
        if args.config:
            parser.error("Resume uses its saved configuration; use key=value for permitted runtime overrides")
        payload = load_payload(args.resume)
        if payload.get("format") != FORMAT:
            parser.error("--resume requires an FPO++ checkpoint")
        cfg = OmegaConf.create(payload["cfg"])
    else:
        cfg = OmegaConf.load(args.config or POLICY_ROOT / "configs/fpopp_robotwin2.yaml")
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.set_struct(cfg, False)
    for name in ("checkpoint", "resume", "task_name", "task_config", "device", "seed"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    if not cfg.resume and not cfg.checkpoint:
        parser.error("Supply --checkpoint PATH (or --resume PATH)")
    if args.output_dir:
        cfg.output_dir = str(args.output_dir.expanduser().resolve())
    if not cfg.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = str(POLICY_ROOT / "output" / f"{cfg.task_name}_{timestamp}_seed{cfg.seed}")
    cfg.output_dir = str(Path(cfg.output_dir).expanduser().resolve())
    for field in ("max_iterations", "total_action_steps", "checkpoint_every"):
        if int(cfg[field]) < 1:
            parser.error(f"{field} must be positive")
    for group, fields in {"rollout": ("n_envs", "n_steps", "env_seed_stride", "worker_threads", "worker_timeout_sec"),
                          "fpo": ("mc_samples", "mc_batch_size"),
                          "ppo": ("batch_size", "grad_accum_steps", "update_epochs")}.items():
        if any(int(cfg[group][field]) < 1 for field in fields):
            parser.error(f"{group} counts must be positive")
    if cfg.rollout.backend not in ("serial", "subprocess"):
        parser.error("rollout.backend must be serial or subprocess")
    if cfg.ppo.amp_dtype not in ("bfloat16", "float16"):
        parser.error("ppo.amp_dtype must be bfloat16 or float16")
    if cfg.env.scene_validation.mode != "cheap":
        parser.error("Use the shared cheap scene validator for this comparison")
    if cfg.validation.interval < 0 or cfg.validation.episodes < 1:
        parser.error("Invalid validation interval or episode count")
    if cfg.validation.interval and not cfg.validation.seed_file:
        parser.error("Set validation.seed_file to an independent validation bank")
    if not args.dry_run and not cfg.resume and any((Path(cfg.output_dir) / "checkpoints").glob("*.ckpt")):
        parser.error("Output already contains checkpoints; use --resume or a new --output-dir")
    return cfg, args


def main():
    cfg, args = parse_config()
    if args.dry_run:
        payload = load_payload(cfg.resume or cfg.checkpoint)
        policy_cfg = policy_configuration(payload)
        start = int(policy_cfg.n_obs_steps)-1
        end = min(start+int(policy_cfg.n_action_steps), int(policy_cfg.horizon))
        print(json.dumps({"checkpoint": cfg.resume or cfg.checkpoint,
                          "maniflow_source": str(POLICY_ROOT.parent / "ManiFlow/ManiFlow"),
                          "policy_cfg": OmegaConf.to_container(policy_cfg, resolve=True),
                          "action_window": [start, end], "effective_action_steps": end-start,
                          "training_config": OmegaConf.to_container(cfg, resolve=True)}, ensure_ascii=False, indent=2))
        return
    from fpopp.workspace import Workspace
    Workspace(cfg).run()


if __name__ == "__main__":
    main()
