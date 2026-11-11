#!/usr/bin/env python3
"""One real-checkpoint/expert-data noise update and policy sample, without a simulator."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT / "ManiFlow"))

import hydra
from omegaconf import OmegaConf
import torch

from maniflow.workspace.train_reinflow_rl_robotwin2_workspace import TrainReinFlowRLRoboTwinWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--task", default="place_empty_cup")
    parser.add_argument("--setting", default="demo_messy_ur5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--check-resume", action="store_true", help="Also write/read full RL checkpoint in a temporary directory")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with hydra.initialize_config_dir(config_dir=str(POLICY_ROOT / "ManiFlow/maniflow/config"), version_base="1.2"):
        cfg = hydra.compose(config_name="reinflow_rl_pointcloud_robotwin2_expert_interp", overrides=[
            f"task_name={args.task}", f"setting={args.setting}", f"task_config={args.setting}",
            f"training.device={args.device}", "rl.actor.actor_old_device=cpu", "logging.mode=disabled",
            f"rl.pretrained_checkpoint_path='{args.checkpoint.resolve()}'",
            f"rl.noise_guidance.dataset_path='{args.dataset.resolve()}'",
            "rl.noise_guidance.batch_size=1", "rl.noise_guidance.gradient_steps=1",
            f"rl.noise_guidance.sample_all_steps={str(args.all_steps).lower()}",
            "rl.ppo.use_amp=false", "rl.eval.enabled=false", "rl.resume=false",
        ])
    with tempfile.TemporaryDirectory(prefix="rafe-expert-smoke-") as output_dir:
        workspace = TrainReinFlowRLRoboTwinWorkspace(cfg, output_dir=output_dir)
        legacy_modules = {"plan_predictor", "plan_proj", "coarse_action_operator"}
        for module in (workspace.model, workspace.ppo_actor, workspace.ppo_actor.actor_old):
            assert not any(legacy_modules.intersection(name.split(".")) for name, _ in module.named_modules())
            assert not any(legacy_modules.intersection(name.split(".")) for name, _ in module.named_parameters())
        metrics = workspace.noise_nll_update()
        batch = next(workspace.expert_data)
        actions, chains, logprobs = workspace.ppo_actor.get_actions(batch["obs"])
        assert torch.isfinite(actions).all() and torch.isfinite(logprobs).all()
        metrics.update(action_shape=list(actions.shape), chain_shape=list(chains.shape),
                       checkpoint_normalizer=True, no_action_reference_modules=True)
        if args.check_resume:
            checkpoint = workspace.save_resume_checkpoint()
            saved_steps = workspace.noise_update_steps
            before = [p.detach().clone() for p in workspace.ppo_actor.noise_head.parameters()]
            workspace.noise_nll_update()
            workspace.load_resume_checkpoint(checkpoint)
            assert workspace.noise_update_steps == saved_steps
            assert all(torch.equal(old, p) for old, p in zip(before, workspace.ppo_actor.noise_head.parameters()))
            metrics["resume_roundtrip"] = True
        print(json.dumps(metrics, sort_keys=True), flush=True)
    print("EXPERT_INTERPOLATION_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
