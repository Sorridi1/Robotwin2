#!/usr/bin/env python3
"""Read-only checkpoint/sampler/CFM diagnostics (synthetic observations, no RL update)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpopp.runtime import configure_paths

configure_paths()

import torch

from fpopp.actor import FPOActor, execution_mask
from fpopp.checkpoint import load_payload, load_policy
from fpopp.loss import surrogate_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="FPO++ checkpoint to inspect")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    torch.set_num_threads(2)
    torch.manual_seed(719)
    payload = load_payload(args.checkpoint)
    base_path = payload["initial_checkpoint"]["path"]
    base = load_payload(base_path)
    original = base["state_dicts"][payload["initial_checkpoint"]["state_key"]]
    current = payload["state_dicts"]["model"]
    report = {"checkpoint": args.checkpoint, "initial_checkpoint": payload["initial_checkpoint"],
              "iteration": payload["iteration"], "observation_source": "synthetic normalizer statistics",
              "device": args.device, "frozen_state": {}}
    for prefix in ("normalizer.", "obs_encoder.", "plan_predictor."):
        names = [key for key in original if key.startswith(prefix)]
        report["frozen_state"][prefix] = {"tensors": len(names),
            "changed": [key for key in names if not torch.equal(original[key], current[key])]}
    flow_names = [key for key in original if key.startswith("model.")]
    report["flow_tensors_changed"] = sum(not torch.equal(original[key], current[key]) for key in flow_names)
    policy, config, _ = load_policy(base, device=args.device)
    actor = FPOActor(policy, seed=719)
    observations = {}
    for field in ("agent_pos", "point_cloud"):
        stats = policy.normalizer[field].get_input_stats()
        mean, std = stats["mean"].to(args.device), stats["std"].to(args.device)
        shape = (2, policy.n_obs_steps, *config.shape_meta.obs[field].shape)
        observations[field] = mean + std.clamp_min(.01) * torch.randn(shape, device=args.device) * .1
    with torch.no_grad():
        features, plan = actor.encode(observations)
        state = actor.rng.get_state()
        actions, targets, _ = actor.sample(features, plan)
        actor.rng.set_state(state)
        latent = torch.randn(targets.shape, device=args.device, generator=actor.rng)
        native = policy.sample_ode(x0=latent, N=actor.inference_steps, vis_cond=features, plan_cond=plan)[-1]
        report["native_sampler_max_abs_error"] = float((targets-native).abs().max())
        report["action_shape"] = list(actions.shape)
        report["action_window"] = list(actor.window)
        actor.final_action_clip = payload["cfg"]["actor"]["final_action_clip"]
        actor.intermediate_clip = payload["cfg"]["actor"]["intermediate_clip"]
        actor.rng.set_state(state)
        _, clipped, clip_fraction = actor.sample(features, plan)
        report["clipping_on_synthetic_obs"] = {"final_clip_fraction": clip_fraction,
            "normalized_action_rms_difference_from_native": float((targets-clipped).square().mean().sqrt()),
            "normalized_action_max_difference_from_native": float((targets-clipped).abs().max())}
        time, noise = actor.draw_cfm(clipped, 2)
    report["unchanged_policy_ratios"] = {}
    for precision in (["float32", "bfloat16"] if str(args.device).startswith("cuda") else ["float32"]):
        from contextlib import nullcontext
        def cast():
            return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
        with torch.no_grad(), cast():
            old = actor.cfm_losses(features, plan, clipped, time, noise)
        # Match update: gradients enabled and the same microbatch shape/AMP mode.
        with cast():
            new = actor.cfm_losses(features, plan, clipped, time, noise)
            _, stats = surrogate_loss(old, new, execution_mask(actor, [actions.shape[1]] * actions.shape[0]),
                                      torch.tensor([1., -1.], device=args.device), clip_coef=.02)
        report["unchanged_policy_ratios"][precision] = {
            **{key: float(value) for key, value in stats.items()},
            "cfm_max_abs_difference": float((old-new.detach()).abs().max())}
        del new, old
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
