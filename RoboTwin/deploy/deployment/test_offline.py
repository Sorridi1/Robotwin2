#!/opt/shared/Prog/anaconda3/envs/rise/bin/python
"""Combined offline tests for Hybrid64 V3.1 deployment.

Requires the ``rise`` conda environment (torch + pytorch3d + MinkowskiEngine + open3d):
    /opt/shared/Prog/anaconda3/envs/rise/bin/python

If any dependency is missing the test will report a clear error — do NOT
silently skip structural tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sd_sha256(sd: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        v = sd[k]
        if hasattr(v, "detach"):
            h.update(torch.as_tensor(v).detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(pickle.dumps(v, protocol=5))
    return h.hexdigest()


def _torch_load(path: str) -> Any:
    return torch.load(os.path.expanduser(str(path)), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Test A: strict checkpoint load
# ---------------------------------------------------------------------------
def test_a_strict_load(checkpoint_path: str) -> Dict[str, Any]:
    """Verify strict=True loading of the Hybrid64 V3.1 checkpoint."""
    print("=" * 70)
    print("TEST A: Strict Checkpoint Load")
    print("=" * 70)

    from tools.bsxyz_mainline.runtime import build_policy

    ckpt = _torch_load(checkpoint_path)
    assert isinstance(ckpt, dict), f"Expected dict, got {type(ckpt)}"

    args_dict = ckpt["args"]
    input_dim = int(ckpt.get("input_dim", 17))
    state = ckpt["policy"]
    assert isinstance(state, dict), f"Expected dict policy, got {type(state)}"

    # Check for wrapper prefixes
    keys = sorted(state.keys())
    prefix = ""
    for cand in ("module.", "model.", "policy."):
        if sum(1 for k in keys if k.startswith(cand)) > len(keys) * 0.5:
            prefix = cand
            break
    if prefix:
        print(f"Removing prefix '{prefix}': {len(state)} keys")
        state = {k[len(prefix):]: v for k, v in state.items()}

    # Build policy from checkpoint metadata
    args_ns = argparse.Namespace(**args_dict)
    policy = build_policy(args_ns, input_dim)

    # strict=True load
    missing, unexpected = policy.load_state_dict(state, strict=True)

    assert missing == [], f"Missing keys: {missing[:10]}"
    assert unexpected == [], f"Unexpected keys: {unexpected[:10]}"

    policy = policy.to("cpu").eval()

    print(f"  checkpoint_type: {ckpt.get('checkpoint_type')}")
    print(f"  epoch: {ckpt.get('epoch')}")
    print(f"  policy_source: {ckpt.get('policy_source')}")
    print(f"  input_dim: {input_dim}")
    print(f"  state_dict keys: {len(state)}")
    print(f"  state_dict SHA256: {_sd_sha256(state)}")
    print(f"  prefix removed: '{prefix}'")
    print(f"  missing_keys: {missing}")
    print(f"  unexpected_keys: {unexpected}")
    print("✅ Test A PASSED: strict load OK\n")

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_type": str(ckpt.get("checkpoint_type")),
        "epoch": int(ckpt.get("epoch", -1)),
        "policy_source": str(ckpt.get("policy_source")),
        "input_dim": input_dim,
        "state_dict_key_count": len(state),
        "state_dict_sha256": _sd_sha256(state),
        "prefix_removed": prefix or "none",
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


# ---------------------------------------------------------------------------
# Test B: evaluator equivalence
# ---------------------------------------------------------------------------
def test_b_evaluator_equivalence(
    checkpoint_path: str,
    val_data_path: str,
    eval_seed: int = 12345,
    max_batches: int = 5,
) -> Dict[str, Any]:
    """Compare evaluator output vs adapter output on same samples."""
    print("=" * 70)
    print("TEST B: Evaluator Equivalence")
    print("=" * 70)

    import argparse
    from torch.utils.data import DataLoader
    from tools.bsxyz_mainline.runtime import build_policy, make_generator
    from dataset.gwm_gaussian_gs import GWMGaussianVoxelDataset, collate_sparse_voxel
    from deployment.hybrid64_v31_policy_adapter import Hybrid64V31PolicyAdapter

    # ---- Reference evaluator path ----------------------------------------
    ckpt = _torch_load(checkpoint_path)
    args_dict = ckpt["args"]
    input_dim = int(ckpt.get("input_dim", 17))
    state = ckpt["policy"]

    args_ns = argparse.Namespace(**args_dict)
    policy_ref = build_policy(args_ns, input_dim)
    policy_ref.load_state_dict(state, strict=True)
    policy_ref = policy_ref.to("cpu").eval()

    # ---- Adapter path ----------------------------------------------------
    adapter = Hybrid64V31PolicyAdapter(
        checkpoint_path=checkpoint_path,
        device="cpu",
        eval_seed=eval_seed,
    )
    adapter.load_checkpoint_strict()

    # ---- Load val data ---------------------------------------------------
    ds = GWMGaussianVoxelDataset(val_data_path, require_input_xyz=True)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_sparse_voxel, drop_last=False,
    )

    differences = []
    for batch_id, batch in enumerate(loader):
        if batch_id >= max_batches:
            break

        coords = batch["input_coords"]
        feats = batch["input_feats"]
        input_xyz = batch["input_xyz"]
        input_batch = batch["input_batch"]
        bs = 1
        gen = make_generator(torch.device("cpu"), eval_seed + batch_id)

        # Reference evaluator inference
        with torch.no_grad():
            action_ref, _ = policy_ref(
                coords, feats, actions=None, batch_size=bs,
                return_aux=True, input_xyz=input_xyz,
                input_batch=input_batch, num_samples=1, generator=gen,
            )
        action_ref_np = action_ref.detach().cpu().float().numpy()
        if action_ref_np.ndim == 4:
            action_ref_np = action_ref_np[:, 0]
        action_ref_np = action_ref_np.squeeze(0)

        # Adapter inference (must use same seed → same generator)
        obs = {
            "input_coords": coords[:, 1:].int().numpy(),  # remove batch col
            "input_feats": feats.float().numpy(),
            "input_xyz": input_xyz.float().numpy(),
        }
        adapter._batch_counter = batch_id  # align counter
        action_adapter_np = adapter.predict_action_chunk(obs, seed=eval_seed + batch_id)

        # Compare
        max_diff = float(np.abs(action_ref_np - action_adapter_np).max())
        all_finite = bool(np.all(np.isfinite(action_ref_np)) and np.all(np.isfinite(action_adapter_np)))
        same_shape = action_ref_np.shape == action_adapter_np.shape

        differences.append({
            "batch_id": batch_id,
            "max_abs_diff": float(max_diff),
            "all_finite": all_finite,
            "same_shape": same_shape,
            "shape_ref": list(action_ref_np.shape),
            "shape_adapter": list(action_adapter_np.shape),
        })

        status = "✅" if (max_diff < 1e-5 and all_finite and same_shape) else "❌"
        print(f"  batch {batch_id}: max_diff={max_diff:.2e}  finite={all_finite}  shape_ok={same_shape}  {status}")

    all_ok = all(
        d["max_abs_diff"] < 1e-5 and d["all_finite"] and d["same_shape"]
        for d in differences
    )
    print(f"{'✅' if all_ok else '❌'} Test B {'PASSED' if all_ok else 'FAILED'}: "
          f"evaluator equivalence\n")

    return {
        "num_batches_tested": len(differences),
        "all_equivalent": bool(all_ok),
        "max_abs_diff_overall": float(max(d["max_abs_diff"] for d in differences)) if differences else None,
        "per_batch": differences,
    }


# ---------------------------------------------------------------------------
# Test C: dry-run with recorded sample
# ---------------------------------------------------------------------------
def test_c_dry_run(
    checkpoint_path: str,
    sample_path: str,
    camera_serial: str = "243222076209",
) -> Dict[str, Any]:
    """Full dry-run pipeline with one recorded window."""
    print("=" * 70)
    print("TEST C: Recorded Dry-Run")
    print("=" * 70)

    from deployment.hybrid64_v31_policy_adapter import Hybrid64V31PolicyAdapter
    from deployment.action_executor import ActionExecutor, denormalize_action_chunk

    # ---- Load policy -----------------------------------------------------
    adapter = Hybrid64V31PolicyAdapter(
        checkpoint_path=checkpoint_path, device="cpu",
    )
    adapter.load_checkpoint_strict()

    # ---- Load recorded sample --------------------------------------------
    sample = _torch_load(sample_path)
    windows = sample.get("windows", [sample] if "coords" not in sample else [sample])
    w = windows[0]

    obs = {
        "input_coords": w["coords"].int().numpy(),
        "input_feats": w["feats"].float().numpy(),
        "input_xyz": w.get("input_xyz", w["coords"].float().numpy() * 0.005).float().numpy(),
    }

    obs_summary = {
        "num_voxels": int(obs["input_coords"].shape[0]),
        "feature_dim": int(obs["input_feats"].shape[1]),
        "coords_range": {
            "x": [int(obs["input_coords"][:, 0].min()), int(obs["input_coords"][:, 0].max())],
            "y": [int(obs["input_coords"][:, 1].min()), int(obs["input_coords"][:, 1].max())],
            "z": [int(obs["input_coords"][:, 2].min()), int(obs["input_coords"][:, 2].max())],
        },
        "input_xyz_range": {
            "x": [float(obs["input_xyz"][:, 0].min()), float(obs["input_xyz"][:, 0].max())],
            "y": [float(obs["input_xyz"][:, 1].min()), float(obs["input_xyz"][:, 1].max())],
            "z": [float(obs["input_xyz"][:, 2].min()), float(obs["input_xyz"][:, 2].max())],
        },
    }

    # ---- Predict ---------------------------------------------------------
    action_norm = adapter.predict_action_chunk(obs)

    assert action_norm.shape == (20, 10), f"Expected [20,10], got {action_norm.shape}"
    assert np.all(np.isfinite(action_norm)), "Non-finite values in output"

    # ---- Prepare commands ------------------------------------------------
    executor = ActionExecutor(camera_serial=camera_serial, enable_robot_motion=False)
    plan = executor.prepare_commands(action_norm)

    # ---- Save outputs ----------------------------------------------------
    out_dir = Path("deployment_dry_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Observation summary
    (out_dir / "observation_summary.json").write_text(
        json.dumps(obs_summary, indent=2, default=str)
    )

    # Sparse tensor summary
    (out_dir / "sparse_tensor_summary.json").write_text(json.dumps({
        "num_voxels": obs_summary["num_voxels"],
        "feature_dim": obs_summary["feature_dim"],
        "feature_layout": "raw17",
        "coord_dtype": "int32",
        "coord_layout": "centered_voxel_xyz_without_batch",
    }, indent=2))

    # Model output
    np.save(str(out_dir / "model_output.npy"), action_norm)

    # Decoded action contract
    (out_dir / "decoded_action_contract.json").write_text(json.dumps({
        "shape": [20, 10],
        "position_indices": [0, 1, 2],
        "rotation_indices": [3, 4, 5, 6, 7, 8],
        "gripper_index": 9,
        "rotation_representation": "rotation_6d",
        "action_frame": "camera",
        "absolute_or_relative": "absolute",
        "gripper_semantics": "continuous_width_metres_normalized",
        "normalization": "decoded_and_clamped_inside_policy",
    }, indent=2))

    # Base-frame command preview
    np.save(str(out_dir / "base_frame_command_preview.npy"), plan["tcp_base_rot6d"])

    # Safety check
    (out_dir / "safety_check.json").write_text(json.dumps({
        "all_safe": plan["all_safe"],
        "checks": plan["checks"],
        "gripper_change_steps": plan["gripper_change_steps"],
    }, indent=2))

    # ---- Print summary ---------------------------------------------------
    print(f"  Observation: {obs_summary['num_voxels']} voxels, "
          f"feature_dim={obs_summary['feature_dim']}")
    print(f"  Model output shape: {action_norm.shape}")
    print(f"  All finite: {np.all(np.isfinite(action_norm))}")
    print(f"  Range: [{action_norm.min():.4f}, {action_norm.max():.4f}]")
    print(f"  All safety checks: {plan['all_safe']}")
    for k, v in plan["checks"].items():
        print(f"    {k}: {v}")
    print(f"  Gripper change steps: {plan['gripper_change_steps']}")
    print(f"  Base-frame TCP (step 0): {plan['tcp_base_rot6d'][0]}")
    print("  Outputs saved to deployment_dry_run/")

    # ---- Write report ----------------------------------------------------
    dry_run_report = {
        "checkpoint": checkpoint_path,
        "sample": sample_path,
        "observation_summary": obs_summary,
        "model_output_shape": list(action_norm.shape),
        "model_output_range": [float(action_norm.min()), float(action_norm.max())],
        "all_finite": bool(np.all(np.isfinite(action_norm))),
        "safety_all_safe": plan["all_safe"],
        "gripper_change_steps": plan["gripper_change_steps"],
        "base_frame_first_step": plan["tcp_base_rot6d"][0].tolist(),
        "coordinate_transform_T_camera_base": executor.transform.T_camera_base.tolist(),
    }
    (out_dir / "dry_run_report.json").write_text(
        json.dumps(dry_run_report, indent=2, default=str)
    )

    print("\n✅ Test C PASSED: dry-run complete\n")
    return dry_run_report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    import argparse as ap
    p = ap.ArgumentParser(description="Offline deployment tests for Hybrid64 V3.1")
    p.add_argument("--checkpoint",
                   default="runs/hybrid64_v31_bsxyz_canonical_w02_seed233_ep600_20260627/final.pt")
    p.add_argument("--val-data", default="")
    p.add_argument("--sample", default="")
    p.add_argument("--skip-b", action="store_true", help="Skip evaluator equivalence (needs val data)")
    p.add_argument("--skip-c", action="store_true", help="Skip dry-run (needs recorded sample)")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = os.path.expanduser(args.checkpoint)

    # --- dependency check -------------------------------------------------
    missing = []
    for mod in ("torch", "pytorch3d", "MinkowskiEngine", "open3d"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"⚠️  Missing dependencies: {missing}")
        print("   PyTorch3D is required for coordinate transforms and rotation conversions.")
        print("   MinkowskiEngine is required for the sparse 3D encoder.")
        print("   open3d is required for online RGBD→point cloud conversion.")
        print()
        print("   Running only Test A (checkpoint inspection + schema validation)...")
        print()

    # Test A always runs (only needs torch)
    result_a = test_a_strict_load(checkpoint)
    with open("deployment_dry_run/checkpoint_schema.json", "w") as f:
        json.dump(result_a, f, indent=2, default=str)

    # Test B: needs val data + full deps
    if not args.skip_b and args.val_data and not missing:
        test_b_evaluator_equivalence(checkpoint, args.val_data)

    # Test C: needs recorded sample + full deps
    if not args.skip_c and args.sample and not missing:
        test_c_dry_run(checkpoint, args.sample)

    print("\n" + "=" * 70)
    print("Test A result saved to deployment_dry_run/checkpoint_schema.json")
    if missing:
        print(f"⚠️  Tests B/C skipped — install: {' '.join(missing)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse  # needed for test functions
    main()
