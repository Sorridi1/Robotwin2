#!/opt/shared/Prog/anaconda3/envs/rise/bin/python
"""Hybrid64 V3.1 Policy Adapter for real-robot deployment.

Requires the ``rise`` conda environment (torch + pytorch3d + MinkowskiEngine + open3d):
    /opt/shared/Prog/anaconda3/envs/rise/bin/python

Loads a Hybrid64 V3.1 final.pt checkpoint using the official build_policy
and evaluator inference path, then exposes a minimal predict_action_chunk()
interface for the action executor.

Usage (offline / dry-run):
    python deployment/hybrid64_v31_policy_adapter.py \\
        --checkpoint runs/hybrid64_v31_bsxyz_canonical_w02_seed233_ep600_20260627/final.pt \\
        --device cpu --dry-run --no-robot

Design rules:
- Does NOT import Agent, UDPClient, or any hardware module.
- Does NOT load Repair payload or compute any auxiliary loss.
- Uses strict=True state-dict loading.
- Reuses tools.bsxyz_mainline.runtime.build_policy for constructor fidelity.
- Follows the exact evaluator inference semantics:
    K=1, num_inference_steps=20, generator from eval_sample_seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap the repo root so relative imports work from anywhere.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Lazy-import heavy dependencies so --help / schema inspection is fast.
# ---------------------------------------------------------------------------
_torch_load = None
_build_policy = None
_make_generator = None
_set_seed = None
_state_dict_sha256 = None


def _lazy_import_runtime():
    global _torch_load, _build_policy, _make_generator, _set_seed, _state_dict_sha256
    if _torch_load is not None:
        return
    from tools.bsxyz_mainline.runtime import (
        build_policy,
        make_generator,
        set_seed,
        state_dict_sha256,
        torch_load,
    )
    _build_policy = build_policy
    _make_generator = make_generator
    _set_seed = set_seed
    _state_dict_sha256 = state_dict_sha256
    _torch_load = torch_load


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = str(
    _REPO_ROOT
    / "runs/hybrid64_v31_bsxyz_canonical_w02_seed233_ep600_20260627/final.pt"
)


# ---------------------------------------------------------------------------
# Policy Adapter
# ---------------------------------------------------------------------------
class Hybrid64V31PolicyAdapter:
    """Load and run a Hybrid64 V3.1 action-token-diffusion policy.

    The adapter is deliberately stateless across calls — it holds the loaded
    policy + metadata but does not accumulate history.  The caller is
    responsible for feeding the correct observation per step.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        model_repo_root: Optional[str] = None,
        eval_seed: int = 12345,
    ):
        self.checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model_repo_root = os.path.abspath(
            os.path.expanduser(str(model_repo_root or _REPO_ROOT))
        )
        self.eval_seed = int(eval_seed)

        # --- populated by load_checkpoint_strict() -----------------------
        self.policy: Optional[torch.nn.Module] = None
        self.ckpt_meta: Dict[str, Any] = {}
        self.args_dict: Dict[str, Any] = {}
        self.input_dim: int = 0
        self.num_action: int = 0
        self.action_dim: int = 0
        self.num_inference_steps: int = 20
        self.state_sha256: str = ""
        self.hybrid_tokenizer_path: str = ""

        # --- batch counter for per-step deterministic sampling -----------
        self._batch_counter: int = 0

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def load_checkpoint_strict(self) -> Dict[str, Any]:
        """Load checkpoint with strict=True, returning a validation report."""
        _lazy_import_runtime()

        ckpt = _torch_load(self.checkpoint_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise TypeError(
                f"Expected checkpoint dict at {self.checkpoint_path}, "
                f"got {type(ckpt).__name__}"
            )

        # ---- validate top-level schema ---------------------------------
        ckpt_type = str(ckpt.get("checkpoint_type", ""))
        if ckpt_type not in ("bsxyz_mainline",):
            raise ValueError(
                f"Unsupported checkpoint_type={ckpt_type!r}. "
                f"Expected 'bsxyz_mainline'."
            )

        args_dict = ckpt.get("args", {})
        if not isinstance(args_dict, dict) or not args_dict:
            raise KeyError("checkpoint missing 'args' dict")

        self.args_dict = dict(args_dict)
        self.input_dim = int(ckpt.get("input_dim", self.args_dict.get("input_dim", 17)))
        self.num_action = int(self.args_dict.get("num_action", 20))
        self.action_dim = int(self.args_dict.get("action_dim", 10))
        self.num_inference_steps = int(self.args_dict.get("num_inference_steps", 20))
        self.hybrid_tokenizer_path = str(self.args_dict.get("hybrid_action_tokenizer_path", ""))

        model_family = str(self.args_dict.get("model_family", ""))
        action_tokenizer_type = str(self.args_dict.get("action_tokenizer_type", ""))
        if model_family == "structured_hybrid_diffusion_v2":
            if action_tokenizer_type != "structured_bsxyz_rotgrip_chunks":
                raise ValueError(
                    "structured_hybrid_diffusion_v2 requires action_tokenizer_type="
                    "'structured_bsxyz_rotgrip_chunks', got "
                    f"{action_tokenizer_type!r}"
                )
        elif action_tokenizer_type not in {"hybrid_bsxyz_rotgrip", "temporal_chunk", "hybrid_bsxyz_rotgrip_dct46"}:
            raise ValueError(
                "Unsupported action_tokenizer_type="
                f"{action_tokenizer_type!r}"
            )
        if model_family not in ("", "structured_hybrid_diffusion_v2", "chunk_scene_diffusion_v1", "hybrid64_bsxyz_rotgrip_dct46"):
            raise ValueError(
                "Unsupported model_family="
                f"{model_family!r}"
            )

        # ---- locate state dict -----------------------------------------
        state = ckpt.get("policy")
        if state is None:
            raise KeyError(
                "checkpoint missing 'policy' key. Available keys: "
                f"{sorted(ckpt.keys())}"
            )
        if not isinstance(state, dict):
            raise TypeError(f"'policy' must be a dict, got {type(state).__name__}")

        # ---- detect wrapper prefix -------------------------------------
        keys = sorted(state.keys())
        prefix = ""
        for candidate in ("module.", "model.", "policy."):
            if sum(1 for k in keys if k.startswith(candidate)) > len(keys) * 0.5:
                prefix = candidate
                break

        if prefix:
            old_keys = list(state.keys())
            state = {k[len(prefix):]: v for k, v in state.items()}
            new_keys = sorted(state.keys())
            print(
                f"[Adapter] Removed '{prefix}' prefix: "
                f"{len(old_keys)} keys → {len(new_keys)} keys"
            )

        self.state_sha256 = _state_dict_sha256(state)

        # ---- build policy from checkpoint metadata ---------------------
        args_ns = argparse.Namespace(**self.args_dict)
        policy = _build_policy(args_ns, self.input_dim)

        missing, unexpected = policy.load_state_dict(state, strict=True)

        report = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_type": ckpt_type,
            "epoch": int(ckpt.get("epoch", -1)),
            "policy_source": str(ckpt.get("policy_source", "")),
            "input_dim": self.input_dim,
            "num_action": self.num_action,
            "action_dim": self.action_dim,
            "model_family": model_family,
            "action_tokenizer_type": action_tokenizer_type,
            "action_representation": str(self.args_dict.get("action_representation", "")),
            "native_latent_dim": int(self.args_dict.get("native_latent_dim", 0)),
            "internal_latent_dim": int(self.args_dict.get("pca_dim", 0)),
            "hybrid_padding_dim": int(self.args_dict.get("hybrid_padding_dim", 0)),
            "num_action_tokens": int(self.args_dict.get("num_action_tokens", 0)),
            "num_structured_tokens": int(self.args_dict.get("num_structured_tokens", 0)),
            "rotgrip_pca_enabled": (
                action_tokenizer_type == "hybrid_bsxyz_rotgrip"
                or bool(self.args_dict.get("rotgrip_pca_enabled", False))
            ),
            "state_dict_key_count": len(state),
            "state_dict_sha256": self.state_sha256,
            "prefix_removed": prefix,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "hybrid_tokenizer_path": self.hybrid_tokenizer_path,
        }

        if missing or unexpected:
            raise RuntimeError(
                f"strict load failed:\n"
                f"  missing={missing[:10]}\n"
                f"  unexpected={unexpected[:10]}"
            )

        self.policy = policy.to(self.device).eval()
        self.ckpt_meta = report
        self._batch_counter = 0

        return report

    # ------------------------------------------------------------------
    # Observation → batch
    # ------------------------------------------------------------------
    def _build_batch(
        self,
        input_coords: np.ndarray,
        input_feats: np.ndarray,
        input_xyz: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Convert raw17 sparse voxels to the batch dict expected by policy.forward."""
        if input_coords.shape[0] == 0:
            raise ValueError("Empty observation (no voxels after preprocessing)")

        M = input_coords.shape[0]
        coords = torch.from_numpy(input_coords).to(dtype=torch.int32)
        feats = torch.from_numpy(input_feats).to(dtype=torch.float32)
        xyz = torch.from_numpy(input_xyz).to(dtype=torch.float32)

        batch_col = torch.zeros((M, 1), dtype=torch.int32)
        coords_b = torch.cat([batch_col, coords], dim=1)
        input_batch = torch.zeros((M,), dtype=torch.long)

        return {
            "coords": coords_b,
            "feats": feats,
            "input_xyz": xyz,
            "input_batch": input_batch,
            "batch_size": 1,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(
        self,
        observation: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Run policy inference and return [20, 10] float32 numpy array.

        Args:
            observation: Dict from ``preprocess_observation`` with keys
                         ``input_coords``, ``input_feats``, ``input_xyz``.
            seed: Override per-step generator seed (default: eval_seed + batch_idx).

        Returns:
            action  [20, 10]  float32  camera-frame normalized action chunk.
        """
        if self.policy is None:
            raise RuntimeError("load_checkpoint_strict() must be called first")

        batch = self._build_batch(
            observation["input_coords"],
            observation["input_feats"],
            observation["input_xyz"],
        )

        step_seed = int(seed) if seed is not None else (self.eval_seed + self._batch_counter)
        gen = _make_generator(self.device, step_seed)

        coords = batch["coords"].to(self.device)
        feats = batch["feats"].to(self.device)
        input_xyz = batch["input_xyz"].to(self.device)
        input_batch = batch["input_batch"].to(self.device)

        action, _aux = self.policy(
            coords,
            feats,
            actions=None,
            batch_size=1,
            return_aux=True,
            input_xyz=input_xyz,
            input_batch=input_batch,
            num_samples=1,
            generator=gen,
        )

        self._batch_counter += 1

        action_np = action.detach().cpu().to(torch.float32).numpy()
        if action_np.ndim == 4:
            action_np = action_np[:, 0]  # [1, K, 20, 10] → [1, 20, 10] with K=1
        action_np = action_np.squeeze(0)  # [1, 20, 10] → [20, 10]

        # ---- contract validation ---------------------------------------
        assert action_np.shape == (self.num_action, self.action_dim), (
            f"Unexpected output shape {action_np.shape}, "
            f"expected ({self.num_action}, {self.action_dim})"
        )
        if not np.all(np.isfinite(action_np)):
            raise RuntimeError(
                f"Policy output contains non-finite values: "
                f"finite={np.isfinite(action_np).sum()}/{action_np.size}"
            )

        return action_np

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def schema_summary(self) -> Dict[str, Any]:
        return {
            **self.ckpt_meta,
            "num_action": self.num_action,
            "action_dim": self.action_dim,
            "num_inference_steps": self.num_inference_steps,
            "hybrid_tokenizer_path": self.hybrid_tokenizer_path,
            "device": str(self.device),
        }

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Read-only inspection: print top-level keys, SHA256, etc. (no model build).

        This method avoids importing the full runtime (and thus pytorch3d, MinkowskiEngine,
        etc.) so it works in minimal environments.
        """
        import hashlib
        import pickle

        def _load(path):
            return torch.load(os.path.expanduser(str(path)), map_location="cpu", weights_only=False)

        def _sd_sha256(state_dict):
            h = hashlib.sha256()
            for key in sorted(state_dict.keys()):
                h.update(key.encode("utf-8"))
                v = state_dict[key]
                if hasattr(v, "detach"):
                    h.update(torch.as_tensor(v).detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(pickle.dumps(v, protocol=5))
            return h.hexdigest()

        ckpt = _load(checkpoint_path)
        report: Dict[str, Any] = {
            "top_level_keys": sorted(ckpt.keys()),
            "checkpoint_type": str(ckpt.get("checkpoint_type", "")),
            "epoch": int(ckpt.get("epoch", -1)),
            "policy_source": str(ckpt.get("policy_source", "")),
            "input_dim": int(ckpt.get("input_dim", -1)),
        }
        args = ckpt.get("args", {})
        if isinstance(args, dict):
            report["policy_module"] = str(args.get("policy_module", ""))
            report["policy_class"] = str(args.get("policy_class", ""))
            report["action_tokenizer_type"] = str(args.get("action_tokenizer_type", ""))
            report["hybrid_action_tokenizer_path"] = str(args.get("hybrid_action_tokenizer_path", ""))
            report["num_action"] = int(args.get("num_action", -1))
            report["action_dim"] = int(args.get("action_dim", -1))
            report["internal_latent_dim"] = int(args.get("pca_dim", -1))
            report["hybrid_padding_dim"] = int(args.get("hybrid_padding_dim", 0))
            report["num_action_tokens"] = int(args.get("num_action_tokens", -1))

        state = ckpt.get("policy")
        if isinstance(state, dict):
            report["state_dict_key_count"] = len(state)
            report["state_dict_sha256"] = _sd_sha256(state)
        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _bool_flag(name: str, default: bool) -> bool:
    val = os.environ.get(f"HYBRID64_{name.upper()}", str(default)).strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hybrid64 V3.1 Policy Adapter — load + inspect + dry-run inference.",
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to final.pt checkpoint.")
    p.add_argument("--device", default="cpu",
                   help="Torch device (cpu / cuda / cuda:0).")
    p.add_argument("--model-repo-root", default=str(_REPO_ROOT),
                   help="Root of the RISE-fzc repository.")
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="Base evaluation seed for deterministic sampling.")
    p.add_argument("--dry-run", action="store_true",
                   help="Perform one offline forward pass with dummy data.")
    p.add_argument("--recorded-sample", default="",
                   help="Path to a .pt window file for offline inference.")
    p.add_argument("--no-robot", action="store_true", default=True,
                   help="Disable robot motion (always on).")
    p.add_argument("--inspect", action="store_true",
                   help="Only inspect checkpoint schema, no model build.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.inspect:
        report = Hybrid64V31PolicyAdapter.inspect_checkpoint(args.checkpoint)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return

    adapter = Hybrid64V31PolicyAdapter(
        checkpoint_path=args.checkpoint,
        device=args.device,
        model_repo_root=args.model_repo_root,
        eval_seed=args.eval_seed,
    )

    report = adapter.load_checkpoint_strict()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("✅ Strict load OK — missing=[] unexpected=[]")

    if args.dry_run:
        print("\n=== Dry-run forward pass ===")
        if args.recorded_sample:
            _lazy_import_runtime()
            sample = _torch_load(os.path.expanduser(args.recorded_sample), map_location="cpu")
            windows = sample.get("windows", [sample] if "coords" not in sample else [sample])
            w = windows[0]
            obs = {
                "input_coords": w["coords"].int().numpy(),
                "input_feats": w["feats"].float().numpy(),
                "input_xyz": w.get("input_xyz", w["coords"].float() * 0.005).float().numpy(),
            }
        else:
            # dummy single-voxel input
            obs = {
                "input_coords": np.array([[0, 0, 0]], dtype=np.int32),
                "input_feats": np.random.randn(1, 17).astype(np.float32),
                "input_xyz": np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            }

        action = adapter.predict_action_chunk(obs, seed=args.eval_seed)
        print(f"Output shape: {action.shape}")
        print(f"All finite:  {np.all(np.isfinite(action))}")
        print(f"Range:       [{action.min():.4f}, {action.max():.4f}]")
        print("First 3 steps:")
        for i in range(min(3, action.shape[0])):
            print(f"  step {i:2d}: pos={np.array2string(action[i, :3], precision=3, suppress_small=True)}  "
                  f"rot6d={np.array2string(action[i, 3:9], precision=3, suppress_small=True)}  "
                  f"grip={action[i, 9]:.4f}")


if __name__ == "__main__":
    main()
