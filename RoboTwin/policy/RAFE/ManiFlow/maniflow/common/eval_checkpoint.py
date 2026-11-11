"""Read evaluation settings from the checkpoint that supplies the weights."""

from pathlib import Path

import dill
from omegaconf import OmegaConf
import torch


def load_evaluation_config(run_dir, tag, fallback_cfg):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint = checkpoint_dir / f"{tag}.ckpt"
    if tag == "best" and not checkpoint.is_file():
        monitor_key = fallback_cfg.checkpoint.topk.monitor_key
        candidates = []
        for path in checkpoint_dir.glob("*.ckpt"):
            if "latest" in path.name:
                continue
            try:
                score = float(path.name.split(f"{monitor_key}=", 1)[1].split(".ckpt")[0])
            except (IndexError, ValueError):
                continue
            candidates.append((score, path))
        if not candidates:
            raise ValueError(f"No checkpoints found with monitor key: {monitor_key}")
        select = min if "loss" in monitor_key else max
        checkpoint = select(candidates, key=lambda item: item[0])[1]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    payload = torch.load(
        str(checkpoint), map_location="cpu", mmap=True,
        pickle_module=dill, weights_only=False,
    )
    if "cfg" not in payload:
        raise ValueError(f"Evaluation checkpoint has no saved configuration: {checkpoint}")
    cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=False))
    return cfg, checkpoint
