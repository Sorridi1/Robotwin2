import dill
from omegaconf import OmegaConf
import pytest
import torch

from maniflow.common.eval_checkpoint import load_evaluation_config


def save_checkpoint(root, tag, cfg):
    path = root / "checkpoints" / f"{tag}.ckpt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"cfg": OmegaConf.create(cfg)}, path, pickle_module=dill)
    return path


def test_original_policy_ignores_later_architecture_and_inference_edits(tmp_path):
    save_checkpoint(tmp_path, "latest", {"policy": {"num_inference_steps": 10}})
    edited = OmegaConf.create({"policy": {
        "num_inference_steps": 2, "n_layer": 3,
    }})
    cfg, checkpoint = load_evaluation_config(tmp_path, "latest", edited)
    assert cfg.policy.num_inference_steps == 10
    assert "n_layer" not in cfg.policy
    assert checkpoint == tmp_path / "checkpoints/latest.ckpt"
    assert edited.policy.n_layer == 3


def test_explicit_checkpoint_uses_its_own_configuration(tmp_path):
    save_checkpoint(tmp_path, "latest", {"policy": {"num_inference_steps": 2}})
    expected = save_checkpoint(tmp_path, "epoch=0500-val_loss=0.3", {
        "policy": {"num_inference_steps": 10, "n_layer": 12},
        "shape_meta": "${task.shape_meta}", "task": {"shape_meta": {"action": 14}},
    })
    cfg, checkpoint = load_evaluation_config(
        tmp_path, "epoch=0500-val_loss=0.3", OmegaConf.create({}),
    )
    assert checkpoint == expected
    assert cfg.policy.n_layer == 12
    assert cfg.policy.num_inference_steps == 10
    assert cfg.shape_meta.action == 14


def test_best_tag_uses_matching_weights_and_configuration(tmp_path):
    save_checkpoint(tmp_path, "epoch=0500-val_loss=0.4", {"epoch": 500})
    expected = save_checkpoint(tmp_path, "epoch=0400-val_loss=0.3", {"epoch": 400})
    cfg, checkpoint = load_evaluation_config(tmp_path, "best", OmegaConf.create({
        "checkpoint": {"topk": {"monitor_key": "val_loss"}},
    }))
    assert checkpoint == expected and cfg.epoch == 400


def test_missing_checkpoint_fails_before_policy_initialization(tmp_path):
    with pytest.raises(FileNotFoundError, match="Evaluation checkpoint"):
        load_evaluation_config(tmp_path, "latest", OmegaConf.create({}))
