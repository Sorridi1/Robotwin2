"""Learned-noise deployment never reinterprets older bounded-sigma weights."""
import importlib.util
from pathlib import Path

import dill
import hydra
from omegaconf import OmegaConf
import pytest
import torch


@pytest.mark.parametrize("change", ["no_cfg", "old_version", "no_version", "no_mode", "no_marker", "wrong_gain"])
def test_ppo_deploy_rejects_incompatible_noise_semantics_before_instantiation(tmp_path, monkeypatch, change):
    path = Path(__file__).resolve().parents[1] / "deploy_ppo_actor_policy.py"
    spec = importlib.util.spec_from_file_location("rafe_ppo_deploy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = {
        "rafe_resume_version": 2,
        "cfg": OmegaConf.create({"rl": {"actor": {
            "noise_parameterization": "log_residual", "residual_scale": 1.0,
        }}}),
        "ppo_actor": {"_extra_state": {"version": 1, "noise_parameterization": "log_residual", "residual_scale": 1.0}},
    }
    if change == "no_cfg":
        payload.pop("cfg")
    elif change == "old_version":
        payload["rafe_resume_version"] = 1
    elif change == "no_version":
        payload.pop("rafe_resume_version")
    elif change == "no_mode":
        payload["cfg"].rl.actor.pop("noise_parameterization")
    elif change == "no_marker":
        payload["ppo_actor"].pop("_extra_state")
    else:
        payload["ppo_actor"]["_extra_state"]["residual_scale"] = 0.5
    checkpoint = tmp_path / "noise.ckpt"
    torch.save(payload, checkpoint, pickle_module=dill)

    def unexpected_instantiation(*args, **kwargs):
        raise AssertionError("Invalid noise semantics must fail before policy instantiation")

    monkeypatch.setattr(hydra.utils, "instantiate", unexpected_instantiation)
    with pytest.raises(ValueError, match="Incompatible RAFE noise checkpoint"):
        module.PPOActorDeployModel({
            "task_name": "smoke", "alg_name": "rafe", "addition_info": "test",
            "training_seed": 0, "ppo_ckpt_path": str(checkpoint), "device": "cpu",
        })
