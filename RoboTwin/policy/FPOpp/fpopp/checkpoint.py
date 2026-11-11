"""Strict checkpoint-native ManiFlow loading and atomic FPO++ saves."""

from pathlib import Path
import os
import random

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

FORMAT = "fpopp_maniflow_v1"


def load_payload(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(str(path), map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)


def policy_configuration(payload):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    if "policy_cfg" in payload:
        result = OmegaConf.create(payload["policy_cfg"])
    elif "cfg" in payload:
        cfg = OmegaConf.create(payload["cfg"])
        result = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
    else:
        raise ValueError("Checkpoint must contain its saved policy configuration")
    expected = "maniflow.policy.maniflow_pointcloud_policy.ManiFlowTransformerPointcloudPolicy"
    if result.get("_target_") != expected:
        raise ValueError(f"Unsupported checkpoint policy: {result.get('_target_')}")
    return result


def restore_normalizer(policy, state):
    nested = {}
    prefix = "normalizer.params_dict."
    for key, value in state.items():
        if key.startswith(prefix):
            fields = key[len(prefix):].split(".")
            group = nested
            for field in fields[:-1]:
                group = group.setdefault(field, {})
            group[fields[-1]] = value

    def convert(group):
        return nn.ParameterDict({key: convert(value) if isinstance(value, dict)
                                 else nn.Parameter(value.detach().clone(), requires_grad=False)
                                 for key, value in group.items()})
    if not all(field in nested for field in ("action", "agent_pos", "point_cloud")):
        raise ValueError("Checkpoint is missing action/agent_pos/point_cloud normalizer parameters")
    policy.normalizer.params_dict = convert(nested)
    policy.normalizer.requires_grad_(False)


def load_policy(payload, state_key="auto", device="cpu"):
    policy_cfg = policy_configuration(payload)
    states = payload.get("state_dicts", {})
    if state_key == "auto":
        state_key = "ema_model" if payload.get("format") != FORMAT and "ema_model" in states else "model"
    if state_key not in states:
        raise KeyError(f"Checkpoint has no state_dicts.{state_key}; available: {list(states)}")
    policy = hydra.utils.instantiate(policy_cfg)
    state = states[state_key]
    # Bypass ManiFlow's legacy allowance for randomly initialized plan keys.
    nn.Module.load_state_dict(policy, state, strict=True)
    restore_normalizer(policy, state)
    return policy.to(device), policy_cfg, state_key


def rng_state(actor):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "actor": actor.rng.get_state()}


def restore_rng(state, actor):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    actor.rng.set_state(state["actor"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary, pickle_module=dill)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
