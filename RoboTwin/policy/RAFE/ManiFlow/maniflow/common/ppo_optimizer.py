"""Actor/noise learning rates and lossless migration of legacy PPO AdamW state."""

import math

import torch


def build_actor_optimizer(actor, config, include_noise=True):
    actor_lr = float(config["actor_lr"])
    noise_lr = config.get("noise_lr")
    # Archived configurations without this option retain their original rate.
    noise_lr = actor_lr if noise_lr is None else float(noise_lr)
    for name, rate in (("actor_lr", actor_lr), ("noise_lr", noise_lr)):
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"rl.ppo.{name} must be finite and nonnegative, got {rate}")

    groups = [
            {"name": "actor", "params": [p for p in actor.base_actor.parameters() if p.requires_grad],
             "lr": actor_lr},
        ]
    if include_noise:
        groups.append({"name": "noise", "params": [p for p in actor.noise_head.parameters() if p.requires_grad],
                       "lr": noise_lr})
    return torch.optim.AdamW(
        groups,
        lr=actor_lr,
        weight_decay=float(config["actor_weight_decay"]),
        betas=tuple(config["betas"]),
    )


def load_actor_optimizer_state(optimizer, saved):
    """Restore all Adam moments, but keep the current configuration's LRs.

    Old actor_parameters() yielded trainable base_actor parameters followed by
    noise_head parameters. Splitting that exact sequence changes only parameter
    groups, not parameter IDs or the corresponding saved Adam state.
    """
    current = optimizer.param_groups
    names = [group.get("name") for group in current]
    if names not in (["actor", "noise"], ["actor"]):
        raise ValueError("Expected named actor/noise optimizer parameter groups")
    rates = [group["lr"] for group in current]
    groups = saved["param_groups"]
    if names == ["actor"] and [group.get("name") for group in groups] != ["actor"]:
        raise ValueError("RAFE actor-only optimizer requires a matching expert-interpolation resume checkpoint")
    migrated = len(groups) == 1 and names == ["actor", "noise"]
    if migrated:
        legacy = groups[0]
        sizes = [len(group["params"]) for group in current]
        if len(legacy["params"]) != sum(sizes):
            raise ValueError("Legacy actor optimizer parameter count differs; check architecture/frozen parameters")
        offset = 0
        groups = []
        for target, size in zip(current, sizes):
            group = dict(legacy)
            group.update(name=target["name"], params=legacy["params"][offset:offset + size])
            groups.append(group)
            offset += size
    elif [group.get("name") for group in groups] != names:
        raise ValueError("Saved actor optimizer groups do not match actor/noise ordering")

    for source, target in zip(groups, current):
        if len(source["params"]) != len(target["params"]):
            raise ValueError(f"Actor optimizer parameter count differs for group {target['name']}")
        for saved_id, parameter in zip(source["params"], target["params"]):
            state = saved["state"].get(saved_id, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError(f"Actor optimizer {key} shape differs for group {target['name']}")

    optimizer.load_state_dict({**saved, "param_groups": groups})
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate
        if "initial_lr" in group:
            group["initial_lr"] = rate
    return migrated
