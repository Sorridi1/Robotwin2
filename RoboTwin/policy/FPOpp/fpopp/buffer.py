"""Owned CPU storage for FPO++ chunks; no denoising chains or log-probs."""

import torch


def cpu_copy(value):
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if value is None:
        return None
    return torch.as_tensor(value).detach().to("cpu").clone()


def chunk_gae(rewards, values, terminated, truncated, last_values, gamma=0.99, gae_lambda=0.95):
    if rewards.shape != values.shape or rewards.ndim != 2:
        raise ValueError("Rewards/values must have matching [T,E] shapes")
    done = terminated.bool() | truncated.bool()
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_values if step == rewards.shape[0]-1 else values[step+1]
        alive = (~done[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * alive * next_value - values[step]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[step] = carry
    return advantages, advantages + values


class FPOBuffer:
    def __init__(self):
        self.records = []

    def add(self, **record):
        self.records.append(cpu_copy(record))

    def finish(self, last_values, gamma=0.99, gae_lambda=0.95, reward_scaler=None, reward_scale=1.0):
        if not self.records:
            raise ValueError("Cannot update from an empty rollout")
        rewards = torch.stack([r["reward"] for r in self.records]).float()
        values = torch.stack([r["value"] for r in self.records]).float()
        terminated = torch.stack([r["terminated"] for r in self.records]).bool()
        truncated = torch.stack([r["truncated"] for r in self.records]).bool()
        if reward_scaler is not None:
            firsts = torch.zeros_like(rewards)
            # Every collected rollout starts with a reset in this implementation.
            firsts[0] = 1
            firsts[1:] = (terminated | truncated)[:-1].float()
            rewards = torch.from_numpy(reward_scaler(rewards.numpy(), firsts.numpy()))
        advantage, returns = chunk_gae(rewards * reward_scale, values, terminated, truncated,
                                       cpu_copy(last_values), gamma, gae_lambda)
        output = {}
        for key in self.records[0]:
            if key == "obs":
                output[key] = {field: torch.stack([r[key][field] for r in self.records]).flatten(0, 1)
                               for field in self.records[0][key]}
            elif self.records[0][key] is not None:
                output[key] = torch.stack([r[key] for r in self.records]).flatten(0, 1)
        output["advantage"], output["returns"] = advantage.flatten(), returns.flatten()
        valid = output["mask"].sum(-1) > 0
        output = {key: ({field: item[valid] for field, item in value.items()} if isinstance(value, dict)
                        else value[valid]) for key, value in output.items()}
        if not valid.any():
            raise RuntimeError("Rollout executed no actions; cannot perform an FPO++ update")
        adv = output["advantage"]
        output["advantage"] = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        return output
