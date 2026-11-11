"""Manipulation FPO++ objective, adapted from fpo-control's loss equations.

Reference: manipulation_experiments/{src/flow_model.py,finetune_online_rl.py}
at 8f460e4415a2a6debfbeb689093f5e36140bd178. No simulator code is vendored.
"""

import torch


def modified_huber(error, delta=0.5):
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    absolute = error.abs()
    return torch.where(absolute <= delta, error.square(), 2 * delta * absolute - delta**2)


def ste_clamp(value, minimum=None, maximum=None):
    return value + (value.clamp(min=minimum, max=maximum) - value).detach()


def surrogate_loss(old_loss, current_loss, mask, advantages, clip_coef=0.01,
                   old_loss_cap=4.0, logratio_cap=5.0, average_chunk=False):
    """Inputs [B,N,H], [B,N,H], [B,H], [B]; ratios retain MC axis N."""
    if old_loss.shape != current_loss.shape or current_loss.ndim != 3:
        raise ValueError("CFM losses must have the same [B,N,H] shape")
    if mask.shape != (current_loss.shape[0], current_loss.shape[2]):
        raise ValueError("Execution mask must have shape [B,H]")
    if advantages.shape != (current_loss.shape[0],):
        raise ValueError("Advantages must have shape [B]")
    if not 0 < clip_coef < 1:
        raise ValueError("clip_coef must be in (0,1)")
    old = old_loss.detach().float()
    current = current_loss.float()
    if old_loss_cap is not None:
        old = old.clamp(max=float(old_loss_cap))
    weights = mask.detach().float().unsqueeze(1)
    old = (old * weights).sum(-1)
    current = (current * weights).sum(-1)
    if average_chunk:
        denominator = weights.sum(-1).clamp_min(1)
        old, current = old / denominator, current / denominator
    raw_logratio = old - current
    logratio = raw_logratio
    if logratio_cap is not None:
        logratio = ste_clamp(logratio, -float(logratio_cap), float(logratio_cap))
    ratio = logratio.exp()
    advantage = advantages.detach().float().unsqueeze(1)
    losses = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1-clip_coef, 1+clip_coef))
    valid = mask.sum(-1) > 0
    # Empty execution windows have no policy-gradient contribution.
    loss = losses[valid].mean() if valid.any() else current.sum() * 0
    metrics = {
        "actor_loss": loss.detach(),
        "ratio_mean": ratio[valid].mean().detach() if valid.any() else ratio.new_tensor(1),
        "clip_fraction": ((ratio[valid]-1).abs() > clip_coef).float().mean().detach()
                         if valid.any() else ratio.new_tensor(0),
        "raw_logratio_abs": raw_logratio.abs().mean().detach(),
        "old_cfm_mean": old_loss.detach().float().mean(),
        "current_cfm_mean": current_loss.detach().float().mean(),
    }
    return loss, metrics
