"""The AM and GM objectives from experiment_plan section 5."""

import torch


def group_advantages(rewards, group_size=8, delta=1e-6):
    grouped = rewards.float().reshape(-1, group_size)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (std + delta)
    return advantages.flatten().detach(), std.flatten().detach()


def sequence_log_ratio(delta, mask, operator):
    lengths = mask.sum(-1)
    if torch.any(lengths == 0):
        raise ValueError("A sequence must contain at least one generated token")
    if operator == "am":
        return (
            torch.logsumexp(delta.masked_fill(~mask, -torch.inf), -1)
            - lengths.to(delta.dtype).log()
        )
    if operator == "gm":
        return delta.masked_fill(~mask, 0).sum(-1) / lengths
    raise ValueError(operator)


def sequence_loss(
    current, old, reference, mask, advantages, operator, epsilon=0.2, beta=0.01
):
    # No hidden log-ratio clamp. Nonfinite objectives are a recorded run failure.
    current, old, reference = current.float(), old.float(), reference.float()
    log_ratio = sequence_log_ratio(current - old, mask, operator)
    ratio = log_ratio.exp()
    surrogate = torch.minimum(
        ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    )
    difference = current - reference
    token_kl = torch.expm1(-difference) + difference
    kl = token_kl.masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
    loss = -(surrogate - beta * kl).mean()
    gap = sequence_log_ratio(current - old, mask, "am") - sequence_log_ratio(
        current - old, mask, "gm"
    )
    return loss, {
        "kl": kl.detach().mean(),
        "clip_fraction": ((ratio < 1 - epsilon) | (ratio > 1 + epsilon))
        .float()
        .mean()
        .detach(),
        "ratio": ratio.detach(),
        "am_gm_log_gap": gap.detach().mean(),
    }
