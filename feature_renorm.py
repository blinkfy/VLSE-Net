from __future__ import annotations

import torch
import torch.nn as nn


def _stats_2d(t: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = t.mean(dim=(2, 3), keepdim=True)
    var = t.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = (var + eps).sqrt()
    return mean, std


def match_input_stats_2d(
    x: torch.Tensor,
    ref: torch.Tensor,
    *,
    eps: float = 1e-5,
    detach_stats: bool = True,
) -> torch.Tensor:
    """Re-normalize x to match per-sample per-channel stats of ref.

    Both x and ref are expected to be NCHW.

    This is a lightweight way to prevent feature distribution drift after fusion
    (e.g., FiLM/conditioning), while keeping the *content* changes.
    """
    if x.ndim != 4 or ref.ndim != 4:
        raise ValueError(f"Expected NCHW tensors, got x.ndim={x.ndim}, ref.ndim={ref.ndim}.")
    if x.shape[:2] != ref.shape[:2]:
        raise ValueError(
            f"Channel mismatch: x.shape={tuple(x.shape)}, ref.shape={tuple(ref.shape)}."
        )

    x_mean, x_std = _stats_2d(x, eps=eps)
    ref_mean, ref_std = _stats_2d(ref, eps=eps)

    if detach_stats:
        x_mean = x_mean.detach()
        x_std = x_std.detach()
        ref_mean = ref_mean.detach()
        ref_std = ref_std.detach()

    x_norm = (x - x_mean) / x_std.clamp_min(eps)
    return x_norm * ref_std + ref_mean


def match_input_std_2d(
    x: torch.Tensor,
    ref: torch.Tensor,
    *,
    eps: float = 1e-5,
    detach_stats: bool = True,
) -> torch.Tensor:
    """Match only the per-channel std of ref, but keep x's mean.

    This is often a better fit for FiLM-style fusion because the mean shift (beta)
    can carry useful semantic information; hard-matching the mean can wash it out.
    """
    if x.ndim != 4 or ref.ndim != 4:
        raise ValueError(f"Expected NCHW tensors, got x.ndim={x.ndim}, ref.ndim={ref.ndim}.")
    if x.shape[:2] != ref.shape[:2]:
        raise ValueError(
            f"Channel mismatch: x.shape={tuple(x.shape)}, ref.shape={tuple(ref.shape)}."
        )

    x_mean, x_std = _stats_2d(x, eps=eps)
    _, ref_std = _stats_2d(ref, eps=eps)

    if detach_stats:
        x_mean = x_mean.detach()
        x_std = x_std.detach()
        ref_std = ref_std.detach()

    x_norm = (x - x_mean) / x_std.clamp_min(eps)
    return x_norm * ref_std + x_mean


def apply_feature_renorm_2d(
    x: torch.Tensor,
    ref: torch.Tensor,
    *,
    mode: str,
    eps: float = 1e-5,
    detach_stats: bool = True,
    alpha: float = 1.0,
) -> torch.Tensor:
    if mode == "none":
        return x
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if mode == "match_input_stats":
        y = match_input_stats_2d(x, ref, eps=eps, detach_stats=detach_stats)
    elif mode == "match_input_std":
        y = match_input_std_2d(x, ref, eps=eps, detach_stats=detach_stats)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if float(alpha) == 1.0:
        return y
    return (1.0 - float(alpha)) * x + float(alpha) * y


class FeatureReNormalization2d(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "match_input_stats",
        eps: float = 1e-5,
        detach_stats: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"match_input_stats", "match_input_std"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.mode = str(mode)
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return apply_feature_renorm_2d(
            x,
            ref,
            mode=self.mode,
            eps=self.eps,
            detach_stats=self.detach_stats,
            alpha=self.alpha,
        )
