# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMS normalization (PyTorch native)."""
    weight = obj.weight
    eps = obj.variance_epsilon

    if residual is not None:
        x = x + residual
        residual_out = x

    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    out = (weight * x).to(weight.dtype)

    if residual is not None:
        return out, residual_out
    return out
