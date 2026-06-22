# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation.
    """
    from vllm import _custom_ops as ops

    if residual is not None:
        # fused_add_rms_norm mutates x and residual in-place, returns None
        ops.fused_add_rms_norm(x, residual, obj.weight, obj.epsilon)
        return x, residual
    else:
        out = torch.empty_like(x)
        ops.rms_norm(out, x, obj.weight, obj.epsilon)
        return out
