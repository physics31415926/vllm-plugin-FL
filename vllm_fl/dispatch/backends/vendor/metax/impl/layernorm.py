# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# TODO: rms_norm standalone function was removed in vLLM 0.20.2; use forward_native instead.
#       Can be revisited if vLLM re-exposes a standalone rms_norm function.
from vllm.model_executor.layers.layernorm import fused_add_rms_norm

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation.
    """
    add_residual = residual is not None
    if add_residual:
        return fused_add_rms_norm(x, residual, obj.weight, obj.variance_epsilon)
    else:
        # vLLM 0.20.2 removed standalone rms_norm(); use RMSNorm.forward_native as fallback
        return obj.forward_native(x)
