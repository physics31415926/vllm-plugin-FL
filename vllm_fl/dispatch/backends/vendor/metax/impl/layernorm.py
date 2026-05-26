# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# vLLM 0.20.2 removed the standalone rms_norm() function from layernorm.py.
# We provide a PyTorch-native fallback and use fused_add_rms_norm when available.
# TODO: remove this workaround when vLLM restores a standalone rms_norm() API.

import torch


def _rms_norm_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """PyTorch-native RMS normalization (fallback for MetaX MACA)."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return (weight * x).to(orig_dtype)


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation.
    Falls back to PyTorch-native when vLLM's fused ops are unavailable.
    """
    add_residual = residual is not None
    if add_residual:
        try:
            from vllm.model_executor.layers.layernorm import fused_add_rms_norm
            return fused_add_rms_norm(x, residual, obj.weight, obj.epsilon)
        except (ImportError, AttributeError):
            # fused_add_rms_norm not available — use PyTorch native
            x = x + residual
            out = _rms_norm_pytorch(x, obj.weight, obj.epsilon)
            return out, x
    else:
        return _rms_norm_pytorch(x, obj.weight, obj.epsilon)
