# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Uses FlagGems unified operator library which provides optimized
    implementations for MetaX GPUs via MACA Triton compatibility.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Uses FlagGems unified operator library which provides optimized
    implementations for MetaX GPUs via MACA Triton compatibility.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.fused import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)
