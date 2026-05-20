# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using FlagGems for MetaX MACA backend.

    Uses FlagGems unified operator library which provides optimized
    implementations for MetaX GPUs via MACA Triton compatibility.
    """
    from flag_gems.modules.normalization import gems_rms_forward

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)
