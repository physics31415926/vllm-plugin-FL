# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def rotary_embedding_maca(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding via obj.forward_cuda.
    forward_cuda handles slicing, cos/sin lookup, and kernel dispatch internally.
    """
    return obj.forward_cuda(positions, query, key)
