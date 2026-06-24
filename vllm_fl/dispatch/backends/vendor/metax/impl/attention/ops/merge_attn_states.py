# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as _triton_merge_attn_states,
)


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    """Merge attention states using Triton kernel (no mcoplib dependency)."""
    return _triton_merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
    )
