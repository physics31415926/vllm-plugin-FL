# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

logger = logging.getLogger(__name__)
from vllm.platforms import current_platform


def _reshape_and_cache_flash_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Pure PyTorch fallback for reshape_and_cache_flash.

    KV cache layout: [num_blocks, block_size, num_heads, head_dim]
    key/value shape: [num_tokens, num_heads, head_dim]
    slot_mapping: [num_tokens] - maps each token to a flat slot index

    This implementation avoids boolean indexing and dynamic shapes so that it
    is compatible with CUDAGraph capture (no CPU-GPU sync).

    TODO(gems): Request FlagGems to implement a native Triton/MACA kernel
    for better performance.
    """
    num_tokens = slot_mapping.shape[0]
    block_size = key_cache.shape[1]

    # Clamp invalid slots (< 0) to 0 so we can compute indices without
    # dynamic shapes.  We will mask out the writes for these positions.
    clamped_slots = torch.clamp(slot_mapping, min=0)

    # Compute block index and offset within block
    block_indices = clamped_slots // block_size
    block_offsets = clamped_slots % block_size

    # Prepare key/value data (only first num_tokens rows are valid)
    k = key[:num_tokens].to(key_cache.dtype)
    v = value[:num_tokens].to(value_cache.dtype)

    # Build a mask for valid tokens (slot_mapping >= 0)
    valid_mask = (slot_mapping >= 0).unsqueeze(-1).unsqueeze(-1)
    # valid_mask shape: [num_tokens, 1, 1] to broadcast with [num_tokens, num_heads, head_dim]

    # Zero out invalid tokens so they don't corrupt the cache at slot 0
    k = k * valid_mask
    v = v * valid_mask

    # Scatter write to cache - invalid tokens write zeros to slot 0 (harmless)
    key_cache[block_indices, block_offsets] = k
    value_cache[block_indices, block_offsets] = v


if current_platform.is_out_of_tree():
    get_scheduler_metadata = None
    reshape_and_cache_flash = _reshape_and_cache_flash_pytorch
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )

    # Note: In maca this need to be None since
    # metax flash_attn api does not have parameter
    # for `fa_version`.
    return None


def flash_attn_supports_fp8() -> bool:
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8"
    )
    return False


def flash_attn_supports_sinks() -> bool:
    # maca fa2 supports sinks
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
