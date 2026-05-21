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

    TODO(gems): Request FlagGems to implement a native Triton/MACA kernel
    for better performance.
    """
    num_tokens = slot_mapping.shape[0]
    block_size = key_cache.shape[1]

    # Filter out padding tokens (slot_mapping < 0)
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    valid_key = key[:num_tokens][valid_mask]
    valid_value = value[:num_tokens][valid_mask]

    # Compute block index and offset within block
    block_indices = valid_slots // block_size
    block_offsets = valid_slots % block_size

    # Write to cache
    # key_cache[block_indices, block_offsets] = valid_key
    # value_cache[block_indices, block_offsets] = valid_value
    key_cache[block_indices, block_offsets] = valid_key.to(key_cache.dtype)
    value_cache[block_indices, block_offsets] = valid_value.to(
        value_cache.dtype
    )


if current_platform.is_out_of_tree():
    get_scheduler_metadata = None
    reshape_and_cache_flash = _reshape_and_cache_flash_pytorch
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401

    get_scheduler_metadata = None


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
