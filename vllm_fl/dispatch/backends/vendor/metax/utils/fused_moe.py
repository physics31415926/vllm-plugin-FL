# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Local copy of vllm_metax/utils/fused_moe.py

from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as vllm_TritonExperts,
    fused_experts as vllm_fused_experts,
    logger,
)

from . import envs as mx_envs


def _get_mx_triton_experts():
    """Lazily import MetaX TritonExperts."""
    from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts as mx_TritonExperts,
    )
    return mx_TritonExperts


def _get_mx_fused_experts():
    """Lazily import MetaX fused_experts."""
    from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as mx_fused_experts,
    )
    return mx_fused_experts


def get_triton_experts_cls():
    if mx_envs.USE_VLLM_TRITON_EXPERT:
        logger.info(
            "Using vLLM's fused MoE implementation for debugging and comparison."
        )
        return vllm_TritonExperts
    return _get_mx_triton_experts()


def get_fused_experts_fn():
    if mx_envs.USE_VLLM_TRITON_EXPERT:
        logger.info(
            "Using vLLM's fused MoE implementation for debugging and comparison."
        )
        return vllm_fused_experts
    return _get_mx_fused_experts()
