# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX rotary embedding operator implementations.
"""


def topk_softmax_maca(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from vllm._custom_ops import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_indices


def invoke_fused_moe_triton_kernel_maca(
    A,
    B,
    C,
    A_scale,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
    compute_type,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    block_shape=None,
    B_bias=None,
):
    """
    Invoke MetaX's precompiled fused MoE triton kernel via mcoplib.

    FlagGems fused_moe_kernel uses 3D tl.trans which is incompatible with
    MACA triton backend (PassManager::run failed). This routes to MetaX's
    own implementation instead.
    """
    from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel as _metax_invoke_fused_moe_kernel,
    )

    _metax_invoke_fused_moe_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
    )
