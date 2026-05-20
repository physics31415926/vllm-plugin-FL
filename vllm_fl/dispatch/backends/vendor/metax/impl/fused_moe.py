# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX fused MoE operator implementations using FlagGems.
"""


def topk_softmax_maca(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from flag_gems import topk_softmax

    try:
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
    except Exception:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices
