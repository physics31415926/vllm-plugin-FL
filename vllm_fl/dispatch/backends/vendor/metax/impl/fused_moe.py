# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX fused MoE operator implementations (PyTorch native).
"""

import torch


def topk_softmax_maca(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    """Top-k softmax gating (PyTorch native)."""
    k = topk_weights.shape[-1]
    topk_w, topk_idx = torch.topk(gating_output, k=k, dim=-1)
    topk_w = torch.softmax(topk_w.float(), dim=-1).to(topk_weights.dtype)

    if renormalize:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    topk_weights.copy_(topk_w)
    topk_indices.copy_(topk_idx)

    # Fill token_expert_indices (flat mapping)
    num_tokens = topk_indices.shape[0]
    token_expert_indices[:num_tokens * k] = topk_indices.reshape(-1)

    return topk_weights, topk_indices
