# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------
# Note: vLLM's Triton topk_topp kernel fails to compile on MetaX
#       Triton backend (PassManager::run failed in ttgir stage).
#       Patch apply_top_k_top_p to always use PyTorch fallback.
#
# TODO(gems): Request FlagGems to implement a MetaX-compatible
#             topk_topp Triton kernel for better performance.
# _____________________________________________________

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
