# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# Fix: AttributeError: module 'torch.accelerator' has no attribute 'empty_cache'
# vllm 0.20+ calls torch.accelerator.empty_cache() but MetaX's torch build
# does not expose this attribute. Patch it to fall back to torch.cuda.empty_cache().
# TODO: Remove when MetaX torch build exposes torch.accelerator.empty_cache.

import torch


def _patch_accelerator_empty_cache() -> None:
    if not hasattr(torch, "accelerator"):
        return
    if hasattr(torch.accelerator, "empty_cache"):
        return
    # Fall back to cuda empty_cache which works on MACA
    torch.accelerator.empty_cache = torch.cuda.empty_cache


_patch_accelerator_empty_cache()
