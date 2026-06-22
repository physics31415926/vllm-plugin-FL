# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Patch apply_top_k_top_p to use PyTorch path instead of Triton.
#
# Root cause: MetaX's Triton backend passes is_active() (so HAS_TRITON=True),
# but the PassManager fails when compiling _topk_topp_kernel on MetaX hardware.
# Patching HAS_TRITON directly is too late — topk_topp_sampler.py already
# consumed it at import time. Instead we monkey-patch the function itself.
#
# Three call sites must be patched because each module binds apply_top_k_top_p
# at import time via a module-level reference.
#
# TODO: remove once MetaX Triton backend supports _topk_topp_kernel compilation.

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return apply_top_k_top_p_pytorch(logits, k, p)


import vllm.v1.sample.ops.topk_topp_sampler as _m1
_m1.apply_top_k_top_p = apply_top_k_top_p

try:
    import vllm.v1.sample.rejection_sampler as _m2
    _m2.apply_top_k_top_p = apply_top_k_top_p
except Exception:
    pass  # module may not exist in all vllm 0.20.x builds

try:
    import vllm.v1.worker.gpu.sample.states as _m3
    _m3.apply_top_k_top_p = apply_top_k_top_p
except Exception:
    pass  # module may not exist in all vllm 0.20.x builds
