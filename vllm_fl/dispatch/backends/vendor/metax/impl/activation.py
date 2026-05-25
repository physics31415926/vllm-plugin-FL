# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication (PyTorch native)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.nn.functional.silu(x1) * x2


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """GELU activation followed by element-wise multiplication (PyTorch native)."""
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.nn.functional.gelu(x1, approximate=approximate) * x2
