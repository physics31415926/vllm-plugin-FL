# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    return obj.forward_cuda(x)


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    return obj.forward_cuda(x)
