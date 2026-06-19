# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)


def _register(cls):
    """Register OOT class only if not already registered."""
    if cls.__name__ in op_registry_oot:
        return

    @cls.register_oot
    class _Maca(cls):
        def forward_oot(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

    _Maca.__name__ = f"Maca{cls.__name__}"
    _Maca.__qualname__ = f"Maca{cls.__name__}"


for _cls in [
    FatreluAndMul,
    SiluAndMul,
    SiluAndMulWithClamp,
    MulAndSilu,
    GeluAndMul,
    SwigluOAIAndMul,
    SwigluStepAndMul,
    NewGELU,
    FastGELU,
    QuickGELU,
]:
    _register(_cls)
