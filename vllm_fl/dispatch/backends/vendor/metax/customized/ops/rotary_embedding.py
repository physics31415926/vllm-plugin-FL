# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.xdrope import XDRotaryEmbedding


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


for _cls in [RotaryEmbedding, MRotaryEmbedding, XDRotaryEmbedding]:
    _register(_cls)
