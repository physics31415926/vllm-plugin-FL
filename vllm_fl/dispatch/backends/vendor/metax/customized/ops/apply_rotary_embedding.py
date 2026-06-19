# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
import torch


def _register():
    """Register MacaApplyRotaryEmb, skip if vllm_metax already registered it."""
    from vllm.model_executor.custom_op import op_registry_oot
    if "ApplyRotaryEmb" in op_registry_oot:
        return

    @ApplyRotaryEmb.register_oot
    class MacaApplyRotaryEmb(ApplyRotaryEmb):
        def forward_oot(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> torch.Tensor:
            """Same as forward_cuda"""
            from flash_attn.layers.rotary import apply_rotary_emb

            x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

            interleaved = not self.is_neox_style
            output = apply_rotary_emb(x, cos, sin, interleaved)

            output = self._post_process(output, origin_shape, origin_dtype)
            return output


_register()
