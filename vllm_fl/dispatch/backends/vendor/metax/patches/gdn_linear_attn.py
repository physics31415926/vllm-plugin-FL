# SPDX-License-Identifier: Apache-2.0
# Fix for MetaX / torch 2.8: `hasattr(self, "in_proj_qkv")` in
# GatedDeltaNetAttention.forward_cuda() causes TorchDynamo to generate an
# attribute-existence guard.  When the guard is evaluated, dynamo calls
# `getattr(obj, "in_proj_qkv")` directly (without try/except), which raises
# AttributeError on objects constructed with create_in_proj_qkvz=True.
# torch 2.8 does not catch this inside guards.py:eval(), so it surfaces as
# InternalTorchDynamoError.
#
# Fix: register MacaGatedDeltaNetAttention as an OOT subclass that:
#   1. stores create_in_proj_qkvz as self.create_in_proj_qkvz in __init__
#   2. overrides forward_cuda, replacing `hasattr(self, "in_proj_qkv")` with
#      `not self.create_in_proj_qkvz` — a compile-time-constant bool that
#      dynamo statically specialises without generating attribute guards.
#
# NOTE: forward() dispatches via self._forward_method which is bound to
# self.forward_cuda in __init__ (for non-XPU platforms).  We therefore must
# override forward_cuda, not forward.
#
# Ported from vllm_metax/customized/pluggable_layer/gdn_linear_attn.py.
# Difference: uses _encode_layer_name(self.prefix) for gdn_attention_core
# (upstream convention); vllm_metax re-registers the op with bare self.prefix.
#
# TODO: remove once MetaX ships pytorch >= 2.10 where dynamo handles
#       attribute-existence guards correctly.

import torch
from einops import rearrange

from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GatedDeltaNetAttention,
    _encode_layer_name,
)
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.config import VllmConfig


@GatedDeltaNetAttention.register_oot
class MacaGatedDeltaNetAttention(GatedDeltaNetAttention):
    """MetaX-safe subclass of GatedDeltaNetAttention.

    Overrides forward_cuda to replace the `hasattr(self, "in_proj_qkv")`
    branch with a stored bool flag, preventing TorchDynamo from generating
    an attribute-existence guard that crashes on torch 2.8.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        create_in_proj_qkvz: bool = True,
        gqa_interleaved_layout: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            vllm_config=vllm_config,
            prefix=prefix,
            create_in_proj_qkvz=create_in_proj_qkvz,
            gqa_interleaved_layout=gqa_interleaved_layout,
        )
        # Store as a plain bool so dynamo can statically specialise instead of
        # generating a hasattr() guard.
        self.create_in_proj_qkvz = create_in_proj_qkvz

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Identical to upstream forward_cuda except hasattr replaced by flag."""
        num_tokens = hidden_states.size(0)

        # ------------------------------------------------------------------ #
        # Part 1: Input Projection
        # ------------------------------------------------------------------ #
        if not self.create_in_proj_qkvz:
            # LoRA path (Qwen3.5 only): in_proj_qkv and in_proj_z are separate
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba
                )
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value),
                )
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ------------------------------------------------------------------ #
        # Part 2: Core Attention (custom op)
        # ------------------------------------------------------------------ #
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        # ------------------------------------------------------------------ #
        # Part 3: Output Projection
        # ------------------------------------------------------------------ #
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
