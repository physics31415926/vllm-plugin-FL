# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for METAX GPUs (MACA platform).
Attention backends are registered here and their implementations live in
vllm_metax (C extensions) or in this plugin's impl/ directory (pure Python).
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


# Register attention backends for MACA.
# Pure-Python backends implemented in this plugin use plugin-local class_path.
# Backends that depend on vllm_metax C extensions use vllm_metax class_path.
def register_attention_backends():
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        backend=AttentionBackendEnum.FLASHMLA_SPARSE,
        class_path="vllm_metax.v1.attention.backends.mla.flashmla_sparse.MacaFlashMLASparseBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.TRITON_MLA,
        class_path="vllm_metax.v1.attention.backends.mla.triton_mla.MacaTritonMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.FLASHINFER,
        class_path="vllm_metax.v1.attention.backends.flashinfer.MacaFlashInferBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.TRITON_ATTN,
        class_path="vllm_metax.v1.attention.backends.triton_attn.MacaTritonAttentionBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.TREE_ATTN,
        class_path="vllm_metax.v1.attention.backends.tree_attn.MacaTreeAttentionBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.FLEX_ATTENTION,
        class_path="vllm_metax.v1.attention.backends.flex_attention.MacaFlexAttentionBackend",
    )
    register_backend(
        backend=AttentionBackendEnum.TURBOQUANT,
        class_path="vllm_metax.v1.attention.backends.turboquant_attn.MacaTurboQuantAttentionBackend",
    )


class MacaBackend(Backend):
    """
    METAX backend for operator implementations.

    This backend uses MACA libraries to provide high-performance
    operator implementations for METAX GPUs.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "maca"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        """Check if Metax hardware and libraries are available."""
        if MacaBackend._available is None:
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    MacaBackend._available = True
                else:
                    MacaBackend._available = False
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_maca
        return silu_and_mul_maca(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import gelu_and_mul_maca
        return gelu_and_mul_maca(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.layernorm import rms_norm_maca
        return rms_norm_maca(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.rotary_embedding import rotary_embedding_maca
        return rotary_embedding_maca(
            obj, query, key, cos, sin, position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Select the attention backend class path for MACA.

        Priority (mirrors vllm_metax platform.py get_attn_backend logic):
        - use_mla + use_sparse  → FLASHMLA_SPARSE
        - use_mla               → FLASHMLA
        - default               → FLASH_ATTN
        Full priority-based selection (TRITON_MLA, FLASHINFER, etc.) is handled
        by vllm_metax.platform.MacaPlatformBase.get_attn_backend() when
        vllm_metax is the active platform plugin.
        """
        # Register all backends before selection so get_path() resolves correctly.
        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_maca
        return topk_softmax_maca(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )
