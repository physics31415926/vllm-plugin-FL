# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for METAX GPUs.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


# Register attention backends for MACA
def register_attention_backends():
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
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
                # Check if Metax device is available
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    MacaBackend._available = True
                else:
                    MacaBackend._available = False
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)
        - FLASHMLA_SPARSE (when use_mla and use_sparse are both True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # register before selection
        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()
