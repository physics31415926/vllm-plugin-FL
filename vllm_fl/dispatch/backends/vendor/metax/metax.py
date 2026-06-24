# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


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
    """METAX (MACA) backend operator implementations."""

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "maca"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        if MacaBackend._available is None:
            try:
                MacaBackend._available = (
                    torch.cuda.is_available() and torch.cuda.device_count() > 0
                )
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        return AttentionBackendEnum.FLASH_ATTN.get_path()
