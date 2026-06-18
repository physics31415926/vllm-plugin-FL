# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Local copy of vllm_metax/v1/attention/backends/fa_utils.py

from vllm.logger import init_logger

logger = init_logger(__name__)

# Re-export flash_attn functions (installed on MetaX container)
try:
    from flash_attn import flash_attn_varlen_func  # noqa: F401
    from flash_attn import flash_attn_with_kvcache  # noqa: F401
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> int | None:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )
    return 2


def is_fa_version_supported(fa_version: int) -> bool:
    return fa_version == 2


def flash_attn_supports_fp8() -> bool:
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8"
    )
    return False


def flash_attn_supports_quant_query_input() -> bool:
    return False


def flash_attn_supports_sinks() -> bool:
    return True


def flash_attn_supports_mla():
    return False
