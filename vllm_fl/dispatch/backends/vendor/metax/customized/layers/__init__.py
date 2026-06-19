# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import layers that don't depend on vllm_metax internal modules.
# mm_encoder_attention, sparse_attn_indexer, deepseek_v4_attention
# depend on vllm_metax.v1.attention and vllm_metax._custom_ops,
# so they require vllm_metax package to be installed.
import logging

logger = logging.getLogger(__name__)

try:
    from . import mm_encoder_attention  # noqa: F401
except (ImportError, AssertionError) as e:
    logger.debug(f"Skipping mm_encoder_attention registration: {e}")

try:
    from . import sparse_attn_indexer  # noqa: F401
except (ImportError, AssertionError) as e:
    logger.debug(f"Skipping sparse_attn_indexer registration: {e}")

try:
    from . import minimax_text01_rmsnorm  # noqa: F401
except (ImportError, AssertionError) as e:
    logger.debug(f"Skipping minimax_text01_rmsnorm registration: {e}")
