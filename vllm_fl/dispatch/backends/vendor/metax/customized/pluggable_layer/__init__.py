# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import logging

_logger = logging.getLogger(__name__)

try:
    from . import gdn_linear_attn  # noqa: F401
except (ImportError, AssertionError) as e:
    _logger.debug(f"Skipping gdn_linear_attn registration: {e}")
