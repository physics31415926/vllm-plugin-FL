# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import logging

_logger = logging.getLogger(__name__)

_modules = [
    "kda",
    "lora",
    "chunk_delta_h",
    "rejection_sampler",
    "eagle",
    "topk_topp_sampler",
    "mrv2",
]

for _name in _modules:
    try:
        __import__(f".{_name}", globals(), locals(), ["*"], level=1)
    except Exception as e:
        _logger.debug("Skipping triton_support patch %s: %s", _name, e)
