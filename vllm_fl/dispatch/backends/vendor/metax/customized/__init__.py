# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import OOT registrations that FL doesn't already cover.
# FL registers: SiluAndMul, GeluAndMul, RMSNorm, RotaryEmbedding, FusedMoE, UnquantizedFusedMoEMethod
# We register: ApplyRotaryEmb, GatedDeltaNetAttention, extra layers
#
# NOTE: vllm_metax may already be installed and have registered these ops.
# All registrations must be idempotent (skip if already registered).

import logging

_logger = logging.getLogger(__name__)

_modules = [
    ".ops",
    ".pluggable_layer",
    ".layers",
]

for _mod in _modules:
    try:
        __import__(_mod, globals(), locals(), ["*"], level=1)
    except (AssertionError, ImportError) as e:
        # AssertionError: "Duplicate op name" when vllm_metax already registered
        # ImportError: missing optional dependencies
        _logger.debug("Skipping customized registration %s: %s", _mod, e)
