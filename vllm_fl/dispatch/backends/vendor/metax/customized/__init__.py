# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import OOT registrations that FL doesn't already cover.
# FL registers: SiluAndMul, GeluAndMul, RMSNorm, RotaryEmbedding, FusedMoE, UnquantizedFusedMoEMethod
# We register: ApplyRotaryEmb, extra activation/layernorm/rotary variants, pluggable layers
#
# NOTE: This module is imported from register_model() (general_plugins hook),
# NOT from import_kernels(). At that point all vllm modules (kernels, layers)
# are fully loaded, so we won't hit circular imports.
#
# NOTE: vllm_metax may already be installed and have registered these ops.
# All registrations must be idempotent (skip if already registered).

import logging

_logger = logging.getLogger(__name__)

_modules = [
    ".ops",
    ".layers",
    ".pluggable_layer",
]

for _mod in _modules:
    try:
        __import__(_mod, globals(), locals(), ["*"], level=1)
    except (AssertionError, ImportError) as e:
        # AssertionError: "Duplicate op name" when vllm_metax already registered
        # ImportError: missing optional dependencies
        _logger.debug("Skipping customized registration %s: %s", _mod, e)
