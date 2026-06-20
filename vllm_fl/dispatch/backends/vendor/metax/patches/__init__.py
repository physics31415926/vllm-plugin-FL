# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import logging

_logger = logging.getLogger(__name__)

# --- Existing FL patches ---
from . import fix_standalone_compile  # noqa: F401
from . import pynccl_wrapper  # noqa: F401
from . import cuda_wrapper  # noqa: F401
from . import utils_patch  # noqa: F401
from . import chunk_delta_h  # noqa: F401

# --- vllm_metax hotfixes ---
from . import fix_cudagraph_sizes  # noqa: F401
from . import fix_compilation_backend  # noqa: F401

# --- vllm_metax patches ---
# These patches may depend on vllm modules (fused_moe, kernels.linear) that
# are partially initialized during import_kernels(). Wrap each in try/except
# so one failure doesn't prevent others from loading.
_metax_patches = [
    "distributed",
    "device_allocator",
    "model_executor",
    "quant_kernels",
    "triton_support",
    "chores",
    "optimizations",
    "MRV2",
    "transformers_utils",
]

for _name in _metax_patches:
    try:
        __import__(f".{_name}", globals(), locals(), ["*"], level=1)
    except Exception as e:
        _logger.debug("Skipping metax patch %s: %s", _name, e)

# --- OOT registrations (customized ops/layers) ---
# NOTE: customized is now registered in register_model() (general_plugins hook)
# to avoid circular imports with vllm.kernels during import_kernels().
