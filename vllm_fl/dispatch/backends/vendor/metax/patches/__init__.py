# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

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
from . import distributed  # noqa: F401
from . import device_allocator  # noqa: F401
from . import model_executor  # noqa: F401
from . import quant_kernels  # noqa: F401
from . import triton_support  # noqa: F401
from . import chores  # noqa: F401
from . import optimizations  # noqa: F401
from . import MRV2  # noqa: F401
from . import transformers_utils  # noqa: F401
# from . import dp_fix  # noqa: F401
# from . import lora  # noqa: F401

# --- OOT registrations (customized ops/layers) ---
from .. import customized  # noqa: F401
