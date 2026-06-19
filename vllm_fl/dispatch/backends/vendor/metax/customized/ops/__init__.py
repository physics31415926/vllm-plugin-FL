# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import ops that FL doesn't register via its own OOT_OPS dict.
# FL already registers: SiluAndMul, GeluAndMul, RMSNorm, RotaryEmbedding
#
# Each module uses guards to skip if vllm_metax already registered.
from . import apply_rotary_embedding  # noqa: F401
from . import activation  # noqa: F401
from . import layernorm  # noqa: F401
from . import rotary_embedding  # noqa: F401
