# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import ops that FL doesn't register via its own OOT_OPS dict.
# FL already registers: SiluAndMul, GeluAndMul, RMSNorm, RotaryEmbedding
from . import apply_rotary_embedding  # noqa: F401
