# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Only import OOT registrations that FL doesn't already cover.
# FL registers: SiluAndMul, GeluAndMul, RMSNorm, RotaryEmbedding, FusedMoE, UnquantizedFusedMoEMethod
# We register: ApplyRotaryEmb, GatedDeltaNetAttention, extra layers

from .ops import apply_rotary_embedding  # noqa: F401
from . import pluggable_layer  # noqa: F401
from . import layers  # noqa: F401
