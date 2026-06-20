# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# Deferred imports: these patches depend on modules (fused_moe.oracle,
# kernels.linear) that may be partially initialized during import_kernels().
# scaled_mm and mp_linear_kernel already handle deferral internally.
# fp8 needs the same treatment.
import sys
import logging

_logger = logging.getLogger(__name__)

from . import scaled_mm  # noqa: F401 (self-defers)
from . import mp_linear_kernel  # noqa: F401 (self-defers)

# fp8 imports vllm.model_executor.layers.fused_moe.oracle.fp8 which chains
# into kernels.linear. Defer if kernels.linear is not fully loaded.
if "vllm.model_executor.kernels.linear" in sys.modules:
    try:
        from . import fp8  # noqa: F401
    except Exception as e:
        _logger.debug(f"Skipping quant_kernels.fp8 patch: {e}")
else:
    import atexit as _atexit

    def _deferred_fp8_patch():
        try:
            if "vllm.model_executor.kernels.linear" in sys.modules:
                from . import fp8  # noqa: F401
        except Exception as e:
            _logger.debug(f"Skipping deferred quant_kernels.fp8 patch: {e}")

    _atexit.register(_deferred_fp8_patch)
