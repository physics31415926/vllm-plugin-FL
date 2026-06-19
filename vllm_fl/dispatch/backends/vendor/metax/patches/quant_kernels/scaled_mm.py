# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ----------------------------------------------------
# Note: add plugin option to cutlass kernel dispatch
# Deferred to avoid circular import with
# vllm.model_executor.kernels.linear during import_kernels()
# ----------------------------------------------------

import sys


def _patch():
    """Apply patch when vllm.model_executor.kernels.linear is fully loaded."""
    import importlib
    from typing import Any
    import torch

    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassInt8ScaledMMLinearKernel,
        Int8ScaledMMLinearLayerConfig,
    )

    from vllm.platforms import PlatformEnum

    from vllm import _custom_ops as ops
    from ...utils import envs as mx_envs

    _mctlass_modname = (
        "vllm_metax.model_executor.layers.quantization._python_api_ops"
        if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
        else "vllm_metax.model_executor.layers.quantization._cutlass_ops"
    )
    mctlass_ops: Any = importlib.import_module(_mctlass_modname)

    class MctlassScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
        @classmethod
        def is_supported(
            cls, compute_capability: int | None = None
        ) -> tuple[bool, str | None]:
            return True, None

        @classmethod
        def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
            return True, None

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

            symmetric = azp_adj is None
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                x.contiguous(), i_s, i_zp, symmetric=symmetric
            )

            if x_zp is not None:
                static = i_zp is not None
                azp = None if static else x_zp
                return mctlass_ops.cutlass_scaled_mm_azp(
                    x_q,
                    w_q,
                    scale_a=x_s,
                    scale_b=w_s,
                    out_dtype=x.dtype,
                    azp_adj=azp_adj,
                    azp=azp,
                    bias=bias,
                )
            return mctlass_ops.cutlass_scaled_mm(
                x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
            )

    import vllm.model_executor.kernels.linear

    vllm.model_executor.kernels.linear._POSSIBLE_INT8_KERNELS = {
        PlatformEnum.OOT: [MctlassScaledMMLinearKernel]
    }


if "vllm.model_executor.kernels.linear" in sys.modules:
    _patch()
else:
    import atexit as _atexit

    def _deferred_patch():
        try:
            if "vllm.model_executor.kernels.linear" in sys.modules:
                _patch()
        except Exception:
            pass

    _atexit.register(_deferred_patch)
