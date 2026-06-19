# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------------
# Note: This patch is to add mp_linear_kernel for oot dispatch
# -----------------------------------------------------------

# Deferred import to avoid circular import with
# vllm.model_executor.kernels.linear during import_kernels()

import sys

def _patch():
    """Apply patch when vllm.model_executor.kernels.linear is fully loaded."""
    from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
        ExllamaLinearKernel as vllm_ExllamaLinearKernel,
        MPLinearLayerConfig,
    )
    from vllm.platforms import PlatformEnum, current_platform

    import torch

    # --------------------------------
    # ensure maca gptq is registered
    import vllm_metax.quant_config.gptq  # noqa F401


    class MacaExllamaLinearKernel(vllm_ExllamaLinearKernel):
        @classmethod
        def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
            if not current_platform.is_cuda_alike():
                return (
                    False,
                    "Exllama is only supported on CUDA and ROCm and Maca",
                )

            if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
                return (
                    False,
                    "Act reordering currently not supported by Exllama, "
                    "when the input features are partitioned across "
                    "devices",
                )

            if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
                return (
                    False,
                    "Output features must be a multiple of the pack "
                    "factor (32 / num_bits) so that we can correctly "
                    "pack the zero points",
                )

            # ------------------------------------------------
            # On maca we support both float16 and bfloat16
            # ------------------------------------------------
            if c.act_type not in (torch.float16, torch.bfloat16):
                return False, "Exllama only supports float16 and bfloat16 activations"

            if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
                return (
                    False,
                    f"Quant type ({c.weight_type}) not supported by "
                    "Exllama, supported types are: "
                    f"{cls.SUPPORTED_QUANT_TYPES}",
                )

            if c.full_weight_shape[0] % c.group_size != 0:
                return (
                    False,
                    f"Group size ({c.group_size}) does not evenly divide"
                    " the number of input features "
                    f"({c.full_weight_shape[0]})",
                )

            return True, None

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            c = self.config

            w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

            assert w_zp is not None, "Zero points are required by Exllama"
            assert w_g_idx is not None, "Group index is required by Exllama"

            return torch.ops.vllm._apply_gptq(
                x,
                w_q,
                w_s,
                w_zp,
                bias,
                w_g_idx,
                True,
                c.weight_type.size_bits,
                c.group_size,
                c.has_g_idx,
            )

    import vllm.model_executor.kernels.linear

    vllm.model_executor.kernels.linear._POSSIBLE_KERNELS = {
        PlatformEnum.OOT: [MacaExllamaLinearKernel]
    }


# Only apply if vllm.model_executor.kernels.linear is already fully loaded
if "vllm.model_executor.kernels.linear" in sys.modules:
    _patch()
else:
    # Register a post-import hook via import the module later
    # The patch will be applied when quant_linear.py calls register_quant_kernel_dicts()
    import atexit as _atexit

    def _deferred_patch():
        try:
            if "vllm.model_executor.kernels.linear" in sys.modules:
                _patch()
        except Exception:
            pass

    _atexit.register(_deferred_patch)
