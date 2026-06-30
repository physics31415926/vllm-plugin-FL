# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from vllm/model_executor/layers/fused_moe/layer.py (v0.24.0)

import torch

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from vllm_fl.ops.fused_moe.router import (
    FusedTopKRouterFL,
    GroupedTopKRouterFL,
    FusedTopKBiasRouterFL,
)
from .fused_moe_utils import select_unquantized_moe_backend_oot


class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    """OOT replacement for UnquantizedFusedMoEMethod that routes computation
    through flaggems operators."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = (
            select_unquantized_moe_backend_oot(moe_config=self.moe)
        )

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if self.experts_cls is None:
                return True
            return self.experts_cls.is_monolithic()
        return self.moe_kernel.is_monolithic


def FusedMoEFL(*args, **kwargs) -> MoERunner:
    """
    OOT factory replacement for FusedMoE (vllm >= 0.24.0).

    In vllm 0.24.0, FusedMoE changed from a class to a factory function that
    returns a MoERunner instance.  FusedMoEFL mirrors this pattern: it
    delegates to the standard FusedMoE() factory and then replaces the router
    and quant_method on the returned MoERunner with FL-customised versions.

    Registration: op_registry_oot maps FusedMoE -> FusedMoEFL so that all
    MoE layers in a model use flaggems operators transparently.
    """
    # 1. Build the standard MoERunner via the upstream factory.
    runner: MoERunner = FusedMoE(*args, **kwargs)

    moe_config: FusedMoEConfig = runner.moe_config

    # 2. Replace quant_method with FL version.
    fl_quant_method = UnquantizedFusedMoEMethodFL(moe_config)
    runner._replace_quant_method(fl_quant_method)

    # 3. Replace router with FL version, copying all attributes from the
    #    original so we don't lose any configuration.
    original_router = runner.router

    def _get(obj, *attrs, default=None):
        """Helper: return the first attribute found on obj, or default."""
        for attr in attrs:
            val = getattr(obj, attr, _SENTINEL)
            if val is not _SENTINEL:
                return val
        return default

    _SENTINEL = object()

    if isinstance(original_router, GroupedTopKRouter):
        runner.router = GroupedTopKRouterFL(
            moe_config=moe_config,
            top_k=_get(original_router, "top_k"),
            num_expert_group=_get(original_router, "num_expert_group"),
            topk_group=_get(original_router, "topk_group"),
            scoring_func=_get(original_router, "scoring_func", default="softmax"),
            correction_bias=_get(original_router, "correction_bias"),
            routed_scaling_factor=_get(
                original_router, "routed_scaling_factor", default=1.0
            ),
        )
    elif isinstance(original_router, FusedTopKBiasRouter):
        runner.router = FusedTopKBiasRouterFL(
            moe_config=moe_config,
            top_k=_get(original_router, "top_k"),
            scoring_func=_get(original_router, "scoring_func", default="softmax"),
            correction_bias=_get(original_router, "correction_bias"),
            routed_scaling_factor=_get(
                original_router, "routed_scaling_factor", default=1.0
            ),
        )
    elif isinstance(original_router, FusedTopKRouter):
        runner.router = FusedTopKRouterFL(
            moe_config=moe_config,
            top_k=_get(original_router, "top_k"),
            scoring_func=_get(original_router, "scoring_func", default="softmax"),
            correction_bias=_get(original_router, "correction_bias"),
            routed_scaling_factor=_get(
                original_router, "routed_scaling_factor", default=1.0
            ),
        )
    # If the router type is unrecognised, leave it as-is (safe fallback).

    return runner


__all__ = ["FusedMoEFL", "UnquantizedFusedMoEMethodFL"]
