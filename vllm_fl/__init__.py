# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetaX MACA compatibility patches — applied at import time in ALL processes
# (main process, EngineCore, Worker_TP*) because worker processes don't go
# through the vLLM plugin register() call.
# ---------------------------------------------------------------------------
def _apply_metax_compat_patches():
    """Apply MetaX-specific compatibility patches.

    These patches fix gaps between vLLM 0.20.x API and MetaX PyTorch.
    Applied at import time so they run in ALL processes (main, EngineCore,
    Worker_TP*) — worker processes don't go through the vLLM plugin
    register() call.
    Safe to call multiple times (idempotent).
    """
    if not os.path.isdir("/opt/maca"):
        return  # Not a MetaX system — skip all patches.

    import torch

    # Patch 1: torch.accelerator.empty_cache() missing in MetaX PyTorch.
    # vllm_fl/worker/model_runner.py calls this during profiling/cleanup.
    # TODO: remove when MetaX PyTorch adds torch.accelerator.empty_cache.
    if not hasattr(torch.accelerator, "empty_cache"):
        torch.accelerator.empty_cache = torch.cuda.empty_cache
        logger.info("MetaX compat: patched torch.accelerator.empty_cache = torch.cuda.empty_cache")

    # Note: topk_topp_sampler.HAS_TRITON=False is patched in
    # vllm_fl/worker/model_runner.py immediately after the sampler import,
    # because that's the only place where the module is guaranteed to be
    # in sys.modules before it's used.


_apply_metax_compat_patches()
# ---------------------------------------------------------------------------


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _is_metax() -> bool:
    """Detect MetaX hardware without requiring CUDA device initialization.

    Uses multiple detection strategies in order of reliability:
    1. /opt/maca directory (MetaX MACA SDK install path)
    2. flag_gems DeviceDetector
    3. torch_maca module
    """
    import os
    # Most reliable: MACA SDK is always installed at /opt/maca on MetaX systems.
    if os.path.isdir("/opt/maca"):
        return True
    try:
        from flag_gems.runtime.backend.device import DeviceDetector
        return DeviceDetector().vendor_name == "metax"
    except Exception:
        pass
    try:
        import torch_maca  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _patch_metax_triton_sampler():
    """MetaX C550: MetaX Triton compiler cannot compile vLLM's topk_topp kernel.
    Patch topk_topp_sampler.HAS_TRITON = False so apply_top_k_top_p always
    takes the PyTorch-native path.

    Uses a sys.meta_path import hook so the patch is applied the moment
    vllm.v1.sample.ops.topk_topp_sampler is imported (which happens inside
    the worker process, after register() has already returned).

    TODO: remove when MetaX Triton supports vLLM's topk_topp kernel.
    """
    import sys

    if not _is_metax():
        return

    _TARGET = "vllm.v1.sample.ops.topk_topp_sampler"

    # If already imported, patch immediately.
    mod = sys.modules.get(_TARGET)
    if mod is not None:
        mod.HAS_TRITON = False
        logger.info("MetaX: patched topk_topp_sampler.HAS_TRITON=False (already imported).")
        return

    # Not yet imported — install a one-shot import hook.
    class _TritonPatchHook:
        """One-shot meta_path hook: patches HAS_TRITON after module load."""

        def find_module(self, fullname, path=None):
            if fullname == _TARGET:
                return self  # claim this import
            return None

        def load_module(self, fullname):
            # Remove ourselves first to avoid recursion.
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            # Let the normal import machinery load the module.
            import importlib
            mod = importlib.import_module(fullname)
            # Now patch it.
            mod.HAS_TRITON = False
            logger.info("MetaX: patched topk_topp_sampler.HAS_TRITON=False (via import hook).")
            return mod

    sys.meta_path.insert(0, _TritonPatchHook())
    logger.info("MetaX: registered import hook for topk_topp_sampler Triton patch.")


def _patch_metax_torch_accelerator():
    """MetaX C550: torch.accelerator.empty_cache() does not exist in MetaX PyTorch.
    Add it as an alias for torch.cuda.empty_cache().
    TODO: remove when MetaX PyTorch adds torch.accelerator.empty_cache.
    """
    if not _is_metax():
        return
    import torch
    if not hasattr(torch.accelerator, "empty_cache"):
        torch.accelerator.empty_cache = torch.cuda.empty_cache
        logger.info("MetaX: patched torch.accelerator.empty_cache = torch.cuda.empty_cache.")


def register():
    """Register the FL platform."""
    _patch_transformers_compat()
    # Note: MetaX compat patches (torch.accelerator.empty_cache,
    # topk_topp_sampler.HAS_TRITON) are applied at module import time
    # in _apply_metax_compat_patches() and model_runner.py respectively,
    # so they work in all spawned worker processes too.

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
