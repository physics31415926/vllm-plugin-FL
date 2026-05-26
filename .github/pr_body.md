## MetaX C550: vLLM 0.20.2 Adaptation Patches

Qwen3-4B TP=2 offline inference verified on MetaX C550 (8x 64GB, MACA 2.33.0, torch 2.8.0+metax3.5.3.9).

---

### Changes & Rationale

#### 1. Core Platform (vllm_fl/platform.py)

| Change | Why | Required? |
|--------|-----|-----------|
| Insert stub vllm._C module into sys.modules | MetaX cannot compile vLLM's native _C.so (requires NVIDIA CUDA). Without the stub, import vllm._C raises ImportError. | **Yes** |
| Dynamically register all _C op schemas (parsed from csrc/torch_bindings.cpp) | vLLM's compile backend accesses torch.ops._C.op at module-level import time. Without schemas, AttributeError crashes the import. | **Yes** (for enforce_eager=False) |
| Provide weak_ref_tensor CUDA impl via torch.library.impl | CUDAGraph capture calls this op at runtime. Schema-only registration is insufficient; a real implementation is needed. Uses torch.as_strided to create a view sharing the same storage. | **Yes** (for CUDAGraph) |
| Skip super().import_kernels() for MetaX, load mcoplib first | mcoplib registers ops via C++ TORCH_LIBRARY. The base class registers fallback fake schemas first, causing duplicate schema crash when mcoplib loads. | **Yes** |
| manual_seed_all fallback for ptpu | torch_ptpu.ptpu lacks manual_seed_all. | Conditional (ptpu only) |

#### 2. MetaX Runtime Patches (vllm_fl/dispatch/backends/vendor/metax/patches/)

| File | What | Why | Required? |
|------|------|-----|-----------|
| accelerator_compat.py | Patch torch.accelerator.empty_cache | torch 2.8+metax lacks this API; vLLM calls it during shutdown. | **Yes** |
| topk_topp_sampler.py | Bypass Triton topk_topp kernel, use PyTorch fallback | MetaX Triton fails to compile this kernel (PassManager::run error in ttgir stage). | **Yes** |
| functorch_config_patch.py | Register autograd_cache_normalize_inputs config key | torch 2.8+metax doesn't define this key; vLLM's compile backend patches it, causing AttributeError. | **Yes** (for enforce_eager=False) |
| pynccl_wrapper.py (new section) | Replace CudaCommunicator.all_reduce with torch.distributed path | pynccl uses ctypes (_SimpleCData) which TorchDynamo cannot trace. torch.distributed is natively supported by Dynamo. | **Yes** (for torch.compile + TP>1) |

#### 3. MetaX Vendor Backend

| File | What | Why |
|------|------|-----|
| metax.yaml | Route silu_and_mul, rms_norm, rotary_embedding, topk_softmax to `reference` backend | These ops use pure PyTorch; no vendor-specific optimization needed. Avoids duplicating reference implementations. |
| metax.py + register_ops.py | Only register `attention_backend` op | Attention is the only op requiring MetaX-specific dispatch (MACA flash_attn). |
| fa_utils.py | Pure PyTorch reshape_and_cache_flash (CUDAGraph-compatible) | vllm._C not compiled; no native KV cache scatter kernel available. Uses clamp+mask instead of boolean indexing to avoid FlagGems nonzero().item() sync during graph capture. |
| flash_attn.py | is_quantized_kv_cache -> get_kv_quant_mode | vLLM 0.20.2 API change. |
| merge_attn_states.py | Delegate to vLLM's Triton merge kernel | Removes mcoplib dependency for prefix caching merge. |
| attention/mla/common.py | Fix AttentionMetadataBuilder import path | vLLM 0.20.2 moved this class. |

#### 4. Compilation (vllm_fl/compilation/graph.py)

| Change | Why |
|--------|-----|
| synchronize() only on NPU before graph replay | On MetaX, unconditional sync causes performance loss and conflicts with CUDAGraph stream semantics. |

---

### Dispatch Architecture

```
metax.yaml op_backends:
  attention_backend → vendor:metax (MACA flash_attn)
  silu_and_mul      → reference (PyTorch native)
  rms_norm          → flagos → reference
  rotary_embedding  → flagos → reference
  topk_softmax      → reference
```

The MetaX vendor backend is now minimal: it only provides the attention backend (which requires MACA-specific flash_attn integration). All other ops fall through to the `reference` backend (pure PyTorch) or `flagos` (Triton/FlagGems) via the dispatch YAML config.

---

### Test Results

| Test | Result | Notes |
|------|--------|-------|
| Qwen3-4B TP=2 enforce_eager=True | Pass | ~0.26 tok/s (PyTorch fallbacks, no CUDAGraph) |
| Qwen3-4B TP=2 enforce_eager=False (torch.compile + CUDAGraph FULL_AND_PIECEWISE) | Pass | ~14.28 tok/s, PIECEWISE 51/51 + FULL 35/35 captured, compile 13.22s |

### Known Limitations

- **Performance**: ~14.28 tok/s with CUDAGraph (FULL_AND_PIECEWISE). Without CUDAGraph (enforce_eager=True), ~0.26 tok/s due to PyTorch fallback kernel launch overhead.
- **Missing FlagGems ops for MetaX**: reshape_and_cache_flash (using PyTorch fallback), topk_topp.
- **MLA attention (DeepSeek series)**: `mla/common.py` still imports `vllm._custom_ops` for `gather_and_maybe_dequant_cache`, `cp_gather_cache`, and `concat_and_cache_mla`. These ops require mcoplib or PyTorch native replacements. Not addressed in this PR since current testing targets Qwen3 (non-MLA).

### Test Environment

- MetaX C550 (8x 64GB)
- MACA 2.33.0
- torch 2.8.0+metax3.5.3.9
- vLLM 0.20.2 (CPU-only wheel, VLLM_TARGET_DEVICE=empty)
- flash_attn (MetaX build)
- **mcoplib uninstalled**: mcoplib's C++ TORCH_LIBRARY registrations conflict with our stub _C op schemas (duplicate registration crash). We bypass it entirely and rely on FlagGems + PyTorch reference fallbacks.

### Note on _C stub conditional logic

The stub vllm._C injection and op schema registration only activates when the native _C.so cannot be imported (i.e., non-NVIDIA platforms). On NVIDIA GPUs, the original vllm._C extension loads normally and no stubs are injected.
