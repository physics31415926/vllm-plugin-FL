# MetaX C550 Testing Skill

Test the vllm-plugin-FL on MetaX C550 hardware via Docker container.

## Overview

This skill automates running the plugin test suite on MetaX C550 GPUs:
- Uses official MetaX Docker image as base (torch + MACA runtime pre-installed)
- Installs vLLM 0.20.2 (CPU-only via `VLLM_TARGET_DEVICE=empty`) + vllm-plugin-fl + FlagGems
- Runs unit, functional, and e2e tests via the platform-aware test runner

## SSH Connection

The MetaX machine is configured in `~/.ssh/config` as `metax_c550`:

```
Host metax_c550
  HostName 124.236.26.209
  User lxwang@secure@172.16.1.15
  Port 2224
  IdentityFile C:\Users\BAAI\Downloads\test.jumpserver.pem
```

Machine: `bm-turing-hz1-zone1-MC550-64G-1-15` (8x MetaX C550 64GB, MACA 2.33.0)

All remote commands:
```bash
ssh metax_c550 "<command>"
# Or for docker exec:
ssh metax_c550 "docker exec vllm-plugin-test env PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin bash -c '<command>'"
```

## Workflow

### Phase 0: Container Setup (skip if container already running)

Check if container exists:
```bash
ssh metax_c550 "docker ps -a --filter name=vllm-plugin-test --format '{{.Status}}'"
```

If not running, pull image and start container:
```bash
# Login and pull (credentials may be needed)
ssh metax_c550 "docker pull cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.19.0-maca.ai3.5.3.502-torch2.8-py312-ubuntu22.04-amd64"

# Start container — MUST use --network host (bridge mode has DNS issues)
ssh metax_c550 "docker run -d --name vllm-plugin-test \
  --privileged \
  --network host \
  --device /dev/dri \
  -v /home/secure/flagos-test:/workspace \
  cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.19.0-maca.ai3.5.3.502-torch2.8-py312-ubuntu22.04-amd64 \
  sleep infinity"
```

Container provides: Python 3.12, torch 2.8.0+metax, MACA runtime, conda at `/opt/conda`.

### Phase 1: Environment Setup (inside container)

All commands below run inside the container via `docker exec`.

1. **Configure apt sources (aliyun mirror)**:
   ```bash
   echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" > /etc/apt/sources.list
   apt-get update
   ```

2. **Install system tools**:
   ```bash
   apt-get install -y git
   ```

3. **Configure pip mirror (Tsinghua)**:
   ```bash
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **Configure git proxy** (for GitHub access):
   ```bash
   git config --global http.proxy "http://lxwang:d6fHoPs2TTgsaG4a@114.111.17.35:3128"
   git config --global https.proxy "http://lxwang:d6fHoPs2TTgsaG4a@114.111.17.35:3128"
   ```

5. **Clone and install vLLM 0.20.2 (CPU-only)**:
   ```bash
   cd /workspace
   git clone https://github.com/vllm-project/vllm.git
   cd vllm && git checkout v0.20.2
   SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2 VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
   ```

6. **Clone and install vllm-plugin-FL**:
   ```bash
   cd /workspace
   git clone https://github.com/flagos-ai/vllm-plugin-FL.git
   cd vllm-plugin-FL
   SETUPTOOLS_SCM_PRETEND_VERSION=0.1.1 pip install --no-build-isolation -e .
   ```

7. **Install FlagGems**:
   ```bash
   pip install -U scikit-build-core==0.11 pybind11 ninja cmake
   cd /workspace
   git clone https://github.com/FlagOpen/FlagGems.git
   cd FlagGems
   pip install --no-build-isolation .
   ```

8. **Remove conflicting packages** (CRITICAL):
   ```bash
   # The base MetaX image includes vllm-metax 0.19.0 and mcoplib which conflict with our setup
   pip uninstall -y mcoplib vllm-metax
   ```

9. **Install pytest plugins**:
   ```bash
   pip install pytest-cov pytest-json-report pytest-timeout
   ```

10. **Verification checkpoint** — all must pass:
   ```bash
   python3 -c "import vllm; print('vllm: OK')"
   python3 -c "import vllm_fl; print('vllm_fl: OK')"
   python3 -c "import flag_gems; print(f'flag_gems: {flag_gems.__version__}')"
   python3 -c "import torch; print(f'torch: {torch.__version__}, GPUs: {torch.cuda.device_count()}')"
   ```
   Expected: vllm 0.20.2+empty, vllm_fl OK, flag_gems 5.0.2, torch 2.8.0+metax, 8 GPUs.

   If any import fails, do NOT proceed — fix the issue first.

### Phase 2: Model Preparation

Download test models inside the container. Default model path: `/workspace/models/Qwen/Qwen3-4B`.

**Option A — Use modelscope (recommended, faster in China)**:
```bash
pip install modelscope
# Use tmux for long-running download to avoid SSH timeout
tmux new-session -d -s model-dl 'python3 -c "from modelscope import snapshot_download; path = snapshot_download(\"Qwen/Qwen3-4B\", cache_dir=\"/workspace/models\"); print(f\"Downloaded to: {path}\")"'

# Check progress
tmux capture-pane -t model-dl -p | tail -20

# Check if done
tmux has-session -t model-dl 2>/dev/null && echo "still running" || echo "done"
```

**Option B — Use huggingface-cli (needs proxy or mirror)**:
```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir /workspace/models/Qwen/Qwen3-4B
```

**Verification checkpoint** — model must be valid:
```bash
ls /workspace/models/Qwen/Qwen3-4B/
du -sh /workspace/models/Qwen/Qwen3-4B/
python3 -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('/workspace/models/Qwen/Qwen3-4B'); print(f'Tokenizer loaded, vocab_size={t.vocab_size}')"
```
Expected: 3 safetensors files (~7.6GB total), tokenizer loads with vocab_size=151643.

If the model is stored at a different path, update `tests/models/qwen3/4b_tp2.yaml` or symlink it.

**Unit Test Checkpoint** — verify plugin code before running GPU tests:
```bash
export VLLM_PLUGINS='fl'
cd /workspace/vllm-plugin-FL
python3 tests/run.py --platform metax --device c550 --scope unit
```
Expected: All unit tests pass (349 passed). If tests fail or crash, do NOT proceed to functional/e2e tests.

### Phase 3: Run Tests

Execute from the `vllm-plugin-FL` project root inside the container.

```bash
export VLLM_PLUGINS='fl'
cd /workspace/vllm-plugin-FL
```

**Unit tests** (no GPU required):
```bash
python tests/run.py --platform metax --device c550 --scope unit
```

**Functional tests** (GPU required):
```bash
python tests/run.py --platform metax --device c550 --scope functional
```

**E2E tests** (GPU + model required):
```bash
python tests/run.py --platform metax --device c550 --scope e2e
```

**Single e2e case**:
```bash
python tests/run.py --platform metax --device c550 --scope e2e --task inference --model qwen3 --case 4b_tp2
```

**Dry run** (show commands without executing):
```bash
python tests/run.py --platform metax --device c550 --dry-run
```

### Phase 4: Results

After tests complete, check:
- Console output for pass/fail summary
- `test-results-metax.xml` — JUnit XML report
- `test-results-metax.json` — JSON report with details

Report any failures with:
1. The test name and scope (unit/functional/e2e)
2. The error message / traceback
3. Whether it's a platform-specific issue or a general regression

## Key Constraints

- **MUST** use `--network host` when creating the container (bridge mode breaks DNS)
- **MUST** use `VLLM_TARGET_DEVICE=empty` when installing vLLM (CPU-only, no CUDA extensions)
- **MUST** use `SETUPTOOLS_SCM_PRETEND_VERSION` if git history is incomplete
- **DO NOT** modify MACA SDK libraries or low-level drivers
- **DO NOT** modify FlagGems source code — install as-is from upstream
- **DO NOT** change files under `vllm_fl/dispatch/backends/vendor/metax/` unless fixing test-only issues
- Only use proxy for git operations, NOT for pip/apt (use mirrors instead)

## Platform Config

The platform configuration lives at `tests/platforms/metax.yaml`. It defines:
- Device type: c550 (64GB)
- Tolerance: rtol=1e-3, atol=1e-3 for inference
- Test matrix: qwen3/4b_tp2 for both inference and serving
- All unit and functional tests included by default

## Troubleshooting

### Common Setup Issues

| Issue | Solution |
|-------|----------|
| DNS resolution fails in container | Container must use `--network host`, not bridge |
| apt-get slow/fails | Use aliyun mirrors in `/etc/apt/sources.list` |
| git clone fails | Set git proxy: `git config --global http.proxy <proxy>` |
| Fatal abort on import (mcoplib._C) | Uninstall `vllm-metax` and `mcoplib` — they conflict with vLLM 0.20.2 |
| `mx-smi` not found | MACA driver not installed or not in PATH |
| `import vllm_fl` fails | Plugin not installed — run `pip install -e .` |
| MACA OOM during e2e | Reduce `gpu_memory_utilization` in model yaml or use fewer TP |
| `libmcruntime.so` not found | MACA runtime not in LD_LIBRARY_PATH |
| pynccl/mccl errors | Check that the MetaX MCCL library path is correct |
| numpy version conflicts | Ignore warnings from vllm-metax (it's the old 0.19 package) |

### vLLM 0.20.2 API Compatibility Issues

When adapting MetaX vendor backend from vLLM 0.19.x to 0.20.2, several API changes require fixes:

**Problem-Solving Process:**
1. Run functional tests to identify import/API errors
2. For each error, search the entire MetaX backend for similar patterns
3. Fix all occurrences in one commit
4. Test incrementally (unit → functional → inference)

**Known API Changes:**

| Error | Root Cause | Fix | Files Affected |
|-------|------------|-----|----------------|
| `cannot import name 'rms_norm'` | vLLM 0.20.2 removed standalone `rms_norm` function | Replace with FlagGems: `gems_rms_forward` | `layernorm.py` |
| `cannot import name 'SiluAndMul'` | Class removed from vLLM | Replace with FlagGems: `gems_silu_and_mul` | `activation.py` |
| `cannot import name 'rotary_embedding' from 'vllm._custom_ops'` | Not available in empty build | Replace with FlagGems: `gems_rope_forward` | `rotary_embedding.py` |
| `cannot import name 'topk_softmax' from 'vllm._custom_ops'` | Not available in empty build | Replace with FlagGems: `flag_gems.topk_softmax` | `fused_moe.py` |
| `cannot import name 'is_quantized_kv_cache'` | Replaced by `get_kv_quant_mode` returning `KVQuantMode` enum | Update import and usage: `get_kv_quant_mode(dtype) != KVQuantMode.NONE` | `flash_attn.py` |
| `cannot import name 'AttentionMetadataBuilder' from 'vllm.v1.attention.backends.utils'` | Moved to `vllm.v1.attention.backend` | Split import: `AttentionMetadataBuilder` from `backend`, others from `backends.utils` | `mla/common.py` |

**Proactive Checking:**
After fixing an import error, always search for similar patterns across all MetaX backend files:
```bash
# Example: Check for other uses of removed API
grep -r "is_quantized_kv_cache" vllm_fl/dispatch/backends/vendor/metax/
grep -r "from vllm._custom_ops" vllm_fl/dispatch/backends/vendor/metax/
grep -r "AttentionMetadataBuilder" vllm_fl/dispatch/backends/vendor/metax/
```

**Remaining Known Issues:**
- `fa_utils.py` and `mla/common.py` still import `vllm._custom_ops` for cache ops
- These are runtime calls, only triggered for specific attention paths (MLA models)
- For standard MHA models (Qwen3-4B), these paths are not executed
- If needed, replace with FlagGems equivalents or dispatch system
