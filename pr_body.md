## 概述

将 vllm-plugin-FL 适配到 vLLM 0.28.0（从 0.24.0 升级）。vLLM 0.28.0 包含多项破坏性 API 变更，本 PR 逐一修复。

## 主要修改

### 1. MoE 层重构 (layer.py, custom_ops.py, fused_moe_utils.py)
- `FusedMoE` 类重命名为 `FusedMoEFactory` 函数
- `FlashinferMoeBackend` 枚举完全移除，改用字符串常量
- `FusedMoEParallelConfig` 移至 `vllm.model_executor.layers.fused_moe.config`
- Worker 中类型检查从 `FusedMoE` 改为 `MoERunner`

### 2. Worker 适配 (worker.py, model_runner.py)
- `prepare_communication_buffer_for_model` 已在 0.28 中移除，加 try/except 兼容
- 新增 `use_v2_model_runner` property（0.28 kernel_warmup 需要）
- `SparseWeightPatch` 移至 `vllm.distributed.weight_transfer.sparse_weight_patch`
- `get_total_cp_world_size` 移除，替换为 `parallel_config.decode_context_parallelism_size`
- `LateInteractionRunner` 移至 `vllm.v1.worker.gpu.pool.late_interaction`

### 3. FlagGems 可选化 (utils.py, platform.py, builtin_ops.py)
- 新增 `_FLAGGEMS_AVAILABLE` 标记 + `VLLM_FL_DISABLE_FLAGGEMS` 环境变量
- platform.py 在 FlagGems 不可用时回退到 NVIDIA 默认值
- builtin_ops.py 中 FlagGems 注册可通过环境变量禁用

### 4. Dispatch 更新 (cuda.py)
- `attention_backend` 方法默认返回 FlashInfer（对齐 vLLM 0.28 默认）

### 5. 依赖更新 (pyproject.toml)
- vllm >= 0.28.0

## 测试结果

### 单元测试（NVIDIA A800）
- **357 passed**, 19 skipped, 3 failed (e2e serving 测试，需运行中的服务，与本升级无关)

### 功能测试
- test_graph_capture.py: 6 passed
- test_collective_ops.py: 10 passed
- test_ops_correctness.py: 7 passed

### 离线推理验证
- 模型: Qwen2.5-0.5B
- 输出: 生成连贯文本，吞吐 27.07 toks/s
- FlashInfer decode backend (sm80) 正常工作
- KV cache: 265,264 tokens 分配成功

## 已知限制 / TODO

- [ ] FlagGems 集成测试（需要兼容 triton 3.7.1 的 FlagGems 版本）
- [ ] V2 model runner 路径测试（vLLM 0.28 部分模型自动启用 V2 runner）
- [ ] OOT rotary embedding（目前通过 VLLM_FL_OOT_ENABLED=0 禁用）
- [ ] deep_gemm 路径污染修复（0.24.0 venv 残留，非阻塞性警告）

## 测试环境
- NVIDIA A800, CUDA 13.0, Driver 580.126.20
- torch 2.13.0+cu130, triton 3.7.1
- flashinfer-python 0.6.16.post3
