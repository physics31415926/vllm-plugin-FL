# vLLM 版本升级指南

> 参考 Commit: [4dfed0f 0.18.1](https://github.com/flagos-ai/vllm-plugin-FL/commit/4dfed0f)、[42f8d6f 0.19.0](https://github.com/flagos-ai/vllm-plugin-FL/commit/42f8d6f)、[PR #158 0.20.2](https://github.com/flagos-ai/vllm-plugin-FL/pull/158)

---

## 概览

vLLM-plugin-FL 作为 vLLM 的 OOT 插件，需要跟随 vLLM 上游版本升级。由于插件通过 monkey-patch、子类继承和 import 内部模块的方式与 vLLM 耦合，每次 vLLM 升级都可能引入 breaking changes。

### 升级历史

| 版本跨度 | Commit/PR | 改动文件 | 新增/删除行 | 主要影响 |
|----------|-----------|----------|-------------|----------|
| → 0.18.1 | 4dfed0f | 41 | +2438/-4805 | model_runner 重写、attention 路径迁移、模型文件清理 |
| 0.18.1 → 0.19.0 | 42f8d6f | 7 | +493/-412 | model_runner 更新、platform 微调 |
| 0.19.0 → 0.20.2 | PR #158 | 21 | +780/-360 | MoE 重构、API 重命名、speculative decoding |

---

## 升级影响范围

根据历史升级经验，以下文件/模块是每次升级的必改项：

### 核心文件（每次必改）

| 文件 | 说明 | 典型改动 |
|------|------|----------|
| `vllm_fl/worker/model_runner.py` | GPU ModelRunner 的 OOT 实现 | 最大改动量，跟随 vLLM 的 execute_model/attention/speculative 变更 |
| `vllm_fl/worker/worker.py` | Worker 入口 | import 路径变更、新增方法签名适配 |
| `vllm_fl/platform.py` | 平台层 | 新增 Platform 抽象方法实现 |

### 高频变更文件

| 文件 | 说明 | 触发条件 |
|------|------|----------|
| `vllm_fl/ops/fused_moe/` | MoE 算子层 | vLLM MoE 架构重构时 |
| `vllm_fl/dispatch/backends/*/` | 算子后端 | vLLM 算子接口签名变更时 |
| `vllm_fl/__init__.py` | 插件注册入口 | 注册 API 变更时 |
| `vllm_fl/compilation/graph.py` | 图编译 | CUDAGraph 接口变更时 |

### 偶尔变更文件

| 文件 | 说明 | 触发条件 |
|------|------|----------|
| `vllm_fl/attention/` | Attention 工具 | Attention 后端接口变更 |
| `vllm_fl/quantization/` | 量化支持 | 量化 kernel 注册方式变更 |
| `vllm_fl/configs/` | 模型配置 | Config 注册 API 变更 |

---

## 升级步骤

### Step 1: 确认目标版本差异

```bash
# 查看 vLLM 上游 changelog
pip show vllm  # 当前版本
# 对比目标版本的 breaking changes
```

重点关注 vLLM 的：
- `vllm/v1/worker/gpu_model_runner.py` — 对应我们的 `model_runner.py`
- `vllm/v1/worker/gpu_worker.py` — 对应我们的 `worker.py`
- `vllm/platforms/cuda.py` — 对应我们的 `platform.py`
- `vllm/model_executor/layers/fused_moe/` — 对应我们的 `ops/fused_moe/`

### Step 2: 升级 vLLM 依赖

```bash
pip install vllm==<target_version>
```

### Step 3: 修复 import 错误

最常见的 breaking change 是 import 路径变更。按以下优先级修复：

```python
# 典型 import 路径迁移示例

# 0.18.1 → 0.19.0: attention 路径迁移
- from vllm.attention.backends.registry import AttentionBackendEnum
+ from vllm.v1.attention.backends.registry import AttentionBackendEnum

- from vllm.attention.selector import AttentionSelectorConfig
+ from vllm.v1.attention.selector import AttentionSelectorConfig

# 0.19.0 → 0.20.2: 工具函数迁移
- from vllm.v1.attention.backend import is_quantized_kv_cache
+ from vllm.utils.torch_utils import is_quantized_kv_cache

# 0.19.0 → 0.20.2: 模块路径变更
- from vllm.v1.worker.gpu.mm.encoder_cudagraph import EncoderCudaGraphManager
+ from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
```

### Step 4: 修复 API 签名变更

```python
# 典型 API 重命名示例

# 0.19.0 → 0.20.2: fused_moe kernel 重命名
- dispatch_fused_moe_kernel(A, B, C, A_scale, B_scale, B_zp, ...)
+ invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, ...)
# 注意：B_zp 参数被移除

# 0.19.0 → 0.20.2: Router 签名新增参数
- def _compute_routing(self, hidden_states, router_logits, indices_type):
+ def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):

# 0.19.0 → 0.20.2: logger 移除 scope 参数
- logger.info_once("...", scope="local")
+ logger.info_once("...")
```

### Step 5: 适配新增抽象方法

vLLM 的 `Platform` 基类可能新增抽象方法，需要在 `PlatformFL` 中实现：

```python
# 0.19.0 → 0.20.2 新增方法
@classmethod
def import_ir_kernels(cls) -> None:
    """Import IR kernel modules."""
    import vllm.kernels  # noqa: F401

@classmethod
def support_deep_gemm(cls) -> bool:
    """Currently, only Hopper and Blackwell GPUs are supported."""
    if cls.device_type == "cuda" and cls.vendor_name == "nvidia":
        return cls.is_device_capability(90) or cls.is_device_capability_family(100)
    return False

@classmethod
def manual_seed_all(cls, seed: int) -> None:
    cls.torch_device_fn.manual_seed_all(seed)

@classmethod
def is_integrated_gpu(cls, device_id: int = 0) -> bool:
    return False
```

### Step 6: 同步 model_runner.py

这是每次升级最大的工作量。策略：

1. **对比 vLLM 上游 diff**：
   ```bash
   # 在 vLLM 仓库中
   git diff v0.19.0..v0.20.2 -- vllm/v1/worker/gpu_model_runner.py
   ```

2. **逐段同步**：将上游变更应用到 `vllm_fl/worker/model_runner.py`

3. **保留 FL 特有逻辑**：
   - `managed_inference_mode()` 装饰器
   - dispatch 相关的 `call_op` 调用
   - FlagGems 相关的初始化
   - 自定义 memory profiling

### Step 7: 同步 MoE 层（如有变更）

vLLM 的 MoE 架构变更频繁，需要特别关注：

```python
# 0.19.0 → 0.20.2 的 MoE 重构
# 旧模式：forward_oot() 覆写
class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    def forward_oot(self, layer, x, topk_weights, topk_ids, ...):
        return fused_experts(...)

# 新模式：MoERunner + TritonExpertsFL
class UnquantizedFusedMoEMethodFL(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend_oot(...)
```

### Step 8: 更新 dispatch 配置

```yaml
# 如果有算子重命名，更新 YAML 配置
# dispatch/config/nvidia.yaml
op_backends:
  invoke_fused_moe_triton_kernel:  # 旧名: dispatch_fused_moe_kernel
    - vendor
    - flagos
    - reference
```

### Step 9: 验证

```bash
# 1. 基本 import 测试
python -c "import vllm_fl; print('OK')"

# 2. 离线推理测试
python examples/offline_inference.py --model Qwen/Qwen3-0.6B

# 3. 多 GPU 测试
python examples/offline_inference.py --model <large_model> --tensor-parallel-size 8

# 4. 运行单元测试
pytest tests/unit_tests/ -v
```

---

## 常见问题模式

### 模式 1: import 路径迁移

**症状**：`ImportError: cannot import name 'xxx' from 'vllm.yyy'`

**解决**：在 vLLM 源码中搜索该符号的新位置：
```bash
# 在 vLLM 安装目录搜索
python -c "import vllm; print(vllm.__file__)"
grep -r "class XXX\|def XXX\|XXX =" /path/to/vllm/
```

### 模式 2: 方法签名变更

**症状**：`TypeError: xxx() got an unexpected keyword argument 'yyy'` 或 `missing required argument`

**解决**：对比 vLLM 上游对应方法的新签名，更新调用处。

### 模式 3: 基类新增抽象方法

**症状**：`TypeError: Can't instantiate abstract class PlatformFL with abstract method xxx`

**解决**：在 `PlatformFL` 中实现该方法，参考 vLLM 的 `CudaPlatform` 实现。

### 模式 4: 内部类/函数被移除或重构

**症状**：`AttributeError: module 'vllm.xxx' has no attribute 'yyy'`

**解决**：
1. 检查是否被重命名（搜索相似功能）
2. 检查是否被合并到其他模块
3. 如果被移除，评估是否需要在 FL 中自行实现

### 模式 5: 算子接口变更

**症状**：运行时 kernel 调用参数不匹配

**解决**：
1. 更新 dispatch 后端中的方法签名
2. 更新 `register_ops.py` 中的 `op_name`
3. 更新 YAML 配置中的算子名称

---

## 升级 Checklist

```
□ 准备阶段
  □ 确认目标 vLLM 版本
  □ 阅读 vLLM release notes / changelog
  □ 备份当前工作分支

□ 依赖升级
  □ pip install vllm==<target>
  □ 确认 torch / transformers 版本兼容性

□ 核心文件修复
  □ vllm_fl/worker/model_runner.py — 同步上游变更
  □ vllm_fl/worker/worker.py — import 路径 + 新方法
  □ vllm_fl/platform.py — 新增抽象方法实现

□ 算子层修复
  □ vllm_fl/ops/fused_moe/ — MoE 接口适配
  □ vllm_fl/dispatch/backends/ — 后端方法签名更新
  □ vllm_fl/dispatch/config/*.yaml — 算子名称更新

□ 其他修复
  □ vllm_fl/__init__.py — 注册 API 变更
  □ vllm_fl/compilation/graph.py — 图编译接口
  □ vllm_fl/quantization/ — 量化 kernel 注册
  □ vllm_fl/attention/ — Attention 工具

□ 验证
  □ python -c "import vllm_fl" 无报错
  □ 单模型离线推理通过
  □ 多 GPU TP 推理通过
  □ 单元测试通过
  □ 各 vendor 后端验证（CUDA/MUSA/Ascend/MetaX）

□ 清理
  □ 移除不再需要的兼容代码
  □ 更新文件头部版本注释
  □ 更新 CI 镜像版本
```

---

## 升级难度评估

| 版本跨度 | 预期难度 | 主要原因 |
|----------|----------|----------|
| patch 版本（如 0.19.0 → 0.19.1） | 低 | 通常只有 bug fix，无 breaking change |
| minor 版本（如 0.18.x → 0.19.0） | 中~高 | 可能有 API 重构、模块迁移 |
| 跨多个 minor（如 0.11 → 0.18） | 极高 | 大量 breaking change 累积 |

**经验法则**：
- `model_runner.py` 的改动量 ≈ 总工作量的 60-70%
- 每跨一个 minor 版本，预期 5-15 个文件需要修改
- MoE 相关变更是最复杂的部分（涉及 dispatch + layer + backend 三层联动）

---

## 与其他适配的关系

vLLM 升级可能影响已有的硬件适配和模型适配：

| 影响 | 说明 | 应对 |
|------|------|------|
| vendor 后端 | 算子接口签名变更 → 所有 vendor 后端需同步更新 | 升级后逐个验证 vendor |
| 模型适配 | Config 注册 API 变更 → 已有模型注册可能失效 | 检查 `__init__.py` 注册代码 |
| patches | vLLM 内部实现变更 → monkey-patch 可能失效 | 逐个验证 patch 目标函数是否存在 |
| 图编译 | CUDAGraph API 变更 → graph.py 需适配 | 检查 GraphWrapper 兼容性 |

---

## 相关文档

- [architecture.md](architecture.md) — 项目整体架构
- [hardware_adaptation_guide.md](hardware_adaptation_guide.md) — 硬件适配指南
- [model_adaptation_guide.md](model_adaptation_guide.md) — 模型适配指南
- [agent_automation_guide.md](agent_automation_guide.md) — Agent 自动化指南
