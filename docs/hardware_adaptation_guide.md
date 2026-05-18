# vLLM-plugin-FL 新硬件适配指南

> 参考 PR: [#97 add musa support](https://github.com/flagos-ai/vllm-plugin-FL/pull/97) (Moore Threads MUSA) 和 [#148 add sunrise device support](https://github.com/flagos-ai/vllm-plugin-FL/pull/148) (Sunrise/平头哥)

---

## 概览

适配一个新硬件到 vLLM-plugin-FL，核心工作分为 **6 个步骤**：

| # | 步骤 | 涉及文件 | 必要性 |
|---|------|----------|--------|
| 1 | 注册设备映射 | `vllm_fl/utils.py` | 必须 |
| 2 | 平台层适配 | `vllm_fl/platform.py` | 必须 |
| 3 | 创建 vendor 后端 | `vllm_fl/dispatch/backends/vendor/<name>/` | 必须 |
| 4 | 编写调度配置 | `vllm_fl/dispatch/config/<name>.yaml` | 必须 |
| 5 | 图编译适配 | `vllm_fl/compilation/graph.py` | 按需 |
| 6 | 通信层适配 | `vllm_fl/distributed/device_communicators/flagcx.py` | 按需 |

---

## Step 1: 注册设备映射

**文件**: `vllm_fl/utils.py`

### 1.1 添加 VENDOR_DEVICE_MAP 条目

```python
VENDOR_DEVICE_MAP: dict[str, dict[str, str]] = {
    "nvidia": {"device_type": "cuda", "device_name": "nvidia"},
    "ascend": {"device_type": "npu", "device_name": "npu"},
    "metax": {"device_type": "cuda", "device_name": "metax"},
    "mthreads": {"device_type": "musa", "device_name": "musa"},
    # ↓ 新增你的硬件 ↓
    "sunrise": {"device_type": "ptpu", "device_name": "ptpu"},
}
```

字段说明：
- **key** (`vendor_name`): FlagGems `DeviceDetector` 检测到的厂商标识
- **device_type**: PyTorch 设备类型（如 `"cuda"`, `"npu"`, `"musa"`, `"ptpu"`）
- **device_name**: vLLM 平台使用的设备名称

### 1.2 添加到 supported_device 列表

```python
class DeviceInfo:
    def __init__(self):
        self.supported_device = ["nvidia", "ascend", "metax", "mthreads", "sunrise"]
        #                                                                  ^^^^^^^^ 新增
```

---

## Step 2: 平台层适配

**文件**: `vllm_fl/platform.py`

根据硬件特性，需要修改 `PlatformFL` 类中的以下方法：

### 2.1 设备能力查询 (get_device_capability)

```python
@classmethod
def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
    if cls.device_type == "npu":
        return None
    # ↓ 新增：如果你的设备没有 CUDA compute capability 概念 ↓
    if cls.device_type == "ptpu":
        return None
    if cls.device_type == "musa":
        major, minor = torch.musa.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)
    major, minor = torch.cuda.get_device_capability(device_id)
    return DeviceCapability(major=major, minor=minor)
```

### 2.2 CUDA 兼容性判断 (is_cuda_alike / is_cuda)

如果你的设备是 CUDA 兼容的（如 MUSA），需要在这些方法中声明：

```python
def is_cuda_alike(self) -> bool:
    if self.vendor_name == "musa":
        return True  # MUSA 兼容 CUDA
    return self.device_type == "cuda"

def is_cuda(self) -> bool:
    if self.vendor_name == "musa":
        return True
    return self.device_type == "cuda" and self.vendor_name == "nvidia"
```

### 2.3 KV Cache block_size 配置 (check_and_update_config)

```python
if cache_config and cache_config.block_size is None:
    if cls.device_type == "npu":
        cache_config.block_size = 128
    elif cls.device_type == "musa":
        cache_config.block_size = 64    # ← 根据硬件特性设置
    else:
        cache_config.block_size = 16
```

### 2.4 Pin Memory 支持

```python
@classmethod
def is_pin_memory_available(cls):
    if cls.device_type in ["cuda", "xpu", "npu", "musa"]:  # ← 按需添加
        return True
    return False
```

---

## Step 3: 创建 Vendor 后端

这是工作量最大的部分。需要在 `vllm_fl/dispatch/backends/vendor/` 下创建新目录。

### 目录结构

```
vllm_fl/dispatch/backends/vendor/<your_vendor>/
├── __init__.py           # 导出 Backend 类
├── <vendor>.py           # Backend 主类（继承 Backend 基类）
├── register_ops.py       # 算子注册（OpImpl 列表）
├── impl/                 # 具体算子实现
│   ├── __init__.py
│   ├── activation.py     # silu_and_mul, gelu_and_mul
│   ├── normalization.py  # rms_norm
│   ├── rotary.py         # rotary_embedding
│   └── attention.py      # attention backend（可选，自定义注意力）
└── patch.py              # 模型 patch（可选）
```

### 3.1 Backend 主类

```python
# vllm_fl/dispatch/backends/vendor/sunrise/sunrise.py

from vllm_fl.dispatch.backends.base import Backend

class SunriseBackend(Backend):
    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "sunrise"

    @property
    def vendor(self) -> Optional[str]:
        return "sunrise"

    def is_available(self) -> bool:
        """检测硬件是否可用"""
        if SunriseBackend._available is None:
            SunriseBackend._available = (
                torch.ptpu.is_available() and torch.ptpu.device_count() > 0
            )
        return SunriseBackend._available

    # ==================== 算子实现 ====================
    def attention_backend(self, use_mla=False, use_sparse=False) -> str:
        """返回注意力后端的完整类路径"""
        return "vllm_fl.dispatch.backends.vendor.sunrise.impl.attention.AttentionFLBackend"
```

### 3.2 算子注册 (register_ops.py)

```python
# vllm_fl/dispatch/backends/vendor/sunrise/register_ops.py

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

def register_builtins(registry) -> None:
    from .sunrise import SunriseBackend

    backend = SunriseBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.sunrise",          # 格式: "vendor.<name>"
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="sunrise",                  # 必须与 VENDOR_DEVICE_MAP key 对应
            priority=BackendPriority.VENDOR,    # 100
        ),
        # 按需注册更多算子...
    ]

    registry.register_many(impls)
```

### 3.3 需要实现的算子（按优先级）

| 算子 | 说明 | 必要性 |
|------|------|--------|
| `attention_backend` | 注意力后端选择 | 必须（核心） |
| `rms_norm` | RMS 归一化 | 推荐 |
| `silu_and_mul` | SiLU 激活 | 推荐 |
| `rotary_embedding` | 旋转位置编码 | 推荐 |
| `moe_align_block_size` | MoE 对齐 | 可选 |
| `dispatch_fused_moe_kernel` | MoE 融合 kernel | 可选 |

**最小可行适配**：只实现 `attention_backend`（如 Sunrise PR #148），其余算子由 FlagGems (DEFAULT) 或 Reference 后端兜底。

---

## Step 4: 编写调度配置

**文件**: `vllm_fl/dispatch/config/<vendor_name>.yaml`

系统会根据当前硬件自动加载对应的 YAML 配置。

```yaml
# vllm_fl/dispatch/config/sunrise.yaml

# 全局偏好后端
prefer: flagos

# 严格模式（false = 失败时自动 fallback）
strict: false

# 每个算子的后端执行顺序
op_backends:
  attention_backend:
    - vendor:sunrise    # 优先用厂商实现
    - flagos            # 其次 FlagGems
    - reference         # 最后 PyTorch 原生

  rms_norm:
    - flagos            # 优先 FlagGems（Triton 跨芯片）
    - vendor:sunrise
    - reference

  silu_and_mul:
    - flagos
    - vendor:sunrise
    - reference

  rotary_embedding:
    - flagos
    - vendor:sunrise
    - reference

# FlagGems 算子黑名单（已知不兼容的算子）
flagos_blacklist: []
  # - scaled_dot_product_attention  # 示例：MUSA 需要屏蔽此算子

# OOT 算子黑名单
# oot_blacklist:
#   - fused_moe
```

---

## Step 5: 图编译适配（按需）

**文件**: `vllm_fl/compilation/graph.py`

如果硬件支持图捕获（类似 CUDA Graph），需要注册对应的 Graph 类：

```python
class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
    elif current_platform.device_type == "ptpu":
        graph = torch.ptpu.PTPUGraph  # ← 如果支持的话
    else:
        raise NotImplementedError("not support graph")
```

---

## Step 6: 通信层适配（按需）

**文件**: `vllm_fl/distributed/device_communicators/flagcx.py`

如果使用 FlagCX 分布式通信，需要添加设备上下文切换逻辑：

```python
# 在 PyFlagcxCommunicator 中
if self.device.type == "musa":
    device_ctx = torch.musa.device(self.device)
elif self.device.type == "ptpu":
    device_ctx = torch.device(self.device)  # ← 新增
else:
    device_ctx = torch.cuda.device(self.device)
```

---

## 两个 PR 的对比总结

| 维度 | PR #97 (MUSA) | PR #148 (Sunrise) |
|------|---------------|-------------------|
| 设备类型 | `musa` (CUDA 兼容) | `ptpu` (非 CUDA) |
| 改动文件数 | 15 | 13 |
| 新增代码行 | ~501 | ~1412 |
| 实现算子数 | 4 (silu_and_mul, rms_norm, rotary, attention) | 1 (attention_backend) |
| 注意力实现 | 复用 vLLM 内置 (FlashAttn/TritonMLA) | 完全自定义 (946 行) |
| 图编译 | 支持 (torch.musa.MUSAGraph) | 支持 (torch.ptpu.PTPUGraph) |
| is_cuda_alike | True | False |
| block_size | 64 | 默认 16 |
| 额外 patch | 无 | 有 (patch.py, vocab_parallel_embedding) |

### 关键差异

- **CUDA 兼容设备**（如 MUSA）：改动较分散但每处较小，主要是在现有分支中加 `elif`，算子可以复用 vLLM 内置 kernel
- **非 CUDA 设备**（如 Sunrise）：需要自定义注意力后端实现（工作量大），但平台层改动较少

---

## Checklist：新硬件适配清单

```
□ vllm_fl/utils.py
  □ VENDOR_DEVICE_MAP 添加条目
  □ DeviceInfo.supported_device 列表添加

□ vllm_fl/platform.py
  □ get_device_capability() 处理新设备类型
  □ is_cuda_alike() / is_cuda() 按需修改
  □ is_pin_memory_available() 按需添加
  □ check_and_update_config() 设置 block_size

□ vllm_fl/dispatch/backends/vendor/<name>/
  □ __init__.py
  □ <name>.py (Backend 类 + is_available + 算子方法)
  □ register_ops.py (OpImpl 注册)
  □ impl/ (具体算子实现)

□ vllm_fl/dispatch/config/<name>.yaml
  □ prefer / strict 配置
  □ op_backends 每算子执行顺序
  □ flagos_blacklist（不兼容的 FlagGems 算子）

□ vllm_fl/compilation/graph.py（如支持图捕获）
  □ 添加 Graph 类映射

□ vllm_fl/distributed/device_communicators/flagcx.py（如使用 FlagCX）
  □ 添加设备上下文切换

□ 可选
  □ requirements/<name>.txt（硬件特定依赖）
  □ vllm_fl/ops/custom_ops.py（自定义 OOT 算子）
  □ patch.py（模型层 monkey-patch）
```
