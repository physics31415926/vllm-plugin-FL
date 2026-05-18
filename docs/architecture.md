# vLLM-plugin-FL 代码架构与设计文档

## 1. 项目概述

vLLM-plugin-FL 是 FlagOS 推理生态的核心插件，通过 vLLM 的 OOT (Out-of-Tree) 插件机制，为 vLLM 提供多芯片（NVIDIA、昇腾、MetaX、Iluvatar、Moore Threads、Sunrise）统一适配能力。

核心设计目标：
- **多芯片统一**：一套代码适配多种硬件平台
- **算子可调度**：通过 dispatch 机制灵活选择最优算子实现
- **无侵入扩展**：不修改 vLLM 源码，通过 entry point 注册

---

## 2. 代码结构

```
vllm_fl/
├── __init__.py              # 插件入口：register() + register_model()
├── platform.py              # PlatformFL：OOT 平台类，设备抽象层
├── utils.py                 # DeviceInfo、设备检测工具
│
├── dispatch/                # 核心：算子调度系统
│   ├── __init__.py          # 公共 API：call_op(), resolve_op()
│   ├── types.py             # 核心类型：OpImpl, BackendImplKind, BackendPriority
│   ├── registry.py          # OpRegistry：线程安全的算子注册表
│   ├── manager.py           # OpManager：调度管理器（resolve + call + fallback）
│   ├── policy.py            # SelectionPolicy：策略选择（prefer/strict/per_op）
│   ├── ops.py               # VLLMFLBackendBase：后端抽象基类
│   ├── builtin_ops.py       # 内置算子注册入口
│   └── backends/            # 具体后端实现
│       ├── flaggems/        # DEFAULT 后端（FlagGems，Triton 实现）
│       │   ├── register_ops.py
│       │   ├── flaggems.py  # FlagGemsBackend 类
│       │   ├── activation.py
│       │   ├── normalization.py
│       │   ├── rotary.py
│       │   ├── attention.py
│       │   └── fused_moe.py
│       ├── reference/       # REFERENCE 后端（PyTorch 原生实现）
│       │   ├── register_ops.py
│       │   ├── activation.py
│       │   ├── normalization.py
│       │   └── rotary.py
│       └── vendor/          # VENDOR 后端（厂商特定实现）
│           ├── cuda/        # NVIDIA CUDA
│           ├── ascend/      # 华为昇腾 NPU
│           ├── metax/       # 沐曦 MetaX
│           └── iluvatar/    # 天数智芯 Iluvatar
│
├── worker/                  # WorkerFL：自定义 worker
├── attention/               # 注意力机制相关
├── distributed/             # 分布式通信（FlagCX）
├── compilation/             # 图编译（GraphWrapper）
├── models/                  # 模型适配
└── kv_connector/            # KV Cache 连接器（PD 分离）
```

---

## 3. 设计思路

### 3.1 插件注册机制

通过 `pyproject.toml` 的 entry points 注册：

```toml
[project.entry-points."vllm.platform_plugins"]
fl = "vllm_fl:register"

[project.entry-points."vllm.general_plugins"]
fl = "vllm_fl:register_model"
```

- `register()` → 返回 `"vllm_fl.platform.PlatformFL"` 类路径，vLLM 加载为平台实现
- `register_model()` → 注册 FlagCX connector、量化 kernel、自定义 router 等

### 3.2 算子调度系统（Dispatch）

这是项目最核心的设计，采用 **策略模式 + 注册表模式 + 优先级排序**：

```
┌─────────────────────────────────────────────────────────┐
│                    call_op("rms_norm", ...)              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              OpManager.call(op_name, *args)              │
│  1. ensure_initialized() → 注册所有后端                   │
│  2. resolve(op_name) → 选择最优实现                       │
│  3. fn(*args, **kwargs) → 执行                           │
│  4. fallback → 失败时尝试下一个实现（非 strict 模式）      │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  OpRegistry  │ │  Policy  │ │ match_token  │
│ {op: impls}  │ │ prefer   │ │ "flagos"     │
│              │ │ strict   │ │ "vendor"     │
│              │ │ per_op   │ │ "reference"  │
│              │ │ deny/    │ │ "vendor:cuda"│
│              │ │ allow    │ │ "impl:id"    │
└──────────────┘ └──────────┘ └──────────────┘
```

### 3.3 三层后端优先级

| 层级 | Kind | 优先级 | 说明 |
|------|------|--------|------|
| DEFAULT | `flagos` | 150 | FlagGems（Triton 实现），跨芯片统一 |
| VENDOR | `vendor` | 100 | 厂商特定优化（CUDA kernels、NPU ops） |
| REFERENCE | `reference` | 50 | PyTorch 原生实现，兜底保证正确性 |

默认选择顺序：`flagos → vendor → reference`（可通过环境变量或配置覆盖）

### 3.4 策略配置

通过环境变量控制调度行为：

| 环境变量 | 作用 |
|----------|------|
| `VLLM_FL_PREFER` | 设置偏好后端（`flagos`/`vendor`/`reference`） |
| `VLLM_FL_STRICT` | 严格模式，失败不 fallback |
| `VLLM_FL_CONFIG` | YAML 配置文件路径，支持 per-op 精细控制 |
| `VLLM_FL_DISPATCH_DEBUG` | 打印调度详情 |

---

## 4. 完整调用流程示例

以 `rms_norm` 算子为例，展示从 vLLM 启动到算子执行的完整链路：

### Step 1: vLLM 加载插件

```python
# vLLM 启动时扫描 entry points
# → 发现 vllm.platform_plugins.fl = "vllm_fl:register"
# → 调用 vllm_fl.register()

def register():
    # 1. patch transformers
    # 2. 设置 multiprocessing start method
    # 3. 返回平台类路径
    return "vllm_fl.platform.PlatformFL"
```

### Step 2: PlatformFL 初始化

```python
class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()           # 自动检测当前硬件
    device_type = device_info.device_type  # "cuda" / "npu" / "musa" / "ptpu"
    vendor_name = device_info.vendor_name  # "nvidia" / "ascend" / "metax" / ...
    dist_backend = "flagcx" if "FLAGCX_PATH" in os.environ else "nccl"
```

### Step 3: 配置更新

```python
# vLLM 调用 PlatformFL.check_and_update_config(vllm_config)
@classmethod
def check_and_update_config(cls, vllm_config):
    # 设置自定义 worker
    parallel_config.worker_cls = "vllm_fl.worker.worker.WorkerFL"
    # 根据设备类型设置 block_size
    if cls.device_type == "npu":
        cache_config.block_size = 128
    elif cls.device_type == "musa":
        cache_config.block_size = 64
    else:
        cache_config.block_size = 16
```

### Step 4: 算子调用（核心流程）

```python
# 模型前向传播中调用 rms_norm
from vllm_fl.dispatch import call_op

output = call_op("rms_norm", x, residual, weight, epsilon=1e-6)
```

内部执行流程：

```python
# call_op → get_default_manager().call("rms_norm", x, residual, weight, epsilon=1e-6)

class OpManager:
    def call(self, op_name, *args, **kwargs):
        # 1. resolve: 选择最优实现
        fn = self.resolve(op_name)
        # 2. 执行
        return fn(*args, **kwargs)

    def resolve(self, op_name):
        # 1. 确保初始化（首次调用时注册所有后端）
        self.ensure_initialized()

        # 2. 获取当前策略
        policy = get_policy()  # 默认: prefer="flagos", strict=False

        # 3. 检查缓存
        cache_key = (op_name, policy.fingerprint(), epoch)
        if cache_key in self._dispatch_cache:
            return self._dispatch_cache[cache_key]

        # 4. 获取所有注册的实现
        candidates = registry.get_implementations("rms_norm")
        # → [OpImpl("default.flagos", priority=150),
        #    OpImpl("vendor.cuda", priority=100),
        #    OpImpl("reference.pytorch", priority=50)]

        # 5. 过滤：vendor allow/deny + is_available()
        candidates = [c for c in candidates if c.is_available()]

        # 6. 按策略顺序选择
        order = ["flagos", "vendor", "reference"]  # 默认顺序
        for token in order:
            matches = [c for c in candidates if match_token(c, token)]
            if matches:
                matches.sort(key=lambda x: x.priority, reverse=True)
                chosen = matches[0]  # → "default.flagos" (priority=150)
                break

        # 7. 缓存并返回
        self._dispatch_cache[cache_key] = chosen.fn
        return chosen.fn
```

### Step 5: 后端注册（初始化时）

```python
# ensure_initialized() 触发 builtin_ops.register_builtins(registry)
# → 各后端的 register_ops.py 被调用

# flaggems/register_ops.py
def register_builtins(registry):
    backend = FlagGemsBackend()
    impls = [
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=backend.rms_norm,
            priority=BackendPriority.DEFAULT,  # 150
        ),
        # ... 其他算子
    ]
    # 只注册 FlagGems 可用的算子
    filtered = [impl for impl in impls if use_flaggems_op(impl.op_name)]
    registry.register_many(filtered)
```

### Step 6: Fallback 机制（非 strict 模式）

```python
# 如果 flagos 实现失败，自动尝试下一个
def call(self, op_name, *args, **kwargs):
    candidates = self.resolve_candidates(op_name)
    # → [flagos(150), vendor(100), reference(50)]

    for impl in candidates:
        try:
            return impl.fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Op '{op_name}' impl '{impl.impl_id}' failed: {e}")
            self._failed_impls.setdefault(op_name, set()).add(impl.impl_id)
            continue

    raise RuntimeError(f"All implementations failed for op='{op_name}'")
```

---

## 5. 关键设计模式

| 模式 | 应用位置 | 说明 |
|------|----------|------|
| 策略模式 | `SelectionPolicy` | 运行时可切换的选择策略 |
| 注册表模式 | `OpRegistry` | 线程安全的算子实现注册 |
| 单例模式 | `get_default_manager()` | 全局唯一的 OpManager |
| 模板方法 | `VLLMFLBackendBase` | 后端抽象基类定义接口 |
| 装饰器模式 | `_bind_is_available()` | 为函数绑定可用性检查 |
| 缓存模式 | `_dispatch_cache` | 避免重复 resolve 开销 |
| Context Manager | `policy_context()` | 临时切换策略 |

---

## 6. 支持的算子

| 算子名 | 说明 | 可用后端 |
|--------|------|----------|
| `silu_and_mul` | SiLU 激活 + 乘法 | flagos, vendor, reference |
| `gelu_and_mul` | GELU 激活 + 乘法 | flagos, vendor, reference |
| `rms_norm` | RMS 归一化 | flagos, vendor, reference |
| `rotary_embedding` | 旋转位置编码 | flagos, vendor, reference |
| `attention_backend` | 注意力后端选择 | flagos, vendor |
| `moe_align_block_size` | MoE 对齐 | flagos, vendor |
| `moe_sum` | MoE 求和 | flagos, vendor |
| `topk_softmax` | TopK Softmax | flagos, vendor |
| `dispatch_fused_moe_kernel` | 融合 MoE kernel | flagos, vendor |
| `grouped_topk` | 分组 TopK | flagos, vendor |

---

## 7. 快速上手

```python
# 使用 vLLM 正常启动，插件自动生效
from vllm import LLM

llm = LLM(model="Qwen/Qwen2-7B")  # PlatformFL 自动接管

# 手动调用 dispatch API（高级用法）
from vllm_fl.dispatch import call_op, resolve_op

# 调用 rms_norm
output = call_op("rms_norm", x, residual, weight, epsilon=1e-6)

# 只 resolve 不调用
fn = resolve_op("rms_norm")
output = fn(x, residual, weight, epsilon=1e-6)
```

### 环境变量配置示例

```bash
# 优先使用厂商实现
export VLLM_FL_PREFER=vendor

# 严格模式（不 fallback）
export VLLM_FL_STRICT=1

# 调试模式（打印调度详情）
export VLLM_FL_DISPATCH_DEBUG=1

# 使用 FlagCX 通信
export FLAGCX_PATH=/path/to/flagcx
```
