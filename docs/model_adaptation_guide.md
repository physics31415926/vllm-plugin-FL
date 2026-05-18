# vLLM-plugin-FL 新模型适配指南

> 参考 Commit: [dc701e0 GLM-5](https://github.com/flagos-ai/vllm-plugin-FL/commit/dc701e0)、[20cdfc0 Qwen3.5](https://github.com/flagos-ai/vllm-plugin-FL/commit/20cdfc0)、[37db70d Kimi-K2.5](https://github.com/flagos-ai/vllm-plugin-FL/commit/37db70d)

---

## 概览

当新模型发布后，如果 vLLM 上游尚未支持，可以通过 vLLM-plugin-FL 的 OOT 插件机制快速适配。根据模型与现有架构的差异程度，适配工作分为三个级别：

| 级别 | 典型场景 | 涉及文件 | 代码量 | 参考 |
|------|----------|----------|--------|------|
| **最小适配** | 模型架构已被 vLLM 支持，只需注册入口 | 2-3 个 | 50-100 行 | Kimi-K2.5 |
| **中等适配** | 模型基于已有架构变体，需要 config 桥接 + 补丁 | 3-5 个 | 100-350 行 | GLM-5 |
| **完整适配** | 全新架构，需要完整模型实现 | 5-7 个 | 500-1500 行 | Qwen3.5 MoE |

---

## 适配决策流程

```
新模型发布
    │
    ├─ 模型架构 vLLM 已支持？
    │   ├─ 是 → 直接使用 vLLM，无需适配
    │   └─ 否 ↓
    │
    ├─ 模型基于已有架构（如 DeepseekV2/V3）？
    │   ├─ 是 → config_type 不同？
    │   │       ├─ 是 → 中等适配（注册 config + 补丁）
    │   │       └─ 否 → 最小适配（注册模型类）
    │   └─ 否 → 完整适配（实现模型类 + config）
    │
    └─ 是否需要 transformers 兼容性修复？
        ├─ 是 → 编写 patches
        └─ 否 → 跳过
```

---

## Level 1: 最小适配（参考 Kimi-K2.5）

### 适用场景

- 模型架构基于 vLLM 已支持的模型（如 DeepseekV2/V3）
- HuggingFace config 中的 `model_type` 已被 transformers 识别
- 只需要一个轻量包装类来适配 vLLM 接口

### 步骤

#### 1. 创建模型文件

```python
# vllm_fl/models/kimi_k25.py

from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.interfaces import SupportsPP

class KimiK25ForConditionalGeneration(nn.Module, SupportsPP):
    """Kimi-K2.5: 基于 DeepseekV3 的文本生成模型"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # 提取 text_config（多模态模型常见模式）
        text_config = getattr(config, "text_config", config)

        # 复用 DeepseekV2Model 作为语言骨干
        self.language_model = DeepseekV2Model(...)
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(...)

    def forward(self, ...):
        # 调用语言模型 forward
        hidden_states = self.language_model(...)
        return hidden_states

    def load_weights(self, weights):
        # 处理权重名称映射
        ...
```

#### 2. 注册模型

在 `vllm_fl/__init__.py` 的 `register_model()` 中添加：

```python
def register_model():
    from vllm import ModelRegistry

    # Register Kimi-K2.5 model
    try:
        ModelRegistry.register_model(
            "KimiK25ForConditionalGeneration",
            "vllm_fl.models.kimi_k25:KimiK25ForConditionalGeneration"
        )
    except Exception as e:
        logger.error(f"Register KimiK25 model error: {str(e)}")
```

**关键点**：
- `ModelRegistry.register_model()` 的第一个参数是 HuggingFace config 中 `architectures` 字段的值
- 第二个参数是 `"模块路径:类名"` 格式

#### 3. 涉及文件

```
vllm_fl/
├── __init__.py              # 添加 ModelRegistry.register_model()
└── models/
    └── kimi_k25.py          # 模型实现（包装已有模型）
```

---

## Level 2: 中等适配（参考 GLM-5）

### 适用场景

- 模型基于已有架构（如 GLM-5 基于 DeepseekV2）
- HuggingFace config 中使用了新的 `model_type`（如 `"glm_moe_dsa"`）
- 当前 transformers 版本不识别该 `model_type`
- 需要兼容性补丁（如 transformers 版本差异）

### 步骤

#### 1. 创建 Config 类

```python
# vllm_fl/configs/glm_moe_dsa.py

from transformers import DeepseekV2Config

class GlmMoeDsaConfig(DeepseekV2Config):
    model_type = "glm_moe_dsa"

    def __init__(
        self,
        # 模型特有字段
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        indexer_rope_interleave=True,
        num_nextn_predict_layers=1,
        moe_layer_freq=1,
        scoring_func="sigmoid",
        ep_size=1,
        head_dim=None,
        rope_parameters=None,
        dtype="bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        # ... 设置所有特有字段
```

**关键点**：
- 继承最接近的已有 Config 类
- `model_type` 必须与 HuggingFace checkpoint 的 `config.json` 中的 `model_type` 一致
- 在 `__init__` 中声明模型特有字段，并传递 `**kwargs` 给父类

#### 2. 注册 Config

在 `vllm_fl/__init__.py` 的 `register_model()` 中添加：

```python
def register_model():
    # Register GLM-5 config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
```

**注意**：如果模型架构与父类完全兼容（如 GLM-5 可以复用 DeepseekV2 的模型实现），则不需要注册 `ModelRegistry`，只需注册 config 即可。vLLM 会根据 config 的继承关系自动匹配模型实现。

#### 3. 编写补丁（按需）

```python
# vllm_fl/patches/glm_moe_dsa.py

def apply_platform_patches():
    """在 register() 阶段调用，修复平台级兼容性"""
    patch_tokenizer_compat()

def apply_model_patches():
    """在 register_model() 阶段调用，修复模型级兼容性"""
    patch_is_deepseek_mla()
    patch_fp8_mqa_logits_dim()
```

常见补丁类型：

| 补丁类型 | 说明 | 示例 |
|----------|------|------|
| tokenizer 兼容 | transformers 版本差异导致 tokenizer 加载失败 | `patch_tokenizer_compat()` |
| 模型识别 | vLLM 内部逻辑不识别新 model_type | `patch_is_deepseek_mla()` |
| 精度修复 | FP8 量化时维度不匹配 | `patch_fp8_mqa_logits_dim()` |

补丁实现模式（monkey-patch）：

```python
def patch_is_deepseek_mla():
    """让 vLLM 识别 glm_moe_dsa 为 MLA 架构"""
    from vllm.model_executor.models import deepseek_v2

    _orig_fn = deepseek_v2._is_deepseek_mla

    def _patched_fn(config):
        if getattr(config, "model_type", None) == "glm_moe_dsa":
            return True
        return _orig_fn(config)

    deepseek_v2._is_deepseek_mla = _patched_fn
```

#### 4. 在 register() 中调用平台补丁

```python
def register():
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    return "vllm_fl.platform.PlatformFL"
```

#### 5. 涉及文件

```
vllm_fl/
├── __init__.py              # 注册 config + 调用补丁
├── configs/
│   └── glm_moe_dsa.py      # Config 类
└── patches/
    ├── __init__.py
    └── glm_moe_dsa.py       # 兼容性补丁
```

---

## Level 3: 完整适配（参考 Qwen3.5 MoE）

### 适用场景

- 全新模型架构，vLLM 没有可复用的实现
- 模型有独特的注意力机制（如 linear attention + full attention 混合）
- 需要自定义算子（如 FLA ops）

### 步骤

#### 1. 创建 Config 类

```python
# vllm_fl/configs/qwen3_5_moe.py

from transformers.configuration_utils import PretrainedConfig

class Qwen3_5MoeTextConfig(PretrainedConfig):
    model_type = "qwen3_5_moe_text"
    base_config_key = "text_config"

    def __init__(self, vocab_size=248320, hidden_size=2048, ...):
        # 完整定义所有模型参数
        ...

class Qwen3_5MoeConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {"text_config": Qwen3_5MoeTextConfig}

    def __init__(self, text_config=None, ...):
        if isinstance(text_config, dict):
            text_config = Qwen3_5MoeTextConfig(**text_config)
        self.text_config = text_config or Qwen3_5MoeTextConfig()
        ...
```

**多模态模型**通常有嵌套 config 结构（text_config + vision_config）。

#### 2. 实现模型类

```python
# vllm_fl/models/qwen3_5.py

class Qwen3_5MoeForConditionalGeneration(nn.Module, SupportsPP):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = config.text_config

        self.model = Qwen3_5MoeModel(vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(...)

    def forward(self, input_ids, positions, intermediate_tensors, ...):
        hidden_states = self.model(input_ids, positions, ...)
        return hidden_states

    def compute_logits(self, hidden_states, sampling_metadata):
        logits = self.logits_processor(self.lm_head, hidden_states, ...)
        return logits

    def load_weights(self, weights):
        # 定义权重映射和加载逻辑
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        ...
```

#### 3. 实现自定义算子（如需要）

```python
# vllm_fl/models/fla_ops.py

import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel(...):
    """Linear attention 的 Triton kernel 实现"""
    ...

def chunk_linear_attn(q, k, v, ...):
    """Linear attention 的 Python 接口"""
    ...
```

#### 4. 注册 Config + 模型

```python
def register_model():
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm_fl.configs.qwen3_5_moe import Qwen3_5MoeConfig

    # 注册 config
    _CONFIG_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeConfig

    # 注册模型
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_fl.models.qwen3_5:Qwen3_5MoeForConditionalGeneration"
    )
```

#### 5. 涉及文件

```
vllm_fl/
├── __init__.py              # 注册 config + model
├── configs/
│   └── qwen3_5_moe.py      # Config 类（可能含嵌套 config）
├── models/
│   ├── qwen3_5.py           # 完整模型实现
│   └── fla_ops.py           # 自定义算子（Triton kernel）
└── worker/
    └── worker.py            # 如需修改 worker 逻辑
```

---

## 通用模式总结

### 注册 API

| API | 用途 | 调用位置 |
|-----|------|----------|
| `_CONFIG_REGISTRY[model_type] = ConfigClass` | 注册新的 model_type，让 vLLM 能解析 HF config | `register_model()` |
| `ModelRegistry.register_model(arch, path)` | 注册模型实现类，让 vLLM 能加载模型 | `register_model()` |

### Config 继承选择

| 基类 | 适用场景 |
|------|----------|
| `DeepseekV2Config` | MLA 架构、MoE 模型（GLM-5、Kimi-K2.5） |
| `Qwen2Config` | Qwen 系列变体 |
| `LlamaConfig` | LLaMA 架构变体 |
| `PretrainedConfig` | 全新架构，无法复用已有 config |

### 模型实现复用

| 策略 | 说明 | 示例 |
|------|------|------|
| 直接包装 | 用已有模型类作为子模块 | Kimi-K2.5 包装 DeepseekV2Model |
| 继承修改 | 继承已有模型类，覆写部分方法 | — |
| 全新实现 | 从头实现 forward/load_weights | Qwen3.5 MoE |

### 权重加载模式

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    # 1. 定义堆叠参数映射（QKV、Gate/Up 合并）
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

    # 2. 定义需要忽略的权重前缀
    ignore_prefixes = ["visual.", "multi_modal_projector."]

    # 3. 处理权重名称重映射
    for name, loaded_weight in weights:
        # 跳过不需要的权重
        if any(name.startswith(p) for p in ignore_prefixes):
            continue
        # 处理堆叠参数
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                param = self.state_dict()[name.replace(weight_name, param_name)]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
        else:
            # 默认加载
            param = self.state_dict()[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

---

## Checklist：新模型适配清单

```
□ 确认适配级别
  □ 模型架构是否已被 vLLM 支持？
  □ model_type 是否已被 transformers 识别？
  □ 是否需要自定义算子？

□ Config 适配
  □ 创建 vllm_fl/configs/<model>.py
  □ 继承合适的基类（DeepseekV2Config / PretrainedConfig / ...）
  □ 设置正确的 model_type
  □ 声明模型特有字段
  □ 在 __init__.py 中注册到 _CONFIG_REGISTRY

□ 模型实现（如需要）
  □ 创建 vllm_fl/models/<model>.py
  □ 实现 __init__、forward、compute_logits、load_weights
  □ 处理权重名称映射
  □ 在 __init__.py 中注册到 ModelRegistry

□ 补丁（如需要）
  □ 创建 vllm_fl/patches/<model>.py
  □ 实现 apply_platform_patches()（register 阶段）
  □ 实现 apply_model_patches()（register_model 阶段）
  □ 在 __init__.py 中调用补丁

□ 验证
  □ python -c "from vllm import LLM; llm = LLM(model='<path>')"
  □ 检查模型能否正常加载
  □ 运行推理测试
```

---

## 三个 Commit 的对比

| 维度 | Kimi-K2.5 (37db70d) | GLM-5 (dc701e0) | Qwen3.5 (20cdfc0) |
|------|---------------------|-----------------|-------------------|
| 适配级别 | 最小 | 中等 | 完整 |
| 改动文件 | 4 | 8 | 7 |
| 新增代码 | ~256 行 | ~315 行 | ~1323 行 |
| Config | 无（复用 HF 已有） | 新建（继承 DeepseekV2Config） | 新建（继承 PretrainedConfig） |
| 模型实现 | 包装 DeepseekV2Model | 无（复用 vLLM 内置） | 全新实现 |
| 补丁 | 无 | 3 个（tokenizer + MLA 识别 + FP8） | 无 |
| 自定义算子 | 无 | 无 | 有（FLA linear attention） |
| 注册方式 | ModelRegistry | _CONFIG_REGISTRY | 两者都有 |

---

## 常见问题

### Q: 什么时候需要注册 Config vs 注册 Model？

- **只注册 Config**：模型架构与父类完全兼容，vLLM 能通过 config 继承关系自动匹配模型实现（如 GLM-5 → DeepseekV2）
- **只注册 Model**：config 已被 transformers 识别，但 vLLM 没有对应的模型实现（如 Kimi-K2.5）
- **两者都注册**：新的 model_type + 新的模型实现（如 Qwen3.5）

### Q: 补丁应该放在 register() 还是 register_model()？

- `register()` 阶段（平台补丁）：transformers 兼容性、tokenizer 修复等全局性补丁
- `register_model()` 阶段（模型补丁）：模型特定逻辑修复（如 MLA 识别、量化维度修复）

### Q: 如何确定 model_type？

查看模型 HuggingFace 仓库中的 `config.json`：
```json
{
  "model_type": "glm_moe_dsa",
  "architectures": ["GlmMoeDsaForCausalLM"],
  ...
}
```
- `model_type` → 用于 `_CONFIG_REGISTRY` 注册
- `architectures[0]` → 用于 `ModelRegistry.register_model()` 注册

### Q: 模型适配后如何测试？

```python
# examples/<model>_offline_inference.py

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    trust_remote_code=True,
    tensor_parallel_size=1,  # 按需调整
)

prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=100))
for output in outputs:
    print(output.outputs[0].text)
```
