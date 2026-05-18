# 使用 AI Agent 自动化适配指南

> 基于 vLLM-plugin-FL 项目的架构文档、硬件适配指南和模型适配指南

---

## 概述

根据我们整理的文档（[architecture.md](architecture.md)、[hardware_adaptation_guide.md](hardware_adaptation_guide.md) 和 [model_adaptation_guide.md](model_adaptation_guide.md)），vLLM-plugin-FL 有两类适配需求：

1. **硬件适配**：添加新的硬件厂商支持（如 MUSA、Sunrise）
2. **模型适配**：添加新的模型架构支持（如 GLM-5、Qwen3.5、Kimi-K2.5）

这两类适配都可以通过配置 AI Agent（如 Claude Code、自定义 Agent）来半自动化或全自动化完成。

---

## 一、适配类型对比

| 维度 | 硬件适配 | 模型适配 |
|------|----------|----------|
| **核心工作** | 实现算子后端 + 平台层集成 | 注册模型配置 + 打补丁 + 模型实现 |
| **涉及文件数** | 10-15 个 | 2-7 个（视适配级别） |
| **代码量** | 500-1500 行 | 50-1500 行（视适配级别） |
| **复杂度** | 高（需要硬件知识 + 算子实现） | 低~高（取决于架构差异程度） |
| **可自动化程度** | 中等（算子实现需要人工） | 高（Level 1/2 高度模板化） |
| **适配级别** | 单一级别 | 三级：最小/中等/完整（见 [model_adaptation_guide.md](model_adaptation_guide.md)） |

---

## 二、硬件适配的 Agent 配置

### 2.1 适配流程

硬件适配分为 **6 个步骤**（见 [hardware_adaptation_guide.md](hardware_adaptation_guide.md)）：

```
1. 注册设备映射 (utils.py)
2. 平台层适配 (platform.py)
3. 创建 vendor 后端 (dispatch/backends/vendor/<name>/)
4. 编写调度配置 (<name>.yaml)
5. 图编译适配 (graph.py) [可选]
6. 通信层适配 (flagcx.py) [可选]
```

### 2.2 Agent Memory 配置 (CLAUDE.md)

在项目根目录或 `~/.claude/` 创建 `CLAUDE.md`，包含以下内容：

```markdown
# vLLM-plugin-FL 硬件适配 Agent 配置

## 项目架构
- 阅读 docs/architecture.md 了解整体设计
- 阅读 docs/hardware_adaptation_guide.md 了解适配步骤

## 硬件适配 Workflow

当用户请求"适配 <硬件名> 硬件"时，按以下步骤执行：

### Step 1: 收集硬件信息
询问用户以下信息：
- vendor_name: FlagGems DeviceDetector 检测到的厂商标识（如 "sunrise"）
- device_type: PyTorch 设备类型（如 "cuda", "npu", "musa", "ptpu"）
- device_name: vLLM 平台使用的设备名称
- 是否 CUDA 兼容？
- 推荐的 KV cache block_size（默认 16）
- 是否支持图捕获（CUDA Graph 类似功能）？
- 硬件 SDK 提供的 PyTorch 接口（如 torch.musa, torch.ptpu）

### Step 2: 执行适配 Checklist
按照 docs/hardware_adaptation_guide.md 的 Checklist 逐项完成：

1. **utils.py**
   - 在 VENDOR_DEVICE_MAP 添加条目
   - 在 DeviceInfo.supported_device 列表添加

2. **platform.py**
   - get_device_capability(): 添加设备类型判断
   - is_cuda_alike() / is_cuda(): 如果 CUDA 兼容，添加判断
   - check_and_update_config(): 设置 block_size
   - is_pin_memory_available(): 按需添加

3. **dispatch/backends/vendor/<name>/**
   - 创建目录结构（__init__.py, <name>.py, register_ops.py, impl/）
   - 实现 Backend 类（继承 Backend 基类）
   - 实现 is_available() 方法（检测硬件）
   - 最小实现：只实现 attention_backend() 方法
   - 注册算子（register_ops.py）

4. **dispatch/config/<name>.yaml**
   - 创建调度配置文件
   - 设置 prefer, strict, op_backends
   - 配置 flagos_blacklist（如有不兼容算子）

5. **可选：graph.py**
   - 如果支持图捕获，添加 Graph 类映射

6. **可选：flagcx.py**
   - 如果使用 FlagCX，添加设备上下文切换

### Step 3: 参考现有实现
- CUDA 兼容设备：参考 vllm_fl/dispatch/backends/vendor/musa/
- 非 CUDA 设备：参考 vllm_fl/dispatch/backends/vendor/sunrise/

### Step 4: 验证
- 运行 `python -c "from vllm_fl.utils import DeviceInfo; print(DeviceInfo().vendor_name)"`
- 检查设备是否被正确识别

## 重要约束
- 算子实现（impl/）需要硬件 SDK 知识，Agent 只能生成模板，具体实现需要人工完成
- attention_backend 是核心，必须实现
- 其他算子（rms_norm, silu_and_mul, rotary_embedding）可以依赖 FlagGems 兜底
```

### 2.3 自定义 Skill 配置

创建 `/hardware-adapt` skill（需要 Claude Agent SDK 或 MCP Server）：

```python
# ~/.claude/skills/hardware_adapt.py

from claude_agent_sdk import Skill, Parameter

class HardwareAdaptSkill(Skill):
    name = "hardware-adapt"
    description = "适配新硬件到 vLLM-plugin-FL"

    parameters = [
        Parameter("vendor_name", str, "厂商标识（如 sunrise）"),
        Parameter("device_type", str, "设备类型（如 ptpu）"),
        Parameter("cuda_compatible", bool, "是否 CUDA 兼容", default=False),
        Parameter("block_size", int, "KV cache block size", default=16),
    ]

    async def execute(self, vendor_name, device_type, cuda_compatible, block_size):
        # 1. 读取 hardware_adaptation_guide.md
        guide = await self.read_file("docs/hardware_adaptation_guide.md")

        # 2. 生成 checklist
        checklist = self.generate_checklist(vendor_name, device_type)

        # 3. 逐步执行
        await self.update_utils_py(vendor_name, device_type)
        await self.update_platform_py(device_type, cuda_compatible, block_size)
        await self.create_vendor_backend(vendor_name, device_type)
        await self.create_yaml_config(vendor_name)

        return f"硬件 {vendor_name} 适配完成，请人工实现算子 impl/"
```

### 2.4 使用示例

```bash
# 在 Claude Code 中
/hardware-adapt vendor_name=sunrise device_type=ptpu cuda_compatible=false block_size=16

# 或者直接对话
"请帮我适配 Sunrise (ptpu) 硬件到这个项目，参考 docs/hardware_adaptation_guide.md"
```

---

## 三、模型适配的 Agent 配置

### 3.1 适配流程

模型适配分为 **三个级别**（见 [model_adaptation_guide.md](model_adaptation_guide.md)）：

| 级别 | 典型场景 | Agent 自动化程度 |
|------|----------|------------------|
| **Level 1 最小适配** | 模型基于已有架构，只需注册入口 | 95%（几乎全自动） |
| **Level 2 中等适配** | 新 model_type + config 桥接 + 补丁 | 80%（补丁需人工审核） |
| **Level 3 完整适配** | 全新架构，需完整模型实现 | 40%（模型实现需人工） |

```
Level 1: 创建模型包装类 → ModelRegistry.register_model()
Level 2: 创建 Config 类 → _CONFIG_REGISTRY 注册 → 编写补丁
Level 3: 创建 Config 类 → 实现完整模型类 → 注册 Config + Model → 编写补丁
```

### 3.2 Agent Memory 配置 (CLAUDE.md)

```markdown
## 模型适配 Workflow

当用户请求"适配 <模型名> 模型"时，按以下步骤执行：

### Step 1: 判断适配级别
阅读 docs/model_adaptation_guide.md，根据以下信息判断适配级别：
- 模型 HuggingFace config.json 中的 model_type 和 architectures
- 模型是否基于已有架构（DeepseekV2/V3、Qwen2MoE、LLaMA 等）
- vLLM 当前版本是否已支持该 model_type
- transformers 当前版本是否已识别该 model_type

判断规则：
- model_type 已被 transformers 识别 + 架构基于已有模型 → Level 1（最小适配）
- model_type 未被识别 + 架构基于已有模型 → Level 2（中等适配）
- 全新架构 → Level 3（完整适配）

### Step 2: 收集模型信息
询问用户：
- 模型名称和 HuggingFace 仓库地址
- model_type 字符串（从 config.json 获取）
- architectures 列表（从 config.json 获取）
- 基础架构类型（如 DeepseekV2, Qwen2MoE, LLaMA）
- 特殊配置字段（如 GLM-5 的 index_topk, index_n_heads）
- 是否需要 tokenizer 补丁？
- 是否需要模型层补丁（如 attention、MoE）？
- 是否为多模态模型（有 text_config 嵌套）？

### Step 3: 按级别执行

#### Level 1（最小适配）：
1. 创建 vllm_fl/models/<model>.py（包装已有模型类）
2. 在 __init__.py 添加 ModelRegistry.register_model()
3. 创建 examples/<model>_offline_inference.py

#### Level 2（中等适配）：
1. 创建 vllm_fl/configs/<model>.py（继承已有 Config）
2. 在 __init__.py 添加 _CONFIG_REGISTRY 注册
3. 创建 vllm_fl/patches/<model>.py（兼容性补丁）
4. 在 register() 中调用 apply_platform_patches()
5. 创建 examples/<model>_offline_inference.py

#### Level 3（完整适配）：
1. 创建 vllm_fl/configs/<model>.py（完整 Config 类）
2. 创建 vllm_fl/models/<model>.py（完整模型实现）
3. 在 __init__.py 添加 _CONFIG_REGISTRY + ModelRegistry 注册
4. 创建 vllm_fl/patches/<model>.py（按需）
5. 创建 examples/<model>_offline_inference.py

### Step 4: 参考现有实现
- Level 1 参考：vllm_fl/models/kimi_k25.py
- Level 2 参考：vllm_fl/configs/glm_moe_dsa.py + vllm_fl/patches/glm_moe_dsa.py
- Level 3 参考：vllm_fl/configs/qwen3_5_moe.py + vllm_fl/models/qwen3_5.py

### Step 5: 验证
- 运行 examples/<model>_offline_inference.py
- 检查模型是否能正确加载和推理

## 重要约束
- Level 3 的模型 forward 实现需要深入理解模型架构，Agent 只能生成骨架
- 补丁编写需要理解 vLLM 内部实现，Agent 可以参考已有补丁模式
- 多模态模型的视觉编码器部分通常需要人工实现
```

### 3.3 自定义 Skill 配置

```python
# ~/.claude/skills/model_adapt.py

class ModelAdaptSkill(Skill):
    name = "model-adapt"
    description = "适配新模型到 vLLM-plugin-FL"

    parameters = [
        Parameter("model_name", str, "模型名称（如 GLM-5）"),
        Parameter("model_type", str, "model_type 字符串"),
        Parameter("architectures", str, "architectures 类名"),
        Parameter("base_config", str, "基类配置（如 DeepseekV2Config）", default=None),
        Parameter("level", int, "适配级别 1/2/3", default=2),
        Parameter("special_fields", dict, "特殊配置字段", default={}),
    ]

    async def execute(self, model_name, model_type, architectures, base_config, level, special_fields):
        # 1. 读取 model_adaptation_guide.md
        guide = await self.read_file("docs/model_adaptation_guide.md")

        if level == 1:
            # Level 1: 最小适配
            await self.create_model_wrapper(model_name, architectures, base_config)
            await self.register_model(architectures, model_name)

        elif level == 2:
            # Level 2: 中等适配
            await self.create_config_class(model_name, model_type, base_config, special_fields)
            await self.register_config(model_type)
            if await self.ask_user("是否需要创建兼容性补丁？"):
                await self.create_patch_file(model_name)

        elif level == 3:
            # Level 3: 完整适配
            await self.create_config_class(model_name, model_type, base_config, special_fields)
            await self.create_model_implementation(model_name, architectures)
            await self.register_config(model_type)
            await self.register_model(architectures, model_name)

        # 创建推理示例
        await self.create_inference_example(model_name)

        return f"模型 {model_name} (Level {level}) 适配完成"
```

---

## 四、Agent 能力边界

### 4.1 Agent 可以自动化的部分

✅ **完全自动化**：
- 文件结构创建（目录、__init__.py）
- 模板代码生成（Backend 类骨架、register_ops.py 模板）
- 配置文件生成（YAML、模型 config）
- 注册代码更新（utils.py、__init__.py）
- 模型包装类生成（Level 1 适配）
- Config 桥接类生成（Level 2 适配）
- 推理示例脚本生成
- 文档生成（README、注释）

✅ **半自动化**（Agent 生成模板，人工填充）：
- 算子实现（impl/activation.py, impl/attention.py）
- 兼容性补丁（patches/<model>.py）— Agent 可参考已有模式
- 硬件特定逻辑（is_available() 检测）
- 完整模型实现（Level 3 适配）— Agent 可生成骨架

### 4.2 Agent 无法自动化的部分

❌ **需要人工完成**：
- **算子具体实现**：需要硬件 SDK 知识和性能优化经验
- **Attention 后端实现**：946 行的自定义 attention（如 Sunrise PR #148）
- **硬件调试**：block_size 调优、内存布局优化
- **复杂模型 forward 逻辑**：如 Qwen3.5 的 linear attention + sliding window 混合
- **多模态视觉编码器**：需要理解图像/视频处理流程
- **性能测试**：benchmark、精度验证
- **monkey-patch 调试**：需要理解 vLLM 内部实现细节

### 4.3 按模型适配级别的自动化程度

| 适配级别 | Agent 可完成 | 需人工审核 | 需人工实现 |
|----------|-------------|-----------|-----------|
| Level 1 | 模型包装类、注册代码、推理示例 | load_weights 映射 | 无（或极少） |
| Level 2 | Config 类、注册代码、补丁模板 | 补丁逻辑正确性 | 复杂补丁 |
| Level 3 | Config 类、模型骨架、注册代码 | 整体架构设计 | forward 实现、自定义层 |

---

## 五、推荐的 Agent 配置方案

### 方案 A：Claude Code + CLAUDE.md（推荐）

**优点**：
- 无需额外开发，直接使用 Claude Code
- CLAUDE.md 提供项目上下文和 workflow
- 适合快速原型和一次性适配

**配置**：
```bash
# 项目根目录创建 .claude/CLAUDE.md
mkdir -p .claude
cat > .claude/CLAUDE.md << 'EOF'
# vLLM-plugin-FL 适配指南

## 项目架构
- 阅读 docs/architecture.md 了解整体设计

## 硬件适配
- 阅读 docs/hardware_adaptation_guide.md 了解适配步骤

## 模型适配
- 阅读 docs/model_adaptation_guide.md 了解三级适配流程
- Level 1 参考：vllm_fl/models/kimi_k25.py
- Level 2 参考：vllm_fl/configs/glm_moe_dsa.py + vllm_fl/patches/glm_moe_dsa.py
- Level 3 参考：vllm_fl/configs/qwen3_5_moe.py + vllm_fl/models/qwen3_5.py

## Workflow
[粘贴上面的 workflow 内容]
EOF
```

### 方案 B：自定义 MCP Server（高级）

**优点**：
- 可以封装复杂逻辑（如自动读取 PR diff、生成代码）
- 可以集成外部工具（如硬件 SDK 文档查询）
- 适合团队重复使用

**实现**：
```python
# mcp_server_vllm_fl.py

from mcp import MCPServer, Tool

server = MCPServer("vllm-fl-adapter")

@server.tool()
async def hardware_adapt(vendor_name: str, device_type: str):
    """适配新硬件"""
    # 读取 guide
    guide = await read_file("docs/hardware_adaptation_guide.md")

    # 生成代码
    files = generate_hardware_files(vendor_name, device_type)

    # 返回待创建的文件列表
    return {"files": files, "next_steps": "请人工实现算子"}

@server.tool()
async def model_adapt(model_name: str, model_type: str, architectures: str, level: int = 2):
    """适配新模型（支持三级适配）"""
    # 读取 guide
    guide = await read_file("docs/model_adaptation_guide.md")

    if level == 1:
        # 最小适配：生成包装类 + 注册代码
        model_code = generate_model_wrapper(model_name, architectures)
        register_code = generate_model_register(architectures, model_name)
        return {"model": model_code, "register": register_code}

    elif level == 2:
        # 中等适配：生成 config + 注册 + 补丁模板
        config_code = generate_config_class(model_name, model_type)
        register_code = generate_config_register(model_type)
        patch_code = generate_patch_template(model_name)
        return {"config": config_code, "register": register_code, "patch": patch_code}

    elif level == 3:
        # 完整适配：生成 config + 模型骨架 + 注册
        config_code = generate_config_class(model_name, model_type)
        model_code = generate_model_skeleton(model_name, architectures)
        register_code = generate_full_register(model_type, architectures, model_name)
        return {
            "config": config_code,
            "model": model_code,
            "register": register_code,
            "next_steps": "请人工实现模型 forward 逻辑"
        }
```

### 方案 C：Agent SDK 自定义 Agent（最灵活）

**优点**：
- 完全自定义 agent 行为
- 可以实现多轮对话、状态管理
- 适合复杂的适配流程

**实现**：
```python
# hardware_adapter_agent.py

from claude_agent_sdk import Agent, State

class HardwareAdapterAgent(Agent):
    def __init__(self):
        super().__init__(
            name="hardware-adapter",
            system_prompt=self.load_prompt(),
            tools=[self.create_files, self.update_registry]
        )

    def load_prompt(self):
        return """
        你是 vLLM-plugin-FL 硬件适配专家。
        参考 docs/hardware_adaptation_guide.md 执行适配。
        """

    async def run(self, vendor_name, device_type):
        # 多轮对话收集信息
        info = await self.collect_hardware_info(vendor_name, device_type)

        # 生成文件
        await self.generate_files(info)

        # 验证
        await self.verify_adaptation()
```

---

## 六、实战示例

### 示例 1：使用 Claude Code 适配 Sunrise 硬件

```
User: 请帮我适配 Sunrise (ptpu) 硬件，参考 docs/hardware_adaptation_guide.md

Claude Code:
1. 读取 docs/hardware_adaptation_guide.md
2. 询问：是否 CUDA 兼容？block_size 推荐值？
3. 执行 Checklist：
   - 更新 vllm_fl/utils.py (VENDOR_DEVICE_MAP)
   - 更新 vllm_fl/platform.py (get_device_capability)
   - 创建 vllm_fl/dispatch/backends/vendor/sunrise/
   - 创建 vllm_fl/dispatch/config/sunrise.yaml
4. 生成 TODO：请人工实现 impl/attention.py
```

### 示例 2：使用 Claude Code 适配 GLM-6 模型（Level 2）

```
User: 请帮我适配 GLM-6 模型，model_type 是 glm6，基于 DeepseekV2 架构

Claude Code:
1. 读取 docs/model_adaptation_guide.md
2. 判断：新 model_type + 基于已有架构 → Level 2 中等适配
3. 执行：
   - 创建 vllm_fl/configs/glm6.py（继承 DeepseekV2Config）
   - 在 __init__.py 添加 _CONFIG_REGISTRY["glm6"] = Glm6Config
   - 创建 vllm_fl/patches/glm6.py（补丁模板）
   - 在 register() 中调用 apply_platform_patches()
   - 创建 examples/glm6_offline_inference.py
4. 提示：请检查补丁逻辑是否正确
```

### 示例 3：使用自定义 Skill 适配 Kimi-K3 模型（Level 1）

```bash
/model-adapt model_name=Kimi-K3 model_type=kimi_k3 architectures=KimiK3ForCausalLM base_config=DeepseekV2Config level=1
```

### 示例 4：使用 Claude Code 适配全新架构模型（Level 3）

```
User: 请帮我适配 Qwen4-MoE 模型，这是全新架构，config.json 中 model_type 是 qwen4_moe

Claude Code:
1. 读取 docs/model_adaptation_guide.md
2. 判断：全新架构 → Level 3 完整适配
3. 执行：
   - 创建 vllm_fl/configs/qwen4_moe.py（完整 Config 类）
   - 创建 vllm_fl/models/qwen4_moe.py（模型骨架 + TODO 标记）
   - 在 __init__.py 添加 _CONFIG_REGISTRY + ModelRegistry 注册
   - 创建 examples/qwen4_moe_offline_inference.py
4. 生成 TODO：请人工实现模型 forward 逻辑和自定义层
```

---

## 七、总结

| 适配类型 | Agent 自动化程度 | 推荐方案 | 人工工作量 |
|----------|------------------|----------|------------|
| 硬件适配 | 60%（模板生成） | Claude Code + CLAUDE.md | 中等（算子实现） |
| 模型适配 Level 1 | 95%（几乎全自动） | Claude Code + CLAUDE.md | 极低（审核即可） |
| 模型适配 Level 2 | 80%（高度模板化） | Claude Code + CLAUDE.md | 低（补丁编写） |
| 模型适配 Level 3 | 40%（骨架生成） | Claude Code + CLAUDE.md | 高（模型实现） |

**关键建议**：
1. **先用 CLAUDE.md**：最快上手，适合 1-2 次适配
2. **再考虑 MCP Server**：团队重复使用，封装最佳实践
3. **最后用 Agent SDK**：需要复杂状态管理和多轮对话

**Agent 的价值**：
- ✅ 减少重复劳动（文件创建、模板生成、注册代码）
- ✅ 保证一致性（遵循项目规范和命名约定）
- ✅ 降低学习成本（新人快速上手，自动判断适配级别）
- ✅ 自动化决策（根据模型信息判断 Level 1/2/3）
- ❌ 无法替代专业知识（算子优化、模型调试、复杂 forward 实现）

**相关文档**：
- [architecture.md](architecture.md) — 项目整体架构与设计
- [hardware_adaptation_guide.md](hardware_adaptation_guide.md) — 硬件适配 6 步流程
- [model_adaptation_guide.md](model_adaptation_guide.md) — 模型适配三级流程
