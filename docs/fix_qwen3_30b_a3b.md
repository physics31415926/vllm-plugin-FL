# Fix: Qwen3-30B-A3B (MoE) Offline Inference

## 模型信息
- 模型路径：`/nfs/wlx/models/Qwen3-30B-A3B`
- 架构：`Qwen3MoeForCausalLM`（MoE）
- 脚本：`examples/qwen3_30b_a3b_offline_inference.py`

---

## 错误记录

### Error 1: FusedTopKRouter.__init__() got unexpected keyword argument 'moe_config'

**完整报错：**
```
(EngineCore pid=...) TypeError: FusedTopKRouter.__init__() got an unexpected keyword argument 'moe_config'
```

**触发位置：** `vllm_fl/ops/fused_moe/layer.py` line 107 — `FusedMoEFL` factory function

**根因（已查清）：**

`FusedMoEFL` 重建三种 router 实例时，传入的参数与 vllm 0.24.0 父类签名不匹配：

| 参数 | FL层传入 | 父类实际签名 | 问题 |
|------|---------|------------|------|
| `moe_config` | ✓ | ✗ 不存在 | **报错根因** |
| `correction_bias` | ✓ | ✗ 叫 `e_score_correction_bias` | 参数名错 |
| `global_num_experts` | ✗ | ✓ 必填 | 缺少必填参数 |
| `renormalize` | ✗ | 有默认值 | 漏传 |

vllm 0.24.0 三个 router 父类签名：
- `FusedTopKRouter.__init__(top_k, global_num_experts, scoring_func, renormalize, eplb_state)`
- `FusedTopKBiasRouter.__init__(top_k, global_num_experts, e_score_correction_bias, renormalize, routed_scaling_factor, eplb_state, *, scoring_func, hash_indices_table)`
- `GroupedTopKRouter.__init__(top_k, global_num_experts, num_expert_group, topk_group, renormalize, scoring_func, routed_scaling_factor, e_score_correction_bias, num_fused_shared_experts, eplb_state)`

**修复方案：**

`layer.py` 里根本不需要重建 router 实例。`router.py` 末尾已有 `replace_router_with_fl()` 函数，
它只替换 `_compute_routing` 方法（猴子补丁），完全不碰 `__init__`：

```python
def replace_router_with_fl() -> None:
    FusedTopKRouter._compute_routing = FusedTopKRouterFL._compute_routing
    GroupedTopKRouter._compute_routing = GroupedTopKRouterFL._compute_routing
    FusedTopKBiasRouter._compute_routing = FusedTopKBiasRouterFL._compute_routing
```

**修复：** 删除 `FusedMoEFL` 中 step 3 重建 router 实例的所有代码，改为调用 `replace_router_with_fl()`。

**状态：** ✅ 根因确认，待实施

---

## 修复实施

### layer.py 修改

**删除：** lines 70–116（重建 router 实例的全部逻辑 + `_get` helper + `_SENTINEL`）

**替换为：**
```python
# 3. Replace router _compute_routing with FL version via monkey-patch.
#    replace_router_with_fl() patches the class method so all instances
#    (including the one just built by FusedMoE()) use FL dispatch.
replace_router_with_fl()
```

并在文件顶部 import 中加：
```python
from vllm_fl.ops.fused_moe.router import replace_router_with_fl
```
