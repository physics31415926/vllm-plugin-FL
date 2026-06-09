#!/usr/bin/env python3
"""Check all imports required by minicpmv4_6.py against installed vllm."""
import importlib
import sys

checks = [
    ("vllm.multimodal.inputs", ["MultiModalFeatureSpec", "MultiModalFieldConfig", "NestedTensors"]),
    ("vllm.model_executor.layers.mamba.mamba_utils", ["MambaStateCopyFuncCalculator", "MambaStateDtypeCalculator", "MambaStateShapeCalculator"]),
    ("vllm.model_executor.layers.attention", ["MMEncoderAttention"]),
    ("vllm.model_executor.layers.activation", ["get_act_fn"]),
    ("vllm.model_executor.layers.linear", ["QKVParallelLinear", "RowParallelLinear"]),
    ("vllm.model_executor.model_loader.weight_utils", ["default_weight_loader"]),
    ("vllm.multimodal.parse", ["ImageProcessorItems", "ImageSize", "VideoProcessorItems"]),
    ("vllm.multimodal.processing.processor", ["PromptReplacement", "PromptUpdateDetails"]),
    ("vllm.model_executor.models.idefics2_vision_model", ["Idefics2VisionTransformer"]),
    ("vllm.model_executor.models.interfaces", [
        "HasInnerState", "IsHybrid", "MultiModalEmbeddings",
        "SupportsMRoPE", "SupportsMultiModal", "SupportsPP", "_require_is_multimodal"
    ]),
    ("vllm.model_executor.models.minicpmv", [
        "MiniCPMVDummyInputsBuilder", "MiniCPMVImageEmbeddingInputs",
        "MiniCPMVImageEmbeddingItems", "MiniCPMVImagePixelInputs",
        "MiniCPMVMultiModalProcessor", "MiniCPMVProcessingInfo",
        "MiniCPMVVideoEmbeddingItems",
    ]),
    ("vllm.model_executor.models.module_mapping", ["MultiModelKeys"]),
    ("vllm.model_executor.models.qwen3_5", ["Qwen3_5ForCausalLM"]),
    ("vllm.model_executor.models.utils", [
        "AutoWeightsLoader", "WeightsMapper", "_merge_multimodal_embeddings",
        "flatten_bn", "maybe_prefix"
    ]),
    ("vllm.model_executor.models.vision", ["is_vit_use_data_parallel"]),
    ("vllm.multimodal", ["MULTIMODAL_REGISTRY"]),
    ("vllm.sequence", ["IntermediateTensors"]),
    ("vllm.config", ["VllmConfig"]),
    ("vllm.distributed", ["get_tensor_model_parallel_world_size"]),
    ("transformers", ["MiniCPMV4_6Config"]),
]

failed = []
for module_path, names in checks:
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        print(f"FAIL  import {module_path}: {e}")
        failed.append((module_path, str(e)))
        continue
    for name in names:
        if not hasattr(mod, name):
            print(f"MISS  {module_path}.{name}")
            failed.append((module_path + "." + name, "missing"))
        else:
            print(f"OK    {module_path}.{name}")

print()
if failed:
    print(f"=== {len(failed)} issues found ===")
    sys.exit(1)
else:
    print("=== All imports OK ===")
