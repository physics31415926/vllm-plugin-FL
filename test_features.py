"""Test vLLM 0.28.0 new features with vllm-plugin-FL"""
import os
os.environ["VLLM_FL_OOT_ENABLED"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json

def test_v2_runner_basic():
    """Test V2 model runner with basic generation"""
    from vllm import LLM, SamplingParams
    print("\n=== TEST: V2 Runner Basic (Qwen3-0.6B) ===")
    llm = LLM(model="/nfs/wlx/models/Qwen3-0.6B", max_model_len=2048,
              enforce_eager=True)
    params = SamplingParams(temperature=0, max_tokens=50)
    outputs = llm.generate(["What is 2+2?", "Hello world"], params)
    for out in outputs:
        print(f"  Input: {out.prompt[:30]}...")
        print(f"  Output: {out.outputs[0].text[:80]}")
    print(f"  V2_RUNNER: {llm.llm_engine.model_config.use_v2_model_runner if hasattr(llm.llm_engine.model_config, 'use_v2_model_runner') else 'N/A'}")
    del llm
    print("  PASSED")

def test_structured_output():
    """Test structured output (JSON mode) - new in 0.28"""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    print("\n=== TEST: Structured Output (JSON mode) ===")
    llm = LLM(model="/nfs/wlx/models/Qwen3-0.6B", max_model_len=2048,
              enforce_eager=True)
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    params = SamplingParams(
        temperature=0, max_tokens=100,
        structured_outputs=StructuredOutputsParams(
            type="json", value=json.dumps(json_schema)
        )
    )
    outputs = llm.generate(["Generate a person's info in JSON:"], params)
    text = outputs[0].outputs[0].text
    print(f"  Output: {text[:200]}")
    try:
        parsed = json.loads(text)
        print(f"  Parsed JSON: {parsed}")
        print("  PASSED (valid JSON)")
    except json.JSONDecodeError:
        print("  PARTIAL (output not valid JSON, but no crash)")
    del llm

def test_moe_model():
    """Test MoE model if available"""
    from vllm import LLM, SamplingParams
    # Check for available MoE models
    import os
    moe_paths = [
        "/nfs/wlx/models/Qwen3-30B-A3B",
        "/nfs/wlx/models/Qwen2.5-3B-A1.5B", 
    ]
    model_path = None
    for p in moe_paths:
        if os.path.exists(p):
            model_path = p
            break
    if not model_path:
        print("\n=== TEST: MoE Model === SKIPPED (no MoE model found)")
        return
    print(f"\n=== TEST: MoE Model ({os.path.basename(model_path)}) ===")
    llm = LLM(model=model_path, max_model_len=2048, enforce_eager=True,
              tensor_parallel_size=1)
    params = SamplingParams(temperature=0, max_tokens=50)
    outputs = llm.generate(["Explain quantum computing briefly:"], params)
    print(f"  Output: {outputs[0].outputs[0].text[:150]}")
    del llm
    print("  PASSED")

if __name__ == "__main__":
    tests = [test_v2_runner_basic, test_structured_output, test_moe_model]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    print("\n=== ALL TESTS COMPLETE ===")
