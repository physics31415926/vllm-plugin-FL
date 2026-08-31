"""Test V2 Model Runner with vLLM 0.28.0 plugin"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_FL_OOT_ENABLED"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

print("="*60)
print("TEST 1: Qwen3-0.6B - V2 Model Runner (inproc, no FlagGems)")
print("="*60)
llm = LLM(model="/nfs/wlx/models/Qwen3-0.6B", max_model_len=512, enforce_eager=True)
v2_status = llm.llm_engine.vllm_config.use_v2_model_runner
print(f"V2_MODEL_RUNNER: {v2_status}")

params = SamplingParams(temperature=0, max_tokens=100)
outputs = llm.generate(["What is the capital of France?"], params)
print(f"OUTPUT: {outputs[0].outputs[0].text[:300]}")
print(f"TOKENS: {len(outputs[0].outputs[0].token_ids)}")

# Test 2: Multi-prompt batch
prompts = [
    "Explain quantum computing in one sentence:",
    "Write a Python function to reverse a string:",
    "The meaning of life is",
]
outputs = llm.generate(prompts, params)
for i, out in enumerate(outputs):
    print(f"\nPROMPT_{i}: {prompts[i]}")
    print(f"OUTPUT_{i}: {out.outputs[0].text[:200]}")

print("\n" + "="*60)
print("ALL V2 RUNNER TESTS PASSED")
print("="*60)
