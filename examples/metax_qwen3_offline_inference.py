# Copyright (c) 2025 BAAI. All rights reserved.
# MetaX C550 offline inference test for Qwen3-4B.
# Usage: VLLM_PLUGINS=fl python examples/metax_qwen3_offline_inference.py

import os
import torch
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

print(f"Current Platform: {current_platform}")
print(f"Platform Type: {type(current_platform)}")
print(f"Device: {torch.cuda.get_device_name(0)}")

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen/Qwen3-4B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2 + 2?",
    ]

    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

    print(f"\nLoading model: {MODEL_PATH} (TP={TP_SIZE})")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True,
        trust_remote_code=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    print("\n=== Results ===")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Output: {generated_text!r}")
        print()

    del llm
    torch.cuda.empty_cache()
    print("Offline inference complete.")
