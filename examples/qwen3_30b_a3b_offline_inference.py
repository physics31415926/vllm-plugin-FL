# Copyright (c) 2025 BAAI. All rights reserved.
# Test script for Qwen3-30B-A3B (MoE) offline inference

import os

import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

print(f"Current Platform: {current_platform}")

if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "/nfs/wlx/models/Qwen3-30B-A3B")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    torch.cuda.empty_cache()
    print("\nInference complete, resources cleared.")
