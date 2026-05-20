# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/examples/offline_inference/basic/basic.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    model_path = os.environ.get("MODEL_PATH", "/workspace/models/Qwen/Qwen3-4B")
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    pp_size = int(os.environ.get("PP_SIZE", "1"))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        enforce_eager=True,
    )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
