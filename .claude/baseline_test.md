# Baseline Test: vllm_metax 0.20.0 on MetaX C550

Date: 2025-06-17
Container: `vllm-fl-adapt-0202`
Environment: vLLM 0.20.0 + vllm_metax 0.20.0+gcce172.d20260529.maca3.7.0.38.torch2.8

## Test Script

```python
from vllm import LLM, SamplingParams

def main():
    model_path = "/public-nfs/wlx/models/Qwen/Qwen3___6-27B"

    prompts = [
        "Hello, who are you?",
        "Explain quantum computing in 3 sentences.",
        "Write a Python function to compute Fibonacci numbers.",
        "Translate the following to Chinese: The weather is beautiful today.",
        "What is 2+2? Show your reasoning step by step.",
    ]

    print("Loading model...")
    llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True)

    params = SamplingParams(temperature=0.7, max_tokens=256)
    print("Running inference...")
    outputs = llm.generate(prompts, params)

    sep = "=" * 60
    for i, output in enumerate(outputs):
        print()
        print(sep)
        print(f"Prompt {i+1}: {prompts[i]}")
        print(sep)
        print(output.outputs[0].text)

    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
```

## Configuration

- Model: Qwen3.6-27B (27B params, hybrid attention+mamba architecture)
- GPUs: 4x MetaX C550 64GB (MACA_VISIBLE_DEVICES=0,1,2,3)
- Tensor Parallel: 4
- Sampling: temperature=0.7, max_tokens=256
- Attention backend: FLASH_ATTN (Maca version, supports v2 only)

## Performance Metrics

- Model loading: 13.01 GiB memory, 9.33s
- torch.compile (first run): ~41.7s for compile range (1, 8192)
- KV cache: 672,672 tokens capacity
- CUDA graph capture: 51 piecewise graphs + 35 full graphs

## Results: PASS

All 5 prompts generated valid output.

### Prompt 1: "Hello, who are you?"
> (thinking mode output with valid response)

### Prompt 2: "Explain quantum computing in 3 sentences."
> (valid explanation generated)

### Prompt 3: "Write a Python function to compute Fibonacci numbers."
> Generated correct memoized Fibonacci implementation:
```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
        return memo[n]
```

### Prompt 4: "Translate the following to Chinese: The weather is beautiful today."
> 今天天气很好。

### Prompt 5: "What is 2+2? Show your reasoning step by step."
> Step-by-step reasoning arriving at answer: 4

## Key Observations

1. vllm_metax plugin auto-registers and overrides model classes for MetaX hardware
2. Uses `MACA_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`
3. Qwen3.6 detected as hybrid model (attention + mamba layers)
4. torch.compile cache stored at `/root/.cache/vllm/torch_compile_cache/`
5. CUDA graph memory profiling disabled by plugin (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`)
6. Minor warnings: leaked semaphore/shared_memory on exit (harmless)
7. Model architecture name in container: `Qwen3___6-27B` (dots replaced with `___` in path)
