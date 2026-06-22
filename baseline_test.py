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
