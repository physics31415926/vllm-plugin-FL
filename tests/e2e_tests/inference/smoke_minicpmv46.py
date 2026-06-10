"""
Smoke test for MiniCPM-V 4.6 inference via vllm-plugin-FL.
Uses the real model weights at /workspace/models/openbmb/MiniCPM-V-4___6.
A synthetic RGB image is created locally to avoid network asset downloads.

Run inside container:
  VLLM_PLUGINS=fl python3 tests/e2e_tests/inference/smoke_minicpmv46.py
"""

import os
import sys

from PIL import Image

from vllm import LLM, SamplingParams

# Ensure plugin is loaded before any vllm internals are imported further
os.environ.setdefault("VLLM_PLUGINS", "fl")

MODEL_PATH = "/workspace/models/openbmb/MiniCPM-V-4___6"


def make_test_image() -> Image.Image:
    """Create a simple synthetic image (red square) for smoke testing."""
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    return img


def main():
    print("=" * 60)
    print("MiniCPM-V 4.6 Smoke Test")
    print("=" * 60)

    print(f"\n[1/4] Loading model from {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.7,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    print("    Model loaded OK")

    print("\n[2/4] Building prompt ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # MiniCPM-V 4.6 uses {"type": "image"} content items so that
    # apply_chat_template inserts the correct <|image_pad|> special token.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is this image?"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"    Prompt: {repr(prompt[:120])}...")

    print("\n[3/4] Running generation ...")
    image = make_test_image()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=sampling_params,
    )

    print("\n[4/4] Results:")
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"    Output [{i}]: {repr(text)}")
        assert len(text) > 0, "Output is empty — generation failed"

    print("\n✓ Smoke test PASSED — model loaded and generated output successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
