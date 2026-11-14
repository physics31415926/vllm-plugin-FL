# Sunrise TP2 27B graph token diagnostic

Exact plugin revision 7b43f8aab9bf321046f237592c7cc90f1263390c and the pinned 3b0f388 Flash KV provider ran the original leading requests followed by the same eight concurrent images. The diagnostic requested raw token IDs and enabled the vLLM logit NaN counter.

Image-01, image-02 and image-03 produced their expected prefixes and then emitted ID 248320 126 times through the 128-token cap. Image-05 met the text oracle on its first token and then emitted 127 copies of that ID. The ID is outside Qwen3.6-27B's configured vocabulary and absent from the tokenizer. All four stopped at the length limit. The aggregate corrupted-request counter rose from 8 to 12 across the target batch. The counter cannot identify which requests contributed.

The unchanged image oracle passed five of eight target requests. The server recorded native FULL decode replay and released its owned GPU resources. This diagnostic captures a reproduced failure with instrumentation.
