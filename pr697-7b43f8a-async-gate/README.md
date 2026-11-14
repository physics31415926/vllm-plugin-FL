# Sunrise async adaptation gate evidence

This run tested vllm-plugin-FL `7b43f8aab9bf321046f237592c7cc90f1263390c` and FlagGems-vllm `3b0f3880b2a781699462eacb7dd3e00cc24f2d43` on two Sunrise SR-SUN-S2-X1-PCIE devices. Each group used Sunrise Model Runner V2 with async scheduling, TP=2, 8,192-token maximum model length, eight concurrent sequences and 4 GiB of KV-cache memory. The container image ID and package versions are in `environment/`.

| Model | Mode | Shared checks | Long prefill | Native FULL replays during requests |
|---|---|---:|---:|---:|
| Qwen3.6-27B | eager | 26/26 | 4,118 tokens, passed | N/A |
| Qwen3.6-27B | graph | 25/26 | Not run after quality failure | 332 |
| Qwen3.6-35B-A3B | eager | 26/26 | 4,118 tokens, passed | N/A |
| Qwen3.6-35B-A3B | graph | 25/26 | Not run after quality failure | 425 |

In the 27B graph run, `image-08` returned empty text after 128 generated tokens. The same eager request returned `4`. In the 35B graph run, single-image `image-05` returned `VLLM`, missing `HELLO`. Its eager counterpart returned both words. The unchanged four-group gate reported failure with 102/104 shared requests passing. All four groups shut down their owned servers, and both devices were idle at the end.

The 20 request reports retain the original prompts, outputs, checks, timings and request IDs. Filesystem paths were normalized to `/checkout`, `/gate`, `/models` and `/run`. Server logs are excerpts. Replay counts sum the native FULL rows emitted after the shared gate started. `model-fingerprints.json` records SHA-256 hashes of the observed model metadata, tokenizer, preprocessor and weight index files. The weight shards were not hashed.

Run `sha256sum -c SHA256SUMS` from this directory to verify the distributed files.
