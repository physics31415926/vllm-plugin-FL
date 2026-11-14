# MetaX C550 FlagGems-vllm router gate

The unmodified shared TP4 gate passed **104/104 requests** on MetaX C550. The vllm-plugin-FL commit was `d828c23d982f689acb4699599fd75fb3f76aebd1`, with FlagGems-vllm `e00557103e284db5b57168386db60faa051989ef` installed from its [hash-verified source archive](https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/e00557103e284db5b57168386db60faa051989ef). The published MetaX CI image used digest `sha256:734c6f7d620e2ca3be4fa382a9e2cd6e1d345a310e2e502f11317b5521b886ae`.

| Model and mode | Passed | Positive PIECEWISE replay count |
| --- | ---: | ---: |
| [`Qwen3.6-27B-eager`](groups/Qwen3.6-27B-eager/) | 26/26 | 0 |
| [`Qwen3.6-27B-graph`](groups/Qwen3.6-27B-graph/) | 26/26 | 235 |
| [`Qwen3.6-35B-A3B-eager`](groups/Qwen3.6-35B-A3B-eager/) | 26/26 | 0 |
| [`Qwen3.6-35B-A3B-graph`](groups/Qwen3.6-35B-A3B-graph/) | 26/26 | 202 |

All groups selected upstream `GPUModelRunner` with the initial process environment missing both `VLLM_USE_V2_MODEL_RUNNER` and `VLLM_FL_FLAGOS_WHITELIST`. The reports include the actual commands, per-process environment receipts, selected V2 and provider logs, and all 20 original scenario reports. [Aggregate results](summary.json) · [Model fingerprints](models.json)
