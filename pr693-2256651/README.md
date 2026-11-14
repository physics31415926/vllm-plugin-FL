# MetaX C550 Runner V2 adaptation gate

The first shared TP4 gate passed **103/104 requests** across text, image, and mixed single-request and concurrent scenarios. All four vLLM services selected upstream `vllm.v1.worker.gpu.model_runner.GPUModelRunner`. The initial API server process environment was checked via `/proc/<pid>/environ`. `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_FL_FLAGOS_WHITELIST`, and `FLAGCX_PATH` were absent for all four services. The running graph API servers were checked again after startup. [Live environment recheck](live-graph-environment.json)

| Model and mode | Requests passed | PIECEWISE replay count |
| --- | ---: | ---: |
| [Qwen3.6-27B-eager](groups/Qwen3.6-27B-eager/) | 26/26 | — |
| [Qwen3.6-27B-graph](groups/Qwen3.6-27B-graph/) | 26/26 | 341 |
| [Qwen3.6-35B-A3B-eager](groups/Qwen3.6-35B-A3B-eager/) | 26/26 | — |
| [Qwen3.6-35B-A3B-graph](groups/Qwen3.6-35B-A3B-graph/) | 25/26 | 279 |

The 35B graph text-only request failed the literal term and order checks because the response omitted the word “limitations.” The response described hallucinations, bias, and the distinction between statistical correlations and understanding. All other original requests passed. The earlier passing run used the same engine configuration and request. [Historical comparison](historical-comparison.json)

Each group includes five scenario JSON reports, the exact `vllm serve` command, the initial environment receipt, and selected server logs. Replay counts sum the positive `Count` column entries emitted by vLLM's graph metrics. See the [aggregate results](summary.json) and [launch events](launch-events.log).

The hardware run used commit `2256651ac575dfacce82dce376f6a5256a4210ca`. The [PR #693 revision](https://github.com/flagos-ai/vllm-plugin-FL/pull/693/commits/20cc789d3a9dae89fcf95ea38e27e2ff48549e82) differs only by formatting one Python unit-test signature. The test's AST and the runtime, CI, fixture, and gate tree IDs are identical. [Revision comparison](revision-equivalence.json)

The [container launcher](launch-container.sh) parameterizes the host model mount and advisory lock. It used `harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-metax-ci@sha256:734c6f7d620e2ca3be4fa382a9e2cd6e1d345a310e2e502f11317b5521b886ae`. The [adaptation gate launcher](adaptation-gate.sh) ran the PR's original `tools/adaptation-gate-cases/run_test.sh` with a 900-second request timeout and a 2,400-second startup timeout. The graph launches used `--cudagraph-metrics` to enable replay logging.

The services loaded preexisting local model directories. The original downloads lacked an immutable upstream revision. Config and weight index SHA-256 checksums record the tested model metadata. [Model fingerprints](models.json)


To repeat the gate on eight idle C550s, copy the two launchers into an empty work directory. Create `pr693-2256651.bundle` there from a vllm-plugin-FL checkout whose history includes the tested commit. Put the two model directories and the CI setup model `Qwen3-0.6B` under `MODEL_DIR`.

```bash
git -C /path/to/vllm-plugin-FL bundle create "$PWD/pr693-2256651.bundle" HEAD
MODEL_DIR=/path/to/models bash launch-container.sh adaptation-gate.sh
```

