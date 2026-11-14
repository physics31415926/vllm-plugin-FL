# Kunlunxin P800 vision dispatch diagnostic

The test used plugin commit 3a0d172bff408afbf9d60c669f3ba56baa43b64c, the reviewed vLLM 0.32.0+flagos container, and one P800 TP4 node with Qwen3.6-27B BF16. The diagnostic adds only where_self_out to the platform's existing FlagGems exclusions. The model-runner override and FlagGems operator allowlist were unset. All five unchanged eager cases passed 26/26 requests. The same server passed 232/232 additional requests across two rounds of order and batch-composition changes.

The launch uses Python's vLLM serving module with --enforce-eager, --tensor-parallel-size 4, --max-model-len 16384, --max-num-seqs 256, --block-size 128 and --gpu-memory-utilization 0.7. The five scenario JSON files come directly from the repository's adaptation gate. The batch-order JSON retains the request order, answer text and original checks. The independent four-group run of b560e3309e821e08e38854aae6670a15402e39ce validates the checked-in route.

The 27B eager server reached readiness with the narrow change after both the original and the bmm control stalled. The earlier kernel exception names where_inner_kernel_rank_4. None of the five cases or batch-order tests changed their assertions.
