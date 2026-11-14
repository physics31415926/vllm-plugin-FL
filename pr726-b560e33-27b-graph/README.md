# Kunlunxin P800 graph gate at b560e33

The exact b560e3309e821e08e38854aae6670a15402e39ce build ran the unchanged Qwen3.6-27B BF16 TP4 graph gate. It passed 23/26 requests. Image-01, image-02 and image-03 in the eight-way image group began with the expected answer and shifted to repeated punctuation, finishing on the 128-token limit. The other four scenarios passed. All four workers, including rank 1, completed graph capture. The log contains no NoC timeout for this group.

The server used the same model, image, no environment dispatch overrides, and CLI settings as the exact eager receipt, with --cudagraph-metrics in place of --enforce-eager. During the bad concurrent image batch, graph telemetry shows 124 FULL steps at 3 active tokens padded to 4. This is a batch-shape correlation. The artifact records the full original verdicts and selected capture logs.
