# Sunrise PTPU Runner V2 diagnostics

These targeted probes used plugin revision 7b43f8aab9bf321046f237592c7cc90f1263390c and the pinned FlagGems-vllm 3b0f388 provider. The Qwen3.6-27B BF16 TP2 eager server used asynchronous scheduling and replayed the eight original concurrent image requests on a fresh server. All 8 passed, including image-06. The earlier full eager run had one empty response in this group.

The fresh server's corrupted_requests counter rose from 0 to 8. A separate no-model test ran the vLLM V2 counting kernel and asynchronous CPU copy on known data. All 32 combinations matched the CPU oracle at width 248320. The result establishes the presence of NaNs before sampling in the replayed batch. It does not locate their origin or establish the cause of the earlier empty output.
