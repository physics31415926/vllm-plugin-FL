# Kunlunxin P800 gate at b560e33

The exact plugin revision b560e3309e821e08e38854aae6670a15402e39ce passed all 26 requests across the five original Qwen3.6-27B BF16 TP4 eager scenarios, then 232/232 in two rounds of additional batch-order cases. The server and tests used the checked-in dispatch policy. The launch unsets both allowlist and blacklist overrides and leaves upstream runner selection automatic. The serving log showed no rank 1 NoC timeout in this completed group.

The review container had image ID sha256:d82423a03bf98d5d5d4c59dc7dbdc8346bb4de4c170e4a3093365fd5c59dea02 and uses vLLM 0.32.0+flagos, Python 3.10.20, torch 2.9.0+cu129, FlagGems 5.4.0, Triton 3.6.0, P800 driver 5.0.21.47 and container XRE 5.37.1.

Model directory /data/models/Qwen/Qwen3.6-27B: config SHA256 69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c, weight index SHA256 a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b.

The launch uses Python's vLLM serving CLI with --enforce-eager, --tensor-parallel-size 4, --max-model-len 16384, --max-num-seqs 256, --block-size 128, --gpu-memory-utilization 0.7 and the local gate images. The request timeout is 300s and startup timeout is 1800s. The five JSON files come directly from the repository gate. The remaining graph and 35B groups are tracked in PR #726.
