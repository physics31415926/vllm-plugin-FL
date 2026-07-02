#!/bin/bash
/nfs/wlx/envs/vllm-fl-0.24.0/bin/python /nfs/wlx/adapt/nvidia-vllm-0.24.0/vllm-plugin-FL/examples/qwen3_30b_a3b_offline_inference.py > /nfs/wlx/tmp/qwen3_30b_a3b.log 2>&1
echo "EXIT_CODE:$?" >> /nfs/wlx/tmp/qwen3_30b_a3b.log
