#!/bin/bash
mkdir -p /workspace/adapt-logs
MACA_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=fl \
    /opt/conda/bin/python3 /workspace/vllm-plugin-FL/baseline_test.py \
    > /workspace/adapt-logs/baseline.log 2>&1
echo "EXIT_CODE=$?" >> /workspace/adapt-logs/baseline.log
