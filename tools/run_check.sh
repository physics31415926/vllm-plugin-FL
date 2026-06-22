#!/bin/bash
export VLLM_PLUGINS=fl
/opt/conda/bin/python3 /workspace/vllm-plugin-FL/tools/check_imports.py 2>&1 | grep -E "^OK|^FAIL|activated|circular|ReplicatedLinear"
