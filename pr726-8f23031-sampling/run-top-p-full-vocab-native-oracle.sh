#!/usr/bin/env bash
set -euo pipefail
root=${EVIDENCE_DIR:-$PWD}
cd "$root"
docker run --rm --name kl032-top-p-full-vocab-native-oracle --ipc=host \
  --device /dev/xpuctrl --device /dev/xpu6 -v "$root:/probe" \
  --entrypoint /bin/bash codex-kl032:review-dev -ec '
    source /etc/bash_env.sh
    export CUDA_VISIBLE_DEVICES=0 GEMS_VENDOR=kunlunxin FLAGGEMS_VENDOR=kunlunxin
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_PLUGINS=fl VLLM_FL_FLAGOS_WHITELIST=
    export VLLM_FL_FLAGOS_BLACKLIST_APPEND=le,softmax
    cd /probe/plugin-515fa94
    export PYTHONPATH=$PWD:${PYTHONPATH:-}
    timeout 300 /flagos/bin/python /probe/top-p-full-vocab-native-oracle.py
  '
