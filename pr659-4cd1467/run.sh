#!/usr/bin/env bash
set -euo pipefail
root=/gate-work/tp8-v2-4cd1467
plugin=/gate-work/tp8-v2-4cd1467/plugin
gate="$plugin/tools/adaptation-gate-cases"
export PYTHONPATH="$plugin:${root%/*}/tp8-gate/flag-gems/src:/opt/flagtree"
export GEMS_VENDOR=amd VLLM_PLUGINS=fl USE_FLAGGEMS=1
export HSA_TOOLS_DISABLE_REGISTER=1
unset VLLM_USE_V2_MODEL_RUNNER VLLM_FL_FLAGOS_WHITELIST
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGCX_PATH=/tmp/FlagCX
export LD_LIBRARY_PATH=/tmp/FlagCX/build/lib:/opt/rocm/lib:/usr/local/lib
export VLLM_CACHE_ROOT="$root/vllm-cache"
export TRITON_CACHE_DIR="${root%/*}/tp8-gate/triton-cache"
export FLAGGEMS_CACHE_DIR="${root%/*}/tp8-gate/flaggems-cache"
export TENSOR_PARALLEL_SIZE=8 MAX_MODEL_LEN=32768 SERVICE_TIMEOUT=2400 REQUEST_TIMEOUT=300
python3 - <<'PYTHON'
from importlib.metadata import version
from pathlib import Path

import flag_gems
import torch
import triton
import vllm_fl

assert "/tp8-gate/flag-gems/src/" in flag_gems.__file__, flag_gems.__file__
assert "/tp8-v2-4cd1467/plugin/vllm_fl/" in vllm_fl.__file__, vllm_fl.__file__
assert "/opt/flagtree/" in triton.__file__, triton.__file__
assert torch.cuda.device_count() == 8
print("plugin", vllm_fl.__file__)
print("FlagGems", flag_gems.__file__)
print("FlagTree Triton", triton.__version__, triton.__file__)
print("vLLM", version("vllm"), "Torch", torch.__version__, "FlagCX", version("flagcx"))
for i in range(8):
    print("GPU", i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
    assert torch.cuda.mem_get_info(i)[0] >= 43 * 1024**3
PYTHON
ROCR_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1 timeout 120s torchrun --standalone --nproc-per-node=2 "${root%/*}/tp8-direct/check_flagcx.py"
cd "$gate"
run_group() (
    local model=$1 mode=$2 port=$3
    local label="$model-$mode"
    export MODEL_PATH="/models/$model" SERVED_MODEL_NAME="$model" PORT="$port"
    export SERVER_PID_FILE="$root/$label.pid" RESULTS_DIR="$root/results/$label"
    mkdir -p "$RESULTS_DIR"
    printf '%s START %s\n' "$(date -Is)" "$label"
    if [[ "$mode" = graph ]]; then
        local graph_options=()
        if [[ "$model" == "Qwen3.6-35B-A3B" ]]; then
            graph_options=(
                --kv-cache-memory-bytes 21474836480
                --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
            )
        fi
        setsid bash -c '
            printf "%s\n" "$$" > "$SERVER_PID_FILE"
            exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
                --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
                --max-model-len "$MAX_MODEL_LEN" --allowed-local-media-path "$PWD/images" \
                --trust-remote-code --cudagraph-metrics "$@"
        ' _ "${graph_options[@]}" > "$root/$label.serve.log" 2>&1 &
    else
        setsid bash ./run_serve_eager.sh > "$root/$label.serve.log" 2>&1 &
    fi
    local server=$!
    trap 'kill -TERM -- "-$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
    local status=0
    bash ./run_test.sh > "$root/$label.test.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$root/$label.exit"
    printf '%s END %s status=%s\n' "$(date -Is)" "$label" "$status"
)
run_group Qwen3.6-35B-A3B graph 18843
run_group Qwen3.6-27B graph 18844
run_group Qwen3.6-27B eager 18845
run_group Qwen3.6-35B-A3B eager 18846
for exit_file in "$root"/*.exit; do
    test "$(cat "$exit_file")" = 0 || exit 1
done
