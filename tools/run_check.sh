#!/bin/bash
export VLLM_PLUGINS=fl
echo "=== import check ==="
/opt/conda/bin/python3 /workspace/vllm-plugin-FL/tools/check_imports.py 2>&1 | grep -E "^OK|^FAIL|activated|circular|ReplicatedLinear"

echo "=== mla/common.py direct import ==="
/opt/conda/bin/python3 -c "
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.common import QueryLenSupport
print('mla.common: OK')
" 2>&1 | grep -vE "^INFO|Version|Build_|GIT_|Vllm Op|SGlang|Staring|Release|mcoplib|MACA_VERSION|Check the"

echo "=== flash_attn.py direct import ==="
/opt/conda/bin/python3 -c "
from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn
print('flash_attn: OK')
" 2>&1 | grep -vE "^INFO|Version|Build_|GIT_|Vllm Op|SGlang|Staring|Release|mcoplib|MACA_VERSION|Check the"

echo "=== full MacaBackend import ==="
/opt/conda/bin/python3 -c "
from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend
print('MacaBackend: OK')
" 2>&1 | grep -vE "^INFO|Version|Build_|GIT_|Vllm Op|SGlang|Staring|Release|mcoplib|MACA_VERSION|Check the"
