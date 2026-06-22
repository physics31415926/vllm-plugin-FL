"""
Fix mla/common.py: move AttentionMetadataBuilder and CommonAttentionMetadata
from vllm.v1.attention.backends.utils to vllm.v1.attention.backend.

Strategy: restore from git HEAD~1, then apply the targeted fix.
"""
import subprocess, sys, os

repo = r"C:\Users\BAAI\Documents\GitHub\FlagScale-Agent-fork\work\vllm-plugin-FL"
target = os.path.join(repo, "vllm_fl", "dispatch", "backends", "vendor",
                      "metax", "impl", "attention", "mla", "common.py")

# 1. Get the clean original from git (one commit before the broken PowerShell edit)
result = subprocess.run(
    ["git", "show", "HEAD~1:vllm_fl/dispatch/backends/vendor/metax/impl/attention/mla/common.py"],
    capture_output=True, cwd=repo
)
if result.returncode != 0:
    print("ERROR: git show failed:", result.stderr.decode())
    sys.exit(1)

lines = result.stdout.decode("utf-8").splitlines(keepends=True)
print(f"Original file: {len(lines)} lines")

# 2. Find the broken import block:
#    from vllm.v1.attention.backends.utils import (
#        AttentionMetadataBuilder,
#        CommonAttentionMetadata,
#        get_dcp_local_seq_lens,
#        ...
#    )
# Replace it with two correctly-targeted import blocks.

OLD_BLOCK = (
    "from vllm.v1.attention.backends.utils import (\n"
    "    AttentionMetadataBuilder,\n"
    "    CommonAttentionMetadata,\n"
    "    get_dcp_local_seq_lens,\n"
    "    get_per_layer_parameters,\n"
    "    infer_global_hyperparameters,\n"
    "    split_decodes_and_prefills,\n"
    ")\n"
)

NEW_BLOCK = (
    "from vllm.v1.attention.backend import (\n"
    "    AttentionMetadataBuilder,\n"
    "    CommonAttentionMetadata,\n"
    ")\n"
    "from vllm.v1.attention.backends.utils import (\n"
    "    get_dcp_local_seq_lens,\n"
    "    get_per_layer_parameters,\n"
    "    infer_global_hyperparameters,\n"
    "    split_decodes_and_prefills,\n"
    ")\n"
)

content = "".join(lines)
if OLD_BLOCK not in content:
    print("ERROR: OLD_BLOCK not found in original file. Dump lines 220-240:")
    for i, l in enumerate(lines[219:240], start=220):
        print(f"  {i}: {repr(l)}")
    sys.exit(1)

count = content.count(OLD_BLOCK)
print(f"OLD_BLOCK found {count} time(s)")
fixed = content.replace(OLD_BLOCK, NEW_BLOCK, 1)

# 3. Verify no duplicate backend imports
backend_imports = [i for i, l in enumerate(fixed.splitlines(), 1)
                   if "from vllm.v1.attention.backend import" in l]
print(f"'from vllm.v1.attention.backend import' appears at lines: {backend_imports}")

# 4. Write
with open(target, "w", encoding="utf-8", newline="\n") as f:
    f.write(fixed)

new_lines = fixed.count("\n")
print(f"Written: {new_lines} lines to {target}")

# 5. Verify the fixed area
fixed_lines = fixed.splitlines()
for i, l in enumerate(fixed_lines):
    if "backends.utils import" in l or "backend import" in l:
        print(f"  line {i+1}: {l}")
