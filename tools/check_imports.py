"""Probe every symbol the plugin's mla/common.py needs, against vllm 0.20.2."""
results = []

def check(label, mod, *names):
    try:
        m = __import__(mod, fromlist=names)
        for n in names:
            getattr(m, n)
        results.append(f"OK   {label}: {mod} -> {', '.join(names)}")
    except (ImportError, AttributeError) as e:
        results.append(f"FAIL {label}: {e}")

# plugin's current imports (the broken ones)
check("plugin-1", "vllm.v1.attention.backends.utils",
      "AttentionMetadataBuilder", "CommonAttentionMetadata",
      "get_dcp_local_seq_lens", "get_per_layer_parameters",
      "infer_global_hyperparameters", "split_decodes_and_prefills")

# vllm_metax's corrected import split
check("metax-1", "vllm.v1.attention.backend",
      "AttentionBackend", "AttentionLayer", "MLAAttentionImpl",
      "AttentionMetadata", "AttentionMetadataBuilder", "CommonAttentionMetadata")
check("metax-2", "vllm.v1.attention.backends.utils",
      "get_dcp_local_seq_lens", "get_per_layer_parameters",
      "infer_global_hyperparameters", "split_decodes_and_prefills")

# plugin-only imports not in vllm_metax
check("plugin-only-1", "vllm.model_executor.layers.attention.mla_attention", "get_mla_dims")
check("plugin-only-2", "vllm.model_executor.layers.batch_invariant", "_batch_invariant_MODE")

# vllm_metax extras not in plugin
check("metax-only-1", "vllm.distributed.parallel_state", "is_global_first_rank")
check("metax-only-2", "vllm.utils.flashinfer", "has_nvidia_artifactory")
check("metax-only-3", "vllm.v1.attention.ops.common", "cp_lse_ag_out_rs")

for r in results:
    print(r)
