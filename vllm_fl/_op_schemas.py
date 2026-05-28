# Copyright (c) 2026 BAAI. All rights reserved.

r"""
Hardcoded _C op schemas for vLLM 0.20.2.

These schemas are registered as stubs via torch.library when the native _C.so
extension is not available (non-NVIDIA platforms without compiled C extensions).

To regenerate for a new vLLM version, run from a vLLM editable install:

    python -c "
    import re, vllm, pathlib
    root = pathlib.Path(vllm.__file__).parent.parent
    f = root / 'csrc' / 'torch_bindings.cpp'
    content = re.sub(r'//[^\n]*', '', f.read_text())
    for m in re.finditer(r'ops\.def\(\s*((?:\"[^\"]*\"\s*)+)\)', content):
        parts = re.findall(r'\"([^\"]*)\"', m.group(1))
        schema = ''.join(parts)
        if schema and '->' in schema:
            print(repr(schema) + ',')
    "
"""

# fmt: off
VLLM_C_OP_SCHEMAS: list[str] = [
    "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! y_q, Tensor! y_s,bool use_ue8m0) -> ()",
    "weak_ref_tensor(Tensor input) -> Tensor",
    "get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor",
    "paged_attention_v1(    Tensor! out, Tensor query, Tensor key_cache,    Tensor value_cache, int num_kv_heads, float scale,    Tensor block_tables, Tensor seq_lens, int block_size,    int max_seq_len, Tensor? alibi_slopes,    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,    int tp_rank, int blocksparse_local_blocks,    int blocksparse_vert_stride, int blocksparse_block_size,    int blocksparse_head_sliding_step) -> ()",
    "paged_attention_v2(    Tensor! out, Tensor! exp_sums, Tensor! max_logits,    Tensor! tmp_out, Tensor query, Tensor key_cache,    Tensor value_cache, int num_kv_heads, float scale,    Tensor block_tables, Tensor seq_lens, int block_size,    int max_seq_len, Tensor? alibi_slopes,    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,    int tp_rank, int blocksparse_local_blocks,    int blocksparse_vert_stride, int blocksparse_block_size,    int blocksparse_head_sliding_step) -> ()",
    "merge_attn_states(    Tensor! output,    Tensor!? output_lse,    Tensor prefix_output,    Tensor prefix_lse,    Tensor suffix_output,    Tensor suffix_lse,    int!? prefill_tokens_with_context,    Tensor? output_scale=None) -> ()",
    "convert_vertical_slash_indexes(   Tensor! block_count, Tensor! block_offset,    Tensor! column_count, Tensor! column_index,    Tensor q_seqlens, Tensor q_seqlens,    Tensor vertical_indexes, Tensor slash_indexes,    int context_size, int block_size_M, int block_size_N,    bool causal) -> ()",
    "convert_vertical_slash_indexes_mergehead(   Tensor! block_count, Tensor! block_offset,    Tensor! column_count, Tensor! column_index,    Tensor q_seqlens, Tensor q_seqlens,    Tensor vertical_indexes, Tensor slash_indexes,    Tensor vertical_indices_count, Tensor slash_indices_count,    int context_size, int block_size_M, int block_size_N,    bool causal) -> ()",
    "silu_and_mul(Tensor! result, Tensor input) -> ()",
    "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit) -> ()",
    "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()",
    "silu_and_mul_per_block_quant(Tensor! out, Tensor input, Tensor! scales, int group_size, Tensor? scale_ub=None, bool is_scale_transposed=False) -> ()",
    "mul_and_silu(Tensor! out, Tensor input) -> ()",
    "gelu_and_mul(Tensor! out, Tensor input) -> ()",
    "gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()",
    "fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()",
    "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float limit=7.0) -> ()",
    "gelu_new(Tensor! out, Tensor input) -> ()",
    "gelu_fast(Tensor! out, Tensor input) -> ()",
    "gelu_quick(Tensor! out, Tensor input) -> ()",
    "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()",
    "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()",
    "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, bool is_neox, Tensor position_ids, int forced_token_heads_per_warp=-1) -> ()",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, float eps, int cache_block_size) -> ()",
    "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, Tensor output_mask, Tensor repetition_penalties) -> ()",
    "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()",
    "top_k_per_row_decode(Tensor logits, int next_n, Tensor seq_lens, Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()",
    "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor workspace, int k, int max_seq_len) -> ()",
    "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, Tensor scale, float epsilon) -> ()",
    "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor! residual, Tensor weight, Tensor scale, float epsilon) -> ()",
    "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, Tensor weight, Tensor! scale, float epsilon, Tensor? scale_ub, Tensor!? residual) -> ()",
    "rms_norm_per_block_quant(Tensor! result, Tensor input, Tensor weight, Tensor! scale, float epsilon, Tensor? scale_ub, Tensor!? residual, int group_size, bool is_scale_transposed) -> ()",
    "rotary_embedding(Tensor positions, Tensor! query,                 Tensor!? key, int head_size,                 Tensor cos_sin_cache, bool is_neox, int rope_dim_offset=0, bool inverse=False) -> ()",
    "dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()",
    "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, SymInt split_k_iters) -> Tensor",
    "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor",
    "machete_supported_schedules(   ScalarType a_type,   int b_type,   ScalarType? maybe_group_scales_type,   ScalarType? maybe_group_zeros_type,   ScalarType? maybe_channel_scales_type,   ScalarType? maybe_token_scales_type,   ScalarType? maybe_out_type) -> str[]",
    "machete_mm(   Tensor A,   Tensor B,   int b_type,   ScalarType? out_type,   Tensor? group_scales,   Tensor? group_zeros,   int?    group_size,   Tensor? channel_scales,   Tensor? token_scales,   str?    schedule) -> Tensor",
    "machete_prepack_B(   Tensor B,   ScalarType a_type,   int b_type,   ScalarType? group_scales_type) -> Tensor",
    "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? b_bias_or_none,Tensor b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor",
    "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor",
    "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor",
    "marlin_int4_fp8_preprocess(Tensor qweight, Tensor? qzeros_or_none, bool inplace) -> Tensor",
    "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? dtype) -> Tensor",
    "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor",
    "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor",
    "ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor",
    "ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int top_k, int type, SymInt row, SymInt tokens) -> Tensor",
    "mxfp8_experts_quant( Tensor input, Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets, Tensor! quant_output, Tensor! scale_factor) -> ()",
    "cutlass_mxfp8_grouped_mm( Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor! out, Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets) -> ()",
    "sm100_cutlass_mla_decode(Tensor! out, Tensor! lse, Tensor q_nope,                         Tensor q_pe, Tensor kv_c_and_k_pe_cache,                         Tensor seq_lens, Tensor page_table,                         Tensor workspace, float scale,                         int num_kv_splits) -> ()",
    "sm100_cutlass_mla_get_workspace_size(int max_seq_len, int num_batches,                                     int sm_count, int num_kv_splits) -> int",
    "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, bool use_v2_format, int bit) -> Tensor",
    "gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()",
    "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, (int, int)? group_shape=None) -> ()",
    "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) -> ()",
    "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale, Tensor? scale_ub) -> ()",
    "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,Tensor? azp) -> ()",
    "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, Tensor!? azp) -> ()",
    "selective_scan_fwd(Tensor! u, Tensor! delta,Tensor! A, Tensor! B, Tensor! C,Tensor? D_, Tensor!? z_, Tensor? delta_bias_,bool delta_softplus,Tensor? query_start_loc,Tensor? cache_indices,Tensor? has_initial_state,Tensor! ssm_states,int null_block_id,int block_size,Tensor? block_idx_first_scheduled_token,Tensor? block_idx_last_scheduled_token,Tensor? initial_state_idx,Tensor? cu_chunk_seqlen,Tensor? last_chunk_indices) -> ()",
    "hadacore_transform(Tensor! x, bool inplace) -> Tensor",
    "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, Tensor? b_zeros, bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, Tensor!? b_zeros_reorder, int K, int N, int N_32align) -> ()",
    "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, Tensor? b_qzeros, SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor",
    "minimax_allreduce_rms(Tensor input,Tensor norm_weight,Tensor workspace,int rank,int nranks,float eps) -> Tensor",
    "minimax_allreduce_rms_qk(Tensor qkv,Tensor norm_weight_q,Tensor norm_weight_k,Tensor workspace,int q_size,int kv_size,int rank,int nranks,float eps) -> (Tensor, Tensor)",
    "swap_blocks(Tensor src, Tensor! dst,            int block_size_in_bytes, Tensor block_mapping) -> ()",
    "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,                  Tensor sizes) -> ()",
    "reshape_and_cache(Tensor key, Tensor value,                  Tensor! key_cache, Tensor! value_cache,                  Tensor slot_mapping,                  str kv_cache_dtype,                  Tensor k_scale, Tensor v_scale) -> ()",
    "reshape_and_cache_flash(Tensor key, Tensor value,                        Tensor! key_cache,                        Tensor! value_cache,                        Tensor slot_mapping,                        str kv_cache_dtype,                        Tensor k_scale, Tensor v_scale) -> ()",
    "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,                     Tensor! kv_cache,                     Tensor slot_mapping,                     str kv_cache_dtype,                     Tensor scale) -> ()",
    "concat_and_cache_mla_rope_fused(                     Tensor positions,                     Tensor! q_pe,                     Tensor! k_pe,                     Tensor kv_c,                     Tensor cos_sin_cache,                     bool is_neox,                     Tensor slot_mapping,                     Tensor! kv_cache,                     str kv_cache_dtype,                     Tensor kv_cache_scale) -> ()",
    "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, str kv_cache_dtype) -> ()",
    "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst,                                Tensor block_table, Tensor cu_seq_lens,                                Tensor token_to_seq,                                int num_tokens,                                str kv_cache_dtype,                                Tensor scale, Tensor? seq_starts) -> ()",
    "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()",
    "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, Tensor block_table, Tensor seq_lens, Tensor workspace_starts, int batch_size) -> ()",
    "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor slot_mapping, int quant_block_size, str kv_cache_dtype) -> ()",
    "concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()",
    "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()",
]
# fmt: on
