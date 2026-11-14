# Sunrise 27B nonfinite-logit boundary

With plugin 7b43f8a and Runner V2 TP2, stage-one hooks observed finite hidden states directly before compute_logits. In the first concurrent text prefill, compute_logits yielded 6,621 NaNs and 3,322 infinities in an eight-row batch on both ranks. The first concurrent image sample had finite encoder outputs, gathered multimodal embeddings and sampled hidden rows, then 83,456 NaNs in the returned logits on both ranks. Single text and single image produced finite sampled logits.

This diagnostic copies selected tensors to CPU outside graph capture. The extra synchronization can affect timing. The run localizes the first observed nonfinite values to the logit path. Local LM-head GEMM and TP gather require separate checks. Independent BF16 vision projection shape checks passed 48/48 for the model's unaligned and padded dimensions.
