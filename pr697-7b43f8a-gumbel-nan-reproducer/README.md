# Sunrise vLLM 0.32 model-free sampler reproduction

The exact installed Gumbel sampler for plugin revision 7b43f8a used one Sunrise PTPU. The probe exercised BF16 and FP32, temperature 0 and 0.7, native ATen and the same FlagGems dispatch as the gate. It compared the actual public sampler with the internal block reductions and sampled block choice for 128 rows. Both device paths agreed for each input.

For 248,320 logits and 1,024 entries per block, the last block starts at 247,808 and its first padding position is 248,320. When every logit is NaN, or only logit ID 19 stays finite, the block reduction loses the finite score and the device chooses that padding index. All-negative-infinity input chooses real ID 0. In a diagnostic control, masking NaNs with minus infinity before sampling preserves the finite winner. This identifies how an invalid token can escape the sampler without requiring model or graph execution. The source of the NaNs in the full model remains open.
