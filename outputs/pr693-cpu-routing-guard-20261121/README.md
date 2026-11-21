# PR #693 CPU routing guard regression

Head `b1423592f370551c3335d8d24e050d2cd3b36696` restricts the existing MetaX extended MoE routing bridge to CUDA tensors. The two newly merged upstream CPU tests were entering the GPU-only FlagGems-vllm provider before reaching their mocked generic/native routing paths. CPU inputs now retain the generic path; MACA CUDA inputs retain the existing provider bridge.

The original two tests and every assertion are unchanged. A focused CPU-only diagnosis reproduced the same two failures twice before the fix, then passed both and all 26 related routing contracts after the fix. The diagnostic temporarily supplied capability metadata only while importing the real module, then restored the original method before pytest; no GPU device was exposed or initialized. This diagnostic helper is outside the PR and is not a hardware Gate or a CI workaround.

The existing provider-mock contract now explicitly models logical CUDA metadata while retaining its CPU storage and mathematical assertions. Existing CPU routing coverage includes both absent and all-false padding. Ruff, formatting and whitespace checks passed. [result.json](result.json) retains original counts, tested router fingerprints and raw archive SHA256. Complete unmodified final-head hardware CI remains pending.
