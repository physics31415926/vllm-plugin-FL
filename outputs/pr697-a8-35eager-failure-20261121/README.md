# Current-head Sunrise 35B eager failure

Exact source `a8d306185adc24bee5f05ce60d74f224a0b3d493`, 2026-11-21 04:07:42–04:22:31 UTC. The original five-scenario Gate executed all 26 requests: **25 passed, one failed; 258/260 checks passed**. The first text response omitted the required inference topic/term, causing both the original semantic and ordered-term checks to fail. It ended normally with finish_reason=stop after 349 completion tokens, below the unchanged 512-token limit; this was not a request timeout.

The other 25 text/image/mixed requests passed. The original validator rejected text_single, so the supplemental long-prefill request was not run. Source/installed-file identities and normal process/device/lock cleanup passed, which does not override the quality failure. 35B graph and the complete matrix verifier were not started. No prompt, check, timeout, seed or retry was changed to obtain a pass.

`original-text-single.json` preserves the original input, output and checks. `quality-analysis.json` recomputes the original pure check functions without modifying request data. `qualification.json` records the pinned stack, native per-worker source-flow evidence, failure exit codes, lifecycle and locally retained 78,607,536-byte raw archive SHA256. The root cause remains under investigation; output variation alone does not establish a specific operator defect.
