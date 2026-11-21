# PR #697 current-head 27B eager Gate

Exact head `a8d306185adc24bee5f05ce60d74f224a0b3d493`, tree `8d9d78848e5f49686e93f81b14f023d2a51bce8c`, Sunrise physical cards 3/7, TP2.

The unchanged original Gate passed all five scenarios, 26 requests and 260 checks, plus a supplemental 4,118-token prefill with the original checks in 42.315 seconds. Request timeout stayed 300 seconds. Actual native V2, default async scheduling, eager mode, context 8,192, eight sequences and 4 GiB KV cache per worker were verified. Source before/after (821 tracked paths including one symlink), all 327 installed package Python files and the pinned dependency stack were checked.

The standard native FlagCX per-PID logs recorded rank0/1 world-size2 initialization DONE for the same worker PIDs later reporting V2, weights and readiness. DONE is bootstrap evidence. The unchanged constructor performs native initialization and collective warmup outside the library-load catch; continuation to model startup provides the source-flow inference that the warmup completed. No communicator-property snapshot or extra RPC is fabricated. The separate direct numerical communication preflight remains supporting evidence.

The server and owned container exited naturally. All return codes were zero, source and device cleanup passed, and all four cooperative locks were independently released. [result.json](result.json) retains configuration, native evidence and original validator/raw archive hashes. Only this one profile has completed; the remaining graph/35B profiles and complete final-head CI remain pending.
