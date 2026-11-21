# Historical PR #697 27B eager Gate

Tested plugin head `8f5ad9358d14f77475d2119ef29fab3ff899fe5b`, before synchronization with Hygon #711.

On Sunrise physical cards 3/7, TP2, the unchanged formal Gate passed all five scenarios, 26 shared requests and 260 individual checks. The supplemental long-prefill request measured 4,118 input tokens and passed the original checks in 42.417 seconds. Actual configuration confirmed native Runner V2, async scheduling, eager mode, context 8,192, eight sequences and 4 GiB KV cache per worker. The request timeout remained 300 seconds. Per-group supported compiler caches were initially empty.

The server and owned container exited naturally. All controller, monitor, attach, inside and original Gate return codes were zero. Source before/after verification, installed wheel identity, selected-device idle/handles/ECC, owned-container absence and independent release of all four cooperative locks passed.

The other three profiles were paused when `0.5.0-dev` advanced, so acceptance for the full matrix and newer head `70105a33` remains pending. [result.json](result.json) retains the original validator and raw archive hashes, software provenance and exact result scope. Direct FlagCX numerical communication preflight is recorded separately; package imports alone are not model-dispatch evidence.
