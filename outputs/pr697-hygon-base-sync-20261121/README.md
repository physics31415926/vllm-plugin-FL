# PR #697 synchronization with Hygon #711

Head `70105a334d743c035bcba997de0d0cbd348ec78f` merges the prior `8f5ad935` with `0.5.0-dev` `f0e89ad3`. The conflict was an import block in Sunrise attention: main moved a private variable which the existing Sunrise 0.32 migration had already removed. The existing Sunrise attention blob is retained; the Hygon changes and shared upstream contracts are preserved.

The unchanged current CPU suites plus the existing Hygon rollback suite passed **194 tests, zero failures and zero skips**, under inherited `GEMS_VENDOR=kunlunxin`. The tested candidate tree is identical to the committed tree. These CPU checks do not replace the final-head hardware CI and model matrix, which remain pending.
