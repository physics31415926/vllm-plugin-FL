# PR #693 synchronization with Hygon #711

Head `513834682a8cbc4b2b3a9c776d052012e89240ce` merges the prior `982739fd` with main `f0e89ad3`. Three MetaX attention conflicts arose because main relocated a private import which the 0.32 implementation had already removed. The existing 0.32 attention and native MLA reuse remain intact; all Hygon additions and shared upstream changes are preserved.

In the pinned MetaX Linux image with GPU devices deliberately absent, the existing CPU suites produced **374 passed, 13 GPU skips and nine failures during CUDA-driver initialization before test assertions**. The nine router cases require the hardware CI environment; they are not counted as passed. The Hygon rollback, worker-symbol compatibility and 41 lifecycle checks passed. Source before/after validation and owned-container cleanup passed.

The candidate tree matches the committed tree. Complete CI and new-head hardware acceptance remain pending. [result.json](result.json) records the original counts, failure names, image and raw archive fingerprint.
