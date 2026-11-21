# THead pinned-provider CI download fix

Both platform PRs reproduced the same THead setup failure: pip's partial Git clone of the fixed FlagGems-vllm source lost its connection and could not fetch a required promisor blob. The functional tests did not start.

The fix adds only `--source-archive` to THead's existing installation-helper call. The helper, provider commit `b5c2c2249319277ccda4d41aa0f65164bd9b2a19`, dependency constraints and test policy remain unchanged. The existing archive SHA256 was verified against a fresh official 3,774,603-byte download: `1b47e1eb7982acc606e2fc920561d169e2a17fac7c92ff65c5f11cafa6b8b16e`.

All 27 existing installer/native-runtime preservation CPU contracts and shell syntax checks passed. Complete final-head CI is still required, including an actual successful download from the CI runner. [result.json](result.json) records both applied commits and the original failed jobs.
