# AMD W7900D Runner V2 gate

The exact plugin source was `4cd14672156c8d6fbb8a637ecb2e54c6dd8c221c`. Both models passed 26/26 requests in both modes with eight W7900D cards. The graph logs recorded 197 and 307 request-time FULL executions.

[Summary](summary.json) records the runtime and fingerprints of the local model directories. The model directories do not record immutable upstream revisions. [Launch script](run.sh) shows both environments and commands; provision the corresponding plugin, FlagGems, FlagTree and FlagCX components before launching. The 35B-A3B graph group uses a 20 GiB per-worker KV budget and FULL_DECODE_ONLY capture sizes 1–32.

[Scenarios](scenarios) contains all 20 request JSONs, with host workspace paths normalized. [Graph log excerpts](graph-log-excerpts.txt) preserves startup and runtime replay records. This run used the existing ROCm qualification container and the staged FlagGems tree. The pinned AMD Docker build and the final merged FlagGems commit are pending qualification.
