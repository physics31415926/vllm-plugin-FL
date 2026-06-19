# vllm-plugin-FL MetaX Adaptation

## MetaX C550 Connection

SSH alias: `metax_c550` (configured in `~/.ssh/config`)

Machine: `bm-turing-hz1-zone1-MC550-64G-1-15`
- 8x MetaX C550 64GB
- MACA 3.7.1.5, Kernel Driver 3.8.1
- Host OS: Ubuntu

## Container: vllm-fl-adapt-0202

```bash
# Enter container
ssh metax_c550 "docker exec -it vllm-fl-adapt-0202 bash"

# Run command in container
ssh metax_c550 "docker exec vllm-fl-adapt-0202 bash -c '<command>'"

# With conda env (required for vllm)
ssh metax_c550 "docker exec vllm-fl-adapt-0202 bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate base && <command>'"
```

Container config:
- Image: `cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.20.0-maca.ai3.7.0.107-torch2.8-py312-ubuntu22.04-amd64`
- Mode: `--privileged --network host --ipc host --shm-size 16g`
- Mounts: `/public-nfs/wlx:/public-nfs/wlx`, `/opt/maca:/opt/maca`
- Python: 3.12 (conda base)
- vLLM: 0.20.0
- vllm_metax: 0.20.0+gcce172.d20260529.maca3.7.0.38.torch2.8

## Code Sync (Local → Container)

The container has the repo at `/workspace/vllm-plugin-FL` (editable install).
To sync changes: commit & push locally, then pull in container.

```bash
# Local: commit and push
git add -A && git commit -m "..." && git push origin metax_adapt_0202

# Container: pull
ssh metax_c550 "docker exec vllm-fl-adapt-0202 bash -c 'cd /workspace/vllm-plugin-FL && git pull origin metax_adapt_0202'"
```

If network is slow or blocked, use proxy:
```bash
# In container, set proxy before git operations
ssh metax_c550 "docker exec vllm-fl-adapt-0202 bash -c 'export https_proxy=\"http://lxwang:d6fHoPs2TTgsaG4a@114.111.17.35:3128\" && export http_proxy=\"http://lxwang:d6fHoPs2TTgsaG4a@114.111.17.35:3128\" && cd /workspace/vllm-plugin-FL && git pull origin metax_adapt_0202'"
```

## Models (Shared Storage)

Path: `/public-nfs/wlx/models/`
- `/public-nfs/wlx/models/Qwen/Qwen3___6-27B` (note: dots replaced with `___` in filesystem)

## GPU Visibility

Use `MACA_VISIBLE_DEVICES` to select GPUs (equivalent to CUDA_VISIBLE_DEVICES):
```bash
MACA_VISIBLE_DEVICES=0,1,2,3 python script.py  # Use first 4 GPUs
```

## Long-Running Commands

Container has no tmux, use host-level tmux:
```bash
ssh metax_c550 "tmux new-session -d -s <session> \"docker exec vllm-fl-adapt-0202 bash -c '<command>'\""
ssh metax_c550 "tmux capture-pane -t <session> -p"
ssh metax_c550 "tmux has-session -t <session> 2>/dev/null && echo running || echo done"
ssh metax_c550 "tmux kill-session -t <session>"
```
