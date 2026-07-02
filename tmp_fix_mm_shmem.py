import yaml

path = '/nfs/wlx/adapt/nvidia-vllm-0.24.0/FlagGems/src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml'
with open(path) as f:
    cfg = yaml.safe_load(f)

mm_cfgs = cfg.get('mm', [])

SHMEM_LIMIT = 166912

def shmem(c):
    m = c.get('META', {})
    bm = m.get('BLOCK_M', 0)
    bn = m.get('BLOCK_N', 0)
    bk = m.get('BLOCK_K', 0)
    s  = c.get('num_stages', 1)
    return s * (bm + bn) * bk * 2

removed = [c for c in mm_cfgs if shmem(c) > SHMEM_LIMIT]
kept    = [c for c in mm_cfgs if shmem(c) <= SHMEM_LIMIT]

print('Removed %d oversize config(s):' % len(removed))
for c in removed:
    m = c['META']
    print('  BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d, num_stages=%d -> shmem=%d bytes' % (
        m['BLOCK_M'], m['BLOCK_N'], m['BLOCK_K'], c['num_stages'], shmem(c)))

cfg['mm'] = kept
with open(path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print('Done. Kept %d configs.' % len(kept))
