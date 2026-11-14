import faulthandler
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.topk_topp_compat import (
    apply_topk_topp_compat,
)
from vllm_fl.utils import get_flag_gems_whitelist_blacklist
from vllm.v1.worker.gpu.sample import sampler as sampler_module
import flag_gems


faulthandler.dump_traceback_later(60, repeat=True)


def sample_argmax(logits, *args, **kwargs):
    return logits.argmax(dim=-1)


def check():
    apply_topk_topp_compat()
    sampler_module.gumbel_sample = sample_argmax
    sampler = object.__new__(sampler_module.Sampler)
    sampler.use_fp64_gumbel = False
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert not whitelist, whitelist
    print("OP_CONFIG", whitelist, blacklist, flush=True)
    cases = []
    for batch_size in (1, 4, 8, 16):
        vocab_size = 248320
        generator = torch.Generator().manual_seed(2026 + batch_size)
        host = torch.randn(batch_size, vocab_size, generator=generator)
        host[:,0:3] = torch.tensor([13.0,12.0,11.0])
        index = torch.arange(batch_size)
        sampler.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(batch_size, device='cuda')),
            seeds=SimpleNamespace(gpu=index.cuda()),
        )
        masked_host = host.clone()
        masked_host[:, 64:] = -torch.inf
        for route, k, p in [
            ("grammar_masked_top_p", None, torch.full((batch_size,), 0.7)),
            ("top_p", None, torch.full((batch_size,), 0.7)),
            ("top_k", torch.full((batch_size,), 50, dtype=torch.int32), None),
            ("top_kp", torch.full((batch_size,), 50, dtype=torch.int32), torch.full((batch_size,), 0.7)),
            ("mixed", torch.tensor([50 if i%2 else vocab_size for i in range(batch_size)],dtype=torch.int32),
                torch.tensor([0.7 if i%2 else 1.0 for i in range(batch_size)])),
        ]:
            from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
            from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
            print("REFERENCE", batch_size, vocab_size, route, flush=True)
            inputs = masked_host if route == "grammar_masked_top_p" else host
            cpu = apply_top_k_top_p_pytorch(inputs.clone(), k, p)
            gpu = apply_top_k_top_p_pytorch(
                inputs.cuda(), k.cuda() if k is not None else None,
                p.cuda() if p is not None else None,
            ).cpu()
            cases.append((batch_size, route, inputs, k, p, cpu, gpu))
    flag_gems.enable(unused=blacklist)
    for batch_size, route, inputs, k, p, cpu, expected in cases:
        index = torch.arange(batch_size)
        sampler.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(batch_size, device="cuda")),
            seeds=SimpleNamespace(gpu=index.cuda()),
        )
        print("START", batch_size, vocab_size, route, flush=True)
        with torch.inference_mode():
            device_logits = inputs.cuda()
            sampled, filtered = sampler._sample_random(
                device_logits, index.cuda(), index.numpy(), index.cuda(),
                k.cuda() if k is not None else None,
                p.cuda() if p is not None else None, False,
            )
            torch.cuda.synchronize()
        filtered_cpu = filtered.cpu()
        differences = (filtered_cpu != expected).sum().item()
        cpu_differences = (cpu != expected).sum().item()
        print("COUNTS", batch_size, route, differences, cpu_differences, flush=True)
        torch.testing.assert_close(filtered_cpu, expected, rtol=0, atol=0)
        torch.testing.assert_close(sampled.cpu(), expected.argmax(dim=-1))
        print("PASS", batch_size,vocab_size,route,flush=True)


if __name__ == "__main__":
    check()
