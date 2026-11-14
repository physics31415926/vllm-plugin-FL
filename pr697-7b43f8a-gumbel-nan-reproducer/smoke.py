"""Observe exact installed V2 Triton sampler on controlled PTPU logit rows."""

import json
from pathlib import Path

import flag_gems
import torch
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample import gumbel
from vllm_fl.utils import get_flag_gems_whitelist_blacklist
from vllm_fl.worker.sunrise_v2_model_runner import install_sunrise_v2_runtime

WIDTH = 248320
BLOCK = 1024
N_BLOCKS = (WIDTH + BLOCK - 1) // BLOCK
TAIL = (N_BLOCKS - 1) * BLOCK
current_platform.set_device(torch.device("ptpu:0"))
install_sunrise_v2_runtime()
_, blacklist = get_flag_gems_whitelist_blacklist()

names = (
    "finite-tail-wins", "all-negative-inf", "all-nans", "tail-all-nans-with-finite-head",
    "all-nans-except-finite-head", "all-nans-except-finite-tail",
    "all-inf-except-nans-in-tail", "tail-finite-head-neg-inf",
)
x = torch.full((len(names), WIDTH), -torch.inf, dtype=torch.float32)
x[0].fill_(-200)
x[0, WIDTH - 1] = 100
x[2].fill_(torch.nan)
x[3, 19] = 100
x[3, TAIL:] = torch.nan
x[4].fill_(torch.nan)
x[4, 19] = 100
x[5].fill_(torch.nan)
x[5, WIDTH - 1] = 100
x[6, TAIL + 511] = torch.nan
x[7, TAIL] = 100


def classify(number):
    if number != number:
        return "NaN"
    if number == float("inf"):
        return "+Inf"
    if number == float("-inf"):
        return "-Inf"
    return number


def invoke(inputs, temperature):
    rows = inputs.shape[0]
    device = inputs.to("ptpu:0")
    indices = torch.arange(rows, dtype=torch.int32).to("ptpu:0")
    temperatures = torch.full((rows,), temperature, dtype=torch.float32).to("ptpu:0")
    seeds = torch.arange(917, 917 + rows, dtype=torch.int64).to("ptpu:0")
    positions = torch.full((rows,), 190, dtype=torch.int64).to("ptpu:0")
    local_argmax = device.new_empty((rows, N_BLOCKS), dtype=torch.int64)
    local_max = device.new_empty((rows, N_BLOCKS), dtype=torch.float32)
    gumbel._gumbel_sample_kernel[(rows, N_BLOCKS)](
        local_argmax, local_argmax.stride(0), local_max, local_max.stride(0),
        None, 0, 0, None, None, 0,
        device, device.stride(0), indices, seeds, positions, temperatures, WIDTH,
        BLOCK_SIZE=BLOCK, IS_DRAFTING=False, APPLY_TEMPERATURE=False,
        USE_FP64=False, PER_TOKEN_COL=False,
    )
    selected = local_max.argmax(dim=-1, keepdim=True)
    picked = local_argmax.gather(dim=-1, index=selected).view(-1).cpu().tolist()
    selected_cpu = selected.view(-1).cpu().tolist()
    original = gumbel.gumbel_sample(
        device, indices, temperatures, seeds, positions,
        apply_temperature=False, is_drafting=False,
    ).cpu().tolist()
    block_ids, block_values = local_argmax.cpu(), local_max.cpu()
    finite_or_inf_cpu = torch.where(inputs.isnan(), -torch.inf, inputs)
    out = []
    for i, name in enumerate(names):
        block = selected_cpu[i]
        cpu_selected = block_values[i].argmax().item()
        case = dict(
            name=name, input_nan_count=int(inputs[i].isnan().sum()),
            input_finite_count=int(inputs[i].isfinite().sum()),
            sampled_id=original[i], sampled_id_in_bounds=0 <= original[i] < WIDTH,
            manual_same_kernel_id=picked[i], selected_block=block,
            cpu_argmax_over_same_block_values=cpu_selected,
            cpu_id_over_same_block_outputs=block_ids[i, cpu_selected].item(),
            cpu_argmax_with_nan_removed=finite_or_inf_cpu[i].argmax().item(),
            selected_block_score=classify(block_values[i, block].item()),
            first_block_id=block_ids[i, 0].item(), first_block_score=classify(block_values[i, 0].item()),
            final_block_id=block_ids[i, -1].item(), final_block_score=classify(block_values[i, -1].item()),
            block_value_nan_count=int(block_values[i].isnan().sum()),
            block_value_finite_count=int(block_values[i].isfinite().sum()),
            block_ids=block_ids[i].tolist(),
            block_scores=[classify(n) for n in block_values[i].tolist()],
        )
        out.append(case)
    return out


result = {"logical_width": WIDTH, "block_width": BLOCK, "tail_start": TAIL,
          "kernel_file": str(gumbel.__file__), "experiments": []}
with torch.inference_mode():
    for policy in ("native", "r8-dispatch"):
        if policy == "r8-dispatch":
            flag_gems.enable(unused=blacklist)
        for dtype in (torch.float32, torch.bfloat16):
            raw = x.to(dtype)
            filtered = torch.where(raw.isnan(), -torch.inf, raw)
            for case, data in (("raw", raw), ("nan-masked-control", filtered)):
                for temp in (0.0, 0.7):
                    cases = invoke(data, temp)
                    result["experiments"].append(dict(policy=policy, dtype=str(dtype), case=case, temperature=temp, rows=cases))
                    print(policy, str(dtype), case, temp, [(row["name"], row["sampled_id"], row["selected_block"], row["final_block_id"], row["final_block_score"]) for row in cases], flush=True)
Path("sampler-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
