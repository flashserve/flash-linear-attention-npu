"""Measure one sequence plan plus all KDA layers of a forward on a single stream."""
import argparse
import json
import sys
from itertools import accumulate
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
from mega_recurrent_kda_reference import _make_case
from fla_npu.ops.ascendc import npu_mega_recurrent_kda

MUTABLE = ("conv_state", "replay_u", "replay_k", "replay_decay")
SHARED = ("cu_seqlens", "state_indices", "conv_num_accepted_tokens")
CASES = {
    "model": ([4] * 30, 12, "TH", 501),
    "ragged": ([4] * 15 + [5, 3] + [4] * 13, 12, "TH", 501),
    "bsh": ([4] * 40, 8, "BSH", 40),
}


def make_layers(name, count, device):
    lengths, heads, layout, slots = CASES[name]
    batch, seq = len(lengths), max(lengths)
    host, flat, _ = _make_case(batch, seq, heads, block_num=batch, seed=19)
    host["rms_norm_eps"] = 1e-5
    hidden = flat.view(batch, seq, -1)
    if layout == "TH":
        host["hidden_states"] = torch.cat([row[:n] for row, n in zip(hidden, lengths)])
        host["cu_seqlens"] = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    else:
        host["hidden_states"] = hidden
        host["cu_seqlens"] = None
    args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in host.items()}
    args["a_log"] = args.pop("A_log")
    for key in (*MUTABLE, "checkpoint_state"):
        value = args[key]
        if slots != value.shape[0]:
            pool = value.new_zeros((slots, *value.shape[1:]))
            pool[:value.shape[0]].copy_(value)
            args[key] = pool
    layers = []
    for _ in range(count):
        layers.append({k: v.clone() if isinstance(v, torch.Tensor) and k not in SHARED else v
                       for k, v in args.items()})
    return args, layers


def run(case, mode, count, repeats, warmup):
    device = torch.device("npu", torch.npu.current_device())
    args, layers = make_layers(case, count, device)
    active = len(CASES[case][0])
    tokens = sum(CASES[case][0])
    if mode == "device" and args["cu_seqlens"] is not None:
        from fla_npu.ops.ascendc import npu_mega_recurrent_kda_metadata as build_plan
    else:
        build_plan = None
    wall, stream = [], []
    for iteration in range(warmup + repeats):
        for layer in layers:
            for key in MUTABLE:
                layer[key][:active].copy_(args[key][:active])
        torch.npu.synchronize()
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        begin = perf_counter_ns()
        start.record()
        plan = build_plan(args["cu_seqlens"], tokens) if build_plan else None
        for layer in layers:
            if mode == "device":
                npu_mega_recurrent_kda(**layer, metadata=plan, use_device_metadata=True)
            else:
                npu_mega_recurrent_kda(**layer)
        end.record()
        end.synchronize()
        elapsed = (perf_counter_ns() - begin) / 1e6
        if iteration >= warmup:
            wall.append(elapsed)
            stream.append(start.elapsed_time(end))
    print(json.dumps(dict(case=case, mode=mode, layers=count, metadata_calls=int(build_plan is not None),
                          wall_ms=median(wall), stream_ms=median(stream),
                          wall_samples_ms=wall, stream_samples_ms=stream)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--mode", choices=("legacy", "device"), required=True)
    parser.add_argument("--layers", type=int, default=69)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    opts = parser.parse_args()
    if min(opts.layers, opts.repeats) < 1 or opts.warmup < 0:
        parser.error("layers and repeats must be positive, warmup must be nonnegative")
    with torch.inference_mode():
        run(opts.case, opts.mode, opts.layers, opts.repeats, opts.warmup)
