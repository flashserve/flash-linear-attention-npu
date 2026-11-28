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
    "a2_model": ([4] * 8, 6, "TH", 567),
    "a5_model": ([4] * 30, 12, "TH", 492),
    "model": ([4] * 30, 12, "TH", 501),
    "ragged": ([4] * 15 + [5, 3] + [4] * 13, 12, "TH", 501),
    "bsh": ([4] * 40, 8, "BSH", 40),
}


def make_layers(name, count, device, int8=False, mxfp8=False, qkv_scale_alg=0, hidden_width=7168):
    lengths, heads, layout, slots = CASES[name]
    batch, seq = len(lengths), max(lengths)
    host, flat, _ = _make_case(batch, seq, heads, block_num=batch, seed=19, hidden=hidden_width)
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
    if int8:
        w = host["qkv_projection_weight"].T.contiguous().float()
        sw = (w.abs().amax(dim=0) / 127).bfloat16()
        cols = w.shape[1]
        packed = (w / sw).round().clamp(-127, 127).to(torch.int8).reshape(w.shape[0] // 16, 16, cols // 32, 32)
        args["qkv_projection_weight"] = packed.permute(2, 0, 1, 3).contiguous().to(device)
        args["qkv_projection_weight_scale"] = sw.to(device)
    if mxfp8:
        weight = host["qkv_projection_weight"].T.contiguous().float()
        packed = (weight * 32).to(torch.float8_e4m3fn).unsqueeze(0).to(device)
        args["qkv_projection_weight_scale"] = torch.full(
            (1, weight.shape[0] // 64, weight.shape[1], 2), 122, dtype=torch.uint8, device=device
        )
        args["qkv_scale_alg"] = qkv_scale_alg
    for key in (*MUTABLE, "checkpoint_state"):
        value = args[key]
        if slots != value.shape[0]:
            pool = value.new_zeros((slots, *value.shape[1:]))
            pool[:value.shape[0]].copy_(value)
            args[key] = pool
    layers = []
    for _ in range(count):
        layer = {k: v.clone() if isinstance(v, torch.Tensor) and k not in SHARED else v
                 for k, v in args.items() if not (mxfp8 and k == "qkv_projection_weight")}
        if mxfp8:
            layer["qkv_projection_weight"] = torch_npu.npu_format_cast(packed, 29)
        layers.append(layer)
    return args, layers


def run(case, mode, count, repeats, warmup, graph=False, qkv_scale_alg=0, hidden_width=7168):
    device = torch.device("npu", torch.npu.current_device())
    args, layers = make_layers(case, count, device, int8=mode == "int8",
                              mxfp8=mode == "mxfp8", qkv_scale_alg=qkv_scale_alg, hidden_width=hidden_width)
    active = len(CASES[case][0])
    tokens = sum(CASES[case][0])
    if mode != "legacy" and args["cu_seqlens"] is not None:
        from fla_npu.ops.ascendc import npu_mega_recurrent_kda_metadata as build_plan
    else:
        build_plan = None

    def forward():
        plan = build_plan(args["cu_seqlens"], tokens) if build_plan else None
        for layer in layers:
            if mode != "legacy":
                npu_mega_recurrent_kda(**layer, metadata=plan, use_device_metadata=True)
            else:
                npu_mega_recurrent_kda(**layer)

    submit = forward
    if graph:
        for _ in range(3):
            forward()
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            forward()
        submit = captured.replay

    wall, stream = [], []
    for iteration in range(warmup + repeats):
        for layer in layers:
            for key in MUTABLE:
                layer[key][:active].copy_(args[key][:active])
        torch.npu.synchronize()
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        begin = perf_counter_ns()
        start.record()
        submit()
        end.record()
        end.synchronize()
        elapsed = (perf_counter_ns() - begin) / 1e6
        if iteration >= warmup:
            wall.append(elapsed)
            stream.append(start.elapsed_time(end))
    print(json.dumps(dict(case=case, mode=mode, hidden_width=hidden_width, qkv_scale_alg=qkv_scale_alg if mode == "mxfp8" else None,
                          layers=count, metadata_calls=int(build_plan is not None),
                          graph=graph, wall_ms=median(wall), stream_ms=median(stream),
                          wall_samples_ms=wall, stream_samples_ms=stream)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--graph", action="store_true", help="Replay metadata and KDA layers in a fixed-shape NPU graph")
    parser.add_argument("--mode", choices=("legacy", "device", "int8", "mxfp8"), required=True)
    parser.add_argument("--qkv-scale-alg", type=int, choices=(0, 1), default=0)
    parser.add_argument("--hidden-width", type=int, default=7168)
    parser.add_argument("--layers", type=int, default=69)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    opts = parser.parse_args()
    if opts.graph and opts.mode == "legacy":
        parser.error("--graph requires device metadata (--mode device, int8 or mxfp8)")
    if min(opts.layers, opts.repeats) < 1 or opts.warmup < 0:
        parser.error("layers and repeats must be positive, warmup must be nonnegative")
    with torch.inference_mode():
        run(opts.case, opts.mode, opts.layers, opts.repeats, opts.warmup, opts.graph, opts.qkv_scale_alg, opts.hidden_width)
