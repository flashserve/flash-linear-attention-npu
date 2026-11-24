# SPDX-License-Identifier: BSD-3-Clause
"""Check the full output before measuring synchronized end-to-end time and allocation."""
import argparse
import json
import statistics
import time

import torch
import torch_npu

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cases import check, expected, make_case, on_device
from fla_npu.ops.vdn import vdn_forward
from fla_npu.ops.vdn.vdn_native import library


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heads", type=int, nargs="+", default=[7])
    parser.add_argument("--frames", type=int, default=87)
    parser.add_argument("--rule", choices=("vdn_solve", "newton_schulz"), default="newton_schulz")
    parser.add_argument("--repeats", type=int, default=31)
    options = parser.parse_args()
    if options.repeats < 10:
        parser.error("--repeats must be at least 10")
    device = torch.device("npu", options.device)
    torch.npu.set_device(device)
    torch.npu.matmul.allow_hf32 = False
    native = library(device) is not None
    for heads in options.heads:
        cpu = make_case(options.frames, heads, 405, torch.bfloat16)
        args = on_device(cpu, device)
        tile = min(64, 1 << max(0, (512 // heads).bit_length() - 1))

        def call():
            return vdn_forward(*args, delta_rule=options.rule, fuse_silu=True, fuse_qk_l2norm=True,
                               frame_tile=tile, value_first=True)

        # Head slices keep the independent CPU oracle within host memory for 56 heads.
        actual = call().cpu()
        for start in range(0, heads, 7):
            sub = slice(start, min(start + 7, heads))
            sliced = tuple(x[:, sub] if i < 5 else x[sub] if i == 6 else x
                           for i, x in enumerate(cpu))
            check(actual[:, sub], expected(sliced))
        del actual, cpu
        for _ in range(8):
            call()
        torch.npu.synchronize(device)
        times = []
        before = torch.npu.memory_allocated(device)
        torch.npu.reset_peak_memory_stats(device)
        for _ in range(options.repeats):
            start = time.perf_counter_ns()
            result = call()
            torch.npu.synchronize(device)
            times.append((time.perf_counter_ns() - start) / 1e6)
            del result
        times.sort()
        print(json.dumps(dict(frames=options.frames, heads=heads, frame_tile=tile,
                              device=torch.npu.get_device_name(device), rule=options.rule, native=native,
                              repetitions=len(times), p10_ms=times[round(.1*(len(times)-1))],
                              p50_ms=statistics.median(times), p90_ms=times[round(.9*(len(times)-1))],
                              peak_allocated_mib=torch.npu.max_memory_allocated(device)/2**20,
                              incremental_peak_mib=(torch.npu.max_memory_allocated(device)-before)/2**20,
                              peak_reserved_mib=torch.npu.max_memory_reserved(device)/2**20)), flush=True)


if __name__ == "__main__":
    main()

