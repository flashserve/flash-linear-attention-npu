"""Fixed issue-680 model cases for msprof Task Duration comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options
from gen_recurrent_kda import model_specs
from fla_npu.ops.ascendc import recurrent_kda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--specs", type=Path,
                        help="Fixed representative (46) or full (124) optimization matrix")
    parser.add_argument("--case", action="append", help="Repeat to select model case names")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    specs = model_specs()
    if args.specs:
        from optimization_suite import validate_suite
        if args.case:
            parser.error("--specs requires the whole matrix; --case is not allowed")
        specs = json.loads(args.specs.read_text(encoding="utf-8"))
        validate_suite(specs)
    if args.case:
        unknown = set(args.case) - {spec["name"] for spec in specs}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        specs = [spec for spec in specs if spec["name"] in args.case]
    manifest = []
    for spec in specs:
        inputs = build_inputs(spec, torch.device(f"npu:{args.device}"))
        options = call_options(spec)
        for _ in range(args.warmup):
            recurrent_kda(**inputs, **options)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_allocated()
        for _ in range(args.samples):
            output = recurrent_kda(**inputs, **options)
        torch.npu.synchronize()
        extra = torch.npu.max_memory_allocated() - before
        manifest.append(dict(spec=spec, warmup=args.warmup, samples=args.samples,
                             extra_allocated_bytes=extra))
        del output, inputs
        print("PROFILE_CASE", spec["name"], "extra_bytes", extra, flush=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
