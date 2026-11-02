"""Compare K-first issue680 shapes with installed upstream/current packages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--case-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")
    sys.path.insert(0, str(args.package.resolve()))
    sys.path.insert(0, str(args.case_root.resolve()))
    import fla_npu
    from fla_npu.ops.ascendc import recurrent_kda
    from executor_recurrent_kda import build_inputs, call_options
    from gen_recurrent_kda import model_specs

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    specs = [dict(spec, layout="TND", state_v_first=False, state_dtype=dtype)
             for dtype in ("fp32", "bf16") for spec in model_specs()]
    if args.smoke:
        specs = [dict(spec, B=2, H=2, HV=4)
                 for spec in specs if spec["case_id"] in (0, 3)]
    manifest = []
    print("PACKAGE", fla_npu.__file__, "TORCH", torch.__version__,
          "TORCH_NPU", torch_npu.__version__, flush=True)
    for index, spec in enumerate(specs):
        inputs = build_inputs(spec, torch.device(f"npu:{args.device}"))
        options = call_options(spec)
        description = {name: dict(shape=list(t.shape), stride=list(t.stride()),
                                 dtype=str(t.dtype))
                       for name, t in inputs.items() if isinstance(t, torch.Tensor)}
        for _ in range(args.warmup):
            recurrent_kda(**inputs, **options)
        torch.npu.synchronize()
        for _ in range(args.samples):
            output = recurrent_kda(**inputs, **options)
        torch.npu.synchronize()
        assert all(t is None or bool(torch.isfinite(t).all().item()) for t in output)
        manifest.append(dict(index=index, spec=spec, inputs=description,
                             options=options, warmup=args.warmup, samples=args.samples))
        print("PROFILE_CASE", index, spec["name"], spec["state_dtype"], flush=True)
        del inputs, output
    args.manifest.write_text(json.dumps(dict(package=fla_npu.__file__, cases=manifest),
                                        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
