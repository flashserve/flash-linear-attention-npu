#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
"""A6 functional smoke using the existing independent Stable-ABI CPU reference.

Integer BF16 inputs and dyadic scale give exact expected outputs. This checks
addressing, transposes and tails; it does not replace ATK accuracy acceptance.
"""

import argparse
import importlib.util
from pathlib import Path
import time

import torch


def load_reference():
    path = Path(__file__).resolve().parents[3] / "stable_abi/test_chunk_kda_bwd_prepare.py"
    spec = importlib.util.spec_from_file_location("prepare_stable_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=65)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--save", type=Path, help="save CPU inputs, expected and actual tensors")
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.batch, args.repeat) <= 0:
        parser.error("tokens, heads, batch and repeat must be positive")
    if args.varlen and args.batch != 1:
        parser.error("packed NTD requires batch=1")

    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc

    torch.set_num_threads(1)
    layout = "NTD" if args.varlen else "BNSD"
    print(f"[CPU] {layout} B={args.batch} H={args.heads} T={args.tokens} "
          f"state_v_first={args.state_v_first}", flush=True)
    inputs, kwargs, expected = load_reference()(
        layout, args.tokens, args.state_v_first, batch=args.batch, heads=args.heads)
    print("[CPU] independent reference ready", flush=True)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    device_inputs = tuple(x.npu() for x in inputs)
    torch.npu.synchronize()
    first = None
    for launch in range(args.repeat):
        print(f"[NPU] launch {launch + 1}/{args.repeat}", flush=True)
        start = time.perf_counter()
        outputs = ascendc.chunk_kda_bwd_prepare(*device_inputs, **kwargs)
        torch.npu.synchronize()
        actual = tuple(x.detach().cpu() for x in outputs)
        print(f"[NPU] completed in {time.perf_counter() - start:.1f}s", flush=True)
        assert ascendc.BACKENDS.get("npu_chunk_kda_bwd_prepare") == "stable", ascendc.BACKENDS
        assert not ascendc.FALLBACKS.get("npu_chunk_kda_bwd_prepare", 0), ascendc.FALLBACKS
        assert len(actual) == 3, "unexpected output count"
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(inputs=inputs, kwargs=kwargs, actual=actual, reference=expected), args.save)
        for index, (name, got, want) in enumerate(zip(("d_aqk", "dv", "dq_raw"), actual, expected)):
            assert outputs[index].is_contiguous(), f"{name}: output must be contiguous"
            assert got.dtype == want.dtype, f"{name}: unexpected dtype {got.dtype}"
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            if first is not None:
                torch.testing.assert_close(got, first[index], rtol=0, atol=0)
            print(f"[CMP] {name:8s} shape={tuple(got.shape)} exact PASS", flush=True)
        if first is None:
            first = actual
    print(f"PASS: {args.repeat} launches, three exact outputs and repeated-call consistency", flush=True)


if __name__ == "__main__":
    main()
