"""Run the Qwen GDN model-shape regression with the installed A5 wheel."""
import argparse
from contextlib import chdir
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F
import torch_npu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd, recurrent_gated_delta_rule


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests/atk/chunk_gated_delta_rule_fwd"))
sys.path.insert(0, str(ROOT / "tests/atk/recurrent_gated_delta_rule"))
from gdn_reference import GdnCase, _compute_recurrent_outputs
from reference_recurrent_gated_delta_rule import recurrent_gated_delta_rule_reference


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Got {actual.shape}/{actual.dtype}, expected {expected.shape}/{expected.dtype}")
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    # Use the GDN Example/ST output criteria, computing metrics in blocks.
    dot = norm_a = norm_e = max_abs = 0.0
    bad_elements = 0
    for a, e in zip(actual.split(1_048_576), expected.split(1_048_576)):
        if not (torch.isfinite(a).all() and torch.isfinite(e).all()):
            raise ValueError("Non-finite values in output or golden")
        diff = (a - e).abs()
        max_abs = max(max_abs, diff.max().item())
        bad_elements += (diff > 0.005 + 0.005 * e.abs()).sum().item()
        dot += torch.dot(a, e).item()
        norm_a += torch.dot(a, a).item()
        norm_e += torch.dot(e, e).item()
    cosine = min(1.0, dot / math.sqrt(norm_a * norm_e)) if norm_a and norm_e else 0.0
    return {"passed": bad_elements == 0 and cosine >= 0.999, "elements": actual.numel(),
            "bad_elements": bad_elements, "max_abs": max_abs, "cosine": cosine}


def kernel_sequence(trace_file):
    trace = json.loads(trace_file.read_bytes())
    rows = trace["traceEvents"] if isinstance(trace, dict) else trace
    hw = {e["pid"] for e in rows if e.get("name") == "process_name"
          and e.get("args", {}).get("name") == "Ascend Hardware"}
    events = sorted((e for e in rows if e.get("ph") == "X" and
                     (e.get("cat", "").lower() == "kernel" or (e.get("pid") in hw
                      and str(e.get("args", {}).get("Task Type", "")).startswith("KERNEL")))),
                    key=lambda e: float(e["ts"]))
    names = [str(e["name"]) for e in events]
    expected = ("chunkgateddeltarulefwdprepare", "chunkfwdh", "chunkfwdo")
    position = 0
    for name in names:
        compact = re.sub(r"[^a-z0-9]", "", name.lower())
        if position < len(expected) and expected[position] in compact:
            position += 1
    return {"passed": position == len(expected), "device_kernels": list(dict.fromkeys(names))}


def run_prefill(case, dim, chunk, device, result):
    t, hk, hv = case["tokens"], case["qk_heads"], case["v_heads"]
    # Single dense request, native grouped heads. Gate is a natural log.
    q, k = [F.normalize(torch.randn(1, hk, t, dim), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, hv, t, dim).bfloat16()
    g = -torch.rand(1, t, hv) * 0.5
    beta = torch.rand(1, t, hv).bfloat16()
    scale = dim ** -0.5
    inputs = [x.to(device) for x in (q, k, v, g, beta)]
    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with tempfile.TemporaryDirectory(prefix="gdn-a5-trace-") as scratch, chdir(scratch):
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]) as prof:
            values = chunk_gated_delta_rule_fwd(
                *inputs, chunk_size=chunk, layout="BNSD", use_exp2=False,
                use_qk_l2norm_in_kernel=False, disable_recompute=False, scale=scale)
            torch.npu.synchronize()
        trace_file = Path(scratch) / "trace.json"
        prof.export_chrome_trace(str(trace_file))
        result["route"] = kernel_sequence(trace_file)
    shape = GdnCase(1, hk, hv, t, dim, dim, chunk, scale, "dense")
    output, _ = _compute_recurrent_outputs(q, k, v, g, beta, None, shape)
    result["output"] = compare(values[0].cpu(), output.transpose(1, 2).bfloat16())
    return result["route"]["passed"] and result["output"]["passed"]


def run_decode(case, dim, device, result):
    t, hk, hv = case["tokens"], case["qk_heads"], case["v_heads"]
    q, k = [F.normalize(torch.randn(t, hk, dim), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(t, hv, dim).bfloat16()
    state = torch.randn(t, hv, dim, dim) * 0.1
    beta = torch.rand(t, hv).bfloat16()
    gate = -torch.rand(t, hv) * 0.5
    lengths = torch.ones(t + 1, dtype=torch.int32)
    lengths[0] = 0
    slots = torch.arange(t, dtype=torch.int32)
    scale = dim ** -0.5
    actual_state = state.to(device)
    actual = recurrent_gated_delta_rule(q.to(device), k.to(device), v.to(device), actual_state,
                                       beta=beta.to(device), g=gate.to(device), scale=scale,
                                       actual_seq_lengths=lengths.to(device), ssm_state_indices=slots.to(device))
    torch.npu.synchronize()
    expected, updated = recurrent_gated_delta_rule_reference(
        q, k, v, state, beta, scale, lengths, slots, g=gate)
    result["output"] = compare(actual.cpu(), expected)
    result["state"] = compare(actual_state.cpu(), updated)
    return result["output"]["passed"] and result["state"]["passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((HERE / "cases.json").read_bytes())
    torch.set_num_threads(1)
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"Expected an Ascend950 A5, got {name}")
    try:
        wheel_version = importlib.metadata.version("flash-linear-attention-npu-a5")
    except importlib.metadata.PackageNotFoundError:
        wheel_version = None  # The CI also tests a locally built adapter plus OPP overlay.
    report = {"hardware": name, "device": args.device, "torch": str(torch.__version__),
              "torch_npu": str(torch_npu.__version__),
              "wheel": wheel_version,
              "tested_sha": os.environ.get("GDN_TESTED_SHA"), "seed": cfg["seed"],
              "atol": 0.005, "rtol": 0.005, "min_cosine": 0.999, "cases": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for case in cfg["cases"]:
        torch.manual_seed(cfg["seed"])
        start = time.perf_counter()
        result = dict(case, passed=False)
        print(f"Starting {case['name']}", flush=True)
        try:
            if case["entry"] == "prefill":
                result["passed"] = run_prefill(case, cfg["dimension"], cfg["chunk_size"], device, result)
            elif case["entry"] == "decode":
                result["passed"] = run_decode(case, cfg["dimension"], device, result)
            else:
                raise ValueError(f"Unknown entry: {case['entry']}")
        except Exception:
            result["error"] = traceback.format_exc()
        result["seconds"] = round(time.perf_counter() - start, 2)
        report["cases"].append(result)
        report["passed"] = len(report["cases"]) == 5 and all(r["passed"] for r in report["cases"])
        tmp = args.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.report)
        print(json.dumps(result), flush=True)
    return 0 if len(cfg["cases"]) == 5 and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
