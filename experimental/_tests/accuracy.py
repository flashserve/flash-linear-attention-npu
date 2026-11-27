#!/usr/bin/env python3
"""Direct RWKV8 NPU/CPU-reference accuracy validation; no ATK installation required."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "experimental/tests/atk/fused_recurrent_rwkv8/atk_fused_recurrent_rwkv8.json"


def load_cases(mode, indices=None):
    catalog = json.loads(CASES.read_text(encoding="utf-8"))
    selected = list(range(3 if mode == "quick" else len(catalog))) if indices is None else indices
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Accuracy selection must be nonempty and contain no duplicates")
    cases = []
    for index in selected:
        if index < 0 or index >= len(catalog):
            raise ValueError(f"Case index out of range: {index}")
        entry = catalog[index]
        if entry["standard"]["acc"] != "mixed_tolerance_bm":
            raise ValueError(f"Unsupported accuracy standard for case {index}")
        spec = json.loads(next(x["range_values"] for x in entry["inputs"] if x["name"] == "case_spec"))
        cases.append({"index": index, "name": entry["name"], "spec": spec})
    return cases


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-indices", help="Comma-separated catalog indices for manual diagnosis")
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    cases = load_cases(args.mode, None if args.case_indices is None else
                       [int(x) for x in args.case_indices.split(",")])
    report = {"schema": "experimental-accuracy-v1", "backend": "cpu-reference",
              "standard": "mixed_tolerance_bm-v1-atk26.9.8", "found": True,
              "complete": False, "all_pass": False, "total": 0, "check_pass": 0,
              "check_fail": 0, "exec_fail": 0, "expected": len(cases),
              "case_indices": [case["index"] for case in cases], "device": args.device,
              "launch_requested": os.environ.get("FLA_NPU_STABLE_LAUNCH", "default"), "results": []}
    save(args.output, report)
    try:
        # Vendor selection must happen before torch_npu initializes CANN.
        from fla_npu_experimental import fused_recurrent_rwkv8
        import torch
        import torch_npu  # noqa: F401

        reference = load_module("rwkv8_reference", ROOT / "experimental/_tests/rwkv8_reference.py")
        trace = load_module("experimental_trace", ROOT / "experimental/tests/atk/common/experimental_isolation_trace.py")
        device = torch.device(f"npu:{args.device}")
        torch.npu.set_device(device)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        with torch.inference_mode():
            for case in cases:
                started = time.monotonic()
                item = dict(case)
                try:
                    cpu = reference._rwkv8_inputs(case["spec"], torch.device("cpu"), high_precision=False)
                    # Quantize once. DUT and FP64 golden see precisely the same values.
                    high = {key: value.double() if isinstance(value, torch.Tensor) else value
                            for key, value in cpu.items()}
                    golden = reference.fused_recurrent_rwkv8_reference(**high)
                    del high
                    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                              for key, value in cpu.items()}
                    outputs = fused_recurrent_rwkv8(**inputs)
                    torch.npu.synchronize(device)
                    trace.record_experimental_execution(device)
                    item["outputs"] = reference.check_outputs(outputs, golden, inputs)
                    passed = all(value["pass"] for value in item["outputs"].values())
                    item["status"] = "PASS" if passed else "FAIL"
                    report["check_pass" if passed else "check_fail"] += 1
                    del cpu, golden, inputs, outputs
                except Exception as exc:
                    item.update(status="ERROR", error=str(exc))
                    report["exec_fail"] += 1
                item["seconds"] = round(time.monotonic() - started, 3)
                report["results"].append(item)
                report["total"] += 1
                save(args.output, report)
                print(f"[experimental][accuracy] {case['index']} {case['name']}: {item['status']}", flush=True)
                if item["status"] == "ERROR":
                    break  # A failed NPU call may leave the stream unusable.
        report["complete"] = report["total"] == len(cases)
        report["all_pass"] = report["complete"] and report["check_pass"] == len(cases)
    except Exception as exc:
        report.update(error=str(exc), exec_fail=report["exec_fail"] + 1)
        print(f"[experimental][ERROR] {exc}", file=sys.stderr)
    finally:
        save(args.output, report)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
