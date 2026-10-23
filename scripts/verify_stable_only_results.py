#!/usr/bin/env python3
"""Record public AscendC calls for a before/after Stable-only regression.

Run the SAME test script and seeds in isolated baseline/candidate packages:
  python scripts/verify_stable_only_results.py record --output before.json --script test.py -- --device 0
  python scripts/verify_stable_only_results.py compare before.json after.json

This supplements independent CPU-reference tests. It does not replace ATK or
its numerical tolerances. A matching digest proves byte-identical tensors;
missing calls, exceptions and non-finite floating outputs fail closed.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import threading


def digest_description(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def finite_description(value) -> bool:
    if isinstance(value, dict):
        return value.get("finite", True) is True and all(
            finite_description(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_description(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def tensor_descriptions_valid(value) -> bool:
    if isinstance(value, dict):
        if value.get("tensor") is True:
            required = {"shape", "dtype", "sha256", "finite", "requires_grad",
                        "version", "device", "stride", "storage_offset"}
            if not required <= value.keys():
                return False
            if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
                return False
            if type(value["finite"]) is not bool:
                return False
        return all(tensor_descriptions_valid(v) for v in value.values())
    if isinstance(value, list):
        return all(tensor_descriptions_valid(v) for v in value)
    return True


def record(output: Path, script: Path, script_args: list[str]) -> int:
    report = {
        "format": 2, "script": script.name, "arguments": script_args,
        "calls": [], "exit_code": 1,
    }
    old_argv, old_path = sys.argv[:], sys.path[:]
    stable_originals = {}
    public_originals = {}
    resolver = None
    ascendc = _stable = None
    published = ()
    nesting = threading.local()

    def describe(value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            finite = bool(torch.isfinite(cpu).all()) if cpu.is_floating_point() or cpu.is_complex() else True
            try:
                version = value._version
            except RuntimeError:  # inference tensors have no version counter
                version = None
            return {"tensor": True, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(), "finite": finite,
                    "requires_grad": value.requires_grad, "version": version,
                    "device": str(value.device), "stride": list(value.stride()),
                    "storage_offset": value.storage_offset()}
        if isinstance(value, dict):
            # Preserve key types: converting 1 and "1" to strings loses data.
            return {"kind": "dict", "items": sorted(
                [[describe(k), describe(v)] for k, v in value.items()],
                key=lambda item: json.dumps(item[0], sort_keys=True))}
        if isinstance(value, (list, tuple)):
            return {"kind": type(value).__name__, "items": [describe(v) for v in value]}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return {"kind": type(value).__name__, "value": str(value)}
        raise TypeError(f"Cannot deterministically record {type(value).__name__}")

    def instrument(name, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            active = getattr(nesting, "operators", ())
            # Public resolution wraps the mutation contract outside _stable.
            # Record that outer call once, including its final version bump.
            if active and active[-1] == name:
                return fn(*args, **kwargs)
            nesting.operators = (*active, name)
            entry = {"op": name}
            report["calls"].append(entry)
            try:
                inputs = describe((args, kwargs))
                entry.update(inputs=inputs, input_sha256=digest_description(inputs))
                result = fn(*args, **kwargs)
                entry["outputs"] = describe(result)
                entry["inputs_after"] = describe((args, kwargs))
                entry["finite"] = finite_description(entry["outputs"])
                return result
            except BaseException as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                nesting.operators = active
        return call

    exit_code = 0
    try:
        report["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        import torch
        from fla_npu.ops import ascendc
        from fla_npu.ops.ascendc import _stable

        _stable.load()
        published = tuple(ascendc._ASCENDC_OPS)
        # Verify actual dispatcher registrations, not just Python attributes.
        for name in published:
            _stable._op(name)
        report.update(
            torch=torch.__version__, launcher=str(_stable._loaded_path),
            public_ops=sorted(published),
            signatures_before={n: str(inspect.signature(getattr(ascendc, n))) for n in published},
        )
        # Instrument the implementation entry points. This also catches tests
        # importing _stable directly, while public wrappers and autograd routes
        # reach the same functions and therefore produce only one record each.
        for name in published:
            fn = getattr(_stable, name)
            stable_originals[name] = fn
            setattr(_stable, name, instrument(name, fn))
        resolver = ascendc._get_direct_op
        resolver.cache_clear()
        resolved = {}

        def resolve(name):
            if name in resolved:
                return resolved[name]
            fn = resolver(name)
            if ascendc.BACKENDS.get(name) != "stable":
                raise RuntimeError(f"{name} did not select Stable ABI")
            resolved[name] = instrument(name, fn)
            return resolved[name]

        ascendc._get_direct_op = resolve
        # Replace already-resolved public raw entries if the host imported and
        # used the package before calling record(). Keep autograd aliases intact.
        for name in published:
            public_originals[name] = getattr(ascendc, name)
            setattr(ascendc, name, ascendc._make_raw_wrapper(name))
            short = name[4:]
            public_originals[short] = getattr(ascendc, short)
            if short not in ("fast_gelu_custom", "causal_conv1d"):
                setattr(ascendc, short, getattr(ascendc, name))
        sys.argv = [str(script), *script_args]
        sys.path.insert(0, str(script.resolve().parent))
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        report["test_error"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 1
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        if any("error" in call or call.get("finite") is not True for call in report["calls"]):
            exit_code = 1
        if not report["calls"]:
            exit_code = 1
        report["exit_code"] = exit_code
        if ascendc is not None:
            report["signatures_after"] = {n: str(inspect.signature(getattr(ascendc, n))) for n in published}
        if resolver is not None:
            ascendc._get_direct_op = resolver
            resolver.cache_clear()
        for name, fn in public_originals.items():
            setattr(ascendc, name, fn)
        for name, fn in stable_originals.items():
            setattr(_stable, name, fn)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return exit_code


def compare_reports(left: dict, right: dict, required_ops: tuple[str, ...] = ()) -> dict:
    failures = []
    for label, report in (("before", left), ("after", right)):
        if report.get("format") != 2:
            failures.append(f"{label}: unsupported or absent format")
        for key in ("script", "script_sha256", "torch", "public_ops", "signatures_before", "signatures_after"):
            if not report.get(key):
                failures.append(f"{label}: missing {key}")
        ops = report.get("public_ops") or []
        if len(ops) != len(set(ops)):
            failures.append(f"{label}: duplicate public operators")
        for key in ("signatures_before", "signatures_after"):
            if set(report.get(key) or {}) != set(ops):
                failures.append(f"{label}: incomplete {key}")
        if "test_error" in report:
            failures.append(f"{label}: test_error recorded")
        for index, call in enumerate(report.get("calls", [])):
            if "error" in call:
                failures.append(f"{label} call {index}: operator raised")
            for key in ("op", "inputs", "input_sha256", "outputs", "inputs_after", "finite"):
                if key not in call:
                    failures.append(f"{label} call {index}: missing {key}")
            if call.get("op") not in report.get("public_ops", []):
                failures.append(f"{label} call {index}: unlisted operator")
            for key in ("inputs", "outputs", "inputs_after"):
                if not tensor_descriptions_valid(call.get(key)):
                    failures.append(f"{label} call {index}: incomplete tensor description in {key}")
            if "inputs" in call:
                try:
                    correct_hash = digest_description(call["inputs"])
                except (TypeError, ValueError):
                    correct_hash = None
                if call.get("input_sha256") != correct_hash or correct_hash is None:
                    failures.append(f"{label} call {index}: invalid input hash")
            if call.get("finite") is not True or not finite_description(call.get("outputs")):
                failures.append(f"{label} call {index}: non-finite or unverified output")
        missing = set(required_ops) - {call.get("op") for call in report.get("calls", [])}
        if missing:
            failures.append(f"{label}: required operators not executed: {sorted(missing)}")
    for key in ("public_ops", "script", "script_sha256", "arguments", "torch", "signatures_before"):
        if left.get(key) != right.get(key):
            failures.append(f"{key} differs")
    if left.get("exit_code") != 0 or right.get("exit_code") != 0:
        failures.append("one test script failed")
    a, b = left.get("calls", []), right.get("calls", [])
    if not a or len(a) != len(b):
        failures.append(f"call coverage differs or is empty: {len(a)} vs {len(b)}")
    for index, (old, new) in enumerate(zip(a, b)):
        for key in ("op", "inputs", "input_sha256", "outputs", "inputs_after"):
            if old.get(key) != new.get(key):
                failures.append(f"call {index} {old.get('op')}: {key} differs")
    if right.get("signatures_after") != right.get("signatures_before"):
        failures.append("candidate public signatures changed after first call")
    return {"passed": not failures, "calls": len(a),
            "operators": sorted({x.get("op", "<missing>") for x in a}), "failures": failures}


def compare(before: Path, after: Path, required_ops: tuple[str, ...] = ()) -> int:
    try:
        left, right = (json.loads(p.read_text(encoding="utf-8")) for p in (before, after))
        summary = compare_reports(left, right, required_ops)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        summary = {"passed": False, "failures": [f"invalid report: {exc}"]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--script", type=Path, required=True)
    rec.add_argument("script_args", nargs=argparse.REMAINDER)
    cmp = sub.add_parser("compare")
    cmp.add_argument("before", type=Path)
    cmp.add_argument("after", type=Path)
    cmp.add_argument("--require-op", action="append", default=[],
                     help="operator that must actually execute (repeatable)")
    cmp.add_argument("--require-all", action="store_true",
                     help="require every published AscendC operator to execute")
    args = parser.parse_args()
    if args.mode == "compare":
        required = args.require_op
        if args.require_all:
            required = json.loads(args.before.read_text(encoding="utf-8"))["public_ops"]
        return compare(args.before, args.after, tuple(required))
    rest = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    return record(args.output, args.script, rest)


if __name__ == "__main__":
    raise SystemExit(main())
