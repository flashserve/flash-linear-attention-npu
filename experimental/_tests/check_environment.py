#!/usr/bin/env python3
"""Check experimental build/runtime dependencies before compiling any operators."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REQUIRED = ("torch", "torch_npu", "numpy", "setuptools", "wheel", "pip", "venv")


def probe(name):
    # A fresh interpreter detects broken binary imports without initializing the
    # coordinator's runtime or activating an already installed FLA wheel.
    env = {**os.environ, "FLA_NPU_DISABLE_PTH": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    code = ("import importlib,json; "
            f"m=importlib.import_module({name!r}); "
            "print(json.dumps({'version':str(getattr(m,'__version__','stdlib/available'))}))")
    try:
        result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                                capture_output=True, timeout=90)
        if result.returncode:
            return {"available": False, "error": (result.stderr or result.stdout)[-2000:]}
        return {"available": True, **json.loads(result.stdout.splitlines()[-1])}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def optional_atk():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ("ATK", "kombu"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def check():
    required = {name: probe(name) for name in REQUIRED}
    optional = optional_atk()
    warnings = []
    if optional["ATK"] is None:
        warnings.append("ATK is not installed; optional ATK validation is unavailable. "
                        "Manual validation compares NPU outputs with the CPU reference directly.")
    elif optional["kombu"] != "5.6.2":
        warnings.append("Optional ATK validation needs the verified kombu==5.6.2; "
                        "this does not affect direct CPU-reference validation.")
    result = {"schema": "experimental-dependencies-v1", "python": sys.executable,
              "required": required, "optional": optional, "warnings": warnings,
              "accuracy_backend": "cpu-reference",
              "status": "success" if all(item["available"] for item in required.values()) else "failure"}
    for warning in warnings:
        print(f"[experimental][WARNING] {warning}", flush=True)
    for name, item in required.items():
        if not item["available"]:
            print(f"[experimental][ERROR] Required dependency {name}: {item['error']}", file=sys.stderr)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
