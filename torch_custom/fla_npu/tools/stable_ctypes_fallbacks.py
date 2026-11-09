#!/usr/bin/env python3
"""Reject removed execution backends in the installed Python package.

The historical command name is kept for CI. Standard-library ctypes remains
valid for dynamic-library loading/capability probes; ACLNN descriptor execution
and the torch C++ extension are forbidden product backends. Test-only raw C ABI
error probes are deliberately outside this package inventory.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = SETUP_DIR / "fla_npu"
FORBIDDEN_MODULES = {"_aclnn_ctypes", "_runtime"}
FORBIDDEN_CALLS = {"load_legacy_torch_ops", "_setup_legacy_extension", "call_aclnn"}


def violations(text: str) -> list[tuple[int, str]]:
    tree = ast.parse(text)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FORBIDDEN_CALLS:
            result.append((node.lineno, "removed backend definition: " + node.name))
        if isinstance(node, ast.ImportFrom):
            modules = (node.module or "").split(".") + [a.name for a in node.names]
            if FORBIDDEN_MODULES.intersection(modules):
                result.append((node.lineno, "removed backend import"))
        if isinstance(node, ast.Import):
            if any(FORBIDDEN_MODULES.intersection(a.name.split(".")) for a in node.names):
                result.append((node.lineno, "removed backend import"))
        if isinstance(node, ast.Call):
            function = ast.unparse(node.func)
            if function.split(".")[-1] in FORBIDDEN_CALLS or function.startswith("torch.ops.npu.npu_"):
                result.append((node.lineno, "removed backend call: " + function))
            if function.split(".")[-1] in {"import_module", "__import__"} and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and FORBIDDEN_MODULES.intersection(value.value.split(".")):
                    result.append((node.lineno, "removed backend dynamic import"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "custom_aclnn_extension_lib" in node.value:
            result.append((node.lineno, "removed extension library reference"))
    return result


def delegating_ops(text: str) -> list[str]:
    tree = ast.parse(text)
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and violations(ast.unparse(node))]


def evaluate() -> dict:
    problems = []
    paths = sorted(PACKAGE_DIR.rglob("*.py"))
    if not paths:
        problems.append("package inventory is empty")
    for path in paths:
        if path.stem in FORBIDDEN_MODULES:
            problems.append(f"{path.relative_to(PACKAGE_DIR)}: removed backend file remains")
        for line, reason in violations(path.read_text(encoding="utf-8")):
            problems.append(f"{path.relative_to(PACKAGE_DIR)}:{line}: {reason}")
    return {"checked_modules": len(paths), "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["problems"]:
        print("\n".join(report["problems"]))
    else:
        print(f"OK: {report['checked_modules']} modules use Stable-only execution")
    return bool(report["problems"])


if __name__ == "__main__":
    raise SystemExit(main())
