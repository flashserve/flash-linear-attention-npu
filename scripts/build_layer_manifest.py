"""Describe the Python/Stable/ACLNN closure shipped by either wheel builder."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "torch_custom/fla_npu"


def load_abi():
    spec = importlib.util.spec_from_file_location("_layer_abi_build", SOURCE / "fla_npu/_op_abi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literals(path):
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def requirements():
    abi = load_abi()
    headers = {k: list(v) for k, v in abi.KNOWN_CONTRACTS.items()}
    for directory in sorted({p.parent for p in (ROOT / "fla/ops/ascendc").rglob("aclnn_*.h")}):
        for symbol, signature in abi.public_prototypes(directory).items():
            if symbol in headers and headers[symbol] != list(signature):
                raise RuntimeError(f"Conflicting ACLNN headers for {symbol}")
            headers[symbol] = list(signature)
    spec = importlib.util.spec_from_file_location("_layer_calls", SOURCE / "tools/op_abi_validate.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    for symbol, (_, kinds) in validator.adapter_calls(SOURCE / "csrc/src").items():
        if "?" not in kinds:
            headers.setdefault(symbol + "GetWorkspaceSize", kinds + list(abi._WORKSPACE_TAIL))
            headers.setdefault(abi.EXECUTE_ALIASES.get(symbol, symbol), list(abi._EXECUTE_SIGNATURE))
    tree = ast.parse((SOURCE / "fla_npu/ops/ascendc/_stable.py").read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(name, seen):
        if name in seen or name not in functions:
            return set()
        seen = seen | {name}
        result = set()
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in ("_op", "_bound_op") and node.args and isinstance(node.args[0], ast.Constant):
                    result.add(node.args[0].value)
                result.update(calls(node.func.id, seen))
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "_OP_CACHE" and node.args and isinstance(node.args[0], ast.Constant)):
                result.add(node.args[0].value)
        return result

    def cpp_symbols(path, seen):
        if path in seen or not path.is_file():
            return set()
        text = abi.strip_comments(path.read_text(encoding="utf-8"))
        result = set(re.findall(r"\baclnn[A-Z][A-Za-z0-9]+", text))
        for include in re.findall(r'#include "(stable_\w+\.cpp)"', text):
            result.update(cpp_symbols(path.parent / include, seen | {path}))
        return result

    result = {}
    for op in literals(SOURCE / "fla_npu/ops/ascendc/__init__.py")["_ASCENDC_OPS"]:
        dependencies = calls(op, set()) | {op}
        symbols = set()
        for dependency in dependencies:
            symbols.update(cpp_symbols(SOURCE / "csrc/src" / ("stable_" + dependency.removeprefix("npu_") + ".cpp"), set()))
        contracts = {}
        for symbol in symbols:
            symbol = symbol.removesuffix("GetWorkspaceSize")
            # Built-in CANN operations are resolved from CANN, not a custom vendor.
            if symbol in ("aclnnFastGelu", "aclnnFastGeluGrad", "aclnnFastGeluBackward"):
                continue
            for entry in (abi.EXECUTE_ALIASES.get(symbol, symbol), symbol + "GetWorkspaceSize"):
                if entry not in headers:
                    raise RuntimeError(f"{op}: no source ACLNN contract for {entry}")
                contracts[entry] = headers[entry]
        result[op] = {"requires": contracts, "dispatch": sorted(dependencies)}
    return result


def write_layer_manifest(package: Path):
    from fla_npu_artifacts import get_tier

    ops = requirements()
    raw = os.environ.get("FLA_NPU_OVERLAY_OPS", "").strip()
    selected = sorted({n if n.startswith("npu_") else "npu_" + n
                       for value in raw.split(",") if (n := value.strip())}) if raw else sorted(ops)
    unknown = set(selected) - ops.keys()
    if unknown:
        raise RuntimeError(f"Unknown FLA_NPU_OVERLAY_OPS: {sorted(unknown)}")
    stamp = literals(package / "ops/ascendc/_stable_hash.py")["SOURCE_HASH"]
    data = {"protocol": 1, "tier": get_tier(), "source_hash": stamp,
            "exports": selected, "operators": ops}
    (package / "layer_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
