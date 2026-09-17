#!/usr/bin/env python3
"""Validate op ABI spec JSON against the installed aclnn_*.h prototype.

Usage:
  python op_abi_validate.py --header <aclnn_xxx.h> --spec <op_spec.json>

The header is the single source of truth. This tool compares only the
*GetWorkspaceSize* argument kinds/order (the trailing workspaceSize/executor
outputs are appended by the generic thin executor and excluded).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_KIND_BY_TYPE = [
    (r"aclTensor\s*\*", "tensor"),
    (r"aclIntArray\s*\*", "int_array"),
    (r"const\s+char\s*\*", "char_ptr"),
    (r"int64_t", "int64"),
    (r"float", "float"),
    (r"double", "double"),
    (r"bool", "bool"),
]


def extract_get_workspace_prototype(header_text: str, aclnn_name: str) -> str:
    func = aclnn_name + "GetWorkspaceSize"
    m = re.search(
        re.escape(func) + r"\s*\((.*?)\)\s*;", header_text, re.S | re.M)
    if not m:
        raise RuntimeError(f"{func} prototype not found in header")
    return m.group(1)


def split_params(params: str) -> list[str]:
    result: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def header_kinds(params: list[str]) -> list[str]:
    kinds: list[str] = []
    for raw in params:
        cleaned = re.sub(r"\s+", " ", raw)
        matched = None
        for pattern, kind in _KIND_BY_TYPE:
            if re.search(pattern, cleaned):
                matched = kind
                break
        if matched is None:
            raise RuntimeError(f"cannot map header parameter type: {raw!r}")
        kinds.append(matched)
    return kinds


def spec_kinds(spec: dict) -> list[str]:
    # ``cpp_only`` args are launcher-only parameters: they drive conditional
    # output allocation in C++ and must not reach the aclnn prototype. Skip
    # them so this check mirrors op_spec_codegen's ABI construction exactly.
    return [arg["kind"] for arg in spec["args"] if not arg.get("cpp_only")]


def normalize(kinds: list[str]) -> list[str]:
    aliases = {
        "optional_tensor": "tensor",
        "cpu_int_array": "int_array",
        "out_tensor": "tensor",
    }
    return [aliases.get(kind, kind) for kind in kinds]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    header_text = args.header.read_text(encoding="utf-8", errors="ignore")
    proto = extract_get_workspace_prototype(
        header_text, spec["aclnn_name"])
    header = header_kinds(split_params(proto)[:-2])
    expected = normalize(spec_kinds(spec))
    header = normalize(header)

    if header != expected:
        print("MISMATCH")
        print("header:", header)
        print("spec:  ", expected)
        return 1
    print(f"OK: {spec['aclnn_name']} ABI matches spec "
          f"({len(expected)} params)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
