#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate the operator filter used by ``FLA_NPU_OPS`` / ``build.sh --ops``.

``FLA_NPU_OPS=...`` and ``bash build.sh --ops=...`` are filters only: CMake
silently drops every operator that is not listed, so a mistyped or unsupported
name used to surface far away from its cause, as

    OpFileNotExistsError: File aic-*-ops-info.ini does not exist in directory
    make: *** [prepare_build] Error
    Error: ops prepare build failed.

This module derives the supported operator list from the repository itself
(mirroring ``cmake/func.cmake::op_add_subdirectory()``) and rejects unknown
names before CMake configuration starts, so the error names the offending
operators, the parameter they came from, and the list that is actually
supported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Discovery root: every custom operator of this repository lives under
# fla/ops/ascendc/<group>/<sub_group>/<op_name>/, so the directory names below
# this root are exactly the names accepted by --ops / FLA_NPU_OPS.
OP_SEARCH_ROOTS = ("fla/ops/ascendc",)
# CMake reads these as "compile every operator" instead of an operator name.
BUILD_ALL_SENTINELS = frozenset({"all", "ALL"})
LIST_COMMAND = "bash build.sh --list-ops"
SCRIPT_LIST_COMMAND = "python3 scripts/check_build_ops.py --list"


def discover_supported_ops(repo_root=REPO_ROOT) -> List[str]:
    """Return every operator name the CMake build can be filtered by."""
    repo_root = Path(repo_root)

    ops = set()
    for search_root in OP_SEARCH_ROOTS:
        base = repo_root / search_root
        if not base.is_dir():
            continue
        for cmake_file in base.rglob("CMakeLists.txt"):
            parts = cmake_file.parts
            # cmake skips everything below a tests/ directory.
            if "tests" in parts:
                continue
            if "op_host" in parts:
                op_index = len(parts) - 1 - parts[::-1].index("op_host")
                op_dir = Path(*parts[:op_index])
            else:
                op_dir = cmake_file.parent
            if op_dir.name:
                ops.add(op_dir.name)
    return sorted(ops)


def parse_ops_filter(raw: Optional[str]) -> List[str]:
    """Split a ``--ops``/``FLA_NPU_OPS`` value into individual operator names."""
    if raw is None:
        return []
    return [name.strip() for name in str(raw).split(",") if name.strip()]


def is_build_all(raw: Optional[str]) -> bool:
    """True when the filter means "compile everything" (unset, empty or all)."""
    names = parse_ops_filter(raw)
    if not names:
        return True
    return len(names) == 1 and names[0] in BUILD_ALL_SENTINELS


def unsupported_ops(
    raw: Optional[str], repo_root=REPO_ROOT, supported: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the requested operator names that this repository cannot build."""
    if is_build_all(raw):
        return []
    known = set(supported) if supported is not None else set(discover_supported_ops(repo_root))
    unknown: List[str] = []
    for name in parse_ops_filter(raw):
        if name in known or name in unknown:
            continue
        unknown.append(name)
    return unknown


def format_validation_error(
    raw: Optional[str],
    source: str,
    unknown: Sequence[str],
    supported: Optional[Sequence[str]] = None,
    repo_root=REPO_ROOT,
    origin: Optional[str] = None,
) -> str:
    """Build the message printed when the operator filter cannot be honoured."""
    if supported is None:
        supported = discover_supported_ops(repo_root)
    lines = [
        "[ERROR] Unsupported operator(s) in {}: {}".format(source, ", ".join(unknown)),
    ]
    if origin:
        lines.append("[ERROR] Parameter source: {} (value: {})".format(origin, raw))
    else:
        lines.append("[ERROR] Parameter value: {}".format(raw))
    lines.append(
        "[ERROR] Supported operators ({}): {}".format(len(supported), ", ".join(supported))
    )
    lines.append(
        "[ERROR] Unknown names would be dropped silently by the build, so it is aborted "
        "before CMake configuration."
    )
    lines.append(
        "[ERROR] Print the supported operator list with: {} (or {})".format(
            LIST_COMMAND, SCRIPT_LIST_COMMAND
        )
    )
    return "\n".join(lines)


def validate_ops_filter(
    raw: Optional[str],
    source: str = "--ops",
    origin: Optional[str] = None,
    repo_root=REPO_ROOT,
    supported: Optional[Sequence[str]] = None,
    stream=None,
) -> bool:
    """Print a diagnostic and return False when ``raw`` contains unknown names."""
    known = list(supported) if supported is not None else discover_supported_ops(repo_root)
    unknown = unsupported_ops(raw, supported=known)
    if not unknown:
        return True
    print(
        format_validation_error(
            raw, source=source, unknown=unknown, supported=known, origin=origin
        ),
        file=stream or sys.stderr,
    )
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate FLA_NPU_OPS/--ops operator names for this repository."
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="repository root used to discover supported operators (default: auto-detected)",
    )
    parser.add_argument("--ops", default=None, help="operator filter value to validate")
    parser.add_argument(
        "--source",
        default="--ops",
        help="name of the parameter the value came from, used in the error message",
    )
    parser.add_argument(
        "--origin",
        default=None,
        help="human readable description of where the parameter was set",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="print the supported operators, one per line, and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="with --list, print a JSON object instead of plain names",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    supported = discover_supported_ops(args.repo_root)

    if args.list:
        if args.json:
            print(json.dumps({"count": len(supported), "operators": supported}, indent=2))
        else:
            print("\n".join(supported))
        return 0

    if args.ops is None:
        parser.error("--ops is required unless --list is given")

    if not validate_ops_filter(
        args.ops,
        source=args.source,
        origin=args.origin,
        supported=supported,
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())