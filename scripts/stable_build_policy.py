"""Shared invariants for every Stable-only wheel build entry point."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re

import npu_compat
from fla_npu_artifacts import get_tier
from npu_compat import MIN_TORCH

RUNTIME_REQUIREMENTS = [f"torch>={MIN_TORCH}", f"torch_npu>={MIN_TORCH}"]
LAUNCHER_FILES = ("libfla_npu_stable.so", "ops/ascendc/_stable_hash.py")


def validate_build_options() -> None:
    """Reject removed modes before running a compiler or changing outputs."""
    stable = os.getenv("FLA_NPU_BUILD_STABLE_ABI", "TRUE").strip().upper()
    if stable not in {"1", "TRUE", "YES", "ON"}:
        raise RuntimeError("FLA_NPU_BUILD_STABLE_ABI cannot disable the mandatory Stable-ABI launcher")
    if os.getenv("FLA_NPU_BUILD_LEGACY_EXTENSION", "FALSE").strip().upper() not in {
        "", "0", "FALSE", "NO", "OFF"
    }:
        raise RuntimeError("FLA_NPU_BUILD_LEGACY_EXTENSION is no longer supported; use Stable ABI")
    if "FLA_NPU_SKIP_TORCH_GEN" in os.environ:
        raise RuntimeError("FLA_NPU_SKIP_TORCH_GEN was removed together with legacy code generation")


def reject_legacy_artifacts(package_dir: Path) -> None:
    stale = sorted(package_dir.rglob("custom_aclnn_extension_lib*"))
    if stale:
        raise RuntimeError("Remove stale legacy extension artifacts before building: "
                           + ", ".join(str(path) for path in stale))


def validate_launcher_payload(package_dir: Path) -> None:
    reject_legacy_artifacts(package_dir)
    for name in LAUNCHER_FILES:
        path = package_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required Stable-ABI payload is missing or empty: {path}")
    validate_build_stamp((package_dir / "ops/ascendc/_stable_hash.py").read_bytes())


def validate_build_stamp(source: bytes) -> str:
    """Read the generated source fingerprint without executing package code."""
    try:
        tree = ast.parse(source.decode("utf-8"))
        values = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SOURCE_HASH"
                    for target in node.targets):
                values.append(ast.literal_eval(node.value))
        if len(values) != 1 or not isinstance(values[0], str) or not re.fullmatch(r"[0-9a-f]{32}", values[0]):
            raise ValueError("expected one literal 32-character source fingerprint")
        return values[0]
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid Stable-ABI build stamp (_stable_hash.py): {exc}") from exc


def write_runtime_meta(package_dir: Path) -> None:
    tier = get_tier()
    (package_dir / "_build_meta.py").write_text(
        '"""Generated at wheel build time. Do not edit."""\n'
        f"TIER = {tier!r}\n", encoding="utf-8")
    (package_dir / "_compat.py").write_text(
        '"""Generated from scripts/npu_compat.py. Do not edit."""\n'
        f"MIN_CANN = {npu_compat.MIN_CANN_BY_TIER[tier]!r}\n"
        f"MIN_TORCH = {npu_compat.MIN_TORCH!r}\n"
        f"TORCH_NPU_GDN_FIX_MINIMUMS = {npu_compat.TORCH_NPU_GDN_FIX_MINIMUMS!r}\n",
        encoding="utf-8")
