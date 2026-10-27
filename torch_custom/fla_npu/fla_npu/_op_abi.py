"""Pure helpers for comparing public aclnn contracts between OPP vendors."""

from __future__ import annotations

import re
from pathlib import Path


TENSOR = "tensor"
INT_ARRAY = "int_array"
ACL_SCALAR = "acl_scalar"

_HEADER_KINDS = (
    (re.compile(r"\bconst\s+aclTensor\s*\*|\baclTensor\s*\*"), TENSOR),
    (re.compile(r"\bconst\s+aclIntArray\s*\*|\baclIntArray\s*\*"), INT_ARRAY),
    (re.compile(r"\bconst\s+aclScalar\s*\*|\baclScalar\s*\*"), ACL_SCALAR),
    (re.compile(r"\b(?:const\s+)?char\s*\*"), "char_ptr"),
    (re.compile(r"\bdouble\b"), "double"),
    (re.compile(r"\bfloat\b"), "float"),
    (re.compile(r"\buint64_t\b"), "uint64"),
    (re.compile(r"\bint64_t\b"), "int64"),
    (re.compile(r"\buint32_t\b"), "uint32"),
    (re.compile(r"\bint32_t\b"), "int32"),
    (re.compile(r"\bbool\b"), "bool"),
)
_TRAILING = ("workspaceSize", "executor")


def strip_comments(text: str) -> str:
    """Remove C/C++ comments without treating comment punctuation as syntax."""

    out: list[str] = []
    index, size = 0, len(text)
    quote = ""
    while index < size:
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < size:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and text.startswith("//", index):
            while index < size and text[index] != "\n":
                index += 1
            continue
        if char == "/" and text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = size if end < 0 else end + 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def split_params(text: str) -> list[str]:
    """Split a C parameter list on top-level commas."""

    text = strip_comments(text)
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def parameter_kind(param: str) -> str:
    """Return the ABI-relevant kind of one public aclnn parameter."""

    for pattern, kind in _HEADER_KINDS:
        if pattern.search(param):
            return kind
    # Preserve an unknown spelling instead of silently accepting it. Removing
    # the parameter name avoids treating a source-only rename as an ABI change.
    normalized = re.sub(r"\s+", " ", param).strip()
    normalized = re.sub(r"\b[A-Za-z_]\w*\s*$", "", normalized).strip()
    return f"raw:{normalized}"


def parse_aclnn_signatures(include_dir: Path) -> dict[str, tuple[str, ...]]:
    """Read ``symbol -> parameter kinds`` from public GetWorkspaceSize headers."""

    signatures: dict[str, tuple[str, ...]] = {}
    if not include_dir.is_dir():
        return signatures
    for header in sorted(include_dir.glob("aclnn_*.h")):
        text = header.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(
            r"aclnnStatus\s+(\w+)GetWorkspaceSize\s*\((.*?)\)\s*;",
            text,
            re.S,
        ):
            symbol, params = match.group(1), split_params(match.group(2))
            while params and any(token in params[-1] for token in _TRAILING):
                params.pop()
            signatures[symbol] = tuple(parameter_kind(param) for param in params)
    return signatures


def vendor_aclnn_signatures(vendor_dir: Path) -> dict[str, tuple[str, ...]]:
    return parse_aclnn_signatures(vendor_dir / "op_api" / "include" / "aclnnop")


def compare_overlay_to_base(overlay_dir: Path, base_dir: Path) -> list[str]:
    """Return public aclnn contracts that an existing wheel cannot safely call."""

    overlay = vendor_aclnn_signatures(overlay_dir)
    base = vendor_aclnn_signatures(base_dir)
    problems: list[str] = []
    for symbol, signature in sorted(overlay.items()):
        base_signature = base.get(symbol)
        if base_signature is None:
            problems.append(
                f"{symbol}: not declared by the base wheel (overlay={list(signature)})"
            )
        elif base_signature != signature:
            problems.append(
                f"{symbol}: overlay={list(signature)} base={list(base_signature)}"
            )
    return problems
