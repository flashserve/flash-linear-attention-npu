#!/usr/bin/env python3
"""Check every aclnn call site against the installed ``aclnn_*.h`` prototypes.

The header of the OPP actually being shipped is the only source of truth for an
entry point's parameter list.  Each Stable-ABI adapter
(``FLA_STABLE_EXEC("aclnnX", ...)`` in ``csrc/src/stable_*.cpp``) has to agree
with it.

Disagreement is not a theoretical concern: when the OPP grew a
``bool stateVFirst`` parameter in front of the outputs of
``aclnnChunkGatedDeltaRuleBwdDhu``, a call site that kept the old list came back
as a bare ``161001`` on 910B.  Comparing against the header turns that into an
offline check.

Header parsing is shared with the installation/runtime contract validator.
Descriptor kinds and pointer levels are distinct; unsupported declarations fail.

Usage::

    # both include directories: the vendor OPP's, and CANN's for the built-in
    # operators.  FastGelu ships with CANN, so passing only the OPP reports it
    # as a missing header rather than as a mismatch.
    python tools/op_abi_validate.py \
        --opp-include <opp>/op_api/include/aclnnop <cann>/include/aclnnop
    python tools/op_abi_validate.py --opp-include ... --json report.json

Exits non-zero when a call site disagrees with the header.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SRC_DIR = PACKAGE / "csrc" / "src"

# Load the pure contract helper without importing fla_npu (which loads CANN).
_spec = importlib.util.spec_from_file_location("fla_npu_op_abi", PACKAGE / "fla_npu" / "_op_abi.py")
_abi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_abi)
WILDCARD = "?"
TENSOR, INT_ARRAY, ACL_SCALAR = _abi.TENSOR, _abi.INT_ARRAY, _abi.ACL_SCALAR
strip_comments, split_params = _abi.strip_comments, _abi.split_params
header_kind = _abi.parameter_kind
KNOWN_HEADER_GAPS = {}  # Legacy gaps now have explicit contracts in _op_abi.


def parse_headers(include_dir: Path, wanted: set[str]) -> dict[str, list[str]]:
    signatures = _abi.parse_aclnn_signatures(include_dir, wanted)
    if (include_dir / "aclnn_solve_tri.h").is_file() and "aclnnSolveTri" in wanted:
        signatures.setdefault("aclnnSolveTri", _abi.KNOWN_CONTRACTS["aclnnSolveTriGetWorkspaceSize"][:-2])
    return {s: list(kinds) for s, kinds in signatures.items()}


_EXEC_CALL = re.compile(r"FLA_STABLE_EXEC\(\s*\"(\w+)\"\s*,(.*?)\)\s*;", re.S)

_TENSOR_HOLDERS = {
    "tensor",
    "optional_tensor",
    "out_tensor",
    "nd_tensor",
    "nd_optional_tensor",
    "nd_out_tensor",
    "nd_logical_out_tensor",
    "logical_tensor",
    "logical_optional_tensor",
    "logical_out_tensor",
}

# The two oldest adapters predate the macro: they call the entry point
# themselves (`get_ws(...)` on a `Runtime::symbol(...)` pointer) and build their
# argument list from local `AclTensorView`s.  Their call sites are checked the
# same way, or aclnn signature drift would go unnoticed exactly where the code
# is oldest.
_HAND_WRITTEN_CALL = re.compile(
    r"get_ws\s*\((.*?)\)\s*;\s*\n", re.S)
_SYMBOL_RE = re.compile(r'rt\.symbol\("(\w+)GetWorkspaceSize"\)')


def _enclosing_params(text: str, position: int) -> dict[str, str]:
    """{parameter: declared type} of the definition around *position*.

    Found by walking back to the `{` that opens the enclosing block and parsing
    the parameter list of the declaration in front of it.  A regex over all
    function definitions is not enough: it also matches calls (`size_of(`) and
    control statements, and then the nearest preceding *match* is no longer the
    enclosing function.
    """

    index = position
    depth = 0
    while index > 0:
        index -= 1
        if text[index] == "}":
            depth += 1
        elif text[index] == "{":
            if depth == 0:
                break
            depth -= 1
    if index == 0:
        return {}

    start = index - 1
    while start > 0 and text[start] not in ";{}":
        start -= 1
    declaration = text[start + 1:index]
    close = declaration.rfind(")")
    if close < 0:
        return {}
    depth = 0
    open_paren = close
    while open_paren > 0:
        open_paren -= 1
        if declaration[open_paren] == ")":
            depth += 1
        elif declaration[open_paren] == "(":
            if depth == 0:
                break
            depth -= 1
    if declaration[open_paren] != "(":
        return {}

    params: dict[str, str] = {}
    for param in split_params(declaration[open_paren + 1:close]):
        stripped = re.sub(r"=[^=]*$", "", param).strip()
        if not stripped:
            continue
        name_match = re.search(r"(\w+)\s*$", stripped)
        if not name_match:
            continue
        params[name_match.group(1)] = stripped[:name_match.start()].strip()
    return params


def _scalar_kind(expression: str, params: dict[str, str]) -> str:
    expression = expression.strip()
    declared = params.get(expression, "")
    for token, kind in (("bool", "bool"), ("double", "double"),
                        ("float", "float"), ("int64_t", "int64"),
                        ("size_t", "int64"), ("int32_t", "int32")):
        if token in declared:
            return kind
    cast = re.match(r"(?:static_cast\s*<\s*|[\(])(\w+_t|bool|double|float)",
                    expression)
    if cast:
        return {"int64_t": "int64", "int32_t": "int32", "size_t": "int64",
                "bool": "bool", "double": "double", "float": "float"}.get(
                    cast.group(1), WILDCARD)
    if re.fullmatch(r"true|false", expression):
        return "bool"
    if re.fullmatch(r"[-+]?\d+", expression):
        return "int64"
    if re.fullmatch(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?[fF]?", expression):
        return "double"
    return WILDCARD


def adapter_calls(src_dir: Path) -> dict[str, tuple[Path, list[str]]]:
    """symbol -> (file, argument kinds) for every FLA_STABLE_EXEC call."""

    calls: dict[str, tuple[Path, list[str]]] = {}
    for source in sorted(src_dir.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in _EXEC_CALL.finditer(text):
            symbol, arguments = match.group(1), match.group(2)
            # The api string is outside the capture, so the list starts at the
            # workspace meta and the stream.
            args = split_params(arguments)[2:]
            # The enclosing definition is how `scalar(name)` learns the
            # parameter's declared type.
            owner = _enclosing_params(text, match.start())
            kinds: list[str] = []
            for argument in args:
                call = re.match(r"(\w+)\s*\((.*)\)\s*$", argument, re.S)
                if not call:
                    kinds.append(WILDCARD)
                    continue
                callee, inner = call.group(1), call.group(2)
                if callee == "scalar":
                    kinds.append(_scalar_kind(inner, owner))
                elif callee in {"cstr", "CStrArg"}:
                    kinds.append("char_ptr")
                elif callee == "int_array":
                    kinds.append(INT_ARRAY)
                elif callee in _TENSOR_HOLDERS:
                    kinds.append(TENSOR)
                else:
                    kinds.append(WILDCARD)
            calls[symbol] = (source, kinds)
    return calls


def hand_written_calls(src_dir: Path) -> dict[str, tuple[Path, list[str]]]:
    """symbol -> (file, argument kinds) for the pre-macro adapters.

    Their list is spelled `<view>.get()` for descriptors, a parameter name for
    scalars, and the trailing `&workspace_size, &executor` pair.
    """

    calls: dict[str, tuple[Path, list[str]]] = {}
    for source in sorted(src_dir.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8")
        if "FLA_STABLE_EXEC" in text:
            continue
        symbol = _SYMBOL_RE.search(text)
        call = _HAND_WRITTEN_CALL.search(text)
        if symbol is None or call is None:
            continue
        owner = _enclosing_params(text, call.start())
        args = split_params(call.group(1))
        while args and re.match(r"&(workspace_size|executor)$", args[-1]):
            args.pop()
        kinds: list[str] = []
        for argument in args:
            argument = argument.strip()
            if argument.endswith(".get()"):
                # A descriptor: tensor and int[] are both pointers here.
                kinds.append(WILDCARD)
            else:
                kinds.append(_scalar_kind(argument, owner))
        calls[symbol.group(1)] = (source, kinds)
    return calls


def compare(expected: list[str], actual: list[str]) -> list[str]:
    problems: list[str] = []
    if len(expected) != len(actual):
        problems.append(
            f"parameter count {len(actual)} != header {len(expected)}")
    for index, (want, got) in enumerate(zip(expected, actual)):
        if WILDCARD in (want, got) or want == got:
            continue
        problems.append(f"param {index}: header {want} != call site {got}")
    return problems


def filter_adapters(
    adapters: dict[str, tuple[Path, list[str]]], raw_ops: str
) -> dict[str, tuple[Path, list[str]]]:
    """Limit a scoped build gate to adapters owned by ``--ops`` sources."""

    selected = {
        name.removeprefix("npu_")
        for entry in raw_ops.split(",")
        if (name := entry.strip())
    }
    if not selected:
        return adapters
    matches = {}
    for name in sorted(selected):
        # Shared implementations need not be named stable_<operator>.cpp.
        normalized = name.replace("_", "").lower()
        owned = {
            symbol: value for symbol, value in adapters.items()
            if symbol.removeprefix("aclnn").replace("_", "").lower() == normalized
            or value[0].stem.removeprefix("stable_") == name
        }
        if not owned:
            raise RuntimeError(f"No ABI call sites found for selected operator {name!r}")
        matches.update(owned)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opp-include", required=True, nargs="+",
                        help="one or more op_api/include/aclnnop directories: "
                             "the vendor OPP's, and CANN's for the built-in "
                             "operators (FastGelu)")
    parser.add_argument("--src-dir", default=str(SRC_DIR))
    parser.add_argument(
        "--ops",
        default="",
        help="comma-separated scoped build operator names; only their stable "
        "adapter call sites are checked",
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    adapters = adapter_calls(Path(args.src_dir))
    adapters.update(hand_written_calls(Path(args.src_dir)))
    adapters = filter_adapters(adapters, args.ops)
    wanted = set(adapters)

    headers: dict[str, list[str]] = {}
    for include_dir in args.opp_include:
        for symbol, kinds in parse_headers(Path(include_dir), wanted).items():
            headers.setdefault(symbol, kinds)

    report: dict[str, dict] = {}
    failures = 0
    for symbol in sorted(adapters):
        source, kinds = adapters[symbol]
        source_name = source.name
        if symbol not in headers:
            report[f"adapter:{symbol}"] = {
                "source": source_name,
                "status": ("known-gap" if symbol in KNOWN_HEADER_GAPS
                           else "no-header"),
                "reason": KNOWN_HEADER_GAPS.get(symbol, ""),
                "call_site": kinds}
            if symbol not in KNOWN_HEADER_GAPS:
                failures += 1
            continue
        problems = compare(headers[symbol], kinds)
        report[f"adapter:{symbol}"] = {
            "source": source_name,
            "status": "ok" if not problems else "mismatch",
            "header": headers[symbol], "call_site": kinds,
            "problems": problems}
        if problems:
            failures += 1
            print(f"adapter {symbol} ({source_name}): {'; '.join(problems)}")
            print(f"    header    : {headers[symbol]}")
            print(f"    call site : {kinds}")

    for symbol in sorted(headers):
        if symbol not in adapters:
            report[f"unused:{symbol}"] = {"status": "not-called"}

    gaps = [key for key, value in report.items()
            if value.get("status") == "known-gap"]
    if gaps:
        print(f"{len(gaps)} call site(s) known to have no public header:")
        for key in gaps:
            print(f"    {key}: {report[key]['reason']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"report -> {args.json}")

    checked = sum(1 for value in report.values()
                  if value.get("status") in {"ok", "mismatch"})
    print(f"{checked} call site(s) checked, {failures} mismatch(es) "
          "and/or missing header(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
