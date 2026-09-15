#!/usr/bin/env python3
"""Check every aclnn call site against the installed ``aclnn_*.h`` prototypes.

The header of the OPP actually being shipped is the only source of truth for an
entry point's parameter list.  Two call sites have to agree with it:

* the ctypes reference (``_GET_WORKSPACE_ARGTYPES`` in ``_aclnn_ctypes.py``),
* each Stable-ABI adapter (``FLA_STABLE_EXEC("aclnnX", ...)`` in
  ``csrc/src/stable_*.cpp``).

Disagreement is not a theoretical concern: when the OPP grew a
``bool stateVFirst`` parameter in front of the outputs of
``aclnnChunkGatedDeltaRuleBwdDhu``, the ctypes reference kept passing the old
list and *segfaulted* on 910B, while the adapter came back as a bare
``161001``.  Both are the same defect seen from two sides, and neither is
visible in a same-OPP parity test -- there the reference is wrong in exactly
the same way.  Comparing against the header turns it into an offline check.

Tensor and int[] parameters are both pointers at this level, so those kinds
are compared as "wildcard": a wrong *order*, a missing scalar parameter or a
parameter count change still shows up, which is what actually happens in
practice.

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
import ast
import ctypes
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CTYPES_MODULE = PACKAGE / "fla_npu" / "ops" / "ascendc" / "_aclnn_ctypes.py"
SRC_DIR = PACKAGE / "csrc" / "src"

# Kind of a parameter as far as a mismatch is concerned.  Tensor and int[]
# descriptors are both plain pointers here, so both sides report WILDCARD and
# only the scalars (and the count) carry information.
WILDCARD = "?"
_POINTER = "_pointer"

_HEADER_KINDS = (
    (re.compile(r"\bconst\s+aclTensor\s*\*|\baclTensor\s*\*"), WILDCARD),
    (re.compile(r"\bconst\s+aclIntArray\s*\*|\baclIntArray\s*\*"), WILDCARD),
    (re.compile(r"\bconst\s+aclScalar\s*\*|\baclScalar\s*\*"), WILDCARD),
    (re.compile(r"\b(?:const\s+)?char\s*\*"), "char_ptr"),
    (re.compile(r"\bdouble\b"), "double"),
    (re.compile(r"\bfloat\b"), "float"),
    (re.compile(r"\bint64_t\b"), "int64"),
    (re.compile(r"\bint32_t\b"), "int32"),
    (re.compile(r"\bbool\b"), "bool"),
)

_CTYPES_KINDS = {
    "c_void_p": WILDCARD,
    "c_char_p": "char_ptr",
    "c_double": "double",
    "c_float": "float",
    "c_int64": "int64",
    "c_int32": "int32",
    "c_bool": "bool",
    # A few entries spell the width by hand; on every supported host these are
    # the 64- and 32-bit integer types.
    "c_long": "int64",
    "c_ulong": "int64",
    "c_longlong": "int64",
    "c_uint64": "int64",
    "c_ulonglong": "int64",
    "c_int": "int32",
    "c_uint": "int32",
    "c_short": "int32",
}

# The two out-parameters every GetWorkspaceSize prototype ends with.  The
# generic executor owns them, so they are not part of the comparison.
_TRAILING = ("workspaceSize", "executor")

# Entry points this tree calls that the OPP publishes no public header for.
# Each one is a reason to check by hand rather than a silent pass, so the list
# is a constant instead of a command-line escape hatch.
KNOWN_HEADER_GAPS = {
    "aclnnSolveTri": "the OPP header declares only the l0op::SolveTri "
                     "prototype, not aclnnSolveTriGetWorkspaceSize",
}


def strip_comments(text: str) -> str:
    """Drop C comments, leaving string literals alone.

    An adapter may put a line comment between two arguments (`// the fused
    kernel takes no cu_seqlens/chunk_indices`).  Those comments routinely carry
    a comma, and counting one as a separator adds a phantom parameter -- which
    is exactly how this gate started reporting
    `aclnnChunkGatedDeltaRuleBwdDhu` as one argument too long.
    """

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
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def header_kind(param: str) -> str:
    for pattern, kind in _HEADER_KINDS:
        if pattern.search(param):
            return kind
    raise RuntimeError(f"unrecognised header parameter: {param!r}")


def parse_headers(include_dir: Path, wanted: set[str]) -> dict[str, list[str]]:
    """symbol -> parameter kinds, for the *wanted* entry points declared there.

    Only the wanted prototypes are parsed.  A CANN include directory holds
    thousands of operators whose parameters use types this comparison has no
    opinion about, and reading them all would turn an unrelated header into a
    gate failure.
    """

    found: dict[str, list[str]] = {}
    for header in sorted(include_dir.glob("aclnn_*.h")):
        text = header.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(
                r"aclnnStatus\s+(\w+)GetWorkspaceSize\s*\((.*?)\)\s*;",
                text, re.S):
            symbol, params = match.group(1), split_params(match.group(2))
            if symbol not in wanted:
                continue
            while params and any(token in params[-1] for token in _TRAILING):
                params.pop()
            found[symbol] = [header_kind(param) for param in params]
    return found


def parse_ctypes_table(path: Path) -> dict[str, list[str]]:
    """Evaluate ``_GET_WORKSPACE_ARGTYPES`` without importing the package.

    The module needs torch/torch_npu at import time; the table itself only
    needs ``ctypes``, so it is evaluated straight out of the AST.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name)
                   and target.id == "_GET_WORKSPACE_ARGTYPES"
                   for target in node.targets):
            continue
        table = eval(compile(ast.Expression(node.value), str(path), "eval"),
                     {"ctypes": ctypes})  # noqa: S307 - our own table
        break
    else:
        raise RuntimeError(f"_GET_WORKSPACE_ARGTYPES not found in {path}")

    result: dict[str, list[str]] = {}
    for symbol, argtypes in table.items():
        kinds: list[str] = []
        for entry in argtypes:
            name = getattr(entry, "__name__", "")
            if name == "LP_c_void_p" or name.startswith("LP_"):
                kinds.append(_POINTER)
            elif name in _CTYPES_KINDS:
                kinds.append(_CTYPES_KINDS[name])
            else:
                kinds.append(f"<{name}>")
        while kinds and kinds[-1] == _POINTER:
            kinds.pop()
        result[symbol] = kinds
    return result


_EXEC_CALL = re.compile(r"FLA_STABLE_EXEC\(\s*\"(\w+)\"\s*,(.*?)\)\s*;", re.S)

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
                elif callee == "cstr":
                    kinds.append("char_ptr")
                else:  # tensor / optional_tensor / out_tensor / int_array
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opp-include", required=True, nargs="+",
                        help="one or more op_api/include/aclnnop directories: "
                             "the vendor OPP's, and CANN's for the built-in "
                             "operators (FastGelu)")
    parser.add_argument("--ctypes-path", default=str(CTYPES_MODULE))
    parser.add_argument("--src-dir", default=str(SRC_DIR))
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    table = parse_ctypes_table(Path(args.ctypes_path))
    adapters = adapter_calls(Path(args.src_dir))
    adapters.update(hand_written_calls(Path(args.src_dir)))
    wanted = set(table) | set(adapters)

    headers: dict[str, list[str]] = {}
    for include_dir in args.opp_include:
        for symbol, kinds in parse_headers(Path(include_dir), wanted).items():
            headers.setdefault(symbol, kinds)

    report: dict[str, dict] = {}
    failures = 0
    for label, callers in (("ctypes", table), ("adapter", adapters)):
        for symbol in sorted(callers):
            if label == "adapter":
                source, kinds = callers[symbol]
                source_name = source.name
            else:
                source_name = Path(args.ctypes_path).name
                kinds = callers[symbol]
            if symbol not in headers:
                report[f"{label}:{symbol}"] = {
                    "source": source_name,
                    "status": ("known-gap" if symbol in KNOWN_HEADER_GAPS
                               else "no-header"),
                    "reason": KNOWN_HEADER_GAPS.get(symbol, ""),
                    "call_site": kinds}
                if symbol not in KNOWN_HEADER_GAPS:
                    failures += 1
                continue
            problems = compare(headers[symbol], kinds)
            report[f"{label}:{symbol}"] = {
                "source": source_name,
                "status": "ok" if not problems else "mismatch",
                "header": headers[symbol], "call_site": kinds,
                "problems": problems}
            if problems:
                failures += 1
                print(f"{label} {symbol} ({source_name}): "
                      f"{'; '.join(problems)}")
                print(f"    header    : {headers[symbol]}")
                print(f"    call site : {kinds}")

    for symbol in sorted(headers):
        if symbol not in table and symbol not in adapters:
            report[f"unused:{symbol}"] = {"status": "not-called"}

    # The two call sites are also compared with each other: the ctypes table is
    # what the header is checked against, so agreement between the adapter and
    # the table catches order/type drift without needing an OPP at all.
    cross = 0
    for symbol in sorted(set(table) & set(adapters)):
        source, kinds = adapters[symbol]
        problems = compare(table[symbol], kinds)
        report[f"cross:{symbol}"] = {"source": source.name, "problems": problems,
                                     "ctypes": table[symbol],
                                     "call_site": kinds}
        if problems:
            cross += 1
            print(f"adapter {symbol} ({source.name}) disagrees with the ctypes "
                  f"table: {'; '.join(problems)}")
            print(f"    ctypes    : {table[symbol]}")
            print(f"    call site : {kinds}")
    failures += cross

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
