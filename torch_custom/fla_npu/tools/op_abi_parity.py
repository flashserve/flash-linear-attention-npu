#!/usr/bin/env python3
"""Offline ABI check: schema, adapter signature and the ctypes argument list.

Three things describe the same call, and all three have to agree:

1. the **schema** the dispatcher sees (`kSchema_<op>(...)`),
2. the **adapter signature** the boxed wrapper unpacks positionally
   (`run_<op>(...)`),
3. the **aclnn argument list** the adapter forwards, compared against the
   ctypes reference by ``tools/op_abi_validate.py``.

A mismatch between 1 and 2 is the quiet one: the dispatcher unpacks by position,
so a schema that lists `chunk_size` before `use_exp2` while the adapter takes
them the other way round compiles, registers, and computes something else.
Comparing the two lists by name and type turns that into a build-time failure.

Usage::

    python tools/op_abi_parity.py            # exit 1 on any mismatch
    python tools/op_abi_parity.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETUP_DIR = HERE.parent
SRC_DIR = SETUP_DIR / "csrc" / "src"

# schema spelling -> the C++ type the boxed adapter unpacks into.  `float` maps
# to `double` because the stable value conversions hand over a double and every
# aclnn entry point that takes one declares `double`.
_TYPE_MAP = {
    # The older adapters read the stable tensor handle directly instead of going
    # through `torch::stable::Tensor`; both are the same handle.
    "Tensor": ("Tensor", "AtenTensorHandle"),
    "Tensor?": ("std::optional<Tensor>", "std::optional<AtenTensorHandle>"),
    "int": ("int64_t", "int"),
    "float": ("double", "float"),
    "double": ("double",),
    "bool": ("bool",),
}

# `do` is a C++ keyword, so the adapter spells that parameter `d_o`.
_RENAMES = {"do": "d_o"}

_SCHEMA_RE = re.compile(
    r'constexpr const char\* (kSchema\w*)\s*=\s*((?:"[^"]*"\s*)+);', re.S)


def split_params(text: str) -> list[str]:
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


def schema_calls(text: str) -> list[dict]:
    """[{const, op, inputs: [(name, type)], outputs: [type]}] from one file."""

    found = []
    for match in _SCHEMA_RE.finditer(text):
        literal = "".join(re.findall(r'"([^"]*)"', match.group(2)))
        # `<op>(<inputs>) -> (<outputs>)`; the input list itself contains
        # parentheses (`Tensor(a!) state` for a mutated argument), so the
        # closing one is found by depth rather than by searching for the first.
        open_paren = literal.index("(")
        depth = 0
        close_paren = -1
        for index in range(open_paren, len(literal)):
            if literal[index] == "(":
                depth += 1
            elif literal[index] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = index
                    break
        op = literal[:open_paren].strip()
        inputs_text = literal[open_paren + 1:close_paren]
        tail = literal[close_paren + 1:]
        inputs = []
        for param in split_params(inputs_text):
            pieces = param.rsplit(" ", 1)
            if len(pieces) != 2:
                raise RuntimeError(f"{match.group(1)}: bad parameter {param!r}")
            declared = pieces[0].strip()
            if declared.startswith("Tensor("):  # Tensor(a!) -> Tensor
                declared = "Tensor"
            inputs.append((pieces[1].strip(), declared))
        outputs = [piece.strip() for piece in
                   split_params(tail.split("->", 1)[1].strip().strip("()"))
                   ] if "->" in tail else []
        found.append({"const": match.group(1), "op": op, "inputs": inputs,
                      "outputs": outputs})
    return found


def adapter_signatures(text: str) -> dict[str, list[tuple[str, str]]]:
    """run_/boxed_ function -> its parameter list as (name, declared type)."""

    found: dict[str, list[tuple[str, str]]] = {}
    pattern = re.compile(
        r"^(?:[\w:<>,\s\*&]*?)\b((?:run|boxed)_[a-z0-9_]+)\s*\((.*?)\)\s*\{",
        re.S | re.M)
    for match in pattern.finditer(text):
        params = []
        for param in split_params(match.group(2)):
            stripped = re.sub(r"=\s*[^=]*$", "", param).strip()
            name_match = re.search(r"(\w+)\s*$", stripped)
            if not name_match:
                continue
            params.append((name_match.group(1),
                           stripped[:name_match.start()].strip()))
        found[match.group(1)] = params
    return found


def evaluate() -> dict:
    problems: list[str] = []
    checked = 0
    for source in sorted(SRC_DIR.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8")
        signatures = adapter_signatures(text)
        for call in schema_calls(text):
            if not call["inputs"]:
                continue
            op = call["op"]
            # The adapter for this schema is the run_/boxed_ function whose name
            # ends with the operator, which is the convention the registration
            # list also relies on.  `boxed_<op>` hand-rolls its stack reads, so
            # only the `run_*` form is compared.
            candidates = [name for name in signatures
                          if name.startswith("run_")
                          and (name.endswith(op) or name.endswith(op[4:]))]
            if not candidates:
                problems.append(
                    f"{op}: no adapter signature found in {source.name}")
                continue
            name = sorted(candidates, key=len)[0]
            params = signatures[name]
            # Extra trailing parameters are the adapter's own outputs
            # (`Tensor* out`, `bool* has_final_state`): they are pointers that
            # the caller fills in after the call, not dispatcher arguments.
            extra = params[len(call["inputs"]):]
            if len(params) < len(call["inputs"]) or any(
                    "*" not in declared for _, declared in extra):
                problems.append(
                    f"{op}: schema has {len(call['inputs'])} parameters but "
                    f"{name} takes {len(params)}")
                continue
            checked += 1
            for (schema_name, schema_type), (param, declared) in zip(
                    call["inputs"], params):
                expected_name = _RENAMES.get(schema_name, schema_name)
                if expected_name != param:
                    problems.append(
                        f"{op}: parameter '{schema_name}' is '{param}' in "
                        f"{name}")
                allowed = _TYPE_MAP.get(schema_type)
                if allowed is None:
                    problems.append(
                        f"{op}: unknown schema type {schema_type!r}")
                    continue
                if declared not in allowed:
                    problems.append(
                        f"{op}: '{schema_name}' is {schema_type} in the schema "
                        f"but {declared} in {name}")
    return {"checked": checked, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["problems"] else 0
    for problem in report["problems"]:
        print(f"MISMATCH {problem}")
    print(f"{report['checked']} adapter(s) checked, "
          f"{len(report['problems'])} mismatch(es)")
    print("ABI MATCH: every schema describes the arguments its adapter takes"
          if not report["problems"] else "ABI MISMATCH")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
