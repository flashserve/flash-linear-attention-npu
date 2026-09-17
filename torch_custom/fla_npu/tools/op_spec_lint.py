#!/usr/bin/env python3
"""Lint spec expressions against the symbols the generated C++ actually has.

``outputs[].alloc``, ``outputs[].when`` and ``args[].when`` are spliced into
the generated adapter body verbatim.  If such an expression references an
argument that only exists in the Python signature (``python.ignored``) or a
misspelled name, nothing complains until the wheel build fails minutes later
(``'use_gate_in_kernel' was not declared in this scope``).  This check turns
that into an offline failure.

Usage:
  python op_spec_lint.py --spec op_specs/aclnn_xxx.json
  python op_spec_lint.py --all
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import sys
from pathlib import Path


SPEC_DIR = Path(__file__).resolve().parents[1] / "op_specs"

# C++ / ATen vocabulary the generated body may use freely.
_CPP_TOKENS = {
    "at", "c10", "std", "torch", "int64_t", "int32_t", "size_t", "uint64_t",
    "bool", "double", "float", "int", "void", "char", "string", "vector",
    "Tensor", "TensorOptions", "empty", "empty_like", "sizes", "size", "numel",
    "element_size", "options", "dtype", "device", "kFloat", "kBFloat16",
    "kHalf", "kByte", "defined", "has_value", "value", "true", "false",
    "const", "auto", "static_cast", "reinterpret_cast", "push_back", "reserve",
    "nullptr", "make_unique", "unique_ptr", "if", "else", "return",
    "AclTensorView", "AclIntArrayView", "outputs", "views", "output", "result",
}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_MEMBER = re.compile(r"(\.|->|::)\s*$")
_LITERAL = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")


def helper_symbols(helpers: str) -> set[str]:
    """Names a helper block introduces: function names, parameters, locals."""

    symbols = set()
    # function names: `inline <ret> name(` and `<ret> name(`
    for match in re.finditer(r"\b(\w+)\s*\(", helpers):
        symbols.add(match.group(1))
    # parameters of the function signatures
    for match in re.finditer(r"\(([^()]*)\)", helpers):
        for raw in match.group(1).split(","):
            parts = raw.strip().split()
            if len(parts) >= 2:
                symbols.add(parts[-1].lstrip("*&"))
    # locals declared as `<type> name =` or `const <type> name =`
    for match in re.finditer(r"\b(?:const\s+)?[\w:<>]+\s+(\w+)\s*=", helpers):
        symbols.add(match.group(1))
    return {name for name in symbols if name}


def free_identifiers(expr: str) -> set[str]:
    """Identifiers used as values: skip members, C++ tokens and literals."""

    expr = _LITERAL.sub('""', expr)
    found = set()
    for match in _IDENT.finditer(expr):
        name = match.group(0)
        if _MEMBER.search(expr[:match.start()]):
            continue  # member access / qualified name
        found.add(name)
    return found


def lint_spec(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    name = spec["python_name"]
    declared = set()
    for argument in spec["args"]:
        declared.add(argument["name"])
        if "cpp" in argument:
            declared.add(argument["cpp"])
    helpers = spec.get("helpers", "")
    allowed = declared | helper_symbols(helpers) | _CPP_TOKENS

    problems = []
    expressions: list[tuple[str, str]] = []
    for index, output in enumerate(spec.get("outputs", [])):
        for field in ("alloc", "when"):
            if field in output:
                expressions.append((f"outputs[{index}].{field}", output[field]))
    if "output" in spec:
        for field in ("alloc", "when"):
            if field in spec["output"]:
                expressions.append(
                    (f"output.{field}", spec["output"][field]))
    for argument in spec["args"]:
        if "when" in argument:
            expressions.append((f"args[{argument['name']}].when",
                                argument["when"]))

    for label, expr in expressions:
        for identifier in free_identifiers(str(expr)):
            if identifier in allowed or keyword.iskeyword(identifier):
                continue
            problems.append(
                f"{label}: {identifier!r} is not declared as an arg/cpp arg "
                f"or helper symbol")

    if problems:
        print(f"FAIL {name}: {len(problems)} unresolved symbol(s)")
        for line in problems:
            print("  -", line)
        return 1
    print(f"OK {name}: {len(expressions)} expression(s) reference declared "
          f"symbols only")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        specs = sorted(SPEC_DIR.glob("*.json"))
    elif args.spec:
        specs = [args.spec]
    else:
        parser.error("pass --spec or --all")
    worst = 0
    for path in specs:
        worst = max(worst, lint_spec(path))
    return worst


if __name__ == "__main__":
    sys.exit(main())
