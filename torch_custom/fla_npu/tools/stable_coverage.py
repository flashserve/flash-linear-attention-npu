#!/usr/bin/env python3
"""Offline coverage gate for the Stable-ABI backend (no NPU, no torch).

What this answers is not "did a test pass" but "is every published operator
carried by the thin layer, and is every adapter wired all the way through".
Everything is read from files in the tree:

1. **Adapter coverage** -- each ``npu_*`` operator in the published list has a
   wrapper in ``_stable.py`` and an adapter in ``csrc/src`` (schema +
   ``run_`` definition + registration).
2. **Wiring** -- a schema without an implementation, an implementation without
   its ``run_`` function, or a wrapper without a schema is a FAIL.  These are
   the mistakes that otherwise show up as a dispatcher error at run time.
3. **Enum tables** -- a ``cstr(k<X>Names, arg)`` call site must have the same
   code order as ``_stable._ENUM[op][arg]``, because the two are the only thing
   that keeps a layout string from silently becoming a different layout.
4. **Public contract** -- the historical API and dispatcher argument positions
   in ``tests/stable_public_api.json`` cannot silently disappear or reorder.
   All operators must be covered; no missing-backend waiver is supported.

Usage::

    python tools/stable_coverage.py              # human-readable matrix
    python tools/stable_coverage.py --json       # machine-readable
    python tools/stable_coverage.py --strict     # ignore the baseline
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETUP_DIR = HERE.parent
REPO_ROOT = SETUP_DIR.parents[1]
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
SRC_DIR = SETUP_DIR / "csrc" / "src"
BASELINE = REPO_ROOT / "tests" / "stable_coverage_baseline.json"
PUBLIC_CONTRACT = REPO_ROOT / "tests" / "stable_public_api.json"


def public_ops() -> list[str]:
    """The operator list the package publishes (``_ASCENDC_OPS``)."""

    text = (OPS_DIR / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"_ASCENDC_OPS\s*=\s*\((.*?)\)", text, re.S)
    if match is None:
        raise RuntimeError("_ASCENDC_OPS not found in __init__.py")
    return re.findall(r'"([a-z0-9_]+)"', match.group(1))


def required_public_ops() -> list[str]:
    """Historical public contract, independent of the implementation inventory."""
    names = json.loads(PUBLIC_CONTRACT.read_text(encoding="utf-8"))["operators"]
    if not names or len(names) != len(set(names)):
        raise ValueError("public API contract must contain unique nonempty operators")
    return names


def wrappers() -> dict[str, int]:
    text = (OPS_DIR / "_stable.py").read_text(encoding="utf-8")
    return {name: text[:match.start()].count("\n") + 1
            for name, match in ((m.group(1), m) for m in
                                re.finditer(r"^def (npu_[a-z0-9_]+)\(",
                                            text, re.M))}


def wrapper_dispatches() -> dict[str, list[dict]]:
    """wrapper -> the dispatcher calls it makes for its own operator.

    The dispatcher registers by position, so a wrapper that hands over a
    different number of arguments than the schema declares is passing something
    the adapter will not unpack the way it was written -- and a same-type swap
    inside that list is invisible everywhere else.  Calls to *other* operators
    are skipped: a composed wrapper (`npu_chunk_kda_bwd` splits its launch out
    to a helper) is a legitimate shape.
    """

    tree = ast.parse((OPS_DIR / "_stable.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def dispatch_name(node):
        if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
            return None
        if isinstance(node.func, ast.Name) and node.func.id == "_op":
            return node.args[0].value
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "_OP_CACHE" and node.func.attr == "get":
            return node.args[0].value
        return None

    def walk(function, visited):
        if function in visited:
            return []
        visited = visited | {function}
        node = functions[function]
        assignments = [item for item in ast.walk(node) if isinstance(item, ast.Assign)]
        aliases = {target.id: dispatch_name(item.value) for item in assignments
                   for target in item.targets if isinstance(target, ast.Name) and dispatch_name(item.value)}
        helpers = {target.id: item.value.id for item in assignments
                   for target in item.targets if isinstance(target, ast.Name)
                   and isinstance(item.value, ast.Name) and item.value.id in functions}
        hoisted = {target.id for item in assignments if isinstance(item.value, ast.Call)
                   and isinstance(item.value.func, ast.Name) and item.value.func.id == "_current_stream_ptr"
                   for target in item.targets if isinstance(target, ast.Name)}
        calls = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            op = dispatch_name(call.func)
            if isinstance(call.func, ast.Name):
                op = aliases.get(call.func.id, op)
                helper = helpers.get(call.func.id, call.func.id)
                if helper in functions and helper not in visited:
                    calls.extend(walk(helper, visited))
            if op is None:
                continue
            last = call.args[-1] if call.args else None
            stream_last = (isinstance(last, ast.Call) and isinstance(last.func, ast.Name)
                           and last.func.id == "_current_stream_ptr") or (
                               isinstance(last, ast.Name) and last.id in hoisted)
            calls.append({"op": op, "args": len(call.args),
                          "expressions": [ast.unparse(arg) for arg in call.args],
                          "star": any(isinstance(arg, ast.Starred) for arg in call.args),
                          "stream_last": stream_last})
        return calls

    return {name: walk(name, set()) for name in functions if name.startswith("npu_")}


def _wrapper_problems(op: str, expected: list, dispatch: dict) -> list[str]:
    """Wiring problems for one dispatcher call a wrapper makes for *op*."""

    if dispatch["star"]:
        return [f"{op}: wrapper passes *args to the dispatcher, so its "
                "argument list cannot be checked"]
    problems: list[str] = []
    if expected and dispatch["args"] != len(expected):
        problems.append(
            f"{op}: wrapper passes {dispatch['args']} arguments but the schema "
            f"declares {len(expected)}")
    if not dispatch["stream_last"]:
        problems.append(
            f"{op}: wrapper does not pass the stream as the last argument")
    return problems


def _enclosing_run(text: str, position: int) -> str:
    """Name of the ``run_<op>`` definition whose body contains *position*."""

    head = text[:position]
    names = re.findall(r"^\w[\w:<>,\s\*&]*?\b(run_[a-z0-9_]+)\s*\(", head,
                       re.M)
    if not names:
        return ""
    # The nearest definition that is still open at `position`: walking back
    # over braces is enough because adapters do not nest definitions.
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
    start = index
    for match in reversed(list(re.finditer(
            r"^\w[\w:<>,\s\*&]*?\b(run_[a-z0-9_]+)\s*\(", text[:start],
            re.M))):
        return match.group(1)
    return ""


_SCHEMA_RE = re.compile(
    r'constexpr const char\* (kSchema\w*)\s*=\s*((?:"[^"]*"\s*)+);', re.S)
# One regex, not two: both registration shapes have to come out of the file in
# the order they were written, because ``registrations`` pairs the `m.def` and
# `m.impl` lists positionally.
_IMPL_RE = re.compile(
    r'm\.impl\(\s*"([a-z0-9_]+)"\s*,\s*&[\w:]*'
    r'(?:boxed_adapter<\s*(run_[a-z0-9_]+)\s*>'
    r'|(boxed_[a-z0-9_]+)\))',
    re.S)


def schemas() -> dict[str, dict]:
    """schema constant -> the file that defines it and the op it declares.

    The operator name is read out of the schema *string* rather than from the
    constant's spelling: hand-written adapters name it `kSchema_chunk_fwd_h`
    while the registered operator is `npu_chunk_fwd_h`, and the string is what
    the dispatcher actually sees.
    """

    found: dict[str, dict] = {}
    for source in sorted(SRC_DIR.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8")
        for match in _SCHEMA_RE.finditer(text):
            literal = "".join(re.findall(r'"([^"]*)"', match.group(2)))
            op = literal.split("(", 1)[0].strip()
            found[match.group(1)] = {"file": source.name, "op": op,
                                     "params": _schema_params(literal)}
    return found


def _schema_params(literal: str) -> list[tuple[str, str]]:
    """(name, type) of every schema parameter, for the declared-domain axes."""

    open_paren = literal.find("(")
    if open_paren < 0:
        return []
    depth = 0
    for index in range(open_paren, len(literal)):
        if literal[index] == "(":
            depth += 1
        elif literal[index] == ")":
            depth -= 1
            if depth == 0:
                body = literal[open_paren + 1:index]
                break
    else:
        return []
    params = []
    for param in body.split(","):
        pieces = param.strip().rsplit(" ", 1)
        if len(pieces) == 2:
            params.append((pieces[1].strip(), pieces[0].strip()))
    return params


def registrations() -> list[tuple[str, str]]:
    """(schema constant, adapter function) in registration order.

    The two lists in ``stable_ops.cpp`` are written in the same order, so the
    pairing is the schema the operator has to match.
    """

    text = (SRC_DIR / "stable_ops.cpp").read_text(encoding="utf-8")
    defs = re.findall(r"m\.def\((kSchema\w*)\)", text)
    impls = {match.group(1): match.group(2) or match.group(3)
             for match in _IMPL_RE.finditer(text)}
    table = schemas()
    return [(constant, impls.get(table.get(constant, {}).get("op"), ""))
            for constant in defs]


def adapter_functions() -> dict[str, str]:
    """adapter function name -> the file that defines it."""

    found: dict[str, str] = {}
    for source in sorted(SRC_DIR.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(
                r"^\s*[\w:<>,\s\*&]*?\b((?:run|boxed)_[a-z0-9_]+)\s*\(",
                text, re.M):
            found[match.group(1)] = source.name
    return found


def translation_unit() -> tuple[list[str], list[str]]:
    """(adapter sources included by stable_ops.cpp, ones that are not).

    Every adapter lives in its own `stable_<op>.cpp` and is *included* into
    `stable_ops.cpp` -- nothing here is compiled on its own.  A source nobody
    includes is therefore dead: its operator would silently have no adapter
    while every other check in this file (they scan the directory) stays green.
    """

    text = (SRC_DIR / "stable_ops.cpp").read_text(encoding="utf-8")
    included = re.findall(r'#include\s+"(stable_\w+\.cpp)"', text)
    present = {path.name for path in SRC_DIR.glob("stable_*.cpp")}
    return included, sorted(present - set(included) - {"stable_ops.cpp"})


def adapters() -> dict[str, dict]:
    """Per operator: schema, adapter function, registration and enum tables."""

    schema_table = schemas()
    functions = adapter_functions()
    found: dict[str, dict] = {}
    for constant, function in registrations():
        schema = schema_table.get(constant)
        if schema is None:
            continue
        name = schema["op"]
        entry = found.setdefault(name, {})
        entry["schema"] = schema["file"]
        entry["def"] = True
        entry["impl"] = bool(function)
        entry["adapter"] = function
        entry["run"] = functions.get(function)
        # The declared domain of this operator, as far as the tree can tell:
        # enum tables give the string-valued axes, the parameter list gives the
        # varlen axis and the boolean flags.
        axes: dict[str, list] = {}
        params = schema.get("params", [])
        # Only string-valued axes are put in `axes`: the report matches them
        # against scenario *names*, so a literal like "varlen" would be reported
        # as a gap for every operator whose scenario label does not spell it.
        entry["axes"] = axes
        entry["flags"] = [name for name, declared in params
                          if declared == "bool"]

    for source in sorted(SRC_DIR.glob("stable_*.cpp")):
        text = source.read_text(encoding="utf-8")
        # `cstr` builds the argument inline; `enum_name` resolves it once and
        # the adapter reuses the pointer -- both are enum tables.
        for call in re.finditer(
                r"(?:cstr|enum_name)\((k[A-Za-z0-9]+Names)\s*,\s*([a-z0-9_]+)\)",
                text):
            owner = _enclosing_run(text, call.start())
            table = re.search(
                re.escape(call.group(1)) + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
            if not owner or not table:
                continue
            owner = owner[len("run_"):]
            name = owner if owner.startswith("npu_") else f"npu_{owner}"
            entry = found.setdefault(name, {})
            entry.setdefault("enums", {})[call.group(2)] = {
                "table": call.group(1),
                "names": re.findall(r'"([^"]*)"', table.group(1))}
            entry.setdefault("axes", {})[call.group(2)] = re.findall(
                r'"([^"]*)"', table.group(1))
    return found


def package_enums() -> dict[str, dict[str, dict[str, int]]]:
    """``_stable._ENUM`` without importing torch.

    The tables are plain literals that share the ``_LAYOUT_CODES`` constant, so
    they are evaluated in order into one namespace instead of with
    ``literal_eval``, which cannot resolve the shared name.
    """

    tree = ast.parse((OPS_DIR / "_stable.py").read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    result: dict[str, dict[str, dict[str, int]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("_"):
            continue
        try:
            value = eval(  # noqa: S307 - checked-in literals
                compile(ast.Expression(node.value), "<_stable.py>", "eval"),
                dict(namespace))
        except Exception:
            continue
        namespace[target.id] = value
        if target.id == "_ENUM":
            result = value
    return result


def load_baseline() -> dict:
    if BASELINE.is_file():
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    return {}


def expected_source(op: str) -> str:
    """The file ``op``'s adapter has to be defined in.

    One operator per file: ``npu_chunk_fwd_h`` is adapted in
    ``stable_chunk_fwd_h.cpp``.  Debug entries (``_stream_probe``) are not
    operators and return "".

    This is the invariant a *merge* breaks: a branch that predates the split
    adds its adapter to the shared file it already has (`stable_chunk.cpp`,
    `stable_kda.cpp`, ...), so the new operator's `run_` ends up somewhere
    else while every other check here -- they key off the operator name --
    stays green.
    """

    return f"stable_{op[len('npu_'):]}.cpp" if op.startswith("npu_") else ""


def enum_codes_match(table: dict[str, int], names: list[str]) -> bool:
    """C++ indexes its name array with integer codes, independently of key order."""
    values = list(table.values())
    return (all(type(value) is int for value in values)
            and values == list(range(len(names))))


def evaluate() -> dict:
    published = public_ops()
    wrapper_lines = wrappers()
    dispatches = wrapper_dispatches()
    schema_by_op = {info["op"]: info["params"] for info in schemas().values()}
    contract = json.loads(PUBLIC_CONTRACT.read_text(encoding="utf-8"))
    adapter_info = adapters()
    enums = package_enums()

    rows = []
    blockers = []
    if not published:
        blockers.append("public API: operator inventory is empty")
    if len(published) != len(set(published)):
        blockers.append("public API: duplicate published operator")
    for name in sorted(set(required_public_ops()) - set(published)):
        blockers.append(f"{name}: historical public API was removed")
    for name in sorted(published):
        row = {
            "op": name,
            "wrapper": name in wrapper_lines,
            "schema": "schema" in adapter_info.get(name, {}),
            "run": bool(adapter_info.get(name, {}).get("run")),
            "def": bool(adapter_info.get(name, {}).get("def")),
            "impl": bool(adapter_info.get(name, {}).get("impl")),
            "enums": sorted(adapter_info.get(name, {}).get("enums", {})),
            "axes": adapter_info.get(name, {}).get("axes", {}),
            "flags": adapter_info.get(name, {}).get("flags", []),
            "backend": "stable",
        }
        rows.append(row)
        if name not in wrapper_lines:
            blockers.append(f"{name}: no wrapper in _stable.py")
        for key in ("schema", "run", "def", "impl"):
            if not row[key]:
                blockers.append(f"{name}: adapter is missing its {key}")
        # One operator per file: the run_ function has to live in the file
        # named after the operator, so that a merge that adds an adapter to a
        # shared file (the pre-split layout) is reported instead of accepted.
        wanted = expected_source(name)
        actual = adapter_info.get(name, {}).get("run")
        if wanted and actual and actual != wanted:
            blockers.append(
                f"{name}: adapter is defined in {actual}, expected {wanted} "
                "(one operator per file)")
        if not dispatches.get(name):
            blockers.append(f"{name}: wrapper never reaches a Stable dispatcher call")
        expected_calls = contract.get("dispatch_arguments", {}).get(name)
        if expected_calls is not None:
            actual_calls = [{"op": call["op"], "expressions": call["expressions"]}
                            for call in dispatches.get(name, [])]
            if actual_calls != expected_calls:
                blockers.append(f"{name}: dispatcher argument order/expression differs from public contract")
        for dispatch in dispatches.get(name, []):
            if dispatch["op"] == name:
                blockers.extend(
                    _wrapper_problems(name, schema_by_op.get(name, []),
                                      dispatch))

        for argument, info in adapter_info.get(name, {}).get("enums", {}).items():
            package_table = enums.get(name, {})
            if argument not in package_table:
                blockers.append(
                    f"{name}: enum argument '{argument}' (table "
                    f"{info['table']}) has no _stable._ENUM entry")
            elif list(package_table[argument]) != info["names"]:
                blockers.append(
                    f"{name}: {info['table']} order {info['names']} != "
                    f"_stable._ENUM {list(package_table[argument])}")
            elif not enum_codes_match(package_table[argument], info["names"]):
                blockers.append(
                    f"{name}: enum numeric codes for {argument!r} "
                    f"{list(package_table[argument].values())} must be integer "
                    f"indexes {list(range(len(info['names'])))}")

    for name in sorted(set(adapter_info) - set(published)):
        blockers.append(f"{name}: adapter exists but is not published")

    unit = (SRC_DIR / "stable_ops.cpp").read_text(encoding="utf-8")
    definitions = re.findall(r"m\.def\((kSchema\w*)\)", unit)
    implementations = [m.group(1) for m in _IMPL_RE.finditer(unit)]
    schema_table = schemas()
    for constant in definitions:
        if constant not in schema_table:
            blockers.append(f"{constant}: registered schema is not defined")
    defined_ops = [schema_table[c]["op"] for c in definitions if c in schema_table]
    defined_ops.extend(re.findall(r'm\.def\(\s*"([a-z0-9_]+)\(', unit))
    for name in implementations:
        if name not in defined_ops:
            blockers.append(f"{name}: implementation has no registered schema")
    for items, label in ((definitions, "schema"), (implementations, "implementation")):
        for name in set(items):
            if items.count(name) != 1:
                blockers.append(f"{name}: duplicate {label} registration")

    # Adapter sources are #included into stable_ops.cpp rather than compiled on
    # their own, so a file nobody includes is never built: the operator would
    # fall back (or fail to resolve) while every other check here -- they read
    # the directory, not the translation unit -- still passed.
    for orphaned in translation_unit()[1]:
        blockers.append(
            f"{orphaned}: adapter source is not #included by stable_ops.cpp, "
            "so it is never compiled")
    for included in translation_unit()[0]:
        if not (SRC_DIR / included).is_file():
            blockers.append(f"{included}: included adapter source is missing")

    # ...and the other direction: a table in Python that no adapter consumes is
    # a stale code order waiting to be used (this is how the recurrent KDA
    # layout silently changed meaning when the canonical order was introduced).
    for name, arguments in sorted(enums.items()):
        adapter_enums = adapter_info.get(name, {}).get("enums", {})
        for argument, table in arguments.items():
            if argument not in adapter_enums:
                blockers.append(
                    f"{name}: _stable._ENUM[{argument!r}] has no cstr() table "
                    "in any adapter")
            elif list(table) != adapter_enums[argument]["names"]:
                blockers.append(
                    f"{name}: _stable._ENUM[{argument!r}] {list(table)} != "
                    f"{adapter_enums[argument]['table']} "
                    f"{adapter_enums[argument]['names']}")
            elif not enum_codes_match(table, adapter_enums[argument]["names"]):
                blockers.append(
                    f"{name}: _stable._ENUM[{argument!r}] enum numeric codes "
                    "do not match C++ table indexes")

    return {"rows": rows, "blockers": blockers,
            "adapter_count": len(adapter_info),
            "public_count": len(published)}


def render(report: dict, strict: bool, baseline: dict) -> tuple[str, bool]:
    lines = ["%-42s %-8s %-8s %-8s %-11s %s" % (
        "operator", "wrapper", "schema", "run", "registered", "backend")]
    for row in report["rows"]:
        lines.append("%-42s %-8s %-8s %-8s %-11s %s" % (
            row["op"], row["wrapper"], row["schema"], row["run"],
            row["def"] and row["impl"], row["backend"]))
    unexplained = report["blockers"]
    if unexplained:
        lines.append("")
        lines.append("UNEXPLAINED GAPS:")
        lines += [f"  - {item}" for item in unexplained]
        return "\n".join(lines), False
    recorded = [b for b in report["blockers"] if b not in unexplained]
    if recorded:
        lines.append("")
        lines.append("recorded gaps:")
        lines += [f"  - {item}" for item in recorded]
    return "\n".join(lines), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = evaluate()
    baseline = load_baseline()
    if args.json:
        report["baseline"] = baseline
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if render(report, args.strict, baseline)[1] else 1
    text, ok = render(report, args.strict, baseline)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
