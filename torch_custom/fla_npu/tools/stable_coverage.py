#!/usr/bin/env python3
"""Offline coverage gate for the Stable-ABI backend (no NPU, no torch).

What this answers is not "did a test pass" but "is every operator the ctypes
layer publishes carried by the launcher, and is every adapter wired all the way
through".  Everything is read from files in the tree:

1. **Adapter coverage** -- each ``npu_*`` operator in the published list has a
   wrapper in ``_stable.py`` and an adapter in ``csrc/src`` (schema +
   ``run_`` definition + registration).
2. **Wiring** -- a schema without an implementation, an implementation without
   its ``run_`` function, or a wrapper without a schema is a FAIL.  These are
   the mistakes that otherwise show up as a dispatcher error at run time.
3. **Enum tables** -- a ``cstr(k<X>Names, arg)`` call site must have the same
   code order as ``_stable._ENUM[op][arg]``, because the two are the only thing
   that keeps a layout string from silently becoming a different layout.
4. **Baseline** -- a gap is a FAIL unless ``tests/stable_coverage_baseline.json``
   records it with a reason, so gaps are declared rather than tolerated.

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


def public_ops() -> list[str]:
    """The operator list the package publishes (``_ASCENDC_OPS``)."""

    text = (OPS_DIR / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"_ASCENDC_OPS\s*=\s*\((.*?)\)", text, re.S)
    if match is None:
        raise RuntimeError("_ASCENDC_OPS not found in __init__.py")
    return re.findall(r'"([a-z0-9_]+)"', match.group(1))


def ctypes_ops() -> list[str]:
    text = (OPS_DIR / "_aclnn_ctypes.py").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^def (npu_[a-z0-9_]+)\(", text, re.M)))


def launcher_only_ops() -> list[str]:
    """Operators declared as having no ctypes reference to compare against."""

    text = (OPS_DIR / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"_LAUNCHER_ONLY_OPS[^=]*=\s*\((.*?)\)", text, re.S)
    if match is None:
        return []
    return re.findall(r'"([a-z0-9_]+)"', match.group(1))


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
    found: dict[str, list[dict]] = {}
    for node in tree.body:
        if (not isinstance(node, ast.FunctionDef)
                or not node.name.startswith("npu_")):
            continue
        # `stream = _current_stream_ptr()` hoisted out of the call is the same
        # thing as passing the call inline; both count as "stream is last".
        hoisted = {target.id
                   for assign in ast.walk(node)
                   if isinstance(assign, ast.Assign)
                   and isinstance(assign.value, ast.Call)
                   and isinstance(assign.value.func, ast.Name)
                   and assign.value.func.id == "_current_stream_ptr"
                   for target in assign.targets
                   if isinstance(target, ast.Name)}
        calls: list[dict] = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            inner = call.func
            if not (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_op"
                    and inner.args
                    and isinstance(inner.args[0], ast.Constant)):
                continue
            last = call.args[-1] if call.args else None
            stream_last = (
                isinstance(last, ast.Call)
                and isinstance(last.func, ast.Name)
                and last.func.id == "_current_stream_ptr"
            ) or (isinstance(last, ast.Name) and last.id in hoisted)
            calls.append({"op": inner.args[0].value,
                          "args": len(call.args),
                          "star": any(isinstance(argument, ast.Starred)
                                      for argument in call.args),
                          "stream_last": stream_last})
        found[node.name] = calls
    return found


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
_IMPL_ADAPTER_RE = re.compile(
    r'm\.impl\(\s*"([a-z0-9_]+)"\s*,\s*&[\w:]*boxed_adapter<\s*'
    r'(run_[a-z0-9_]+)\s*>')
_IMPL_BOXED_RE = re.compile(
    r'm\.impl\(\s*"([a-z0-9_]+)"\s*,\s*&(boxed_[a-z0-9_]+)\)')


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
    impls = [match.group(1) or match.group(2)
             for match in list(_IMPL_ADAPTER_RE.finditer(text))
             + list(_IMPL_BOXED_RE.finditer(text))]
    return list(zip(defs, impls))


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
        entry["impl"] = True
        entry["adapter"] = function
        entry["run"] = functions.get(function)
        # The declared domain of this operator, as far as the tree can tell:
        # enum tables give the string-valued axes, the parameter list gives the
        # varlen axis and the boolean flags.  coverage_gap_report.py compares
        # these against the scenario names that actually ran.
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


def evaluate() -> dict:
    published = public_ops()
    ctypes_names = set(ctypes_ops())
    launcher_only = set(launcher_only_ops())
    wrapper_lines = wrappers()
    dispatches = wrapper_dispatches()
    schema_by_op = {info["op"]: info["params"] for info in schemas().values()}
    adapter_info = adapters()
    enums = package_enums()

    rows = []
    blockers = []
    for name in sorted(published):
        row = {
            "op": name,
            "wrapper": name in wrapper_lines,
            "schema": "schema" in adapter_info.get(name, {}),
            "run": "run" in adapter_info.get(name, {}),
            "def": bool(adapter_info.get(name, {}).get("def")),
            "impl": bool(adapter_info.get(name, {}).get("impl")),
            "enums": sorted(adapter_info.get(name, {}).get("enums", {})),
            "axes": adapter_info.get(name, {}).get("axes", {}),
            "flags": adapter_info.get(name, {}).get("flags", []),
            "reference": "launcher-only" if name in launcher_only else "ctypes",
        }
        rows.append(row)
        if name not in wrapper_lines:
            blockers.append(f"{name}: no wrapper in _stable.py")
        for key in ("schema", "run", "def", "impl"):
            if not row[key]:
                blockers.append(f"{name}: adapter is missing its {key}")
        # A new operator with no ctypes wrapper has to say so.  Without a
        # same-kernel reference its parity scenario must bring its own, and that
        # is a decision a reader has to be able to find; declaring one that
        # ctypes still defines is the other half of the same invariant.
        if name not in ctypes_names and name not in launcher_only:
            blockers.append(
                f"{name}: published but absent from the ctypes reference "
                "(add the reference, or declare it in _LAUNCHER_ONLY_OPS)")
        if name in launcher_only and name in ctypes_names:
            blockers.append(
                f"{name}: declared in _LAUNCHER_ONLY_OPS but the ctypes "
                "reference still defines it")
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

    for name in sorted(set(adapter_info) - set(published)):
        blockers.append(f"{name}: adapter exists but is not published")

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

    return {"rows": rows, "blockers": blockers,
            "adapter_count": len(adapter_info),
            "launcher_only": sorted(launcher_only)}


def render(report: dict, strict: bool, baseline: dict) -> tuple[str, bool]:
    known = {} if strict else baseline.get("known_gaps", {})
    lines = ["%-42s %-8s %-8s %-8s %-11s %s" % (
        "operator", "wrapper", "schema", "run", "registered", "reference")]
    for row in report["rows"]:
        lines.append("%-42s %-8s %-8s %-8s %-11s %s" % (
            row["op"], row["wrapper"], row["schema"], row["run"],
            row["def"] and row["impl"], row.get("reference", "ctypes")))
    unexplained = [b for b in report["blockers"]
                   if b.split(":", 1)[0] not in known]
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
        return 0 if not report["blockers"] or not args.strict else 1
    text, ok = render(report, args.strict, baseline)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
