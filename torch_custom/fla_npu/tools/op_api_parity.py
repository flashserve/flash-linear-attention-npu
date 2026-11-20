#!/usr/bin/env python3
"""Compare the public Python signature of every backend against ctypes.

The ctypes module is the reference users have been calling: its parameter
order, which parameters carry defaults, and the default values themselves are
part of the published API.  A stable backend that changes any of those turns a
working call into a ``TypeError`` -- and because the stable path is selected
transparently by ``_get_direct_op``, the caller never asked for that change.

This tool parses the modules with :mod:`ast` (no import, so no torch and
no NPU are needed) and reports, per operator:

* parameters missing from the backend, or present only in the backend;
* a different positional order;
* a positional parameter that is keyword-only (or vice versa);
* a default that changed, appeared, or disappeared.

Usage::

    python tools/op_api_parity.py            # compare everything, exit 1 on drift
    python tools/op_api_parity.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent.parent
OPS_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"

REFERENCE = OPS_DIR / "_aclnn_ctypes.py"
BACKENDS = {
    "stable": OPS_DIR / "_stable.py",
}

# Every signature is expected to match the reference exactly: a drift here
# is a rename or a changed default that a caller switching backends never
# asked for.


def launcher_only_ops() -> set[str]:
    """Operators declared as having no ctypes reference (no shape to preserve)."""

    text = (OPS_DIR / "__init__.py").read_text(encoding="utf-8")
    marker = text.find("_LAUNCHER_ONLY_OPS")
    if marker < 0:
        return set()
    open_paren = text.find("(", marker)
    close_paren = text.find(")", open_paren)
    if open_paren < 0 or close_paren < 0:
        return set()
    body = text[open_paren + 1:close_paren]
    return {piece.strip().strip('"') for piece in body.split(",")
            if piece.strip()}


def _defaults(node: ast.arguments) -> dict[str, str]:
    """Map every parameter with a default to its unparsed default text."""

    positional = list(node.posonlyargs) + list(node.args)
    out: dict[str, str] = {}
    for name, value in zip(positional[-len(node.defaults):], node.defaults):
        out[name.arg] = ast.unparse(value)
    for name, value in zip(node.kwonlyargs, node.kw_defaults):
        if value is not None:
            out[name.arg] = ast.unparse(value)
    return out


def _signature(node: ast.FunctionDef) -> dict:
    node_args: ast.arguments = node.args
    positional = [a.arg for a in node_args.posonlyargs] + \
        [a.arg for a in node_args.args]
    kwonly = [a.arg for a in node_args.kwonlyargs]
    return {
        "positional": positional,
        "keyword_only": kwonly,
        "defaults": _defaults(node_args),
    }


def signatures(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    # Module-level constants, so a default spelled `PAD_SLOT_ID` in one module
    # and `_PAD_SLOT_ID` in the other still compares by value.
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("npu_"):
            out[node.name] = _signature(node)
        elif (isinstance(node, ast.ImportFrom)
              and (node.module or "").endswith("_aclnn_ctypes")):
            # A backend may re-export the reference function instead of defining
            # its own wrapper (the conv1d family shares one implementation);
            # the signature is the reference's by construction.
            for alias in node.names:
                if alias.name.startswith("npu_"):
                    out[alias.asname or alias.name] = {"reexport": True}
    for entry in out.values():
        if "defaults" in entry:
            entry["constants"] = constants
    return out


def _compare(name: str, reference: dict, backend: dict) -> list[str]:
    problems: list[str] = []
    ref_constants = reference.get("constants", {})
    got_constants = backend.get("constants", {})

    def resolved(text: str, constants: dict) -> object:
        return constants.get(text, text)

    ref_pos = reference["positional"]
    got_pos = backend["positional"]
    if ref_pos != got_pos:
        missing = [p for p in ref_pos if p not in got_pos]
        extra = [p for p in got_pos if p not in ref_pos]
        if missing:
            problems.append(f"positional parameters missing: {missing}")
        if extra:
            problems.append(f"unexpected positional parameters: {extra}")
        if not missing and not extra:
            problems.append(
                f"positional order differs: ctypes {ref_pos} vs backend {got_pos}")

    ref_kw = reference["keyword_only"]
    got_kw = backend["keyword_only"]
    drifted_kind = [p for p in ref_pos if p in got_kw]
    if drifted_kind:
        problems.append(
            f"ctypes positional became keyword-only: {drifted_kind} "
            f"(a positional call now raises TypeError)")
    became_positional = [p for p in ref_kw if p in got_pos]
    if became_positional:
        problems.append(
            f"ctypes keyword-only became positional: {became_positional}")
    missing_kw = [p for p in ref_kw if p not in got_kw and p not in got_pos]
    if missing_kw:
        problems.append(f"keyword-only parameters missing: {missing_kw}")
    extra_kw = [p for p in got_kw if p not in ref_kw and p not in ref_pos]
    if extra_kw:
        problems.append(f"unexpected keyword-only parameters: {extra_kw}")

    for parameter, default in reference["defaults"].items():
        if parameter not in backend["defaults"]:
            problems.append(
                f"default lost on {parameter!r} (ctypes {default})")
        else:
            got = backend["defaults"][parameter]
            # Equal values under different constant names are not drift: the
            # modules use private spellings of the same constant.
            if resolved(got, got_constants) == resolved(default, ref_constants):
                continue
            problems.append(
                f"default changed on {parameter!r}: "
                f"ctypes {default} vs backend {got}")
    for parameter, default in backend["defaults"].items():
        if parameter not in reference["defaults"]:
            problems.append(
                f"default added on {parameter!r} (backend {default})")
    return problems


def evaluate() -> dict:
    reference = signatures(REFERENCE)
    backends = {label: signatures(path) for label, path in BACKENDS.items()}
    launcher_only = launcher_only_ops()
    rows: list[dict] = []
    for name in sorted(set(reference) | set(backends.get("stable", {}))):
        if name not in reference:
            # No ctypes wrapper: there is no published shape to stay compatible
            # with, so what is checked is that the operator says so.  A name
            # that reaches the launcher without either is an oversight, not a
            # policy.
            rows.append({
                "op": name,
                "backend": "stable",
                "reference": "launcher-only",
                "problems": [] if name in launcher_only else [
                    "no ctypes reference and not declared in "
                    "_LAUNCHER_ONLY_OPS"],
            })
            continue
        found = False
        for label, table in backends.items():
            if name not in table:
                continue
            found = True
            if table[name].get("reexport"):
                rows.append({"op": name, "backend": label, "problems": []})
                continue
            # Every backend that exposes the operator is compared, not just the
            # one that happens to answer first, so a caller cannot change the
            # backend flag and land on a renamed keyword.
            rows.append({
                "op": name,
                "backend": label,
                "problems": _compare(name, reference[name], table[name]),
            })
        if not found:
            rows.append({"op": name, "backend": None,
                         "reference": "ctypes",
                         "problems": ["no stable backend signature found"]})
    for row in rows:
        row.setdefault("reference", "ctypes")
    return {"rows": rows, "reference": str(REFERENCE.name),
            "launcher_only": sorted(launcher_only)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = evaluate()
    rows = report["rows"]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        drifted = [row for row in rows if row["problems"]]
        print(f"operators compared: {len(rows)}")
        print(f"drifted          : {len(drifted)}")
        if report.get("launcher_only"):
            print("launcher-only    : "
                  f"{', '.join(report['launcher_only'])} "
                  "(no ctypes reference; declared in _LAUNCHER_ONLY_OPS)")
        print()
        for row in rows:
            if not row["problems"]:
                continue
            print(f"{row['op']}  [{row['backend']}]")
            for problem in row["problems"]:
                print(f"    - {problem}")
        if not drifted:
            print("SIGNATURES MATCH: every backend exposes the ctypes call shape")
    return 1 if any(row["problems"] for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
