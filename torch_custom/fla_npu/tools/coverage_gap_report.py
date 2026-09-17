#!/usr/bin/env python3
"""Where is an operator's declared legal domain wider than what it exercises?

The coverage gate (``stable_coverage.py``) answers "does every operator have an
adapter, and is every declared axis representable".  The scenario baseline
(``tests/stable_abi/stable_scenarios.json``) answers "which scenarios ran".  Neither alone
answers "is *this* operator's domain exercised" -- ``chunk_fwd_o`` declaring
four layouts and running one of them looks fully covered to both, and the other
three layouts look covered because *other* operators' scenarios mention them.

So the comparison has to be per operator: the scenario names are filtered to the
ones belonging to the operator under test, and only string-valued axes
(layout / dtype / activation) are judged from names -- boolean flags appear in
scenario labels as abbreviations (``use=1 dis=0``), so they are reported
separately as "check by hand" rather than counted as gaps.  That conservative
split is what makes the output actionable: it found the missing
layout/dtype variants that the suite was quietly not covering.

Usage::

    python tools/coverage_gap_report.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


SETUP_DIR = pathlib.Path(__file__).resolve().parent.parent
BASELINE = (SETUP_DIR.parent.parent / "tests" / "stable_abi"
            / "stable_scenarios.json")


def _load_coverage_tool():
    path = SETUP_DIR / "tools" / "stable_coverage.py"
    spec = importlib.util.spec_from_file_location("stable_coverage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if not BASELINE.exists():
        raise SystemExit(f"no scenario baseline at {BASELINE}")
    coverage = _load_coverage_tool()
    report = coverage.evaluate()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    gaps = 0
    for device, record in sorted(baseline.items()):
        names = list(record["passed"]) + list(record["skipped"])
        print(f"== {device} ({len(record['passed'])} passed, "
              f"{len(record['skipped'])} skipped)")
        for row in report["rows"]:
            axes = row.get("axes") or {}
            if not axes:
                continue
            short = row["op"][4:] if row["op"].startswith("npu_") else row["op"]
            own = " ".join(name for name in names if short in name).lower()
            if not own:
                continue
            missing: list[str] = []
            flags: list[str] = []
            for axis, values in sorted(axes.items()):
                if axis == "flags":
                    flags.extend(values)
                    continue
                strings = [value for value in values
                           if isinstance(value, str)]
                if not strings:
                    flags.append(axis)
                    continue
                missing.extend(
                    f"{axis}={value}" for value in strings
                    if value.lower() not in own
                    and value.lower() not in {"false", "true"})
            if missing:
                gaps += len(missing)
                print(f"  {row['op']:<44} no scenario for: "
                      f"{', '.join(missing)}")
            if row.get("flags"):
                # Boolean axes are named in the schema, not in scenario labels,
                # so they are listed for a human rather than counted as gaps.
                flags.extend(row["flags"])
            if flags:
                print(f"  {row['op']:<44} boolean axes (check the scenario, "
                      f"not the name): {', '.join(sorted(set(flags)))}")
        print()
    print(f"declared layout/dtype values with no scenario for that operator: "
          f"{gaps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
