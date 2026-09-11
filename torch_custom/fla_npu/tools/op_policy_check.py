#!/usr/bin/env python3
"""Check that a thin spec agrees with the authoritative host policy.

A migrated operator owns its host-visible semantics in exactly one place: the
policy helper that the ctypes reference path already consumes (for example
``fla_npu.ops.ascendc._kda_policy.kda_fwd_optional_output_mask``).  The thin
spec restates that policy mechanically, twice:

* ``outputs[].when`` decides whether an aclnn output slot is a real tensor or
  ``at::Tensor()`` (i.e. a null descriptor), and
* ``python.return_code`` decides which slots the Python caller sees, and in
  which order.

Those restatements are exactly where past regressions came from: an A5-only
output that ctypes left null while thin allocated it (aclnn 161002), and a
12-element return tuple that drifted from the ctypes one.  This tool turns that
class of drift into an offline failure by enumerating every flag combination
and comparing both restatements against the policy.

Usage:
  python op_policy_check.py --spec op_specs/aclnn_chunk_kda_fwd.json
  python op_policy_check.py --all

The check needs neither torch nor a real NPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
import sys
import textwrap
from pathlib import Path


SPEC_DIR = Path(__file__).resolve().parents[1] / "op_specs"
PKG_ROOT = Path(__file__).resolve().parents[1]  # torch_custom/fla_npu

# python_name -> how to reach the single authoritative policy.
#
# ``flags`` is the policy function's keyword order; every combination of
# True/False is enumerated.  ``suffix`` names the trailing return elements that
# the policy also covers but that are not aclnn outputs (they are supplied by
# the Python wrapper instead).
POLICIES: dict[str, dict] = {
    "npu_chunk_kda_fwd": {
        "module_file": "fla_npu/ops/ascendc/_kda_policy.py",
        "module": "fla_npu.ops.ascendc._kda_policy",
        "function": "kda_fwd_optional_output_mask",
        "flags": [
            "output_final_state",
            "use_gate_in_kernel",
            "disable_recompute",
            "return_intermediate_states",
        ],
        "suffix": ["initial_state"],
    },
}


class _Sentinel:
    """Stand-in for a tensor/argument while replaying ``return_code``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, item: str) -> "_Sentinel":
        return _Sentinel(f"{self.name}.{item}")

    def __call__(self, *args, **kwargs) -> "_Sentinel":
        return _Sentinel(f"{self.name}()")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


def to_python_expr(expr: str) -> str:
    """Translate the C++ boolean operators used in specs to Python."""

    expr = re.sub(r"!(?!=)", " not ", expr)
    expr = expr.replace("&&", " and ").replace("||", " or ")
    return expr


def eval_when(expr: str, flags: dict) -> bool:
    return bool(eval(to_python_expr(expr), {"__builtins__": {}}, dict(flags)))


def load_policy(entry: dict):
    # Load by path: the policy module must stay importable without torch (and
    # without an installed wheel), so the check can run in plain CI.
    path = PKG_ROOT / entry["module_file"]
    spec = importlib.util.spec_from_file_location(
        "_fla_npu_policy_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry["function"])


def check_spec(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    name = spec["python_name"]
    entry = POLICIES.get(name)
    if entry is None:
        print(f"SKIP {name}: no policy registered (nothing to cross-check)")
        return 0

    outputs = spec.get("outputs", [])
    if len(outputs) != len([a for a in spec["args"] if a["kind"] == "out_tensor"]):
        print(f"FAIL {name}: outputs spec length mismatch")
        return 1
    policy = load_policy(entry)
    suffix = entry.get("suffix", [])
    return_code = spec.get("python", {}).get("return_code")
    if not return_code:
        print(f"FAIL {name}: policy registered but no python.return_code")
        return 1

    failures: list[str] = []
    for combo in itertools.product([False, True], repeat=len(entry["flags"])):
        flags = dict(zip(entry["flags"], combo))
        expected = tuple(policy(**flags))
        if len(expected) != len(outputs) + len(suffix):
            failures.append(
                f"{flags}: policy returns {len(expected)} values, spec has "
                f"{len(outputs)} outputs + {len(suffix)} suffix")
            continue
        # 1) aclnn descriptor mask: outputs[].when must equal the policy mask.
        got_mask = tuple(
            eval_when(o["when"], flags) if "when" in o else True for o in outputs)
        if got_mask != expected[:len(outputs)]:
            failures.append(
                f"{flags}: when-mask {got_mask} != policy {expected[:len(outputs)]}")
        # 2) Python return tuple: replay return_code with sentinels and compare
        #    which positions are non-None, in order.
        namespace = {a["name"]: _Sentinel(a["name"]) for a in spec["args"]
                     if a["kind"] != "out_tensor"}
        namespace.update(flags)
        namespace["result"] = [_Sentinel(f"result[{i}]")
                               for i in range(len(outputs))]
        namespace["torch"] = _Sentinel("torch")
        for extra in suffix:
            namespace[extra] = _Sentinel(extra)
        # ``return_code`` is spliced into the wrapper body, so replay it inside
        # a probe function; names resolve against our sentinel namespace.
        probe = "def __policy_probe__():\n" + textwrap.indent(
            return_code, "    ") + "\n"
        exec(compile(probe, f"<{name}.return_code>", "exec"), namespace)
        returned = namespace["__policy_probe__"]()
        if not isinstance(returned, tuple):
            failures.append(f"{flags}: return_code produced {type(returned)!r}")
            continue
        got_visible = tuple(item is not None for item in returned)
        if got_visible != expected:
            failures.append(
                f"{flags}: return visibility {got_visible} != policy {expected}")

    if failures:
        print(f"FAIL {name}: {len(failures)} flag combination(s) disagree")
        for line in failures[:8]:
            print("  -", line)
        return 1
    print(f"OK {name}: {2 ** len(entry['flags'])} flag combinations match "
          f"{entry['module']}.{entry['function']}")
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
        worst = max(worst, check_spec(path))
    return worst


if __name__ == "__main__":
    sys.exit(main())
