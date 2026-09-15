#!/usr/bin/env python3
"""Customer-facing switch test: a ctypes call site keeps working on the launcher.

A customer never imports ``_aclnn_ctypes`` or ``_stable``; they import
``fla_npu.ops.ascendc`` and call ``npu_*``.  That is all this test exercises,
and it runs the *same* script twice:

* ``old`` -- ``FLA_NPU_STABLE_ABI=ctypes``, i.e. how the package behaved while
  the ctypes layer was the only path;
* ``new`` -- the default, i.e. the launcher.

Both phases record identical observations -- the public signature of every
published operator, plus the return shape, values and in-place contract of a
few real calls -- and the driver diffs them.  Two properties have to hold
together, and neither is enough alone:

1. the records are identical, so nothing a caller can see changed;
2. the recorded backends really differ (``old`` ctypes, ``new`` stable, no
   fallbacks), otherwise a run where both phases silently took the same path
   would pass and prove nothing.

Invalid input is recorded but not compared: the launcher validates less in
Python than the reference does, so the error *kind* is allowed to differ.  The
documented contract is "illegal input errors, and does not have to error the
same way"; what is compared is only whether it errored at all.

A probe that both phases *accept* is not discriminating: it is printed as
`old=accepted new=accepted` rather than counted as coverage, and belongs on the
list of probes to sharpen into a real rejection.

usage::

    FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
        python tests/stable_abi/customer_switch_compat.py [--json report.json]
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest

# torch / torch_npu / fla_npu are imported by _import_runtime(), so that
# --help works on a machine without an NPU runtime.
torch = None
ascendc = None


def _import_runtime() -> None:
    """Import the runtime and the package, once per process."""

    global torch, ascendc
    if torch is not None:
        return
    import torch as torch_module
    import torch_npu  # noqa: F401

    torch_module.npu.config.allow_internal_format = False
    torch_module.npu.set_compile_mode(jit_compile=False)
    import fla_npu.ops.ascendc as package

    torch = torch_module
    ascendc = package


def describe(value) -> dict:
    """A JSON-safe fingerprint of a return value."""

    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sum": float(value.float().sum().item()),
            "absmax": float(value.float().abs().max().item()),
        }
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "items": [describe(item) for item in value]}
    if value is None:
        return {"kind": "none"}
    return {"kind": type(value).__name__}


def api_shape() -> dict:
    """The public signature of every published operator, and its bare alias.

    This is the part of the customer-visible contract that does not need a
    device: parameter names, order, keyword-only-ness and defaults.  It is also
    where a launcher-only operator would quietly degrade to ``(*args,
    **kwargs)`` if the wrapper lost its signature.
    """

    shape: dict[str, dict] = {}
    for name in sorted(ascendc._ASCENDC_OPS):
        entry: dict = {}
        try:
            signature = inspect.signature(getattr(ascendc, name))
            entry["params"] = [parameter.name
                               for parameter in signature.parameters.values()]
            entry["kinds"] = [str(parameter.kind)
                              for parameter in signature.parameters.values()]
            entry["defaults"] = {
                parameter.name: repr(parameter.default)
                for parameter in signature.parameters.values()
                if parameter.default is not inspect.Parameter.empty}
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
        bare = name[4:] if name.startswith("npu_") else name
        entry["bare_alias_same"] = (getattr(ascendc, bare, None)
                                    is getattr(ascendc, name, None))
        shape[name] = entry
    return shape


def cases():
    """(name, call, mutated tensors, illegal call) for the operators exercised.

    The input shapes are the ones the shipped regression suite already uses for
    these operators, so they are known-good on 910B and Ascend950; the point
    here is the customer's call, not the numerics.
    """

    def fast_gelu():
        x = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
        return (lambda: ascendc.npu_fast_gelu_custom(x), (),
                lambda: ascendc.npu_fast_gelu_custom(x.double()))

    def kda_gate_cumsum():
        batch, heads, tokens, kdim, chunk = 1, 4, 128, 64, 64
        g = (torch.randn(batch, tokens, heads, kdim) * 1.25).to(torch.float16)
        g = g.permute(0, 2, 1, 3).contiguous().npu()
        a_log = (torch.randn(heads, dtype=torch.float32) * 0.12).npu()
        dt_bias = (torch.randn(heads * kdim, dtype=torch.float32) * 1.65
                   - 3.0).npu()
        kwargs = dict(A_log=a_log, dt_bias=dt_bias, use_gate_in_kernel=True,
                      safe_gate=False, lower_bound=-5.0)
        return (lambda: ascendc.npu_kda_gate_cumsum(g, chunk, **kwargs), (),
                lambda: ascendc.npu_kda_gate_cumsum(g.double(), chunk,
                                                    **kwargs))

    def recurrent_gated_delta_rule():
        batch, nk, nv, dim, gap, offset = 8, 8, 16, 128, 16384, 12288
        block_stride = nv * dim * dim + gap
        raw = torch.empty((batch + 1) * block_stride * 4, dtype=torch.int8,
                          device="npu")
        state = torch.as_strided(
            raw.view(torch.float32),
            size=(batch + 1, nv, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset)
        state.zero_()
        normalize = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
        query = normalize(torch.randn(batch, nk, dim,
                                      device="npu")).to(torch.bfloat16)
        key = normalize(torch.randn(batch, nk, dim,
                                    device="npu")).to(torch.bfloat16)
        value = torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu")
        beta = torch.rand(batch, nv, dtype=torch.bfloat16, device="npu")
        gate = torch.rand(batch, nv, dtype=torch.float32, device="npu")
        lengths = torch.tensor([0] + [1] * batch, dtype=torch.int32,
                               device="npu")
        indices = torch.arange(batch, dtype=torch.int32, device="npu")
        kwargs = dict(beta=beta, g=gate, scale=dim ** -0.5,
                      actual_seq_lengths=lengths, ssm_state_indices=indices,
                      num_accepted_tokens=None)
        return (lambda: ascendc.npu_recurrent_gated_delta_rule(
            query, key, value, state, **kwargs), (state,),
            lambda: ascendc.npu_recurrent_gated_delta_rule(
                query, key, value, state, beta=beta, scale=dim ** -0.5,
                actual_seq_lengths=lengths, ssm_state_indices=indices,
                num_accepted_tokens=None))

    def causal_conv1d_update():
        def ramp(count, start, shape):
            return (torch.arange(count, dtype=torch.float32)
                    + start).reshape(shape).to(torch.bfloat16).npu()

        x = ramp(2 * 16, 1.0, (2, 16))
        weight = ramp(4 * 16, 101.0, (4, 16))
        bias = ramp(16, 201.0, (16,))
        conv_state = ramp(3 * 3 * 16, 301.0, (3, 3, 16))
        indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        kwargs = dict(activation="silu", conv_state_indices=indices)
        return (lambda: ascendc.npu_causal_conv1d_update(
            x, conv_state, weight, bias, **kwargs), (x, conv_state),
            lambda: ascendc.npu_causal_conv1d_update(
                x.double(), conv_state, weight, bias, **kwargs))

    # Each builder returns (call, mutated, illegal); flatten it so the caller
    # unpacks (name, call, mutated, illegal) in one step.
    return [("fast_gelu", *fast_gelu()),
            ("kda_gate_cumsum", *kda_gate_cumsum()),
            ("recurrent_gated_delta_rule", *recurrent_gated_delta_rule()),
            ("causal_conv1d_update", *causal_conv1d_update())]


def observe() -> dict:
    torch.npu.set_device(0)
    torch.manual_seed(20260914)

    calls: dict[str, dict] = {}
    illegal: dict[str, str] = {}
    for name, call, mutated, rejects in cases():
        versions = [int(tensor._version) for tensor in mutated]
        torch.npu.synchronize()
        try:
            result = call()
            torch.npu.synchronize()
            calls[name] = {
                "return": describe(result),
                "version_delta": [int(tensor._version) - before
                                  for tensor, before in zip(mutated, versions)],
            }
        except Exception as exc:  # noqa: BLE001
            calls[name] = {"error": type(exc).__name__}
        try:
            rejects()
            torch.npu.synchronize()
            illegal[name] = "accepted"
        except Exception as exc:  # noqa: BLE001
            illegal[name] = f"raised {type(exc).__name__}"

    return {
        "stable_abi_env": os.environ.get("FLA_NPU_STABLE_ABI", ""),
        "api": api_shape(),
        "calls": calls,
        "illegal": illegal,
        "backends": dict(sorted(ascendc.BACKENDS.items())),
        "fallbacks": dict(sorted(ascendc.FALLBACKS.items())),
    }


def run_phase(env_extra: dict) -> dict:
    """Run one phase in its own process: the backend is resolved at import."""

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.json")
        environment = dict(os.environ)
        environment.pop("FLA_NPU_STABLE_ABI", None)
        environment.update(env_extra)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--report", path], check=True, env=environment)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def _diff(label: str, old: dict, new: dict) -> list[str]:
    problems = [f"{label}: {key} differs"
                for key in sorted(set(old) | set(new))
                if old.get(key) != new.get(key)]
    if problems:
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                print(f"  {label}[{key}]")
                print(f"    old: {json.dumps(old.get(key), sort_keys=True)}")
                print(f"    new: {json.dumps(new.get(key), sort_keys=True)}")
    return problems


def compare(old: dict, new: dict) -> list[str]:
    """Everything the driver asserts, as a list of human-readable problems."""

    problems = _diff("api", old["api"], new["api"])
    problems += _diff("calls", old["calls"], new["calls"])

    # The error *kind* is allowed to differ; whether it errored is not.
    rejected_old = {name: value != "accepted"
                    for name, value in old["illegal"].items()}
    rejected_new = {name: value != "accepted"
                    for name, value in new["illegal"].items()}
    problems += _diff("rejects_illegal_input", rejected_old, rejected_new)

    # The two phases also have to have taken *different* paths: identical
    # records only mean something if the second run really used the launcher.
    for label, report, expected in (("old", old, "ctypes"),
                                    ("new", new, "stable")):
        served = report["backends"]
        wrong = {name: backend for name, backend in served.items()
                 if backend != expected}
        if not served or wrong:
            problems.append(f"{label}: expected every operator to be served by "
                            f"{expected}, got {served}")
    if new["fallbacks"]:
        problems.append("new: the launcher fell back to ctypes: "
                        f"{new['fallbacks']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="",
                        help="record one phase and exit (used by the driver)")
    parser.add_argument("--json", default="", help="write the two records here")
    args = parser.parse_args()

    _import_runtime()

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(observe(), handle, indent=2, sort_keys=True)
        return 0

    old = run_phase({"FLA_NPU_STABLE_ABI": "ctypes"})
    new = run_phase({})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"old": old, "new": new}, handle, indent=2,
                      sort_keys=True)
        print(f"records -> {args.json}")

    print(f"old phase: FLA_NPU_STABLE_ABI={old['stable_abi_env'] or '<unset>'}")
    print(f"new phase: FLA_NPU_STABLE_ABI={new['stable_abi_env'] or '<unset>'}")

    print("illegal input (kind may differ by design):")
    for name in sorted(set(old["illegal"]) | set(new["illegal"])):
        print(f"  {name:<32} old={old['illegal'].get(name)} "
              f"new={new['illegal'].get(name)}")
    print(f"backends: old={sorted(set(old['backends'].values()))} "
          f"new={sorted(set(new['backends'].values()))} "
          f"(fallbacks: old={old['fallbacks']} new={new['fallbacks']})")

    problems = compare(old, new)

    if problems:
        print(f"\nFAIL: {len(problems)} difference(s) between the ctypes and "
              "launcher runs")
        return 1
    print(f"\nPASS: {len(new['api'])} operators keep their public signature and "
          f"{len(new['calls'])} calls keep their result, now served by the "
          "launcher")
    return 0


class ComparisonTest(unittest.TestCase):
    """The driver's own logic, exercised without an NPU.

    The two device phases are the expensive half of this file; this half runs
    anywhere, so a mistake in the comparison cannot hide behind a machine that
    happens to have no accelerator.
    """

    def _record(self, backend: str, *, calls=None, fallbacks=None) -> dict:
        return {
            "api": {"npu_x": {"params": ["a"],
                              "kinds": ["POSITIONAL_OR_KEYWORD"],
                              "defaults": {}}},
            "calls": calls or {"case": {"return": {"kind": "tensor"}}},
            "illegal": {"case": "raised TypeError"},
            "backends": {"npu_x": backend},
            "fallbacks": fallbacks or {},
        }

    def test_matching_records_pass(self) -> None:
        self.assertEqual(compare(self._record("ctypes"),
                                 self._record("stable")), [])

    def test_a_changed_signature_fails(self) -> None:
        new = self._record("stable")
        new["api"]["npu_x"]["params"] = ["b"]
        self.assertTrue(compare(self._record("ctypes"), new))

    def test_a_changed_result_fails(self) -> None:
        new = self._record("stable", calls={"case": {"return": {"kind": "none"}}})
        self.assertTrue(any("calls" in problem
                            for problem in compare(self._record("ctypes"), new)))

    def test_same_backend_in_both_phases_fails(self) -> None:
        """Identical records are worthless if both runs took the same path."""

        problem = compare(self._record("stable"), self._record("stable"))
        self.assertTrue(any(problem.startswith("old:") for problem in problem),
                        problem)

    def test_a_fallback_fails(self) -> None:
        new = self._record("stable", fallbacks={"npu_x": 1})
        self.assertTrue(any("fell back" in problem
                            for problem in compare(self._record("ctypes"), new)))


if __name__ == "__main__":
    sys.exit(main())
