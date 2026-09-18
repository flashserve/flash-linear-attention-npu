"""全量算子 stable parity：复用 regression_ops 的每个场景，但把
`_launcher` 整体改道到 stable 后端，逐个与 ctypes 参考对比。

用法（221/241，wheel 已装入环境）：
    FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
        python tests/stable_abi/regression_stable_full.py

任何算子/场景在 stable 侧缺失或回退，都会在这里暴露为 AttributeError/差异。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fla_npu.ops.ascendc import _stable  # noqa: E402

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")
BASELINE = Path(__file__).resolve().parent / "stable_scenarios.json"
# One baseline entry covering the Ascend950-only operators: on a non-950 host
# they are a recorded skip, on Ascend950 they are recorded passes.  The fused
# backward belongs here too -- it composes the 950-only finalize kernel, so A2
# refuses it with a parameter error (aclnn 161002) rather than computing it.
A5_SCENARIOS = ("Ascend950-only: fwd_prepare / bwd_finalize / fwd(a5) / "
                "chunk_gated_delta_rule_bwd")
BASELINE_WRITE = os.environ.get("FLA_NPU_BASELINE_WRITE", "").strip().lower() in {
    "1", "true", "yes", "on"}


def check_baseline(device: str, suite, subset: bool = False) -> int:
    """Compare this run's scenario set against the checked-in record.

    The point is the *set*, not the numbers: a scenario that disappears (a
    dropped layout, a flag combination nobody exercises any more) is exactly the
    kind of coverage loss that a green run would otherwise hide.  Diffs are
    recorded too -- they are 0.0 by construction, so a non-zero one means the
    comparison itself changed meaning.
    """

    observed = {"passed": dict(sorted(suite.SCENARIOS.items())),
                "skipped": dict(sorted(suite.SKIPPED.items()))}
    data = {}
    if BASELINE.exists():
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
    if BASELINE_WRITE:
        data[device] = observed
        BASELINE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"baseline written for {device}: "
              f"{len(observed['passed'])} passed, "
              f"{len(observed['skipped'])} skipped -> {BASELINE}")
        return 0
    if subset:
        # A --group run executes a slice of the scenario library by design, so
        # every scenario outside the slice would look "lost".  The bitwise
        # assertion still applies to the ones that ran; only the set check is
        # meaningful for a full run.
        print(f"baseline comparison skipped for a subset run "
              f"({len(observed['passed'])} scenario(s) executed)")
        return 0
    if device not in data:
        print(f"no baseline recorded for {device}; run once with "
              f"FLA_NPU_BASELINE_WRITE=1 to add it")
        return 0
    expected = data[device]
    missing = sorted(set(expected["passed"]) - set(observed["passed"]))
    added = sorted(set(observed["passed"]) - set(expected["passed"]))
    skipped_missing = sorted(set(expected["skipped"]) - set(observed["skipped"]))
    skipped_added = sorted(set(observed["skipped"]) - set(expected["skipped"]))
    for label, items in (("scenario lost", missing),
                         ("scenario added", added),
                         ("recorded skip lost", skipped_missing),
                         ("new skip", skipped_added)):
        for item in items:
            print(f"{label}: {item}")
    bad = [name for name, diff in observed["passed"].items() if diff != 0.0]
    for name in bad:
        print(f"scenario no longer bit-identical: {name}")
    if missing or bad:
        print(f"\nBASELINE MISMATCH for {device} "
              f"({len(missing)} lost, {len(bad)} non-zero)")
        return 1
    print(f"baseline ok for {device}: {len(observed['passed'])} scenarios "
          f"({len(added)} new, {len(skipped_added)} new skip)")
    return 0


class StableShim:
    """Stands in for the `_launcher` module inside the regression suite."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.missing: list[str] = []

    def __getattr__(self, name):
        import fla_npu.ops.ascendc._stable as stable_mod

        try:
            target = getattr(stable_mod, name)
        except AttributeError:
            # No second backend to fall back to: a missing name is a coverage
            # gap, and a gap is a failure here rather than a quiet skip.
            self.missing.append(name)
            raise AttributeError(
                f"stable backend has no adapter for {name!r} (coverage gap)")

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return target(*args, **kwargs)

        return counted


class PublicShim:
    """Stands in for `_launcher` by calling the public API.

    The scenarios then exercise the *whole* dispatch chain -- backend
    selection, the mutation contract, the wrapper -- instead of pinning the
    backend directly.  With FLA_NPU_STABLE_TRACE=1 every operator announces which
    backend served it, and the run fails if anything had to fall back to ctypes
    (FALLBACKS is the counted form of "the launcher did not carry this
    operator", which changes the dependency footprint and must not happen
    silently).
    """

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.missing: list[str] = []

    def __getattr__(self, name):
        import fla_npu.ops.ascendc as ascendc

        try:
            target = getattr(ascendc, name)
        except AttributeError as exc:
            raise AttributeError(
                f"the public API does not expose {name!r}") from exc

        def counted(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return target(*args, **kwargs)

        return counted


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    suite_cli = None

    # Either FLA_NPU_STABLE_LIB points at a build tree, or an installed wheel
    # carries the launcher next to the package -- both are ordinary customer
    # setups, so accept whichever resolves instead of demanding the env var.
    if not _stable.available():
        raise SystemExit(
            "no Stable-ABI launcher: set FLA_NPU_STABLE_LIB to a built "
            "libfla_npu_stable.so, or install a wheel that bundles one")
    import regression_ops as suite

    suite.group_cli(parser)
    args = parser.parse_args()

    shim = StableShim()
    if (os.environ.get("FLA_NPU_DISPATCH") or "").strip().lower() == "public":
        shim = PublicShim()
    suite._launcher = shim  # every scenario now exercises the stable backend
    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scenarios = [
        suite.scenario_fast_gelu,
        suite.scenario_recurrent_gated_delta_rule,
        suite.scenario_recurrent_kda,
        suite.scenario_recompute,
        suite.scenario_pwy_full,
        suite.scenario_pwy,
        suite.scenario_dv_local,
        suite.scenario_pwy_da,
        suite.scenario_gated_fwd_h,
        suite.scenario_chunk_fwd_h,
        suite.scenario_chunk_fwd_o,
        suite.scenario_bwd_dhu,
        suite.scenario_conv1d_new_apis,
        suite.scenario_conv1d_prefill,
        suite.scenario_conv1d_varlen_initial_state,
        suite.scenario_conv1d_update,
        suite.scenario_conv1d_update_offset_state,
        suite.scenario_conv1d_update_paged_state,
        suite.scenario_conv1d_prefill_paged_state,
        suite.scenario_conv1d_gather_padding,
        suite.scenario_conv1d_varlen_pad_slot,
        suite.scenario_conv1d_bwd_bnsd,
        suite.scenario_chunk_kda_fwd,
        suite.scenario_chunk_kda_fwd_variants,
        suite.scenario_chunk_kda_fwd_three_stage,
        suite.scenario_chunk_kda_fwd_finalize,
        suite.scenario_chunk_kda_bwd_intra,
        suite.scenario_chunk_kda_bwd,
        suite.scenario_chunk_kda_bwd_recompute,
        suite.scenario_dqkwg,
        suite.scenario_chunk_local_cumsum,
        suite.scenario_scaled_dot_kkt,
        suite.scenario_solve_tri_dense,
        suite.scenario_solve_tri_guards,
        suite.scenario_kda_gate_cumsum,
        suite.scenario_chunk_gated_delta_rule_fwd,
    ]
    names = [scenario.__name__ for scenario in scenarios]
    suite.missing_from_groups(names)
    if args.list_groups:
        suite.print_groups(names)
        return 0
    chosen = set(suite.select_groups(names, args.group))
    scenarios = [scenario for scenario in scenarios
                 if scenario.__name__ in chosen]
    if args.group:
        # A subset must not be compared against the whole-matrix baseline: the
        # scenarios that were not selected would look like lost coverage.
        global BASELINE_WRITE
        BASELINE_WRITE = False
        print(f"groups {', '.join(args.group)}: {len(scenarios)} of "
              f"{len(names)} scenarios "
              f"({', '.join(suite.select_groups(names, args.group))}); "
              "baseline comparison skipped for a subset run")
    for scenario in scenarios:
        print(f"--- entering {scenario.__name__}", flush=True)
        scenario()

    # Two operators only exist in the Ascend950 OPP
    # (chunk_gated_delta_rule_fwd_prepare / _bwd_finalize).  They are exercised
    # by their own driver instead of being silently absent: on a non-950 host
    # this prints an explicit SKIP naming them, so the coverage record says why
    # rather than counting them as covered.
    device = str(torch.npu.get_device_name(0))
    # They live outside the scenario list (their own module), so the `a5` group
    # selects them explicitly: a run that picked another group must not drag
    # them in, and `--group a5` must run them even though the list is empty.
    run_a5 = not args.group or "a5" in args.group
    if not run_a5:
        print("Ascend950-only scenarios skipped (not selected by --group)")
    elif "950" in device:
        import regression_950_ops as a5

        a5._launcher = shim
        for scenario in (a5.scenario_fwd_prepare, a5.scenario_bwd_finalize,
                         a5.scenario_chunk_gated_delta_rule_fwd_a5,
                         suite.scenario_chunk_gated_delta_rule_bwd):
            print(f"--- entering {scenario.__name__} (Ascend950)",
                  flush=True)
            scenario()
    else:
        reason = (f"requires Ascend950, this host reports {device!r}")
        suite.SKIPPED[A5_SCENARIOS] = reason
        print(f"SKIP {A5_SCENARIOS}: {reason}")
    if run_a5 and "950" in device:
        # The A5 driver keeps its own recorder; fold it into this run's record
        # so one baseline covers whatever the host can actually execute.
        # `a5` is only imported when this run selects the group, so a subset run
        # on a 950 host must not reach for it.
        suite.SCENARIOS.update(a5.SCENARIOS)
    print(f"\nstable ops exercised: {len(shim.calls)}")
    for name in sorted(shim.calls):
        print(f"  {name}: {shim.calls[name]} call(s)")
    if shim.missing:
        print(f"MISSING ADAPTERS: {sorted(set(shim.missing))}")
        return 1
    status = check_baseline(device, suite, subset=bool(args.group))
    if status != 0:
        return status
    if isinstance(shim, PublicShim):
        import fla_npu.ops.ascendc as ascendc

        if ascendc.FALLBACKS:
            print(f"UNEXPECTED FALLBACKS: {ascendc.FALLBACKS}")
            return 1
        served = sorted(set(ascendc.BACKENDS.values()))
        print(f"public dispatch: {len(ascendc.BACKENDS)} operators, "
              f"backends {served}, no fallback")
    print("ALL PASS: full stable parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
