"""L2Norm epsilon regression: analytic CPU oracle plus NPU out/state checks."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("FLA_NPU_STABLE_ABI", "ctypes")

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executor_recurrent_kda import recurrent_kda_reference
from fla_npu.ops.ascendc import recurrent_kda

RTOL = 0.02
ATOL = 0.01
SMALL = 2.0 ** -10
K = V = 128
H, HV = 2, 4
PATTERNS = {
    "normal": (1.0, 1.0),
    "zero_q": (0.0, 1.0),
    "zero_k": (1.0, 0.0),
    "zero_qk": (0.0, 0.0),
    "small_q": (SMALL, 1.0),
    "small_k": (1.0, SMALL),
    "small_qk": (SMALL, SMALL),
    "mixed": (SMALL, SMALL),
}


def make_case(tokens, layout, v_first, pattern):
    q = torch.zeros((1, tokens, H, K), dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    q[..., 0] = PATTERNS[pattern][0]
    k[..., 0] = PATTERNS[pattern][1]
    if pattern == "mixed":
        for t in range(tokens):
            for h in range(H):
                q[0, t, h, 0] = [0.0, SMALL, 1.0][(t + h) % 3]
                k[0, t, h, 0] = [SMALL, 1.0, 0.0][(t + h) % 3]
    state = torch.zeros((1, HV, V, K), dtype=torch.float32)
    state[..., 0] = 0.25
    if not v_first:
        state = state.transpose(-1, -2).contiguous()
    x = {
        "q": q, "k": k,
        "v": torch.full((1, tokens, HV, V), 2.0, dtype=torch.bfloat16),
        "g": torch.zeros((1, tokens, HV, K)),
        "beta": torch.full((1, tokens, HV), 0.5),
        "initial_state": state,
        "cu_seqlens": torch.tensor([0, tokens], dtype=torch.int64),
    }
    if layout == "TND":
        for name in ["q", "k", "v", "g", "beta"]:
            x[name] = x[name].squeeze(0)
    return x


def analytic(x, tokens, layout, v_first, normalize, epsilon=1e-6):
    """One active K coordinate gives an independent scalar recurrence."""
    q = x["q"].reshape(tokens, H, K)
    k = x["k"].reshape(tokens, H, K)
    output = torch.empty((tokens, HV, V), dtype=torch.float64)
    final = torch.zeros((1, HV, V, K), dtype=torch.float64)
    for hv in range(HV):
        state = 0.25
        for t in range(tokens):
            a = float(q[t, hv // (HV // H), 0])
            b = float(k[t, hv // (HV // H), 0])
            if normalize:
                a = a / math.sqrt(a * a + epsilon) if a else 0.0
                b = b / math.sqrt(b * b + epsilon) if b else 0.0
            state += (2.0 - state * b) * 0.5 * b
            output[t, hv].fill_(state * a / math.sqrt(K))
        final[0, hv, :, 0] = state
    if layout == "BSND":
        output = output.unsqueeze(0)
    if not v_first:
        final = final.transpose(-1, -2).contiguous()
    return output.to(torch.bfloat16), final.float()


def compare(actual, expected):
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all(), "nonfinite output or state"
        torch.testing.assert_close(a.float(), e.float(), rtol=RTOL, atol=ATOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-mismatch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch_npu.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    results = []
    for tokens in [1, 4]:
        for layout in ["BSND", "TND"]:
            for v_first in [True, False]:
                for pattern in PATTERNS:
                    # Explicitly cover the disabled path with the same small inputs.
                    for normalize in ([True, False] if pattern == "small_qk" else [True]):
                        x = make_case(tokens, layout, v_first, pattern)
                        kw = dict(
                            layout=layout, state_v_first=v_first,
                            output_final_state=True, inplace_final_state=False,
                            use_qk_l2norm_in_kernel=normalize,
                        )
                        cpu = recurrent_kda_reference(**x, **kw)
                        expected = analytic(x, tokens, layout, v_first, normalize)
                        compare(cpu, expected)
                        # The previous formula must be distinguishable at the fixed tolerance.
                        old_formula = analytic(x, tokens, layout, v_first, normalize, epsilon=0.0)
                        old_detected = False
                        try:
                            compare(old_formula, expected)
                        except AssertionError:
                            old_detected = True
                        npu_x = {name: value.to(device) for name, value in x.items()}
                        actual = recurrent_kda(**npu_x, **kw)
                        torch_npu.npu.synchronize()
                        actual = tuple(a.cpu() for a in actual)
                        passed, error = True, ""
                        try:
                            compare(actual, cpu)
                        except AssertionError as exc:
                            passed, error = False, str(exc)[:700]
                        results.append(dict(
                            T=tokens, layout=layout, state_v_first=v_first,
                            pattern=pattern, normalize=normalize, cpu_analytic_pass=True,
                            old_formula_detected=old_detected, passed=passed, error=error,
                        ))
    failures = [r for r in results if not r["passed"]]
    summary = dict(
        total=len(results), cpu_analytic_pass=len(results),
        npu_pass=len(results)-len(failures), npu_fail=len(failures),
        old_formula_detected=sum(r["old_formula_detected"] for r in results),
        expect_mismatch=args.expect_mismatch, cases=results,
    )
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert len(results) == 72
    assert summary["old_formula_detected"] > 0
    if args.expect_mismatch:
        assert failures, "unfixed baseline unexpectedly passed every epsilon test"
        assert all(r["normalize"] and r["old_formula_detected"] for r in failures)
        for pattern in ["small_q", "small_k", "small_qk"]:
            assert any(r["pattern"] == pattern for r in failures), pattern
    else:
        assert not failures, f"{len(failures)} epsilon regression cases failed"
    print("L2NORM_BOUNDARY", {k: v for k, v in summary.items() if k != "cases"})


if __name__ == "__main__":
    main()
