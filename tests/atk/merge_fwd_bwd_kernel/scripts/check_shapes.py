"""Analytical layout checks and input-range calibration for merge_fwd_bwd_kernel."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "merge_cpu", Path(__file__).with_name("merge_fwd_bwd_kernel_cpu.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _mixed_fail_count(got: torch.Tensor, ref: torch.Tensor, abs_lim=0.01, ulp_mul=32):
    g = got.float()
    r = ref.float()
    abs_err = (g - r).abs()
    sign = torch.where(r == 0, torch.ones_like(r), torch.sign(r))
    ulp = (torch.nextafter(r, r + sign * 1e38) - r).abs()
    ulp = torch.where(ulp == 0, torch.full_like(ulp, torch.finfo(torch.float32).tiny), ulp)
    bad = (abs_err > abs_lim) & (abs_err > ulp_mul * ulp)
    return int(bad.sum()), float(abs_err.max()), float(g.abs().max())


def _analytical_checks() -> int:
    count = 0
    for dtype in (torch.float32, torch.bfloat16):
        for forward in (False, True):
            for transpose in (False, True):
                for world, n, rank in ((1, 1, 0), (4, 1, 2), (4, 2, 2 if forward else 0)):
                    ag = torch.zeros(world, 2, 128, 256, dtype=dtype)
                    base = torch.arange(128, dtype=torch.float32).view(128, 1).expand(128, 128)
                    for source in range(world):
                        ag[source, :, :, :128] = (base + source).to(dtype)
                        ag[source, :, :, 128:] = torch.eye(128, dtype=dtype) * 2
                    sources = (
                        [0]
                        if world == 1
                        else list(range(rank - n, rank))
                        if forward
                        else list(range(rank + n, rank, -1))
                    )
                    expected = ag[sources[0], ..., :128].to(torch.float64)
                    for source in sources[1:]:
                        expected = 2 * expected + ag[source, ..., :128].to(torch.float64)
                    if transpose:
                        expected = expected.transpose(-1, -2)
                    got = module.merge_reference(ag, rank, n, forward, transpose)
                    h = torch.full((2, 128, 128), float("nan"), dtype=dtype)
                    h.copy_(got.to(dtype))
                    assert h.shape == (2, 128, 128) and h.dtype == dtype
                    torch.testing.assert_close(h, expected.to(dtype), rtol=0, atol=0)
                    count += 1
    return count


def _range_calibration() -> None:
    """Freeze He=0.05 / M=0.02: correct FP32 passes mixed tolerance, representative errors fail."""
    spec = {
        "dtype": "fp32",
        "S": 8,
        "HV": 1,
        "pre_or_post_num_ranks": 7,
        "rank": 7,
        "forward": True,
        "state_v_first": False,
        "seed": 20260817,
    }
    golden = module.run_reference(spec, high_precision=True).float()
    correct = module.run_reference(spec, high_precision=False).float()
    bad, max_abs, mag = _mixed_fail_count(correct, golden)
    if bad != 0:
        raise AssertionError(
            f"scaled correct FP32 failed mixed tolerance: bad={bad} max_abs={max_abs} mag={mag}"
        )

    zeros = torch.zeros_like(golden)
    skip_he = module.merge_reference(
        _ag_without_last_he(spec), spec["rank"], spec["pre_or_post_num_ranks"], True, False
    ).float()
    doubled_m = module.merge_reference(
        _ag_scaled_m(spec, 2.0), spec["rank"], spec["pre_or_post_num_ranks"], True, False
    ).float()
    wrong_dir = module.run_reference(dict(spec, forward=False, rank=0), high_precision=False).float()
    for name, wrong in (
        ("zeros", zeros),
        ("skip_last_he", skip_he),
        ("doubled_m", doubled_m),
        ("wrong_direction", wrong_dir),
    ):
        wbad, _, _ = _mixed_fail_count(wrong, golden)
        if wbad == 0:
            raise AssertionError(f"representative error {name} was not caught by mixed tolerance")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec["seed"])
    raw = torch.rand((8, 1, 128, 256), generator=generator) * 10 - 5
    old_fp64 = module.merge_reference(raw.to(torch.float64), 7, 7, True, False)
    old_fp32 = module.merge_reference(raw.to(torch.float32), 7, 7, True, False)
    old_bad, _, old_mag = _mixed_fail_count(old_fp32, old_fp64)
    if old_bad == 0:
        raise AssertionError("unscaled [-5,5] unexpectedly passed mixed tolerance")
    print(
        f"range: scaled mixed_fail={bad} mag={mag:.4f}; "
        f"unscaled[-5,5] mixed_fail={old_bad} mag={old_mag:.4e}"
    )


def _ag_without_last_he(spec: dict) -> torch.Tensor:
    ag = module.make_ag_hm(spec, torch.device("cpu"))
    src = module.src_rank(spec["rank"], spec["pre_or_post_num_ranks"], spec["pre_or_post_num_ranks"] - 1, True, spec["S"])
    ag = ag.clone()
    ag[src, :, :, :128] = 0
    return ag


def _ag_scaled_m(spec: dict, scale: float) -> torch.Tensor:
    ag = module.make_ag_hm(spec, torch.device("cpu")).clone()
    ag[..., 128:] *= scale
    return ag


def main():
    module.self_check()
    count = _analytical_checks()
    _range_calibration()
    print(f"merge: {count} analytical checks passed")


if __name__ == "__main__":
    main()
