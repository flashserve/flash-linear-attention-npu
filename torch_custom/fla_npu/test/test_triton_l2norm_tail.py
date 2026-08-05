"""Regression test for the Triton L2Norm backward tail rows."""

import os

import torch
import torch_npu

from fla.ops.triton.triton_core.l2norm import l2norm_bwd, l2norm_fwd


REPEATS = 6


def torch_l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    y_fp32 = y.float()
    dy_fp32 = dy.float()
    rstd_fp32 = rstd.float().unsqueeze(-1)
    return (
        dy_fp32 * rstd_fp32
        - (dy_fp32 * y_fp32).sum(dim=-1, keepdim=True)
        * y_fp32
        * rstd_fp32
    ).to(y.dtype)


def run_case(rows: int, dtype: torch.dtype) -> None:
    torch.manual_seed(20260726 + rows)
    x = torch.randn(
        rows,
        128,
        dtype=torch.float32,
        device="npu",
    ).to(dtype)
    dy = torch.randn(
        rows,
        128,
        dtype=torch.float32,
        device="npu",
    ).to(dtype)
    y, rstd = l2norm_fwd(x)
    expected = torch_l2norm_bwd(y, rstd, dy)
    outputs = [l2norm_bwd(y, rstd, dy) for _ in range(REPEATS)]

    for repeat, output in enumerate(outputs):
        if not torch.isfinite(output).all():
            raise AssertionError(
                f"rows={rows}, dtype={dtype}, repeat={repeat}: "
                "non-finite output"
            )
        if not torch.equal(output, expected):
            max_abs = (output.float() - expected.float()).abs().max().item()
            raise AssertionError(
                f"rows={rows}, dtype={dtype}, repeat={repeat}: "
                f"formula mismatch, max_abs={max_abs}"
            )
        if not torch.equal(output, outputs[0]):
            raise AssertionError(
                f"rows={rows}, dtype={dtype}, repeat={repeat}: "
                "non-deterministic output"
            )


def main() -> None:
    device = int(os.environ.get("TEST_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    for rows in (15, 16, 17, 390, 3386):
        run_case(rows, torch.bfloat16)
    run_case(17, torch.float16)
    torch.npu.synchronize()
    print("triton_l2norm_tail: PASS")


if __name__ == "__main__":
    main()
