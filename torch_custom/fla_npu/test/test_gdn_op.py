"""Smoke test for gated_delta_net_op.
Usage:
    python test_gdn_op.py [--device 0] [--dtype bf16]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
torch.npu.set_compile_mode(jit_compile=False)

from fla_npu.ops.gated_delta_net import gated_delta_net_op  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════
#  Weight bank helper
# ════════════════════════════════════════════════════════════════════════════════


class _GdnWeightBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_v_heads: int,
        num_k_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        key_dim = num_k_heads * key_head_dim
        value_dim = num_v_heads * value_head_dim
        conv_dim = 2 * key_dim + value_dim

        self.in_proj_qkv_weight = nn.Parameter(torch.empty(conv_dim, hidden_size))
        self.in_proj_z_weight = nn.Parameter(torch.empty(value_dim, hidden_size))
        self.in_proj_b_weight = nn.Parameter(torch.empty(num_v_heads, hidden_size))
        self.in_proj_a_weight = nn.Parameter(torch.empty(num_v_heads, hidden_size))
        self.conv_weight = nn.Parameter(torch.empty(conv_kernel_size, conv_dim))
        self.norm_weight = nn.Parameter(torch.ones(value_head_dim))
        self.out_proj_weight = nn.Parameter(torch.empty(hidden_size, value_dim))
        self.A_log = nn.Parameter(torch.log(torch.rand(num_v_heads) * 15.99 + 0.01))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))

        self._init_weights()

    def _init_weights(self):
        for p in (
            self.in_proj_qkv_weight,
            self.in_proj_z_weight,
            self.in_proj_b_weight,
            self.in_proj_a_weight,
            self.out_proj_weight,
        ):
            nn.init.xavier_uniform_(p)
        nn.init.kaiming_uniform_(self.conv_weight, a=5**0.5)


def _npu_gdn_forward(bank: _GdnWeightBank, x: torch.Tensor, cu_seqlens=None, chunk_size=64) -> torch.Tensor:
    return gated_delta_net_op(
        x,
        in_proj_qkv_weight=bank.in_proj_qkv_weight,
        in_proj_z_weight=bank.in_proj_z_weight,
        in_proj_b_weight=bank.in_proj_b_weight,
        in_proj_a_weight=bank.in_proj_a_weight,
        conv_weight=bank.conv_weight,
        norm_weight=bank.norm_weight,
        out_proj_weight=bank.out_proj_weight,
        A_log=bank.A_log,
        dt_bias=bank.dt_bias,
        num_value_heads=bank.num_v_heads,
        num_key_heads=bank.num_k_heads,
        key_head_dim=bank.key_head_dim,
        value_head_dim=bank.value_head_dim,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ════════════════════════════════════════════════════════════════════════════════


def _run_case(
    name: str,
    hidden_size: int,
    num_v_heads: int,
    num_k_heads: int,
    head_dim: int,
    T: int,
    dtype: torch.dtype,
    cu_seqlens=None,
    chunk_size: int = 64,
) -> bool:
    """单次调用验证:前向输出 finite + 反向梯度 finite。"""
    print(f"\n=== {name} ===")
    torch.manual_seed(42)

    bank = _GdnWeightBank(hidden_size, num_v_heads, num_k_heads, head_dim, head_dim)
    bank = bank.to("npu", dtype=dtype)

    x_npu = torch.randn(1, T, hidden_size, dtype=dtype, device="npu", requires_grad=True)

    # ---- forward ----
    try:
        out = _npu_gdn_forward(bank, x_npu, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        torch.npu.synchronize()
    except Exception as e:
        print(f"  [FAIL] forward raised: {e}")
        return False

    fwd_finite = torch.isfinite(out.float()).all().item()
    print(f"  forward: shape={tuple(out.shape)}, finite={fwd_finite}")
    if not fwd_finite:
        print("  [FAIL] forward output contains NaN/Inf")
        return False

    # ---- backward ----
    try:
        loss = out.float().square().mean()
        loss.backward()
        torch.npu.synchronize()
    except Exception as e:
        print(f"  [FAIL] backward raised: {e}")
        return False

    all_ok = True

    # 检查 x.grad
    grad = x_npu.grad
    if grad is None:
        print("  [FAIL] x.grad is None")
        all_ok = False
    else:
        g_finite = torch.isfinite(grad.float()).all().item()
        g_norm = grad.float().norm().item()
        print(f"  x.grad: finite={g_finite}, norm={g_norm:.6f}")
        all_ok = all_ok and g_finite

    weight_grads = {
        "in_proj_qkv": bank.in_proj_qkv_weight.grad,
        "conv": bank.conv_weight.grad,
        "norm": bank.norm_weight.grad,
        "out_proj": bank.out_proj_weight.grad,
        "A_log": bank.A_log.grad,
        "dt_bias": bank.dt_bias.grad,
    }
    for wname, g in weight_grads.items():
        if g is None:
            print(f"  [FAIL] {wname}.grad is None")
            all_ok = False
        else:
            g_finite = torch.isfinite(g.float()).all().item()
            g_norm = g.float().norm().item()
            print(f"  {wname}.grad: finite={g_finite}, norm={g_norm:.6f}")
            all_ok = all_ok and g_finite

    status = "PASS" if all_ok else "FAIL"
    print(f"  [{status}] {name}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    results = {}

    # 1. basic: num_v == num_k
    results["basic"] = _run_case(
        "basic (v=k=4, T=64)",
        hidden_size=64, num_v_heads=4, num_k_heads=4, head_dim=128, T=64,
        dtype=dtype,
    )

    # 2. group head: num_v > num_k
    results["grouped_heads"] = _run_case(
        "grouped_heads (v=8, k=2, T=64)",
        hidden_size=128, num_v_heads=8, num_k_heads=2, head_dim=128, T=64,
        dtype=dtype,
    )

    # 3. varlen
    cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int64, device="npu")
    results["varlen"] = _run_case(
        "varlen (cu_seqlens=[0,32,64])",
        hidden_size=64, num_v_heads=4, num_k_heads=2, head_dim=128, T=64,
        dtype=dtype, cu_seqlens=cu_seqlens,
    )

    # 4. fp16
    if args.dtype == "bf16":
        results["fp16"] = _run_case(
            "fp16 (v=k=4, T=64)",
            hidden_size=64, num_v_heads=4, num_k_heads=4, head_dim=128, T=64,
            dtype=torch.float16,
        )

    # ---- summary ----
    print("\n" + "=" * 60)
    print("Summary:")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and ok

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
