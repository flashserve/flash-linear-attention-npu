"""比较 aclnn 取数程序的 dhm 与 CPU 标杆，并按 E_r / P_r 两个平面分别给出误差。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np


def report(name: str, a: np.ndarray, b: np.ndarray, verbose: bool, tol: float) -> Tuple[float, float]:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    diff = np.abs(af - bf)
    denom = np.maximum(np.abs(bf), 1e-6)
    rel = diff / denom
    good = float((rel < 2e-2).mean())
    # 与算子链一致的口径：状态用模型 dtype 传递，绝对误差应相对「标杆幅值」衡量，
    # 否则大状态（多 chunk）下的绝对阈值没有意义。
    scale = max(float(np.abs(bf).max()), 1e-12)
    rel_norm = float(diff.max()) / scale
    print(f"{name:12s} max_abs={diff.max():.3e} max_rel={rel.max():.3e} rel_norm={rel_norm:.3e} "
          f"ref_amax={scale:.3e} mean_abs={diff.mean():.3e} "
          f"within2%={good * 100:.1f}% nan_inf={int(np.isnan(a).sum() + np.isinf(a).sum())}")
    if verbose:
        flat = np.argsort(diff.reshape(-1))[::-1][:5]
        for idx in flat:
            pos = np.unravel_index(idx, diff.shape)
            print(f"    worst @ {pos} got={af[pos]:.6g} exp={bf[pos]:.6g}")
        # 按行/列统计误差，帮助判断是否布局问题
        print(f"    err by last-axis(top3)={np.argsort(diff.mean(axis=tuple(range(diff.ndim - 1)))[::-1])[:3].tolist()}")
    return float(diff.max()), rel_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tol", type=float, default=2e-2, help="相对标杆幅值的容差（默认 2%%）")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "case.json")) as f:
        meta = json.load(f)
    hv, k_dim, vk = meta["dhm_shape"]
    got = np.fromfile(os.path.join(args.dir, "dhm_acl.bin"), dtype=np.float32).reshape(hv, k_dim, vk)
    exp = np.fromfile(os.path.join(args.dir, "expected_dhm.bin"), dtype=np.float32).reshape(hv, k_dim, vk)
    v_dim = vk - k_dim
    verbose = os.environ.get("CP_COMPARE_VERBOSE", "1") != "0"
    e_max, e_rel = report("E_r", got[:, :, :v_dim], exp[:, :, :v_dim], verbose, args.tol)
    p_max, p_rel = report("P_r", got[:, :, v_dim:], exp[:, :, v_dim:], verbose, args.tol)
    worst = max(e_rel, p_rel)
    print("PASS" if worst <= args.tol else "FAIL",
          f"(worst_rel_norm={worst:.3e}, worst_abs={max(e_max, p_max):.3e}, tol={args.tol:.1e})")


if __name__ == "__main__":
    main()
