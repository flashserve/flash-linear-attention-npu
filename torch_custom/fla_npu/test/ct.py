"""ct 对比工具的仓内等价 shim。

原 ct 模块为内部工具,本仓未携带;按测试文件中的使用契约实现:
  - ct.dual(actual, golden, golden_high_precision) -> {"success": bool, ...}
  - ct.isclose(actual, expected, diff_thd=0.1) -> bool(超阈值时抛 AssertionError)
"""

import torch

def _diff_stats(actual, expected):
    a = actual.float()
    b = expected.float()
    abs_diff = (a - b).abs()
    max_abs = abs_diff.max().item() if abs_diff.numel() > 0 else 0.0
    denom = b.abs().clamp_min(1e-6)
    max_rel = (abs_diff / denom).max().item() if abs_diff.numel() > 0 else 0.0
    return max_abs, max_rel

def _check(actual, expected, abs_thd, rel_thd, tag):
    max_abs, max_rel = _diff_stats(actual, expected)
    ok = (max_rel <= rel_thd) or (max_abs <= abs_thd)
    print(f"[ct.{tag}] max_abs_diff={max_abs:.6g} max_rel_diff={max_rel:.6g} "
          f"(abs_thd={abs_thd}, rel_thd={rel_thd}) -> {'PASS' if ok else 'FAIL'}")
    return ok, max_abs, max_rel

def dual(actual, golden, golden_high_precision=None, abs_thd=2e-2, rel_thd=2e-2, level=None):
    ok, max_abs, max_rel = _check(actual, golden, abs_thd, rel_thd, "dual")
    ok_hp, max_abs_hp, max_rel_hp = True, 0.0, 0.0
    if golden_high_precision is not None:
        ok_hp, max_abs_hp, max_rel_hp = _check(
            actual, golden_high_precision, abs_thd, rel_thd, "dual.hp")
    success = ok and ok_hp
    return {
        "success": success,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "max_abs_diff_hp": max_abs_hp,
        "max_rel_diff_hp": max_rel_hp,
    }

def isclose(actual, expected, diff_thd=0.1, rel_thd=None):
    if rel_thd is None:
        rel_thd = diff_thd
    ok, _, _ = _check(actual, expected, diff_thd, rel_thd, "isclose")
    if not ok:
        raise AssertionError(
            f"ct.isclose 超阈值: diff_thd={diff_thd}, rel_thd={rel_thd}")
    return True
