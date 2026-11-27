"""fused_recurrent_rwkv8 精度对拍工具。"""
import torch


def rel_rmse(actual: torch.Tensor, golden: torch.Tensor) -> float:
    """相对 RMSE：||actual - golden||_2 / ||golden||_2（分母近零时兜底 1e-12）。"""
    actual, golden = actual.float(), golden.float()
    return ((actual - golden).pow(2).mean().sqrt()
            / golden.pow(2).mean().sqrt().clamp_min(1e-12)).item()


def compare_rel_rmse(actual: torch.Tensor, golden: torch.Tensor,
                     name: str = "tensor", threshold: float = 2e-3) -> bool:
    """rel-RMSE 对拍；先查 shape 一致，再查有限性，最后比阈值。"""
    if actual.shape != golden.shape:
        print(f"  [{name}] FAIL  shape mismatch: actual {tuple(actual.shape)} "
              f"vs golden {tuple(golden.shape)}")
        return False
    if not torch.isfinite(actual.float()).all().item():
        print(f"  [{name}] FAIL  actual contains NaN/Inf")
        return False
    if golden.numel() == 0:
        print(f"  [{name}] PASS  empty tensor (shape {tuple(golden.shape)})")
        return True
    err = rel_rmse(actual, golden)
    max_abs = (actual.float() - golden.float()).abs().max().item()
    ok = err <= threshold
    print(f"  [{name}] {'PASS' if ok else 'FAIL'}  rel_rmse={err:.3e}  "
          f"max_abs={max_abs:.3e}  threshold={threshold:.1e}")
    return ok
