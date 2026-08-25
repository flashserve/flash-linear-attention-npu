"""fused_recurrent_rwkv8 的 NPU accuracy backend runner。

由 tests/operators/fused_recurrent_rwkv8/accuracy/test_fused_recurrent_rwkv8.py
以 subprocess 拉起（FLA_NPU_RUN_OPERATOR_TESTS=1 门控）。每个 manifest accuracy
正例 case：按 seed 结构化造数（z=-kk、b=kk*a，kk 为 L2 归一化——无约束 randn
会让 delta-rule 状态指数爆炸）→ NPU 上跑 fla_npu ctypes wrapper → CPU 上跑
tests/reference 的 PyTorch golden → rel-RMSE 对拍（阈值取 manifest tolerance）。

环境变量：
    FLA_NPU_CASE_MANIFEST  manifest JSON 路径（缺省用 tests/op_cases/ 下标准位置）
    FLA_NPU_CASE_IDS       逗号分隔的 case id 清单（缺省跑全部 accuracy 正例）
    TEST_DEVICE_ID         NPU 卡号（缺省 0）
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - 必须在 NPU 机器上跑
    raise SystemExit(f"torch_npu unavailable: {exc}")

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fla_npu.ops import ascendc as fla_ascendc  # noqa: E402
from tests.reference.fused_recurrent_rwkv8_reference import (  # noqa: E402
    fused_recurrent_rwkv8_reference,
)


def _rel_rmse(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.double() - ref.double()).pow(2).sum().item()
    den = ref.double().pow(2).sum().item()
    if den == 0.0:
        return 0.0 if diff == 0.0 else 1e30
    return math.sqrt(diff / den)


def _make_inputs(case: dict, device: torch.device):
    """结构化造数，配方与仓外 ascendc/scripts/gen_data.py 一致。"""
    s = case["shape"]
    B, T, H, N = (int(s[k]) for k in ("B", "T", "H", "N"))
    K = int(s.get("K", N))
    V = int(s.get("V", N))
    torch.manual_seed(int(case["seed"]))
    q = torch.randn((B, H, T, K), dtype=torch.float32)
    w = -torch.rand((B, H, T, K), dtype=torch.float32) * 2.0 - 0.1
    k = torch.randn((B, H, T, K), dtype=torch.float32)
    v = torch.randn((B, H, T, V), dtype=torch.float32)
    kk = torch.nn.functional.normalize(torch.randn((B, H, T, K), dtype=torch.float32), dim=-1)
    z = -kk
    b = kk * torch.randn((B, H, T, K), dtype=torch.float32)
    initial_state = None
    if case.get("optional_inputs", {}).get("initial_state") is not None:
        initial_state = torch.randn((B, H, K, V), dtype=torch.float32)
    cpu_inputs = dict(q=q, w=w, k=k, v=v, z=z, b=b, initial_state=initial_state)
    npu_inputs = {
        name: (tensor.to(device) if tensor is not None else None)
        for name, tensor in cpu_inputs.items()
    }
    return cpu_inputs, npu_inputs


def _run_case(case: dict, device: torch.device, tol: float) -> dict:
    attrs = case["attrs"]
    chunk_len = int(attrs.get("chunk_len", 16))
    output_s = bool(attrs.get("output_chunk_state", False))
    output_sa = bool(attrs.get("output_sa", False))
    scale = float(attrs.get("scale", 1.0))
    reverse = bool(attrs.get("reverse", False))

    cpu_inputs, npu_inputs = _make_inputs(case, device)

    o_npu, s_npu, sa_npu = fla_ascendc.fused_recurrent_rwkv8(
        npu_inputs["q"], npu_inputs["w"], npu_inputs["k"],
        npu_inputs["v"], npu_inputs["z"], npu_inputs["b"],
        scale=scale,
        initial_state=npu_inputs["initial_state"],
        reverse=reverse,
        output_chunk_state=output_s,
        output_sa=output_sa,
        chunk_len=chunk_len,
    )
    torch.npu.synchronize()

    ref = fused_recurrent_rwkv8_reference(
        cpu_inputs["q"], cpu_inputs["w"], cpu_inputs["k"],
        cpu_inputs["v"], cpu_inputs["z"], cpu_inputs["b"],
        scale=scale,
        initial_state=cpu_inputs["initial_state"],
        reverse=reverse,
        output_chunk_state=output_s,
        output_sa=output_sa,
        chunk_len=chunk_len,
    )

    errors = {}
    for name in case["expect"]["compare_outputs"]:
        got = {"o": o_npu, "s": s_npu, "sa": sa_npu}[name]
        want = {"o": ref.o, "s": ref.s, "sa": ref.sa}[name]
        if want is None or got is None:
            raise AssertionError(f"{case['id']}: {name} expected but got None (got={got is None}, want={want is None})")
        got_cpu = got.float().cpu()
        if not bool(torch.isfinite(got_cpu).all().item()):
            raise AssertionError(f"{case['id']}: {name} contains NaN or Inf")
        errors[name] = _rel_rmse(got_cpu, want.float())
    return errors


def main() -> int:
    manifest_path = os.environ.get(
        "FLA_NPU_CASE_MANIFEST",
        str(ROOT / "tests" / "op_cases" / "fused_recurrent_rwkv8.json"),
    )
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    case_ids = os.environ.get("FLA_NPU_CASE_IDS", "")
    cases = [
        case
        for case in manifest["cases"]
        if "accuracy" in case["tags"]
        and case.get("expect", {}).get("return_code") == "ACLNN_SUCCESS"
    ]
    if case_ids:
        wanted = set(case_ids.split(","))
        cases = [case for case in cases if case["id"] in wanted]
    if not cases:
        print("[FATAL] no accuracy cases selected")
        return 1

    tol = float(manifest.get("tolerance", {}).get("float32", {}).get("rel_rmse", 0.002))
    device = torch.device(f"npu:{int(os.environ.get('TEST_DEVICE_ID', '0'))}")
    torch.npu.set_device(device)

    failed = 0
    for case in cases:
        try:
            errors = _run_case(case, device, tol)
        except Exception as exc:
            print(f"[FAIL] {case['id']}: {exc}")
            failed += 1
            continue
        bad = {name: err for name, err in errors.items() if err > tol}
        detail = "  ".join(f"{name}={err:.3e}" for name, err in errors.items())
        if bad:
            print(f"[FAIL] {case['id']}  {detail}  (tol={tol:.1e})")
            failed += 1
        else:
            print(f"[PASS] {case['id']}  {detail}")

    total = len(cases)
    print(f"fused_recurrent_rwkv8 accuracy: {total - failed}/{total} PASS (rel-RMSE tol={tol:.1e})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
