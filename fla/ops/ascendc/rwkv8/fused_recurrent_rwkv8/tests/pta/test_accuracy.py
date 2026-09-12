"""fused_recurrent_rwkv8 (WKV7) PTA 精度测试。

NPU 输出对拍 CPU 金标（golden.py），用例取自上级目录 cases.py
（正例全跑 + 负例校验拒绝路径）。plain script，非 pytest：

    python test_accuracy.py            # 默认 npu:0
    RWKV8_PTA_DEVICE=npu:1 python test_accuracy.py
"""
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401  注册 NPU 后端

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden import fused_recurrent_rwkv8_golden
from utils import compare_rel_rmse
from cases import CASES, REL_RMSE_THRESHOLD

from fla_npu.ops import ascendc as ascendc_ops

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def make_inputs(case):
    """结构化造数（与仓外 ascendc/scripts/gen_data.py 同配方）：
    q/k/v ~ randn；w = -rand*2-0.1（log 域衰减，保证 decay ∈ (0,1)）；
    kk = L2normalize(randn)，z = -kk，b = kk * randn。
    禁止无约束 randn 造 z/b（delta-rule 状态会指数爆炸）。
    """
    torch.manual_seed(int(case["seed"]))
    B, H, T, K, V = case["B"], case["H"], case["T"], case["K"], case["V"]
    dtype = DTYPES[case["dtype"]]

    q = torch.randn(B, H, T, K, dtype=torch.float32)
    w = -torch.rand(B, H, T, K, dtype=torch.float32) * 2.0 - 0.1
    k = torch.randn(B, H, T, K, dtype=torch.float32)
    v = torch.randn(B, H, T, V, dtype=torch.float32)
    kk = torch.nn.functional.normalize(torch.randn(B, H, T, K), p=2, dim=-1)
    z = -kk
    b = kk * torch.randn(B, H, T, K)

    inputs = [x.to(dtype) for x in (q, w, k, v, z, b)]
    initial_state = None
    if case["initial_state"]:
        initial_state = torch.randn(B, H, K, V, dtype=torch.float32)
    return inputs, initial_state


def run_golden(case, inputs, initial_state):
    return fused_recurrent_rwkv8_golden(
        *inputs,
        scale=case["scale"],
        initial_state=initial_state,
        output_chunk_state=case["output_chunk_state"],
        output_sa=case["output_sa"],
        chunk_len=case["chunk_len"],
    )


def run_npu(case, inputs, initial_state, device):
    npu_inputs = [x.to(device) for x in inputs]
    init_npu = initial_state.to(device) if initial_state is not None else None
    out = ascendc_ops.fused_recurrent_rwkv8(
        *npu_inputs,
        scale=case["scale"],
        initial_state=init_npu,
        output_chunk_state=case["output_chunk_state"],
        output_sa=case["output_sa"],
        chunk_len=case["chunk_len"],
    )
    torch.npu.synchronize()
    return out  # (o, s, sa)，s/sa 关闭时为 None


def run_positive_case(case, device):
    print(f"\n{'=' * 60}")
    print(f"CASE: {case['id']}  B{case['B']} H{case['H']} T{case['T']} "
          f"K{case['K']} V{case['V']} {case['dtype']} "
          f"scale={case['scale']} chunk_len={case['chunk_len']} "
          f"init={case['initial_state']} seed={case['seed']}")
    inputs, initial_state = make_inputs(case)
    golden = run_golden(case, inputs, initial_state)
    npu_out = run_npu(case, inputs, initial_state, device)

    threshold = REL_RMSE_THRESHOLD[case["dtype"]]
    results = []
    for name, g, a in zip(("o", "s", "sa"), (golden.o, golden.s, golden.sa), npu_out):
        if name not in case["compare_outputs"]:
            continue
        assert a is not None and g is not None, f"{name} 应产出但为 None"
        results.append(compare_rel_rmse(a.cpu(), g, name, threshold))
    ok = all(results)
    print(f"  >> {case['id']}: {'PASS' if ok else 'FAIL'}")
    return ok


def run_negative_case(case, device):
    """负例：构造非法输入，wrapper 必须抛错且信息命中关键词。"""
    print(f"\n{'=' * 60}")
    print(f"CASE: {case['id']} (negative, expect error matching {case['negative']!r})")
    inputs, initial_state = make_inputs(case)
    q, w, k, v, z, b = inputs

    if case["id"] == "negative_shape_mismatch":
        w = w[:, :, :, : w.shape[-1] // 2].contiguous()  # k 侧 shape 不一致
    elif case["id"] == "negative_bad_init_shape":
        initial_state = torch.randn(case["B"], case["H"], case["K"], case["V"] * 2,
                                    dtype=torch.float32)

    try:
        ascendc_ops.fused_recurrent_rwkv8(
            q.to(device), w.to(device), k.to(device), v.to(device),
            z.to(device), b.to(device),
            scale=case["scale"],
            initial_state=initial_state.to(device) if initial_state is not None else None,
            output_chunk_state=case["output_chunk_state"],
            output_sa=case["output_sa"],
            chunk_len=case["chunk_len"],
        )
    except Exception as exc:
        hit = case["negative"] in str(exc)
        print(f"  raised {type(exc).__name__}: {exc}")
        print(f"  >> {case['id']}: {'PASS' if hit else 'FAIL (错误信息未命中关键词)'}")
        return hit
    print(f"  >> {case['id']}: FAIL (未抛异常)")
    return False


def main():
    device = torch.device(os.environ.get("RWKV8_PTA_DEVICE", "npu:0"))
    torch.npu.set_device(device)
    print(f"Using device: {device}")

    results = {}
    for case in CASES:
        if case["negative"] is None:
            results[case["id"]] = run_positive_case(case, device)
        else:
            results[case["id"]] = run_negative_case(case, device)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for case_id, ok in results.items():
        print(f"  {case_id}: {'PASS' if ok else 'FAIL'}")
    passed = sum(results.values())
    print(f"\n  {passed}/{len(results)} passed")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
