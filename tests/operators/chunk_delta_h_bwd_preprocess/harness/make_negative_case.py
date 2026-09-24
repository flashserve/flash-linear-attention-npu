"""为反向用例生成（可能非法的）输入文件与 run_case 命令行。

反向用例只验证「算子是否按设计拦截 + 返回码是否符合预期」，因此不需要 CPU 标杆：这里按用例给的
（可能是非法的）shape/dtype 直接写原始字节，交给 run_case 调用 aclnn。

用例 id 与期望返回码来自 tests/op_cases/chunk_delta_h_bwd_preprocess.json 的 negative_cases；
每条用例的"非法参数"由 note/trigger 给出（例如 K=512、Hk=3/Hv=4、B=2、chunk_size=128、
g 为 [B,Hk,T]、gk 为 FP32、cu_seqlens 只有 1 项、T=0），下表把它们按 id 显式固化：

    neg_01_g_and_gk_both              g 与 gk 同时非空
    neg_02_k_too_large                K=512
    neg_03_hv_not_multiple_of_hk      Hk=3, Hv=4
    neg_04_dense_b_greater_than_one   B=2，无 cu_seqlens
    neg_05_varlen_b_greater_than_one  B=2，传 cu_seqlens
    neg_06_chunk_size_not_64          chunk_size=128
    neg_07_g_shape_mismatch           g 为 [B,Hk,T]
    neg_08_gk_dtype_fp32              gk 为 FP32
    neg_09_cu_seqlens_too_short       cu_seqlens 只有 1 项
    neg_10_empty_tensor               T=0

用法：python3 make_negative_case.py --dir <case_dir> <case_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "op_cases", "chunk_delta_h_bwd_preprocess.json"))
MODEL_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

# 合法基线（其余非法用例在此基础上改一个参数）
BASE = {
    "dtype": "bf16",
    "B": 1,
    "Hk": 4,
    "Hv": 4,
    "T": 256,
    "K": 128,
    "V": 128,
    "chunk_size": 64,
    "gate": "none",
}
ILLEGAL = {
    "neg_01_g_and_gk_both": {"gate": "both"},
    "neg_02_k_too_large": {"K": 512},
    "neg_03_hv_not_multiple_of_hk": {"Hk": 3, "Hv": 4},
    "neg_04_dense_b_greater_than_one": {"B": 2},
    "neg_05_varlen_b_greater_than_one": {"B": 2, "cu_seqlens": [0, 128, 256]},
    "neg_06_chunk_size_not_64": {"chunk_size": 128},
    # 注意：只有 Hk != Hv 时 [B,Hk,T] 才与 [B,Hv,T] 可区分；这里用 GVA（Hk=2, Hv=4）
    "neg_07_g_shape_mismatch": {"gate": "g", "g_head": "Hk", "Hk": 2, "Hv": 4},
    "neg_08_gk_dtype_fp32": {"gate": "gk", "gk_dtype": "fp32"},
    "neg_09_cu_seqlens_too_short": {"cu_seqlens": [0]},
    "neg_10_empty_tensor": {"T": 0},
}


def dump(t: torch.Tensor, path: str) -> None:
    t.detach().cpu().contiguous().view(torch.uint8).numpy().tofile(path)


def randn(shape, dtype_name: str, seed: int) -> torch.Tensor:
    if any(d == 0 for d in shape):  # T=0：空张量（算子应在 tiling 拦截）
        return torch.empty(shape, dtype=MODEL_DTYPES[dtype_name])
    torch.manual_seed(seed)
    return torch.randn(*shape).to(MODEL_DTYPES[dtype_name])


def rand01(shape, seed: int, scale: float = 0.05) -> torch.Tensor:
    if any(d == 0 for d in shape):
        return torch.empty(shape, dtype=torch.float32)
    torch.manual_seed(seed)
    return (torch.rand(*shape) * scale).to(torch.float32)


def build(case_id: str, out_dir: str) -> dict:
    spec = json.load(open(SPEC))
    cases = {c["id"]: c for c in spec["negative_cases"]}
    if case_id not in cases:
        print(f"unknown case id {case_id}; available: {', '.join(cases)}", file=sys.stderr)
        sys.exit(2)
    case = cases[case_id]
    a = dict(BASE)
    a.update(ILLEGAL[case_id])
    b, hk, hv, t = a["B"], a["Hk"], a["Hv"], a["T"]
    k_dim, v_dim, cs = a["K"], a["V"], a["chunk_size"]
    dt = a["dtype"]
    gate = a.get("gate", "none")
    os.makedirs(out_dir, exist_ok=True)

    dump(randn((b, hk, t, k_dim), dt, 11), os.path.join(out_dir, "q.bin"))
    dump(randn((b, hk, t, k_dim), dt, 12), os.path.join(out_dir, "k.bin"))
    dump(randn((b, hv, t, k_dim), dt, 13), os.path.join(out_dir, "w.bin"))
    dump(randn((b, hv, t, v_dim), dt, 14), os.path.join(out_dir, "do.bin"))
    dump(randn((b, hv, t, v_dim), dt, 15), os.path.join(out_dir, "dv.bin"))
    if gate in ("g", "both"):
        g_shape = (b, hk, t) if a.get("g_head") == "Hk" else (b, hv, t)
        if a.get("g_dtype") == "fp32":
            dump(rand01(g_shape, 16), os.path.join(out_dir, "g.bin"))
        else:
            # g 默认与 q/k 同 dtype（与 run_case 的 gateType 选择一致）
            dump(randn(g_shape, dt, 16), os.path.join(out_dir, "g.bin"))
    gk_fp32 = a.get("gk_dtype") == "fp32"
    if gate in ("gk", "both"):
        if gk_fp32:
            dump(rand01((b, hv, t, k_dim), 17), os.path.join(out_dir, "gk.bin"))
        else:
            dump(randn((b, hv, t, k_dim), dt, 17), os.path.join(out_dir, "gk.bin"))
    varlen = 0
    if a.get("cu_seqlens") is not None:
        torch.tensor(a["cu_seqlens"], dtype=torch.int64).numpy().tofile(os.path.join(out_dir, "cu_seqlens.bin"))
        varlen = 1

    gate_code = {"none": 0, "g": 1, "gk": 2, "both": 3}[gate]
    g_fp32 = 1 if (a.get("g_dtype") == "fp32" or gk_fp32) else 0
    g_mode = 1 if a.get("g_head") == "Hk" else 0
    dtype_code = 0 if dt == "bf16" else 1
    args = [out_dir, str(dtype_code), str(gate_code), str(g_fp32), str(b), str(hk), str(hv), str(t),
            str(k_dim), str(v_dim), str(cs), repr(float(k_dim ** -0.5)), str(varlen), str(g_mode)]
    with open(os.path.join(out_dir, "case.json"), "w") as f:
        json.dump({"case_id": case_id, "args": a, "expected_return_code": case["expected_return_code"],
                   "trigger": case["trigger"], "note": case["note"], "run_case": " ".join(["./run_case"] + args)},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "run_case_args.txt"), "w") as f:
        f.write(" ".join(args))
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("case_id", nargs="?")
    args = ap.parse_args()
    if args.all:
        for cid in ILLEGAL:
            build(cid, os.path.join(args.dir, cid))
            print("wrote", os.path.join(args.dir, cid))
        return
    if not args.case_id:
        ap.error("需要 case_id（或用 --all）")
    build(args.case_id, args.dir)


if __name__ == "__main__":
    main()
