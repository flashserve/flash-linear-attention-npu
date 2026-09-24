"""为 chunk_delta_h_bwd_preprocess 生成 aclnn 取数程序的输入文件与 CPU 标杆。

生成的 <dir> 内容：
  q.bin k.bin w.bin do.bin dv.bin [g.bin|gk.bin] [cu_seqlens.bin]  按 dtype/shape 的原始字节
  expected_dhm.bin    CPU 标杆（FP32，[Hv, K, V+K]）
  case.json           本次用例的参数（含 run_case 的命令行）

输入生成固定随机种子；NPU 侧的 dtype 是 bf16/fp16，标杆用同一份存下来的数值（cast 回 FP32）计算，
保证两侧看到的是同一条数据。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from reference import preprocess_reference  # noqa: E402

MODEL_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def dump(tensor: torch.Tensor, path: str) -> None:
    tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=list(MODEL_DTYPES))
    ap.add_argument("--gate", default="g", choices=["none", "g", "gk"])
    ap.add_argument("--g-dtype", default="model", choices=["model", "fp32"])
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--Hk", type=int, default=4)
    ap.add_argument("--Hv", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--cu-seqlens", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero", default="", help="逗号分隔的输入名（q,k,w,do,dv,g），置零以逐项隔离")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    torch.manual_seed(args.seed)
    model = MODEL_DTYPES[args.dtype]
    gate_dtype = torch.float32 if (args.gate == "g" and args.g_dtype == "fp32") else model
    scale = args.scale if args.scale is not None else args.K ** -0.5

    q = torch.randn(args.B, args.Hk, args.T, args.K, dtype=torch.float32)
    k = torch.randn(args.B, args.Hk, args.T, args.K, dtype=torch.float32)
    w = (torch.randn(args.B, args.Hv, args.T, args.K, dtype=torch.float32) * 0.1)
    do = torch.randn(args.B, args.Hv, args.T, args.V, dtype=torch.float32)
    dv = (torch.randn(args.B, args.Hv, args.T, args.V, dtype=torch.float32) * 0.1)
    g = gk = None
    if args.gate == "g":
        g = torch.cumsum(-torch.rand(args.B, args.Hv, args.T, dtype=torch.float32) * 0.05, dim=-1)
    elif args.gate == "gk":
        gk = torch.cumsum(-torch.rand(args.B, args.Hv, args.T, args.K, dtype=torch.float32) * 0.05, dim=-2)
    for name in [z.strip() for z in args.zero.split(",") if z.strip()]:
        if name == "q":
            q.zero_()
        elif name == "k":
            k.zero_()
        elif name == "w":
            w.zero_()
        elif name == "do":
            do.zero_()
        elif name == "dv":
            dv.zero_()
        elif name in ("g", "gk"):
            if g is not None:
                g.zero_()
            if gk is not None:
                gk.zero_()

    # 落盘（模型 dtype），标杆用落盘后的数值
    qs, ks, ws, dos, dvs = (x.to(model) for x in (q, k, w, do, dv))
    gs = g.to(gate_dtype) if g is not None else None
    gks = gk.to(model) if gk is not None else None
    dump(qs, os.path.join(args.dir, "q.bin"))
    dump(ks, os.path.join(args.dir, "k.bin"))
    dump(ws, os.path.join(args.dir, "w.bin"))
    dump(dos, os.path.join(args.dir, "do.bin"))
    dump(dvs, os.path.join(args.dir, "dv.bin"))
    if gs is not None:
        dump(gs, os.path.join(args.dir, "g.bin"))
    if gks is not None:
        dump(gks, os.path.join(args.dir, "gk.bin"))
    bos, eos = 0, args.T
    if args.cu_seqlens:
        cu = torch.tensor(args.cu_seqlens, dtype=torch.int64)
        cu.numpy().tofile(os.path.join(args.dir, "cu_seqlens.bin"))
        bos, eos = args.cu_seqlens[0], args.cu_seqlens[1]

    dhm = preprocess_reference(
        qs.float(), ks.float(), ws.float(), dos.float(), dvs.float(),
        None if gs is None else gs.float(), None if gks is None else gks.float(),
        scale=scale, chunk_size=args.chunk_size, bos=bos, eos=eos,
    )
    dhm.numpy().astype("float32").tofile(os.path.join(args.dir, "expected_dhm.bin"))

    dtype_code = 0 if args.dtype == "bf16" else 1
    gate_code = {"none": 0, "g": 1, "gk": 2}[args.gate]
    cmd = [
        "./run_case", args.dir, str(dtype_code), str(gate_code),
        "1" if (args.gate == "g" and args.g_dtype == "fp32") else "0",
        str(args.B), str(args.Hk), str(args.Hv), str(args.T), str(args.K), str(args.V),
        str(args.chunk_size), repr(scale), "1" if args.cu_seqlens else "0",
    ]
    meta = {"args": vars(args), "scale": scale, "run_case": " ".join(cmd),
            "dhm_shape": [args.Hv, args.K, args.V + args.K]}
    with open(os.path.join(args.dir, "case.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.dir, "run_case_args.txt"), "w") as f:
        f.write(" ".join(cmd[1:]))  # 去掉 "./run_case"
    print("case written to", args.dir)
    print("run:", " ".join(cmd))


if __name__ == "__main__":
    main()
