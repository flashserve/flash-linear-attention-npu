"""受控实验：把一个既有 case 的 W 改成"仅第 0 行全 1、其余全 0"，并据此重算 CPU 标杆。

在 NT=1、无门控下：T1 = Wᵀ@K̄ ⇒ T1 只有第 0 行非零，且等于 K̄ 的第 0 行；
P = I - T1，因此 P 的第 0 行 = e_0 - K̄[0,:]，其余行为单位阵行。用它可以直接判断
ColumnMajor 左操作数（Wᵀ）在 Catlass 里的实际语义。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from reference import preprocess_reference  # noqa: E402


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--head", type=int, default=0, help="被改写的 value head")
    args = ap.parse_args()
    d = args.dir
    meta = json.load(open(os.path.join(d, "case.json")))
    a = meta["args"]
    chunk, k_dim, v_dim = a["chunk_size"], a["K"], a["V"]
    hv, hk = a["Hv"], a["Hk"]

    def load_f32(name: str, shape) -> torch.Tensor:
        raw = np.fromfile(os.path.join(d, name + ".bin"), dtype=np.uint8)
        return torch.from_numpy(bf16_to_f32(raw).reshape(shape).copy())

    def dump_bf16(t: torch.Tensor, name: str) -> None:
        t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tofile(os.path.join(d, name + ".bin"))

    # reference 使用 [B, H, T, D]；文件里存的是不带 B 的 rank-local buffer
    q = load_f32("q", (hk, chunk, k_dim)).unsqueeze(0)
    k = load_f32("k", (hk, chunk, k_dim)).unsqueeze(0)
    w = load_f32("w", (hv, chunk, k_dim)).unsqueeze(0)
    do = load_f32("do", (hv, chunk, v_dim)).unsqueeze(0)
    dv = load_f32("dv", (hv, chunk, v_dim)).unsqueeze(0)
    w[0, args.head] = 0.0
    w[0, args.head, 0, :] = 1.0
    dump_bf16(w.squeeze(0), "w")

    bos, eos = 0, chunk
    if a.get("cu_seqlens"):
        bos, eos = a["cu_seqlens"][0], a["cu_seqlens"][1]
    ref = preprocess_reference(q, k, w, do, dv, None, None, scale=meta["scale"], chunk_size=chunk, bos=bos, eos=eos)
    ref.numpy().astype("float32").tofile(os.path.join(d, "expected_dhm.bin"))
    print(f"ctrl W(head={args.head}) 已改写：row0=1 其余 0；expected_dhm.bin 已重算")


if __name__ == "__main__":
    main()
