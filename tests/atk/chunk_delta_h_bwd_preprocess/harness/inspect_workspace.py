"""逐平面核对 cp 反向预处理的 workspace 内容（配合 run_case 落下的 workspace.bin）。

做法：按 host tiling 公式复算各平面偏移（用户 workspace 起点固定为 16 MiB），
再用 case 目录里的输入文件独立算出期望值，逐平面给出 max_abs / 最差位置。
当前实现按 **NT=1**（单 chunk）推导期望值：首轮 dH_old = 0，因此
    dVpre = 0, T1 = Wᵀ@K̄, dVhat = -dv_local, qterm = Q̄sᵀ@do, wterm = Wᵀ@dVhat,
    dH = decayK⊙0 + (qterm + wterm), P_c = diag(decayK) - T1, P_new = P_c @ I = P_c
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="case 目录（需含 workspace.bin 与各 .bin 输入）")
    ap.add_argument("--base", type=int, default=16 * 1024 * 1024, help="用户 workspace 起点（默认 16 MiB）")
    ap.add_argument("--wg", type=int, default=0, help="工作组数（默认为 0，取 Hv）")
    args = ap.parse_args()
    d = args.dir
    with open(os.path.join(d, "case.json")) as f:
        meta = json.load(f)
    hv, hk = meta["args"]["Hv"], meta["args"]["Hk"]
    t_len, k_dim, v_dim = meta["args"]["T"], meta["args"]["K"], meta["args"]["V"]
    chunk = meta["args"]["chunk_size"]
    scale = meta["scale"]
    assert t_len == chunk, "inspect_workspace 目前只支持 NT=1（T == chunk_size）"

    ws = np.fromfile(os.path.join(d, "workspace.bin"), dtype=np.uint8)
    base = args.base

    def load(name: str) -> np.ndarray:
        return bf16_to_f32(np.fromfile(os.path.join(d, name + ".bin"), dtype=np.uint8))

    q = load("q").reshape(hk, chunk, k_dim)
    k = load("k").reshape(hk, chunk, k_dim)
    w = load("w").reshape(hv, chunk, k_dim)
    do = load("do").reshape(hv, chunk, v_dim)
    dv = load("dv").reshape(hv, chunk, v_dim)

    def f32(off: int, n: int) -> np.ndarray:
        return np.frombuffer(ws[base + off : base + off + n * 4].tobytes(), dtype=np.float32)

    def bfp(off: int, n: int) -> np.ndarray:
        return bf16_to_f32(np.frombuffer(ws[base + off : base + off + n * 2].tobytes(), dtype=np.uint16))

    # 每个平面按工作组复制 wg 份；wg == blockDim == min(Hv, 核数)，测试用例 Hv 很小，通常 wg == hv。
    wg = args.wg if args.wg else hv

    # host 侧每个平面总字节 = AlignUp(wg * 单工作组字节, 512)，单工作组字节 = 每工作组份数 * 单份字节。
    # kernel 用 offset + sliceIndex * (planeBytes / sliceCount) 寻址，因此这里必须按同一份数还原。
    def align(x: int) -> int:
        return ((x + 511) // 512) * 512

    def plane(wg_slab: int, slices_per_wg: int, off: int):
        """返回 (单份字节数, 单工作组切片字节数, 下一个平面偏移)。"""
        slab = align(wg * wg_slab)
        return slab // (wg * slices_per_wg), slab // wg, off + slab

    slot_bytes = align(chunk * (3 * k_dim + v_dim) * 2 + k_dim * 4)
    off = 512 + wg * slot_bytes
    dh_unit, dh_slab, off = plane(2 * k_dim * v_dim * 4, 2, off)
    dh_off = off - dh_slab * wg
    dhbf_unit, dhbf_slab, off = plane(k_dim * v_dim * 2, 1, off)
    dhbf_off = off - dhbf_slab * wg
    dvpre_unit, dvpre_slab, off = plane(chunk * v_dim * 4, 1, off)
    dvpre_off = off - dvpre_slab * wg
    dvhat_unit, dvhat_slab, off = plane(chunk * v_dim * 2, 1, off)
    dvhat_off = off - dvhat_slab * wg
    qterm_unit, qterm_slab, off = plane(k_dim * v_dim * 4, 1, off)
    qterm_off = off - qterm_slab * wg
    wterm_unit, wterm_slab, off = plane(k_dim * v_dim * 4, 1, off)
    wterm_off = off - wterm_slab * wg
    t1_unit, t1_slab, off = plane(k_dim * k_dim * 4, 1, off)
    t1_off = off - t1_slab * wg
    pc_unit, pc_slab, off = plane(k_dim * k_dim * 2, 1, off)
    pc_off = off - pc_slab * wg
    p_unit, p_slab, off = plane(2 * k_dim * k_dim * 4, 2, off)
    p_off = off - p_slab * wg
    pbf_unit, pbf_slab, off = plane(k_dim * k_dim * 2, 1, off)
    pbf_off = off - pbf_slab * wg

    slot0 = 512
    qb_off, kb_off, wb_off, dob_off, dec_off = slot0, slot0 + chunk * k_dim * 2, \
        slot0 + 2 * chunk * k_dim * 2, slot0 + 3 * chunk * k_dim * 2, slot0 + 3 * chunk * k_dim * 2 + chunk * v_dim * 2

    def rep(name: str, got: np.ndarray, exp: np.ndarray) -> None:
        g = np.asarray(got, dtype=np.float64).reshape(exp.shape)
        diff = np.abs(g - exp)
        where = np.unravel_index(int(np.argmax(diff)), exp.shape)
        print(f"  {name:10s} max_abs={diff.max():.4e} mean={diff.mean():.4e} worst@{where} "
              f"got={g[where]:.6g} exp={exp[where]:.6g}")

    print(f"case={d} hv={hv} hk={hk} T={t_len} K={k_dim} V={v_dim} chunk={chunk} scale={scale:g}")
    for h in range(hv):
        kbar = bfp(kb_off + h * slot_bytes, chunk * k_dim)
        qbar = bfp(qb_off + h * slot_bytes, chunk * k_dim)
        wsl = bfp(wb_off + h * slot_bytes, chunk * k_dim)
        dos = bfp(dob_off + h * slot_bytes, chunk * v_dim)
        dec = f32(dec_off + h * slot_bytes, k_dim)
        print(f"[hv={h}] slot 面:")
        rep("Qbar", qbar, scale * q[h])
        rep("Kbar", kbar, k[h])
        rep("W", wsl, w[h])
        rep("do", dos, do[h])
        rep("decayK", dec, np.ones(k_dim))
        print(f"[hv={h}] 中间面（NT=1，本工作组第 {h} 份）:")
        rep("dVpre", f32(dvpre_off + h * dvpre_slab, chunk * v_dim), np.zeros((chunk, v_dim)))
        rep("T1", f32(t1_off + h * t1_slab, k_dim * k_dim), w[h].T @ k[h])
        rep("dVhat", bfp(dvhat_off + h * dvhat_slab, chunk * v_dim), -dv[h])
        rep("qterm", f32(qterm_off + h * qterm_slab, k_dim * v_dim), (scale * q[h]).T @ do[h])
        rep("wterm", f32(wterm_off + h * wterm_slab, k_dim * v_dim), w[h].T @ (-dv[h]))
        rep("Pc", bfp(pc_off + h * pc_slab, k_dim * k_dim), np.eye(k_dim) - w[h].T @ k[h])
        rep("PBf", bfp(pbf_off + h * pbf_slab, k_dim * k_dim), np.eye(k_dim))
        # P 的 parity 语义：InitState 写 parity = chunkNum % 2；末 chunk( chunkIdx=0, parity=0 ) 写 parity 0。
        # NT=1 时 P(0) = P_c @ I = I - W^T K̄，P(1) = I（初值）。
        rep("P0", f32(p_off + h * p_slab, k_dim * k_dim), np.eye(k_dim) - w[h].T @ k[h])
        rep("P1", f32(p_off + h * p_slab + p_unit, k_dim * k_dim), np.eye(k_dim))
        # dhBf 是 V4 落下的 dH 新值（模型 dtype）：NT=1 时 = qterm + wterm（上一轮 dH = 0）
        rep("dhBf", bfp(dhbf_off + h * dhbf_slab, k_dim * v_dim), (scale * q[h]).T @ do[h] + w[h].T @ (-dv[h]))


if __name__ == "__main__":
    main()
