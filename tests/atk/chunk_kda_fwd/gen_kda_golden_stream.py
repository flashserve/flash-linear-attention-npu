#!/usr/bin/env python3
"""内存受限复刻 ATK chunk_kda_fwd 的 CPU fp64 golden（流式按 head 输出）。

背景：ATK 金标 worker 在 T16384 export 例上峰值 RSS >32GiB，被 cgroup
memory.limit SIGKILL（WorkerLostError ×N）。本脚本逐 head 复刻
executor_chunk_kda_fwd._reference_model_parallel 的 fp64 数学（每个 torch
算子逐行同序调用，逐 head 流式），输出直接写 fp32 numpy memmap，峰值 RSS
仅数 GB。输入用同一 torch.Generator(seed) 流与同一 _normal_quantized 生成
（以 original dtype 紧凑存放，fp64 提升在取用时做，数值逐位不变）。

用法:
  venv/bin/python3 gen_kda_golden_stream.py --case-id 297 \
      --atk-json tests/atk/chunk_kda_fwd/atk_chunk_kda_fwd.json \
      --executor tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py \
      --out-dir /tmp/golden297 [--check-against <ATK 真金标目录>] [--keep-raw]
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import sys

import numpy as np
import torch


def load_executor(path):
    spec = importlib.util.spec_from_file_location("kda_executor", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kda_executor"] = mod  # dataclass 装饰器要求模块已注册
    spec.loader.exec_module(mod)
    return mod


def get_spec(atk_json, case_id):
    cases = json.load(open(atk_json))
    for case in cases:
        if case["id"] == case_id:
            for item in case["inputs"]:
                if item["name"] == "case_spec":
                    return json.loads(item["range_values"])
    raise SystemExit(f"case {case_id} not found in {atk_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", type=int, required=True)
    ap.add_argument("--atk-json", required=True)
    ap.add_argument("--executor", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--check-against", default=None,
                    help="与 ATK 真金标目录逐输出比对 max|diff|")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()

    ex = load_executor(args.executor)
    spec = get_spec(args.atk_json, args.case_id)

    batch = int(spec["B"]); total_t = int(spec["T"])
    h_num = int(spec["H"]); hv_num = int(spec["HV"])
    k_dim = int(spec["K"]); v_dim = int(spec["V"])
    chunk_size = int(spec["chunk_size"])
    layout = str(spec["layout"])
    assert layout == "BSND", f"仅实现 BSND（本套件全量 BSND），got {layout}"
    assert spec.get("data_profile") == "model_h96", "仅实现 model_h96 数据档案"
    assert not ex._as_bool(spec["initial_state"]), "initial_state 例未实现"

    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec["seed"]))

    q_orig = ex._DTYPES[str(spec["q_dtype"])]
    g_orig = ex._DTYPES[str(spec["g_dtype"])]
    beta_orig = ex._DTYPES[str(spec["beta_dtype"])]
    # 与 _prepare_inputs(high_precision=True) 完全同序消费 RNG；紧凑存放
    # （original dtype 即金标的量化目标，fp64 提升在取用时做，值恒等）。
    q = ex._normal_quantized((batch, total_t, h_num, k_dim), generator,
                             q_orig, q_orig, "cpu", std=float(spec["qk_scale"]),
                             l2_normalize=True)
    k = ex._normal_quantized((batch, total_t, h_num, k_dim), generator,
                             q_orig, q_orig, "cpu", std=float(spec["qk_scale"]),
                             l2_normalize=True)
    v = ex._normal_quantized((batch, total_t, hv_num, v_dim), generator,
                             q_orig, q_orig, "cpu", std=float(spec["v_scale"]))
    g = ex._normal_quantized((batch, total_t, hv_num, k_dim), generator,
                             g_orig, g_orig, "cpu", std=float(spec["gate_scale"]))
    beta = ex._normal_quantized((batch, total_t, hv_num), generator,
                                beta_orig, beta_orig, "cpu",
                                mean=float(spec["beta_bias"]),
                                std=float(spec["beta_scale"]), sigmoid=True)
    A_log = ex._normal_quantized((hv_num,), generator, torch.float32,
                                 torch.float32, "cpu",
                                 std=float(spec["a_log_scale"]))
    dt_bias = ex._normal_quantized((hv_num * k_dim,), generator, torch.float32,
                                   torch.float32, "cpu",
                                   mean=float(spec["dt_bias_mean"]),
                                   std=float(spec["dt_bias_scale"]))

    cu = ex._parse_cu(spec.get("cu_seqlens"))
    spans = ex._spans(batch, total_t, chunk_size, cu)
    seq_num = (len(cu) - 1) if cu is not None else batch
    total_chunks = len(spans)
    group = hv_num // h_num
    scale = float(spec["scale"])
    lower_bound = float(spec["lower_bound"])

    export_full = ex._as_bool(spec["disable_recompute"])
    export_h = export_full or ex._as_bool(spec["return_intermediate_states"])
    expose_gk = (not ex._as_bool(spec["use_gate_in_kernel"])) or export_full

    dt_bias_2d = dt_bias.view(hv_num, k_dim)
    lengths = {end - start for _, _, _, start, end in spans}
    strict_masks = {L: torch.ones((L, L), dtype=torch.bool).tril(-1) for L in lengths}
    eyes = {L: torch.eye(L, dtype=torch.float64) for L in lengths}

    work = args.work_dir or (args.out_dir.rstrip("/") + "_raw")
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    def mm(name, shape):
        return np.memmap(os.path.join(work, name), dtype=np.float32, mode="w+",
                         shape=tuple(shape))

    o_mm = mm("o", (batch, total_t, hv_num, v_dim))
    gk_mm = mm("gk", (batch, total_t, hv_num, k_dim))  # 保存时 permute(0,2,1,3)
    aqk_mm = mm("aqk", (batch, hv_num, total_t, chunk_size))
    akk_mm = mm("akk", (batch, hv_num, total_t, chunk_size))
    w_mm = mm("w", (batch, hv_num, total_t, k_dim)) if export_full else None
    u_mm = mm("u", (batch, hv_num, total_t, v_dim)) if export_full else None
    qg_mm = mm("qg", (batch, hv_num, total_t, k_dim)) if export_full else None
    kg_mm = mm("kg", (batch, hv_num, total_t, k_dim)) if export_full else None
    vn_mm = mm("vn", (batch, hv_num, total_t, v_dim)) if export_full else None
    h_mm = mm("h", (batch, total_chunks, hv_num, k_dim, v_dim)) if export_h else None

    nonfinite = 0
    for hv in range(hv_num):
        q_head = hv // group
        # ---- 逐 head 复刻 _gate_cumsum（fp64，元素级与逐 span cumsum 同序）----
        g_h = g[0, :, hv, :].to(torch.float64)
        raw = g_h + dt_bias_2d[hv].to(torch.float64)
        eig = torch.exp(A_log[hv].to(torch.float64))
        gate = lower_bound * torch.sigmoid(eig * raw)
        gk_h = torch.empty_like(gate)
        for _, _, _, start, end in spans:
            gk_h[start:end] = torch.cumsum(gate[start:end], dim=0) / math.log(2.0)

        state = torch.zeros((seq_num, k_dim, v_dim), dtype=torch.float64)
        for batch_id, seq_id, chunk_id, start, end in spans:
            length = end - start
            q_block = q[batch_id, start:end, q_head].to(torch.float64)
            k_block = k[batch_id, start:end, q_head].to(torch.float64)
            v_block = v[batch_id, start:end, hv].to(torch.float64)
            beta_block = beta[batch_id, start:end, hv].to(torch.float64)
            g_block = gk_h[start:end]

            qk, kk = ex._stable_causal_scores(
                q_block.unsqueeze(0), k_block.unsqueeze(0), g_block.unsqueeze(0),
                scale)
            qk = qk.squeeze(0); kk = kk.squeeze(0)
            lhs = kk.mul(beta_block[:, None]).masked_fill(~strict_masks[length], 0.0)
            inverse = torch.linalg.solve_triangular(
                lhs + eyes[length], eyes[length], upper=False)

            exp_g = torch.exp2(g_block)
            w_block = inverse @ (k_block * beta_block[:, None] * exp_g)
            u_block = inverse @ (v_block * beta_block[:, None])
            last_g = g_block[-1]
            qg_block = q_block * exp_g
            kg_block = k_block * torch.exp2(last_g[None, :] - g_block)
            previous = state[seq_id].clone()
            v_new_block = u_block - w_block @ previous
            state[seq_id] = (torch.exp2(last_g)[:, None] * previous
                             + kg_block.T @ v_new_block)
            out_block = qg_block @ previous * scale + qk @ v_new_block

            o_mm[batch_id, start:end, hv] = out_block.to(torch.float32).numpy()
            gk_mm[batch_id, start:end, hv] = gk_h[start:end].to(torch.float32).numpy()
            aqk_mm[batch_id, hv, start:end, :length] = qk.to(torch.float32).numpy()
            akk_mm[batch_id, hv, start:end, :length] = inverse.to(torch.float32).numpy()
            if export_full:
                w_mm[batch_id, hv, start:end] = w_block.to(torch.float32).numpy()
                u_mm[batch_id, hv, start:end] = u_block.to(torch.float32).numpy()
                qg_mm[batch_id, hv, start:end] = qg_block.to(torch.float32).numpy()
                kg_mm[batch_id, hv, start:end] = kg_block.to(torch.float32).numpy()
                vn_mm[batch_id, hv, start:end] = v_new_block.to(torch.float32).numpy()
            if export_h:
                h_mm[batch_id, chunk_id, hv] = previous.to(torch.float32).numpy()
        if hv % 8 == 0 or hv == hv_num - 1:
            print(f"[gen] head {hv + 1}/{hv_num} done", flush=True)
    for arr in (o_mm, gk_mm, aqk_mm, akk_mm, w_mm, u_mm, qg_mm, kg_mm, vn_mm, h_mm):
        if arr is not None:
            arr.flush()

    # ---- 按可见输出策略（_apply_output_policy + fp32 转换）落盘 .pt ----
    named = [("attn_out", "o", o_mm, False)]
    if expose_gk:
        named.append(("gk", "gk", gk_mm, True))
    named += [("Aqk", "aqk", aqk_mm, False), ("Akk", "akk", akk_mm, False)]
    if export_full:
        named += [("w", "w", w_mm, False), ("u", "u", u_mm, False),
                  ("qg", "qg", qg_mm, False), ("kg", "kg", kg_mm, False),
                  ("v_new", "vn", vn_mm, False)]
    if export_h:
        state_v_first = ex._as_bool(spec["state_v_first"])
        named.append(("h", "h", h_mm, "transpose" if state_v_first else False))

    info = []
    for idx, (name, raw, arr, view) in enumerate(named):
        tensor = torch.from_numpy(np.memmap(
            os.path.join(work, raw), dtype=np.float32, mode="r",
            shape=arr.shape))
        if view is True:
            tensor = tensor.permute(0, 2, 1, 3)      # gk_out
        elif view == "transpose":
            tensor = tensor.transpose(-1, -2)        # h_out(state_v_first)
        nonfinite += int((~torch.isfinite(tensor)).sum().item())
        out_path = os.path.join(args.out_dir, f"output_{idx}.pt")
        torch.save(tensor, out_path)
        info.append({"dtype": str(tensor.dtype),
                     "shape": list(tensor.shape), "stride": list(tensor.stride())})
        print(f"[gen] output_{idx}.pt <- {name} {tuple(tensor.shape)}", flush=True)
    with open(os.path.join(args.out_dir, "output_info.json"), "w") as f:
        json.dump(info, f)
    print(f"[gen] nonfinite elements: {nonfinite}")

    if args.check_against:
        worst = 0.0
        for idx in range(len(named)):
            mine = torch.load(os.path.join(args.out_dir, f"output_{idx}.pt"),
                              map_location="cpu", weights_only=False)
            ref_path = os.path.join(args.check_against, f"output_{idx}.pt")
            if not os.path.exists(ref_path) or os.path.getsize(ref_path) == 0:
                print(f"[chk] output_{idx}: 参考件缺失/0字节（ATK 侧残留），跳过")
                continue
            ref = torch.load(ref_path, map_location="cpu", weights_only=False)
            if tuple(mine.shape) != tuple(ref.shape):
                print(f"[chk] output_{idx}: 形状不一致 {tuple(mine.shape)} vs {tuple(ref.shape)}")
                worst = float("inf")
                continue
            diff = (mine.float() - ref.float()).abs().max().item()
            worst = max(worst, diff)
            print(f"[chk] output_{idx}: max|diff| = {diff:.3e}")
        print(f"[chk] 最坏 max|diff| = {worst:.3e}  （fp64→fp32 双路应≈0）")

    if not args.keep_raw:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
