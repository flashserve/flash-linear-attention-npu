# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""bwd_dhu GVA 双标杆测试：随机输入直接测，无需 example dump。"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import ct
from fla_npu.ops import ascendc as ascendc_ops

import torch

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("test_bwd_dhu_golden", os.path.join(_SCRIPT_DIR, "test_bwd_dhu.py"))
_golden_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden_mod)

chunk_gated_delta_rule_bwd_dhu_cpu = _golden_mod.chunk_gated_delta_rule_bwd_dhu_cpu
create_bwd_dhu_random_inputs = _golden_mod.create_bwd_dhu_random_inputs
effective_scale = _golden_mod.effective_scale
generate_cu_seqlens = _golden_mod.generate_cu_seqlens
prepare_chunk_indices = _golden_mod.prepare_chunk_indices
scale_for_compute_dtype = _golden_mod.scale_for_compute_dtype

DEFAULT_OUT_DIR = os.path.join(_SCRIPT_DIR, "bwd_dhu_out")


@dataclass
class BwdDhuCase:
    name: str
    batch: int
    k_h: int
    v_h: int
    tokens: int
    v_dim: int = 256
    k_dim: int = 128
    chunk_size: int = 64
    varlen: bool = True
    cu_seqlens_len: Optional[int] = None
    dtype: str = "bf16"
    supported: bool = True
    skip_reason: str = ""

    def ktype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bf16" else torch.float16

    def gtype(self) -> torch.dtype:
        return torch.float32


# 覆盖泛化表中的 fixed/varlen、V=128/256、chunk=64/128 和 MHA/GVA 组合。
CASES = [
    BwdDhuCase("smoke_varlen_t256_v256", 1, 16, 32, 256, chunk_size=64, cu_seqlens_len=5),
    BwdDhuCase("smoke_fixed_mha_t257_v128", 1, 4, 4, 257, v_dim=128, chunk_size=128,
               varlen=False),
    BwdDhuCase("general_fixed_b2_h32_t8192_v128", 2, 32, 32, 8192, v_dim=128, chunk_size=64,
               varlen=False),
    BwdDhuCase("fixed_b176_t24_v256", 176, 2, 64, 24, chunk_size=64, varlen=False),
    BwdDhuCase("general_fixed_b711_t196_v128", 711, 4, 32, 196, v_dim=128, chunk_size=128,
               varlen=False),
    BwdDhuCase("fixed_b16_t2048_v256", 16, 21, 63, 2048, chunk_size=64, varlen=False),
    BwdDhuCase("smoke_fixed_t4096_v256", 1, 16, 32, 4096, chunk_size=64, varlen=False),
    BwdDhuCase("general_varlen_t7178_v128_cu17", 1, 4, 32, 7178, v_dim=128, chunk_size=128,
               cu_seqlens_len=17),
    BwdDhuCase("general_varlen_t8999_v256_cu13", 1, 16, 48, 8999, chunk_size=128,
               cu_seqlens_len=13),
    BwdDhuCase("general_varlen_t16387_v256_cu667", 1, 16, 48, 16387, chunk_size=64,
               cu_seqlens_len=667),
    BwdDhuCase("general_varlen_t36621_v128_cu668", 1, 16, 32, 36621, v_dim=128, chunk_size=64,
               cu_seqlens_len=668),
    BwdDhuCase("varlen_t16384_v256_cu2", 1, 21, 63, 16384, chunk_size=64, cu_seqlens_len=2),
    BwdDhuCase("varlen_t16384_v256_cu128", 1, 16, 32, 16384, chunk_size=64, cu_seqlens_len=128),
    BwdDhuCase("varlen_t65536_v256_cu17", 1, 4, 32, 65536, chunk_size=128, cu_seqlens_len=17),
    BwdDhuCase("varlen_t65536_v256_cu172", 1, 8, 32, 65536, chunk_size=128, cu_seqlens_len=172),
    BwdDhuCase("varlen_t65536_v256_cu668", 1, 16, 32, 65536, chunk_size=64, cu_seqlens_len=668),
    BwdDhuCase(
        "varlen_t262144_v256_cu32", 1, 2, 64, 262144, chunk_size=64, cu_seqlens_len=32,
        supported=False,
        skip_reason="CPU fp64 golden T=262144 过慢，暂跳过",
    ),
]


def _dual_check(name: str, npu_out: torch.Tensor, ref_fp64: torch.Tensor, ref_npu: torch.Tensor, level: str = "L1"):
    print(f"================== {name} (dual: fp64 gt / npu-aligned bench) ==================", flush=True)
    result = ct.dual(npu_out.cpu().float(), ref_fp64.cpu().float(), ref_npu.cpu().float(), level=level)
    ok = bool(result.get("success"))
    ratios = result.get("ratios", {})
    checks = result.get("checks", {})
    tag = "PASS" if ok else "FAIL"
    print(f"[{name}] dual {tag}: checks={checks} ratios={ratios}", flush=True)
    return ok, ratios, checks


def _build_inputs(case: BwdDhuCase, seed: int = 0):
    ktype = case.ktype()
    gtype = case.gtype()
    torch.manual_seed(seed)
    q, k, w, do, dv, g = create_bwd_dhu_random_inputs(
        case.batch, case.k_h, case.v_h, case.tokens, case.k_dim, case.v_dim, ktype, gtype,
    )
    cu_seqlens = None
    chunk_indices = None
    if case.varlen:
        sequence_count = case.cu_seqlens_len - 1
        segment_max = max(128, math.ceil(case.tokens / sequence_count))
        cu_seqlens = generate_cu_seqlens(case.cu_seqlens_len, case.tokens, seg_min=1, seg_max=segment_max)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)
    scale = scale_for_compute_dtype(effective_scale(1.0 / math.sqrt(case.k_dim), case.k_dim), ktype)
    return q, k, w, do, dv, g, cu_seqlens, chunk_indices, scale


def run_case(case: BwdDhuCase, device: int, out_root: str, seed: int = 0) -> tuple[str, str]:
    if not case.supported:
        return "SKIP", case.skip_reason

    print(f"\n========== {case.name} ==========", flush=True)
    print(
        f"B={case.batch} Hk/Hv={case.k_h}/{case.v_h} T={case.tokens} "
        f"K={case.k_dim} V={case.v_dim} cs={case.chunk_size} varlen={case.varlen}",
        flush=True,
    )

    q, k, w, do, dv, g, cu_seqlens, chunk_indices, scale = _build_inputs(case, seed=seed)

    dh_fp64, _, dv2_fp64 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, do, dv, cu_seqlens, chunk_indices, g=g, scale=scale,
        chunk_size=case.chunk_size, golden_mode="fp64",
    )
    dh_npu_bench, _, dv2_npu_bench = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, do, dv, cu_seqlens, chunk_indices, g=g, scale=scale,
        chunk_size=case.chunk_size, golden_mode="npu",
    )

    dh_npu, _, dv2_npu = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), w.npu(), do.npu(), dv.npu(),
        scale=scale,
        chunk_size=case.chunk_size,
        g=g.npu(),
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    dh_ok, _, _ = _dual_check("dh", dh_npu, dh_fp64, dh_npu_bench)
    dv2_ok, _, _ = _dual_check("dv2", dv2_npu, dv2_fp64, dv2_npu_bench)

    case_dir = os.path.join(out_root, case.name)
    if os.environ.get("BWD_HU_SAVE_OUT", "1") == "1":
        os.makedirs(case_dir, exist_ok=True)
        torch.save({
            "case": case.name,
            "dh_npu": dh_npu.cpu(),
            "dv2_npu": dv2_npu.cpu(),
            "dh_fp64": dh_fp64.cpu(),
            "dh_npu_bench": dh_npu_bench.cpu(),
            "dv2_fp64": dv2_fp64.cpu(),
            "dv2_npu_bench": dv2_npu_bench.cpu(),
        }, os.path.join(case_dir, "outputs.pt"))

    if dh_ok and dv2_ok:
        return "PASS", f"dh=PASS dv2=PASS | out={case_dir}"
    return "FAIL", f"dh={'PASS' if dh_ok else 'FAIL'} dv2={'PASS' if dv2_ok else 'FAIL'}"


def main():
    device = int(os.environ.get("TEST_DEVICE_ID", 5))
    torch.npu.set_device(device)
    out_root = os.environ.get("BWD_HU_OUT_DIR", DEFAULT_OUT_DIR)
    only = os.environ.get("BWD_HU_CASE", "").strip()

    cases = CASES
    if only:
        cases = [c for c in cases if c.name == only]
        if not cases:
            raise SystemExit(f"unknown BWD_HU_CASE={only}")

    print(f"[MODE] random input + dual(fp64 gt / npu-aligned bench), no example dump", flush=True)
    print(f"device={device} out_root={out_root} cases={len(cases)}", flush=True)

    results = []
    for case in cases:
        try:
            status, detail = run_case(case, device, out_root)
        except Exception as exc:
            status, detail = "FAIL", str(exc)
            print(f"FAIL: {exc}", flush=True)
        results.append((case.name, status, detail))

    print("\n===== SUMMARY =====", flush=True)
    for name, status, detail in results:
        print(f"{name}: {status} ({detail})", flush=True)

    if any(s == "FAIL" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
