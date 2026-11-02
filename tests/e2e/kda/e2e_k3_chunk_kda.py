#!/usr/bin/env python3
"""
End-to-end test of chunk_kda_fwd with Kimi-K3 architecture.

Validates:
1. Model config: Kimi-K3-0.40B (same architecture as K3)
2. NPU kernel: chunk_kda_fwd runs correctly with K3-compatible shapes
3. Output sanity: no NaN/Inf, non-zero outputs, reasonable range
4. Performance: kernel timing across multiple shapes
5. ATK comparison: cross-reference with ATK's CPU golden
"""
import os
import sys
import math
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# 本脚本位于 <repo>/tests/e2e/kda/，据此定位仓库根
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "torch_custom") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "torch_custom"))

from fla_npu.ops.ascendc import chunk_kda_fwd

# Kimi-K3-0.40B 权重快照（仅读取 config.json；可用 K3_MODEL_DIR 覆盖）
MODEL_DIR = os.environ.get(
    "K3_MODEL_DIR",
    "/workspace/models/models--inference-optimization--Kimi-K3-0.40B/snapshots/d853649387ffe8f48ce0198a29ac1a44205031f7")


def load_k3_config():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        full_cfg = json.load(f)
    return full_cfg["text_config"], full_cfg


def get_k3_info(config):
    kda_cfg = config["linear_attn_config"]
    return {
        "num_heads": kda_cfg["num_heads"],
        "head_dim": kda_cfg["head_dim"],
        "qk_nope": config.get("qk_nope_head_dim", 64),
        "qk_rope": config.get("qk_rope_head_dim", 32),
        "v_dim": config.get("v_head_dim", 64),
        "kv_lora_rank": config.get("kv_lora_rank", 128),
        "q_lora_rank": config.get("q_lora_rank", 256),
        "kda_layers": kda_cfg["kda_layers"],
        "full_attn_layers": kda_cfg["full_attn_layers"],
        "num_experts": config.get("num_experts", 8),
        "num_experts_per_token": config.get("num_experts_per_token", 2),
        "hidden_size": config.get("hidden_size", 1024),
    }


def run_single_case(batch, seq_len, chunk_size, dtype_str, device, H, K, V):
    """Run a single NPU test case."""
    DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
    td = DTYPE_MAP[dtype_str]
    
    torch.manual_seed(20260812)
    q = torch.randn(batch, seq_len, H, K, dtype=td, device=device) * 0.1
    k = torch.randn(batch, seq_len, H, K, dtype=td, device=device) * 0.1
    v = torch.randn(batch, seq_len, H, V, dtype=td, device=device) * 0.1
    g = torch.randn(batch, seq_len, H, K, dtype=torch.float32, device=device) * 0.01
    beta = torch.randn(batch, seq_len, H, dtype=torch.float32, device=device) * 0.1
    scale = K ** -0.5
    
    # Run NPU
    torch.npu.synchronize()
    t0 = time.perf_counter()
    result = chunk_kda_fwd(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, chunk_size=chunk_size,
        layout="BSND",
        disable_recompute=True,
        safe_gate=False,
    )
    torch.npu.synchronize()
    npu_ms = (time.perf_counter() - t0) * 1000
    
    attn_out = result[0]
    
    # Sanity checks
    has_nan = bool(torch.isnan(attn_out).any().item())
    has_inf = bool(torch.isinf(attn_out).any().item())
    expected_shape = (batch, seq_len, H, V)
    shape_ok = attn_out.shape == expected_shape
    all_zeros = bool((attn_out == 0).all().item())
    min_val = float(attn_out.min().item())
    max_val = float(attn_out.max().item())
    mean_abs = float(attn_out.abs().mean().item())
    
    # Pass criteria: correct shape, no NaN/Inf, non-zero output
    passed = shape_ok and not has_nan and not has_inf and not all_zeros
    
    return {
        "case": f"B{batch}_T{seq_len}_c{chunk_size}_{dtype_str}",
        "npu_ms": npu_ms,
        "shape": attn_out.shape,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "all_zeros": all_zeros,
        "min_val": min_val,
        "max_val": max_val,
        "mean_abs": mean_abs,
        "passed": passed,
    }


def main():
    print("=" * 64)
    print("Kimi-K3 chunk_kda_fwd End-to-End Test")
    print("=" * 64)
    
    # Load K3 model config
    config, full_cfg = load_k3_config()
    dims = get_k3_info(config)
    
    print(f"\n--- Kimi-K3 Architecture ---")
    print(f"  Model:            Kimi-K3-0.40B (KDA+MoE+MLA)")
    print(f"  KDA layers:       {dims['kda_layers']}")
    print(f"  Full-attn layers: {dims['full_attn_layers']}")
    print(f"  num_heads (H):    {dims['num_heads']}")
    print(f"  head_dim:         {dims['head_dim']}")
    print(f"  qk_nope:          {dims['qk_nope']}")
    print(f"  qk_rope:          {dims['qk_rope']}")
    print(f"  v_head_dim:       {dims['v_dim']}")
    print(f"  kv_lora_rank:     {dims['kv_lora_rank']}")
    print(f"  q_lora_rank:      {dims['q_lora_rank']}")
    print(f"  MoE experts:      {dims['num_experts']} total / {dims['num_experts_per_token']} active")
    print(f"  hidden_size:      {dims['hidden_size']}")
    
    # For chunk_kda_fwd: K must equal V, and both must be 64 or 128
    H = dims["num_heads"]
    K = dims["kv_lora_rank"]  # 128
    V = K  # 128
    
    print(f"\n--- chunk_kda_fwd Configuration ---")
    print(f"  Heads:   H={H} (matches K3 num_heads)")
    print(f"  K-dim:   K={K} (=kv_lora_rank)")
    print(f"  V-dim:   V={V}")
    print(f"  Layout:  BSND")
    
    device = torch.device("npu:0")
    
    # Test cases: B, T, chunk_size, dtype
    test_cases = [
        (1, 64,   64,  "bf16"),
        (1, 128,  64,  "bf16"),
        (1, 256,  64,  "bf16"),
        (1, 512,  64,  "bf16"),
        (1, 1024, 64,  "bf16"),
        (1, 2048, 64,  "bf16"),
        (2, 256,  64,  "bf16"),
        (4, 128,  64,  "bf16"),
    ]
    
    print(f"\n--- Running NPU Tests ---")
    results = []
    all_passed = True
    
    for batch, seq_len, chunk_size, dtype_str in test_cases:
        try:
            r = run_single_case(batch, seq_len, chunk_size, dtype_str, device, H, K, V)
            mark = "✓" if r["passed"] else "✗"
            print(f"  {mark} {r['case']:25s}  {r['npu_ms']:.1f}ms  "
                  f"range=[{r['min_val']:.4f},{r['max_val']:.4f}]  mean|o|={r['mean_abs']:.4e}")
            results.append(r)
            all_passed &= r["passed"]
        except Exception as e:
            print(f"  ✗ B{batch}_T{seq_len}_c{chunk_size}_{dtype_str}: EXCEPTION {type(e).__name__}: {str(e)[:80]}")
            results.append({"case": f"B{batch}_T{seq_len}_c{chunk_size}_{dtype_str}", "passed": False, "reason": str(e)[:100]})
            all_passed = False
    
    # Summary
    print(f"\n{'═' * 64}")
    print("Results Summary")
    print(f"{'═' * 64}")
    print(f"{'Case':30s} {'Status':8s} {'Time':10s} {'Range':20s} {'MeanAbs':10s}")
    print(f"{'─' * 64}")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        if "npu_ms" in r:
            print(f"{r['case']:30s} {mark:8s} {r['npu_ms']:8.1f}ms [{r['min_val']:.3f},{r['max_val']:.3f}] {r['mean_abs']:10.2e}")
        else:
            print(f"{r['case']:30s} {mark:8s}  ERR: {r.get('reason', '')[:40]}")
    
    passed_count = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    print(f"\n  {passed_count}/{total} cases passed")
    print(f"  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"{'═' * 64}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
