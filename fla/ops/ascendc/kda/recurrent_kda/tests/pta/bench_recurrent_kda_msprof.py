#!/usr/bin/env python3
"""Use msprof to collect recurrent_kda performance for the fixed baseline case."""

from __future__ import annotations

import os

import torch
import torch_npu

from fla_npu.ops.ascendc import recurrent_kda


device_id = int(os.environ.get("TEST_DEVICE_ID", "2"))
device = torch.device(f"npu:{device_id}")
torch_npu.npu.set_device(device)


def make_inputs():
    """Build B=1, T=8, Hk=Hv=12, K=V=128 with eight varlen sequences."""

    batch, total_tokens = 1, 8
    key_heads, value_heads = 12, 12
    key_dim, value_dim = 128, 128
    sequence_num = 8
    state_sequence_num = 470

    query = torch.randn(
        (batch, total_tokens, key_heads, key_dim), dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        (batch, total_tokens, key_heads, key_dim), dtype=torch.bfloat16, device=device
    )
    value = torch.randn(
        (batch, total_tokens, value_heads, value_dim), dtype=torch.bfloat16, device=device
    )
    gate = torch.randn(
        (batch, total_tokens, value_heads, key_dim), dtype=torch.float32, device=device
    ) * 0.5
    beta = torch.randn(
        (batch, total_tokens, value_heads), dtype=torch.float32, device=device
    )
    initial_state = torch.randn(
        (state_sequence_num, value_heads, value_dim, key_dim),
        dtype=torch.float32,
        device=device,
    ) * 0.02

    return {
        "query": query,
        "key": key,
        "value": value,
        "gate": gate,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": torch.arange(sequence_num + 1, dtype=torch.int64, device=device),
        "ssm_state_indices": torch.arange(total_tokens, dtype=torch.int64, device=device),
        "num_accepted_tokens": torch.ones(sequence_num, dtype=torch.int64, device=device),
        "A_log": torch.randn((value_heads,), dtype=torch.float32, device=device) * 0.1,
        "dt_bias": torch.randn((value_heads * key_dim,), dtype=torch.float32, device=device) * 0.1,
    }


def run_bench():
    inputs = make_inputs()
    kwargs = {
        "layout": "BSND",
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "lower_bound": -5.0,
        "state_v_first": True,
        "cu_seqlens": inputs["cu_seqlens"],
        "ssm_state_indices": inputs["ssm_state_indices"],
        "num_accepted_tokens": inputs["num_accepted_tokens"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
    }

    print("[Warmup] running 5 iterations...")
    for _ in range(5):
        recurrent_kda(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["gate"],
            inputs["beta"],
            inputs["initial_state"],
            **kwargs,
        )
        torch_npu.npu.synchronize()

    print("[Bench] running 10 iterations for profiling...")
    for _ in range(10):
        output, final_state = recurrent_kda(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["gate"],
            inputs["beta"],
            inputs["initial_state"],
            **kwargs,
        )
        torch_npu.npu.synchronize()

    print(f"[Bench] done, output={output.shape}")


if __name__ == "__main__":
    run_bench()
