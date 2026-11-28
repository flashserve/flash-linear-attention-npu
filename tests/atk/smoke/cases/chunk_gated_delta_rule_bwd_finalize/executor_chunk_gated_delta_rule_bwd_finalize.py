"""A5 Finalize ATK executor using the FLAN stage 0–15 CPU reference."""

import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "common" / "chunk_gated_delta_rule_bwd"),
    str(PROJECT_ROOT / "common"), str(PROJECT_ROOT),
]

from _ascendc_common_executor import _num_chunks, _orig_dtype
from gdn_atk import GdnApi, packed_metadata, uniform
from chunk_gated_delta_rule_bwd_finalize.scripts.cpu_reference import (
    chunk_gated_delta_rule_bwd_finalize_golden,
)

OP = "chunk_gated_delta_rule_bwd_finalize"


def build_inputs(spec, device):
    b, hk, hv, t, k, v = (spec[key] for key in ("B", "HK", "HV", "T", "K", "V"))
    bt, seed = spec["chunk_size"], spec["seed"]
    _, cu, indices = packed_metadata(spec)
    nt = len(indices) // 2 if indices is not None else _num_chunks(t, bt)
    scalar_dtype = _orig_dtype(spec["g_dtype"])
    main_dtype = torch.bfloat16
    q_shape, v_shape, state_shape = (b, hk, t, k), (b, hv, t, v), (b, nt, hv, k, v)
    profiles = (
        ("q", q_shape, 0.2, main_dtype), ("k", q_shape, 0.2, main_dtype),
        ("v", v_shape, 0.5, main_dtype),
        ("v_new", v_shape, 0.03, main_dtype), ("do", v_shape, 0.03, main_dtype),
        ("du", v_shape, 0.03, main_dtype), ("h", state_shape, 0.03, main_dtype),
        ("dh", state_shape, 0.03, main_dtype), ("a", (b, hv, t, bt), 0.03, main_dtype),
        ("g", (b, hv, t), 0.09, scalar_dtype), ("beta", (b, hv, t), 0.09, scalar_dtype),
    )
    inputs = {name: uniform(shape, dtype, device, seed + index, -limit, limit)
              for index, (name, shape, limit, dtype) in enumerate(profiles, start=1)}
    for name in ("q", "k"):
        rstd = None
        if spec["use_qk_l2norm"]:
            rstd = torch.rsqrt(inputs[name].float().square().sum(dim=-1) + 1e-6)
        inputs[f"{name}_rstd"] = rstd
    inputs.update(
        beta_raw=(uniform((b, hv, t), scalar_dtype, device, seed + 12, -0.09, 0.09)
                  if spec["use_beta_sigmoid"] else None),
        cu_seqlens=cu, chunk_indices=indices, scale=spec["scale"], chunk_size=bt,
        use_qk_l2_norm_in_kernel=spec["use_qk_l2norm"],
        use_beta_sigmoid_in_kernel=spec["use_beta_sigmoid"],
        state_v_first=spec["state_v_first"], use_exp2=spec["use_exp2"],
    )
    return inputs


def run_cpu(spec, inputs, high_precision):
    return chunk_gated_delta_rule_bwd_finalize_golden(
        **inputs, nt_first=True, high_precision=high_precision,
    )


def run_npu(spec, inputs):
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_finalize

    return npu_chunk_gated_delta_rule_bwd_finalize(**inputs)


@register(f"executor_{OP}")
class FunctionApi(GdnApi):
    op_name = OP
    build_inputs = staticmethod(build_inputs)
    run_cpu = staticmethod(run_cpu)
    run_npu = staticmethod(run_npu)

    @staticmethod
    def output_contract(inputs):
        return tuple((inputs[name].shape, inputs[name].dtype)
                     for name in ("q", "k", "v", "beta", "g"))
