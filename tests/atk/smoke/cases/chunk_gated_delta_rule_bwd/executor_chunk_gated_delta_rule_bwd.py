"""GDN backward chain. The DUT is a single public Ascend C call."""

import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "common" / "chunk_gated_delta_rule_bwd"),
    str(PROJECT_ROOT / "common"), str(PROJECT_ROOT),
]

from gdn_atk import GdnApi
sys.path.append(str(PROJECT_ROOT / "chunk_fwd_h"))
import fwd_h_cpu
from chunk_gated_delta_rule_bwd.scripts.input_data import build_inputs
from chunk_gated_delta_rule_bwd_dhu.scripts.cpu_reference import chunk_gated_delta_rule_bwd_dhu_cpu
from chunk_gated_delta_rule_bwd_finalize.scripts.cpu_reference import (
    chunk_gated_delta_rule_bwd_finalize_golden,
)
from chunk_gdn_bwd_intra.scripts.cpu_reference import _cpu_ref

OP = "chunk_gated_delta_rule_bwd"


def run_cpu(spec, inputs, high_precision):
    g_dtype, beta_dtype = inputs["g"].dtype, inputs["beta"].dtype
    main_dtype = torch.float64 if high_precision else torch.bfloat16
    gate_dtype = torch.promote_types(g_dtype, beta_dtype)
    common = dict(scale=inputs["scale"], chunk_size=spec["chunk_size"],
                  use_exp2=spec["use_exp2"], cu_seqlens=inputs["cu_seqlens"],
                  chunk_indices=inputs["chunk_indices"])
    q, k, v, d_o, a = (inputs[name].to(main_dtype) for name in ("q", "k", "v", "d_o", "A"))
    g, beta = (inputs[name].to(gate_dtype) for name in ("g", "beta"))
    w, u, dv_local = (value.to(main_dtype) for value in _cpu_ref(
        dict(q=q, k=k, v=v, d_o=d_o, A=a, g=g, beta=beta, main_dtype=main_dtype, **common)
    ))
    h0, dht = (inputs["initial_state"], inputs["dht"])
    if h0 is not None:
        h0 = h0.to(main_dtype)
    fwd_inputs = fwd_h_cpu.PreparedInputs(
        k=k, w=w, u=u, g=g, gk=None, initial_state=h0, cu_seqlens=common["cu_seqlens"],
        chunk_indices=common["chunk_indices"], seqlens=inputs["seqlens"],
    )
    h, v_new, _ = fwd_h_cpu._reference(
        fwd_inputs, output_final_state=False, use_exp2=spec["use_exp2"],
        state_v_first=False, high_precision=high_precision,
    )
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv_local, g=g, h0=h0, dht=dht, **common,
        golden_mode="fp64" if high_precision else "npu", nt_first=True,
    )
    beta_raw = inputs["beta_raw"]
    dq, dk, dv, dbeta, dg = chunk_gated_delta_rule_bwd_finalize_golden(
        q, k, v, v_new, d_o, dv2, g, beta, h, dh.to(main_dtype), a,
        q_rstd=inputs["q_rstd"], k_rstd=inputs["k_rstd"],
        beta_raw=None if beta_raw is None else beta_raw.to(gate_dtype),
        use_qk_l2_norm_in_kernel=spec["use_qk_l2norm"],
        use_beta_sigmoid_in_kernel=spec["use_beta_sigmoid"], nt_first=True,
        high_precision=high_precision, **common,
    )
    if dh0 is not None:
        dh0 = dh0.to(main_dtype)
        if spec["state_v_first"]:
            dh0 = dh0.transpose(-1, -2).contiguous()
    if not high_precision:
        dbeta, dg = dbeta.to(beta_dtype), dg.to(g_dtype)
    return (*(value.transpose(1, 2).contiguous() for value in (dq, dk, dv, dbeta, dg)),
            dh0, None, None)


def prepare_npu_inputs(spec, inputs):
    public = dict(inputs)
    for name in ("q", "k", "v", "d_o", "g", "beta", "beta_raw"):
        if inputs[name] is not None:
            public[name] = inputs[name].transpose(1, 2).contiguous()
    if spec["state_v_first"]:
        for name in ("initial_state", "dht"):
            if inputs[name] is not None:
                public[name] = inputs[name].transpose(-1, -2).contiguous()
    del public["seqlens"]
    return public


def run_npu(spec, inputs):
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd

    return npu_chunk_gated_delta_rule_bwd(
        **inputs, layout=spec["layout"], chunk_size=spec["chunk_size"], use_exp2=spec["use_exp2"],
        use_qk_l2norm_in_kernel=spec["use_qk_l2norm"],
        use_beta_sigmoid_in_kernel=spec["use_beta_sigmoid"], state_v_first=spec["state_v_first"],
    )


@register(f"executor_{OP}")
class FunctionApi(GdnApi):
    op_name = OP
    build_inputs = staticmethod(build_inputs)
    run_cpu = staticmethod(run_cpu)
    run_npu = staticmethod(run_npu)
    prepare_npu_inputs = staticmethod(prepare_npu_inputs)

    @staticmethod
    def output_contract(inputs):
        gradients = tuple((inputs[name].transpose(1, 2).shape, inputs[name].dtype)
                          for name in ("q", "k", "v", "beta", "g"))
        h0 = inputs["initial_state"]
        return (*gradients, None if h0 is None else (h0.shape, h0.dtype), None, None)
