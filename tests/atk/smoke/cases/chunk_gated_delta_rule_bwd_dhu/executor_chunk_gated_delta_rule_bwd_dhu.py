"""A5 Dhu ATK executor using the FLAN fp64/npu CPU reference."""

import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "common" / "chunk_gated_delta_rule_bwd"),
    str(PROJECT_ROOT / "common"), str(PROJECT_ROOT),
]

from _ascendc_common_executor import _num_chunks, _orig_dtype, _randn
from gdn_atk import GdnApi, gate_values, packed_metadata
from chunk_gated_delta_rule_bwd_dhu.scripts.cpu_reference import chunk_gated_delta_rule_bwd_dhu_cpu

OP = "chunk_gated_delta_rule_bwd_dhu"


def build_inputs(spec, device):
    b, hk, hv, t, k, v = (spec[key] for key in ("B", "HK", "HV", "T", "K", "V"))
    lengths, cu, indices = packed_metadata(spec)
    count = b if lengths is None else len(lengths)
    dtype = torch.bfloat16
    seed = spec["seed"]
    return {
        "q": _randn((b, hk, t, k), "bf16", dtype, device, seed + 1),
        "k": _randn((b, hk, t, k), "bf16", dtype, device, seed + 2),
        "w": _randn((b, hv, t, k), "bf16", dtype, device, seed + 3),
        "d_o": _randn((b, hv, t, v), "bf16", dtype, device, seed + 4),
        "dv": _randn((b, hv, t, v), "bf16", dtype, device, seed + 5),
        "g": gate_values(b, hv, t, lengths, seed + 6, _orig_dtype(spec["g_dtype"]), device),
        "h0": (_randn((count, hv, k, v), "bf16", dtype, device, seed + 7)
               if spec["with_state"] else None),
        "dht": (_randn((count, hv, k, v), "bf16", dtype, device, seed + 8)
                if spec["with_dht"] else None),
        "cu_seqlens": cu, "chunk_indices": indices,
        "scale": spec["scale"], "chunk_size": spec["chunk_size"], "use_exp2": spec["use_exp2"],
    }


def run_cpu(spec, inputs, high_precision):
    names = ("q", "k", "w", "d_o", "dv")
    dtype = torch.float64 if high_precision else torch.bfloat16
    outputs = chunk_gated_delta_rule_bwd_dhu_cpu(
        *(inputs[name].to(dtype) for name in names),
        **{name: inputs[name] for name in ("g", "h0", "dht", "cu_seqlens", "chunk_indices",
                                           "scale", "chunk_size", "use_exp2")},
        state_v_first=spec["state_v_first"], nt_first=True,
        golden_mode="fp64" if high_precision else "npu",
    )
    return tuple(value.to(dtype) if value is not None else None for value in outputs)


def run_npu(spec, inputs):
    from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_dhu

    return npu_chunk_gated_delta_rule_bwd_dhu(
        **inputs, transpose_state_layout=spec["state_v_first"],
    )


@register(f"executor_{OP}")
class FunctionApi(GdnApi):
    op_name = OP
    build_inputs = staticmethod(build_inputs)
    run_cpu = staticmethod(run_cpu)
    run_npu = staticmethod(run_npu)

    @staticmethod
    def output_contract(inputs):
        dv, h0 = inputs["dv"], inputs["h0"]
        b, hv, t, v = dv.shape
        k = inputs["q"].shape[-1]
        indices = inputs["chunk_indices"]
        nt = len(indices) // 2 if indices is not None else _num_chunks(t, inputs["chunk_size"])
        return (((b, nt, hv, k, v), dv.dtype),
                None if h0 is None else (h0.shape, h0.dtype),
                (dv.shape, dv.dtype))
