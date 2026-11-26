"""A5 Intra ATK executor using the FLAN CPU reference."""

import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "common" / "chunk_gated_delta_rule_bwd"),
    str(PROJECT_ROOT / "common"), str(PROJECT_ROOT),
]

from _ascendc_common_executor import _orig_dtype, _randn
from gdn_atk import GdnApi, gate_values, packed_metadata, uniform
from chunk_gdn_bwd_intra.scripts.cpu_reference import _cpu_ref

OP = "chunk_gdn_bwd_intra"


def build_inputs(spec, device):
    b, hk, hv, t, k, v = (spec[key] for key in ("B", "HK", "HV", "T", "K", "V"))
    bt, seed = spec["chunk_size"], spec["seed"]
    lengths, cu, indices = packed_metadata(spec)
    dtype = torch.bfloat16
    beta_dtype = _orig_dtype(spec["beta_dtype"])
    return {
        "q": _randn((b, hk, t, k), "bf16", dtype, device, seed + 1, 0.20),
        "k": _randn((b, hk, t, k), "bf16", dtype, device, seed + 2, 0.20),
        "v": _randn((b, hv, t, v), "bf16", dtype, device, seed + 3),
        "g": gate_values(b, hv, t, lengths, seed + 4, _orig_dtype(spec["g_dtype"]), device),
        "beta": uniform((b, hv, t), beta_dtype, device, seed + 5, 0.1, 0.9),
        "A": _randn((b, hv, t, bt), "bf16", dtype, device, seed + 6),
        "d_o": _randn((b, hv, t, v), "bf16", dtype, device, seed + 7, 0.20),
        "scale": spec["scale"], "chunk_size": bt, "use_exp2": spec["use_exp2"],
        "cu_seqlens": cu, "chunk_indices": indices, "main_dtype": dtype,
    }


def run_cpu(spec, inputs, high_precision):
    if high_precision:
        inputs = {key: (value.double() if isinstance(value, torch.Tensor) else value)
                  for key, value in inputs.items()}
        inputs["main_dtype"] = torch.float64
    return tuple(value.to(inputs["main_dtype"]) for value in _cpu_ref(inputs))


def run_npu(spec, inputs):
    from fla_npu.ops.ascendc import npu_chunk_gdn_bwd_intra

    return npu_chunk_gdn_bwd_intra(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A"], inputs["d_o"], inputs["scale"], inputs["chunk_size"],
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"],
        use_exp2=inputs["use_exp2"],
    )


@register(f"executor_{OP}")
class FunctionApi(GdnApi):
    op_name = OP
    build_inputs = staticmethod(build_inputs)
    run_cpu = staticmethod(run_cpu)
    run_npu = staticmethod(run_npu)

    @staticmethod
    def output_contract(inputs):
        v = inputs["v"]
        return ((v.shape, v.dtype),) * 3
