"""ChunkGatedDeltaRuleBwd ATK executor and independent CPU reference chain."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device


OP_NAME = "chunk_gated_delta_rule_bwd"
CHUNK_SIZE = 64
DIM = 128
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_INTRA = _load_module(
    "chunk_gated_delta_rule_bwd_intra_reference",
    Path(__file__).resolve().parents[1]
    / "chunk_gdn_bwd_intra"
    / "executor_chunk_gdn_bwd_intra.py",
)
_FWD_H = _load_module(
    "chunk_gated_delta_rule_bwd_fwd_h_reference",
    Path(__file__).resolve().parents[1]
    / "chunk_fwd_h"
    / "executor_chunk_fwd_h.py",
)
_DHU = _load_module(
    "chunk_gated_delta_rule_bwd_dhu_reference",
    REPO_ROOT / "torch_custom" / "fla_npu" / "test" / "test_bwd_dhu.py",
)
_FINALIZE = _load_module(
    "chunk_gated_delta_rule_bwd_finalize_reference",
    Path(__file__).resolve().parents[1]
    / "chunk_gated_delta_rule_bwd_finalize"
    / "scripts"
    / "chunk_gated_delta_rule_bwd_finalize_cpu.py",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _randn(shape, seed: int, scale: float, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * float(scale)
    return value.to(dtype).to(device)


def _rand(shape, seed: int, low: float, high: float, dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.rand(tuple(shape), generator=generator, dtype=torch.float32)
    value = value * (float(high) - float(low)) + float(low)
    return value.to(dtype).to(device)


def _canonical_metadata(spec: dict[str, Any]):
    seqlens = spec.get("seqlens")
    if not seqlens:
        return None, None, None
    seqlens = [int(length) for length in seqlens]
    cu_seqlens = [0]
    chunk_indices = []
    for sequence, length in enumerate(seqlens):
        cu_seqlens.append(cu_seqlens[-1] + length)
        for local_chunk in range((length + CHUNK_SIZE - 1) // CHUNK_SIZE):
            chunk_indices.extend((sequence, local_chunk))
    return seqlens, tuple(cu_seqlens), tuple(chunk_indices)


def _gate_values(batch: int, heads: int, tokens: int, seqlens, seed: int,
                 dtype: torch.dtype, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    steps = -(torch.rand((batch, heads, tokens), generator=generator) * 0.01 + 0.001)
    gate = torch.empty_like(steps)
    spans = [(0, tokens)] if seqlens is None else []
    if seqlens is not None:
        begin = 0
        for length in seqlens:
            spans.append((begin, begin + int(length)))
            begin += int(length)
    for physical_batch in range(batch):
        for sequence_begin, sequence_end in spans:
            for chunk_begin in range(sequence_begin, sequence_end, CHUNK_SIZE):
                chunk_end = min(chunk_begin + CHUNK_SIZE, sequence_end)
                gate[physical_batch, :, chunk_begin:chunk_end] = torch.cumsum(
                    steps[physical_batch, :, chunk_begin:chunk_end], dim=-1
                )
    return gate.to(dtype).to(device)


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch, key_heads, value_heads, tokens = (
        int(spec[name]) for name in ("B", "HK", "HV", "T")
    )
    g_dtype = (
        torch.float32
        if spec.get("g_dtype", spec.get("scalar_dtype", "fp32")) == "fp32"
        else torch.bfloat16
    )
    beta_dtype = (
        torch.float32
        if spec.get("beta_dtype", spec.get("scalar_dtype", "fp32")) == "fp32"
        else torch.bfloat16
    )
    seed = int(spec.get("seed", 20260911))
    seqlens, cu_seqlens, chunk_indices = _canonical_metadata(spec)
    if seqlens is not None and (batch != 1 or sum(seqlens) != tokens):
        raise ValueError("varlen cases require B=1 and sum(seqlens)=T")

    q = _randn((batch, key_heads, tokens, DIM), seed + 1, 0.12, torch.bfloat16, device)
    k = _randn((batch, key_heads, tokens, DIM), seed + 2, 0.12, torch.bfloat16, device)
    v = _randn((batch, value_heads, tokens, DIM), seed + 3, 0.05, torch.bfloat16, device)
    g = _gate_values(batch, value_heads, tokens, seqlens, seed + 4, g_dtype, device)
    use_beta_sigmoid = _as_bool(spec.get("use_beta_sigmoid", False))
    beta_raw = (
        _randn((batch, value_heads, tokens), seed + 5, 0.5, beta_dtype, device)
        if use_beta_sigmoid
        else None
    )
    beta = torch.sigmoid(beta_raw.float()).to(beta_dtype) if beta_raw is not None else _rand(
        (batch, value_heads, tokens), seed + 5, 0.1, 0.9, beta_dtype, device
    )
    a = _randn((batch, value_heads, tokens, CHUNK_SIZE), seed + 6, 0.02, torch.bfloat16, device)
    d_o = _randn((batch, value_heads, tokens, DIM), seed + 7, 0.12, torch.bfloat16, device)

    use_qk_l2norm = _as_bool(spec.get("use_qk_l2norm", False))
    q_rstd = torch.rsqrt((q.float() * q.float()).sum(-1) + 1.0e-6) if use_qk_l2norm else None
    k_rstd = torch.rsqrt((k.float() * k.float()).sum(-1) + 1.0e-6) if use_qk_l2norm else None

    sequences = batch if seqlens is None else len(seqlens)
    with_state = _as_bool(spec.get("with_state", False))
    with_dht = _as_bool(spec.get("with_dht", False))
    initial_state = (
        _randn((sequences, value_heads, DIM, DIM), seed + 8, 0.02, torch.bfloat16, device)
        if with_state
        else None
    )
    dht = (
        _randn((sequences, value_heads, DIM, DIM), seed + 9, 0.02, torch.bfloat16, device)
        if with_dht
        else None
    )
    reserved = _as_bool(spec.get("reserved_inputs", False))
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A": a,
        "d_o": d_o,
        "q_rstd": q_rstd,
        "k_rstd": k_rstd,
        "beta_raw": beta_raw,
        "initial_state": initial_state,
        "dht": dht,
        "a_log": _randn((value_heads,), seed + 10, 0.1, torch.float32, device) if reserved else None,
        "dt_bias": _randn((value_heads,), seed + 11, 0.1, torch.float32, device) if reserved else None,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(DIM))),
    }


def _intra_reference(inputs: dict[str, Any], use_exp2: bool):
    intra_inputs = {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "g": inputs["g"],
        "beta": inputs["beta"],
        "A": inputs["A"],
        "d_o": inputs["d_o"],
        "scale": inputs["scale"],
        "chunk_size": CHUNK_SIZE,
        "use_exp2": use_exp2,
        "main_dtype": torch.bfloat16,
        "cu_seqlens": inputs["cu_seqlens"],
        "chunk_indices": inputs["chunk_indices"],
    }
    return tuple(tensor.to(torch.bfloat16) for tensor in _INTRA._cpu_ref(intra_inputs))


def run_cpu(spec: dict[str, Any]):
    inputs = build_inputs(spec, torch.device("cpu"))
    g_dtype = inputs["g"].dtype
    beta_dtype = inputs["beta"].dtype
    gate_dtype = torch.float32 if torch.float32 in (g_dtype, beta_dtype) else torch.bfloat16
    inputs["g"] = inputs["g"].to(gate_dtype)
    inputs["beta"] = inputs["beta"].to(gate_dtype)
    if inputs["beta_raw"] is not None:
        inputs["beta_raw"] = inputs["beta_raw"].to(gate_dtype)
    use_exp2 = _as_bool(spec.get("use_exp2", False))
    w, u, dv_local = _intra_reference(inputs, use_exp2)
    fwd_inputs = _FWD_H.PreparedInputs(
        k=inputs["k"],
        w=w,
        u=u,
        g=inputs["g"],
        gk=None,
        initial_state=inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        seqlens=inputs["seqlens"],
    )
    h, v_new, _ = _FWD_H._reference(
        fwd_inputs, output_final_state=False, use_exp2=use_exp2, state_v_first=False,
    )
    dh, dh0, dv2 = _DHU.chunk_gated_delta_rule_bwd_dhu_cpu(
        inputs["q"], inputs["k"], w, inputs["d_o"], dv_local,
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"],
        g=inputs["g"], h0=inputs["initial_state"], dht=inputs["dht"],
        scale=inputs["scale"], chunk_size=CHUNK_SIZE, golden_mode="npu", use_exp2=use_exp2,
        nt_first=True,
    )
    dh = dh.to(torch.bfloat16)
    dh0 = dh0.to(torch.bfloat16) if dh0 is not None else None
    outputs = _FINALIZE.chunk_gated_delta_rule_bwd_finalize_golden(
        inputs["q"], inputs["k"], inputs["v"], v_new, inputs["d_o"], dv2,
        inputs["g"], inputs["beta"], h, dh, inputs["A"],
        q_rstd=inputs["q_rstd"], k_rstd=inputs["k_rstd"],
        beta_raw=inputs["beta_raw"], cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"], scale=inputs["scale"],
        chunk_size=CHUNK_SIZE,
        use_qk_l2_norm_in_kernel=_as_bool(spec.get("use_qk_l2norm", False)),
        use_beta_sigmoid_in_kernel=_as_bool(spec.get("use_beta_sigmoid", False)),
        use_gate_in_kernel=False, state_v_first=False, use_exp2=use_exp2, nt_first=True,
    )
    state_v_first = _as_bool(spec.get("state_v_first", False))
    if dh0 is not None and state_v_first:
        dh0 = dh0.transpose(-1, -2).contiguous()
    dq, dk, dv, d_beta, d_g = outputs
    if str(spec.get("layout", "BNSD")) in {"BSND", "TND"}:
        dq, dk, dv = (
            tensor.transpose(1, 2).contiguous() for tensor in (dq, dk, dv)
        )
    d_beta = d_beta.transpose(1, 2).contiguous().to(beta_dtype)
    d_g = d_g.transpose(1, 2).contiguous().to(g_dtype)
    return dq, dk, dv, d_beta, d_g, dh0, None, None


def _public_layout(tensor, sequence_major: bool):
    if tensor is None or not sequence_major:
        return tensor
    return tensor.transpose(1, 2).contiguous()


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))
    from fla_npu.ops import ascendc

    layout = str(spec.get("layout", "BNSD"))
    sequence_major = layout in {"BSND", "TND"}
    state_v_first = _as_bool(spec.get("state_v_first", False))
    initial_state = inputs["initial_state"]
    dht = inputs["dht"]
    if state_v_first:
        initial_state = initial_state.transpose(-1, -2).contiguous() if initial_state is not None else None
        dht = dht.transpose(-1, -2).contiguous() if dht is not None else None
    outputs = ascendc.npu_chunk_gated_delta_rule_bwd(
        _public_layout(inputs["q"], sequence_major),
        _public_layout(inputs["k"], sequence_major),
        _public_layout(inputs["v"], sequence_major),
        _public_layout(inputs["g"], True),
        _public_layout(inputs["beta"], True),
        inputs["A"],
        _public_layout(inputs["d_o"], True),
        inputs["scale"], chunk_size=CHUNK_SIZE, layout=layout,
        initial_state=initial_state, dht=dht,
        q_rstd=inputs["q_rstd"],
        k_rstd=inputs["k_rstd"],
        beta_raw=_public_layout(inputs["beta_raw"], True),
        a_log=inputs["a_log"], dt_bias=inputs["dt_bias"],
        use_exp2=_as_bool(spec.get("use_exp2", False)),
        use_gate_in_kernel=False,
        use_qk_l2norm_in_kernel=_as_bool(spec.get("use_qk_l2norm", False)),
        use_beta_sigmoid_in_kernel=_as_bool(spec.get("use_beta_sigmoid", False)),
        state_v_first=state_v_first, cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"], return_intermediate_states=False,
    )
    torch.npu.synchronize()
    return outputs


@register("executor_chunk_gated_delta_rule_bwd")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = _case_spec(input_data, OP_NAME)
        if self.device == "cpu":
            outputs = run_cpu(spec)
        elif self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        else:
            raise RuntimeError(f"{OP_NAME} only supports CPU golden and NPU DUT nodes")
        return _finite_tuple(outputs, golden=self.device == "cpu")
