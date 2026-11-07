"""ChunkGatedDeltaRuleBwd ATK executor and independent CPU reference chain."""

from __future__ import annotations

import importlib.util
import math
import os
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


def _randn(
    shape,
    seed: int,
    scale: float,
    dtype: torch.dtype,
    device: torch.device,
    compute_dtype: torch.dtype | None = None,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * float(scale)
    value = value.to(dtype)
    if compute_dtype is not None:
        value = value.to(compute_dtype)
    return value.to(device)


def _rand(
    shape,
    seed: int,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    compute_dtype: torch.dtype | None = None,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.rand(tuple(shape), generator=generator, dtype=torch.float32)
    value = value * (float(high) - float(low)) + float(low)
    value = value.to(dtype)
    if compute_dtype is not None:
        value = value.to(compute_dtype)
    return value.to(device)


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
                 dtype: torch.dtype, device: torch.device,
                 compute_dtype: torch.dtype | None = None):
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
    gate = gate.to(dtype)
    if compute_dtype is not None:
        gate = gate.to(compute_dtype)
    return gate.to(device)


def build_inputs(
    spec: dict[str, Any], device: torch.device, high_precision: bool = False
) -> dict[str, Any]:
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
    compute_dtype = torch.float64 if high_precision else None
    seqlens, cu_seqlens, chunk_indices = _canonical_metadata(spec)
    if seqlens is not None and (batch != 1 or sum(seqlens) != tokens):
        raise ValueError("varlen cases require B=1 and sum(seqlens)=T")

    q = _randn((batch, key_heads, tokens, DIM), seed + 1, 0.12, torch.bfloat16, device, compute_dtype)
    k = _randn((batch, key_heads, tokens, DIM), seed + 2, 0.12, torch.bfloat16, device, compute_dtype)
    v = _randn((batch, value_heads, tokens, DIM), seed + 3, 0.05, torch.bfloat16, device, compute_dtype)
    g = _gate_values(batch, value_heads, tokens, seqlens, seed + 4, g_dtype, device, compute_dtype)
    use_beta_sigmoid = _as_bool(spec.get("use_beta_sigmoid", False))
    beta_raw = (
        _randn((batch, value_heads, tokens), seed + 5, 0.5, beta_dtype, device, compute_dtype)
        if use_beta_sigmoid
        else None
    )
    if beta_raw is not None:
        # Input construction is identical for all roles, before FP64 promotion.
        beta = torch.sigmoid(beta_raw.float()).to(beta_dtype)
        if compute_dtype is not None:
            beta = beta.to(compute_dtype)
    else:
        beta = _rand(
            (batch, value_heads, tokens), seed + 5, 0.1, 0.9,
            beta_dtype, device, compute_dtype
        )
    a = _randn((batch, value_heads, tokens, CHUNK_SIZE), seed + 6, 0.02, torch.bfloat16, device, compute_dtype)
    d_o = _randn((batch, value_heads, tokens, DIM), seed + 7, 0.12, torch.bfloat16, device, compute_dtype)

    use_qk_l2norm = _as_bool(spec.get("use_qk_l2norm", False))
    if use_qk_l2norm:
        norm_dtype = torch.float32
        q_rstd = torch.rsqrt((q.to(norm_dtype) * q.to(norm_dtype)).sum(-1) + 1.0e-6).to(torch.float32)
        k_rstd = torch.rsqrt((k.to(norm_dtype) * k.to(norm_dtype)).sum(-1) + 1.0e-6).to(torch.float32)
    else:
        q_rstd = k_rstd = None

    sequences = batch if seqlens is None else len(seqlens)
    with_state = _as_bool(spec.get("with_state", False))
    with_dht = _as_bool(spec.get("with_dht", False))
    initial_state = (
        _randn((sequences, value_heads, DIM, DIM), seed + 8, 0.02, torch.bfloat16, device, compute_dtype)
        if with_state
        else None
    )
    dht = (
        _randn((sequences, value_heads, DIM, DIM), seed + 9, 0.02, torch.bfloat16, device, compute_dtype)
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
        "a_log": _randn((value_heads,), seed + 10, 0.1, torch.float32, device, compute_dtype) if reserved else None,
        "dt_bias": _randn((value_heads,), seed + 11, 0.1, torch.float32, device, compute_dtype) if reserved else None,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "scale": float(spec.get("scale", 1.0 / math.sqrt(DIM))),
    }


def _arch22_preparation_reference(inputs: dict[str, Any], use_exp2: bool):
    """FP32 benchmark for RecomputeWUFwd + ChunkBwdDvLocal storage boundaries."""
    q, k, v, d_o = (inputs[name].float() for name in ("q", "k", "v", "d_o"))
    a, beta, g = (inputs[name].float() for name in ("A", "beta", "g"))
    if use_exp2:
        # The composite converts the gate to natural-log FP32 before both ops.
        g = g * 0.6931471805599453
    outputs = [torch.zeros_like(v) for _ in range(3)]
    group = v.shape[1] // q.shape[1]
    main_dtype = inputs["q"].dtype

    def stored(value):
        return value.to(main_dtype).float()

    for b, start, end in _INTRA._work_chunks(dict(inputs, chunk_size=CHUNK_SIZE)):
        length = end - start
        valid = torch.ones((length, length), dtype=torch.bool).triu()
        for hv in range(v.shape[1]):
            hk = hv // group
            beta_chunk, g_chunk = beta[b, hv, start:end], g[b, hv, start:end]
            a_chunk = a[b, hv, start:end, :length]
            vb = stored(v[b, hv, start:end] * beta_chunk.unsqueeze(-1))
            kbg = stored(k[b, hk, start:end] *
                         (beta_chunk * torch.exp(g_chunk)).unsqueeze(-1))
            outputs[0][b, hv, start:end] = a_chunk @ kbg
            outputs[1][b, hv, start:end] = a_chunk @ vb
            # Local-dV stores the ungated KQ product before vector processing.
            score = stored(k[b, hk, start:end] @ q[b, hk, start:end].t())
            delta = g_chunk.unsqueeze(0) - g_chunk.unsqueeze(1)
            factor = torch.where(valid, torch.exp(delta), 0.0) * float(inputs["scale"])
            attention = stored(factor * score)
            outputs[2][b, hv, start:end] = attention @ d_o[b, hv, start:end]
    return tuple(tensor.to(main_dtype) for tensor in outputs)


def _arch35_vector_store(value: torch.Tensor, dtype: torch.dtype):
    """Model CAST_ROUND at Vector BF16 stores; Cube stores retain PyTorch RNE."""
    if dtype != torch.bfloat16:
        return value.to(dtype)
    fp32 = value.float().contiguous()
    bits = fp32.view(torch.int32)
    rounded = ((bits + 0x8000) & -0x10000).view(torch.float32).to(dtype)
    return torch.where(torch.isfinite(fp32), rounded, fp32.to(dtype))


def _intra_reference(inputs: dict[str, Any], use_exp2: bool, high_precision: bool):
    if not high_precision and os.environ.get("GDN_ATK_BENCHMARK_ARCH", "arch35") == "arch22":
        return _arch22_preparation_reference(inputs, use_exp2)
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
        "main_dtype": torch.float64 if high_precision else torch.bfloat16,
        "cu_seqlens": inputs["cu_seqlens"],
        "chunk_indices": inputs["chunk_indices"],
    }
    if not high_precision:
        intra_inputs["vector_store"] = _arch35_vector_store
    outputs = _INTRA._cpu_ref(intra_inputs)
    return tuple(tensor.to(torch.float64 if high_precision else torch.bfloat16) for tensor in outputs)


def _fwd_h_p_store_dtype(inputs: dict[str, Any], use_exp2: bool, high_precision: bool):
    if (high_precision or not use_exp2 or
            os.environ.get("GDN_ATK_BENCHMARK_ARCH", "arch35") != "arch22"):
        return None
    key_heads = inputs["k"].shape[1]
    group = inputs["v"].shape[1] // key_heads
    # Match FwdHGatedReusable and its packed odd-group fallback. The gated
    # Cube writes P to FP32 workspace even when recurrent states are BF16.
    if inputs["cu_seqlens"] is not None and group % 2 != 0 and key_heads > 1:
        return None
    return torch.float32


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=high_precision)
    g_dtype = inputs["g"].dtype
    beta_dtype = inputs["beta"].dtype
    gate_dtype = (
        torch.float64
        if high_precision
        else (torch.float32 if torch.float32 in (g_dtype, beta_dtype) else torch.bfloat16)
    )
    inputs["g"] = inputs["g"].to(gate_dtype)
    inputs["beta"] = inputs["beta"].to(gate_dtype)
    if inputs["beta_raw"] is not None:
        inputs["beta_raw"] = inputs["beta_raw"].to(gate_dtype)
    use_exp2 = _as_bool(spec.get("use_exp2", False))
    vector_store = (
        _arch35_vector_store if not high_precision and
        os.environ.get("GDN_ATK_BENCHMARK_ARCH", "arch35") == "arch35" else None
    )
    w, u, dv_local = _intra_reference(inputs, use_exp2, high_precision)
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
        compute_dtype=torch.float64 if high_precision else None,
        vector_store=vector_store,
        p_store_dtype=_fwd_h_p_store_dtype(inputs, use_exp2, high_precision),
    )
    dh, dh0, dv2 = _DHU.chunk_gated_delta_rule_bwd_dhu_cpu(
        inputs["q"], inputs["k"], w, inputs["d_o"], dv_local,
        cu_seqlens=inputs["cu_seqlens"], chunk_indices=inputs["chunk_indices"],
        g=inputs["g"], h0=inputs["initial_state"], dht=inputs["dht"],
        scale=inputs["scale"], chunk_size=CHUNK_SIZE,
        golden_mode="fp64" if high_precision else "npu", use_exp2=use_exp2,
        nt_first=True,
    )
    if not high_precision:
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
        compute_dtype=torch.float64 if high_precision else None,
        vector_store=vector_store,
    )
    state_v_first = _as_bool(spec.get("state_v_first", False))
    if dh0 is not None and state_v_first:
        dh0 = dh0.transpose(-1, -2).contiguous()
    dq, dk, dv, d_beta, d_g = outputs
    if str(spec.get("layout", "BNSD")) in {"BSND", "TND"}:
        dq, dk, dv = (
            tensor.transpose(1, 2).contiguous() for tensor in (dq, dk, dv)
        )
    d_beta = d_beta.transpose(1, 2).contiguous()
    d_g = d_g.transpose(1, 2).contiguous()
    if not high_precision:
        d_beta = d_beta.to(beta_dtype)
        d_g = d_g.to(g_dtype)
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
        self.task_name = str(getattr(task_result, "name", "") or "")
        self.is_benchmark_task = bool(getattr(task_result, "is_benchmark_task", False))
        self.high_precision = self.device == "cpu" and self.is_benchmark_task
        if self.device in {"npu", "pyaclnn"}:
            self.role = "dut"
        elif self.device == "cpu" and self.is_benchmark_task:
            self.role = "golden"
        elif self.device == "cpu" and self.task_name == "cpu_lowp":
            self.role = "benchmark"
            if os.environ.get("GDN_ATK_BENCHMARK_ARCH") not in {"arch22", "arch35"}:
                raise RuntimeError("Set GDN_ATK_BENCHMARK_ARCH=arch22/arch35 or use run_double_benchmark.sh")
        else:
            raise RuntimeError(
                f"{OP_NAME} 无法识别 ATK 角色：device={self.device!r}, "
                f"name={self.task_name!r}, is_benchmark_task={self.is_benchmark_task}"
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        spec = dict(_case_spec(input_data, OP_NAME))
        # ATK assigns hashed IDs above this threshold to accuracy_lt copies.
        # The tensors are generated inside this executor, so varying ATK's
        # marker tensors alone would otherwise repeat the original inputs.
        # All three roles receive the same copy ID and therefore the same seed.
        atk_case_id = int(self.task_result.case_config.id)
        threshold = int(self.task_result.task_info.accuracy_lt_threshold)
        if atk_case_id > threshold:
            spec["seed"] = int(spec.get("seed", 20260911)) + atk_case_id
        print(
            f"[gdn-double-atk] case={spec.get('case_id', '?')} role={self.role} "
            f"target={'cpu_fp64' if self.role == 'golden' else ('cpu_fp32' if self.role == 'benchmark' else OP_NAME)} "
            f"atk_case={atk_case_id} seed={spec.get('seed', 20260911)} "
            f"benchmark_arch={os.environ.get('GDN_ATK_BENCHMARK_ARCH', 'unset')}",
            flush=True,
        )
        if self.device == "cpu":
            outputs = run_cpu(spec, high_precision=self.high_precision)
        elif self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        else:
            raise RuntimeError(f"{OP_NAME} only supports CPU golden and NPU DUT nodes")
        # Keep FP64 accumulation inside the golden chain, then normalize all
        # visible outputs to FP32 so ATK compares the three roles with one
        # dtype.  This does not alter the FP32 benchmark computation.
        normalized = _finite_tuple(outputs, golden=self.role == "golden")
        if self.role == "dut":
            normalized = tuple(output.float() for output in normalized)
        if self.role == "benchmark":
            normalized = tuple(output.float() for output in normalized)
        return normalized
