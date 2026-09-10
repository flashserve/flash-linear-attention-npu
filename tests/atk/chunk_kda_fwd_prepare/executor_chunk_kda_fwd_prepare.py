"""chunk_kda_fwd_prepare 的 ATK executor 与独立 CPU 标杆。"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device


OP_NAME = "chunk_kda_fwd_prepare"
CHUNK_SIZE = 64
HEAD_DIM = 128
SUB_CHUNK = 16
DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_ints(value: Any) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return None
    return tuple(int(item) for item in text.split(","))


def _canonical_chunk_indices(
    cu_seqlens: Optional[tuple[int, ...]],
) -> Optional[tuple[int, ...]]:
    if cu_seqlens is None:
        return None
    indices = []
    for sequence, (begin, end) in enumerate(
        zip(cu_seqlens, cu_seqlens[1:])
    ):
        for chunk in range((end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE):
            indices.extend((sequence, chunk))
    return tuple(indices)


def _layout_from_bsnd(
    tensor: torch.Tensor,
    layout: str,
    *,
    scalar: bool = False,
) -> torch.Tensor:
    if layout == "BSND":
        return tensor.contiguous()
    if layout == "BNSD":
        return tensor.permute(0, 2, 1) if scalar else tensor.permute(0, 2, 1, 3)
    if layout == "TND":
        return tensor.squeeze(0).contiguous()
    if layout == "NTD":
        tensor = tensor.squeeze(0)
        return tensor.permute(1, 0) if scalar else tensor.permute(1, 0, 2)
    raise ValueError(f"{OP_NAME}: unsupported layout {layout!r}")


def _layout_to_bsnd(
    tensor: torch.Tensor,
    layout: str,
    *,
    scalar: bool = False,
) -> torch.Tensor:
    if layout == "BSND":
        return tensor
    if layout == "BNSD":
        return tensor.permute(0, 2, 1) if scalar else tensor.permute(0, 2, 1, 3)
    if layout == "TND":
        return tensor.unsqueeze(0)
    if layout == "NTD":
        tensor = tensor.permute(1, 0) if scalar else tensor.permute(1, 0, 2)
        return tensor.unsqueeze(0)
    raise ValueError(f"{OP_NAME}: unsupported layout {layout!r}")


def _quantized_random(
    shape,
    generator: torch.Generator,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype,
    device: torch.device,
    *,
    low: float,
    high: float,
) -> torch.Tensor:
    value = torch.rand(shape, generator=generator, dtype=torch.float32)
    value = value.mul(float(high) - float(low)).add(float(low))
    return value.to(source_dtype).to(target_dtype).to(device)


@dataclass
class PreparedInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: Optional[torch.Tensor]
    dt_bias: Optional[torch.Tensor]
    cu_seqlens: Optional[tuple[int, ...]]
    chunk_indices: Optional[tuple[int, ...]]


def build_inputs(
    spec: dict[str, Any],
    device: torch.device,
    *,
    high_precision: bool = False,
) -> PreparedInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(spec.get("seed", 20260910)))
    target_dtype = torch.float64 if high_precision else torch.bfloat16
    fp32_target = torch.float64 if high_precision else torch.float32

    batch = int(spec["B"])
    tokens = int(spec["T"])
    key_heads = int(spec["HK"])
    value_heads = int(spec["HV"])
    layout = str(spec["layout"])
    data_scale = float(spec.get("data_scale", 0.08))

    q_bsnd = _quantized_random(
        (batch, tokens, key_heads, HEAD_DIM),
        generator,
        torch.bfloat16,
        target_dtype,
        device,
        low=-data_scale,
        high=data_scale,
    )
    k_bsnd = _quantized_random(
        (batch, tokens, key_heads, HEAD_DIM),
        generator,
        torch.bfloat16,
        target_dtype,
        device,
        low=-data_scale,
        high=data_scale,
    )
    v_bsnd = _quantized_random(
        (batch, tokens, value_heads, HEAD_DIM),
        generator,
        torch.bfloat16,
        target_dtype,
        device,
        low=-data_scale,
        high=data_scale,
    )

    gate_dtype = DTYPES[str(spec["gate_dtype"])]
    gate_target = fp32_target if high_precision else gate_dtype
    gate_scale = float(spec.get("gate_scale", 1.0))
    if _as_bool(spec["use_gate_in_kernel"]):
        gate_low, gate_high = -gate_scale, gate_scale
    else:
        gate_low, gate_high = -0.02 * gate_scale, -0.002 * gate_scale
    g_bsnd = _quantized_random(
        (batch, tokens, value_heads, HEAD_DIM),
        generator,
        gate_dtype,
        gate_target,
        device,
        low=gate_low,
        high=gate_high,
    )

    beta_dtype = DTYPES[str(spec["beta_dtype"])]
    beta_target = fp32_target if high_precision else beta_dtype
    beta_scale = float(spec.get("beta_scale", 1.0))
    beta_bsnd = _quantized_random(
        (batch, tokens, value_heads),
        generator,
        beta_dtype,
        beta_target,
        device,
        low=-beta_scale,
        high=beta_scale,
    )

    a_log = None
    if _as_bool(spec["use_gate_in_kernel"]):
        a_log = _quantized_random(
            (value_heads,),
            generator,
            torch.float32,
            fp32_target,
            device,
            low=-6.0,
            high=-2.0,
        )
    dt_bias = None
    if _as_bool(spec.get("dt_bias", False)):
        dt_bias = _quantized_random(
            (value_heads * HEAD_DIM,),
            generator,
            torch.float32,
            fp32_target,
            device,
            low=-2.0,
            high=2.0,
        )

    cu_seqlens = _parse_ints(spec.get("cu_seqlens"))
    chunk_indices = (
        _canonical_chunk_indices(cu_seqlens)
        if cu_seqlens is not None
        and _as_bool(spec.get("explicit_chunk_indices", False))
        else None
    )
    return PreparedInputs(
        q=_layout_from_bsnd(q_bsnd, layout),
        k=_layout_from_bsnd(k_bsnd, layout),
        v=_layout_from_bsnd(v_bsnd, layout),
        g=_layout_from_bsnd(g_bsnd, layout),
        beta=_layout_from_bsnd(beta_bsnd, layout, scalar=True),
        a_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def _chunk_spans(
    batch: int,
    tokens: int,
    cu_seqlens: Optional[tuple[int, ...]],
):
    if cu_seqlens is None:
        return [
            (batch_id, begin, min(begin + CHUNK_SIZE, tokens))
            for batch_id in range(batch)
            for begin in range(0, tokens, CHUNK_SIZE)
        ]
    return [
        (0, begin, min(begin + CHUNK_SIZE, sequence_end))
        for sequence_begin, sequence_end in zip(cu_seqlens, cu_seqlens[1:])
        for begin in range(sequence_begin, sequence_end, CHUNK_SIZE)
    ]


def _bf16_value(value: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    return value.to(torch.bfloat16).to(compute_dtype)


def _gate_and_beta(
    inputs: PreparedInputs,
    spec: dict[str, Any],
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = str(spec["layout"])
    gate = _layout_to_bsnd(inputs.g, layout).to(compute_dtype)
    if _as_bool(spec["use_gate_in_kernel"]):
        if inputs.a_log is None:
            raise RuntimeError(f"{OP_NAME}: CPU reference requires a_log")
        a = torch.exp(inputs.a_log.to(compute_dtype)).view(
            1, 1, int(spec["HV"]), 1
        )
        raw = gate
        if inputs.dt_bias is not None:
            raw = raw + inputs.dt_bias.to(compute_dtype).view(
                1, 1, int(spec["HV"]), HEAD_DIM
            )
        if _as_bool(spec["safe_gate"]):
            gate = float(spec["lower_bound"]) * torch.sigmoid(a * raw)
        else:
            gate = -a * torch.nn.functional.softplus(raw)

    beta = _layout_to_bsnd(inputs.beta, layout, scalar=True).to(compute_dtype)
    if _as_bool(spec["use_beta_sigmoid_in_kernel"]):
        beta = torch.sigmoid(beta)
        if _as_bool(spec["allow_neg_eigval"]):
            beta = beta * 2.0
    return gate, beta


def _normalize_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    epsilon: float,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stat_shape = q.shape[:-1]
    if not enabled:
        ones = torch.ones(stat_shape, dtype=q.dtype, device=q.device)
        return q, k, ones, ones.clone()
    q_rstd = torch.rsqrt(q.square().sum(dim=-1) + float(epsilon))
    k_rstd = torch.rsqrt(k.square().sum(dim=-1) + float(epsilon))
    return (
        q * q_rstd.unsqueeze(-1),
        k * k_rstd.unsqueeze(-1),
        q_rstd,
        k_rstd,
    )


def _factor(
    value: torch.Tensor,
    use_exp2: bool,
    base2_lower: int,
    base2_upper: int,
) -> torch.Tensor:
    if use_exp2:
        return torch.exp2(value.clamp(float(base2_lower), float(base2_upper)))
    lower = float(base2_lower) * math.log(2.0)
    upper = float(base2_upper) * math.log(2.0)
    return torch.exp(value.clamp(lower, upper))


def _s4_scores(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    use_exp2: bool,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q_hat.shape[0]
    aqk = torch.zeros((rows, CHUNK_SIZE), dtype=compute_dtype, device=q_hat.device)
    lkk = torch.zeros_like(aqk)
    for band_begin in range(0, rows, SUB_CHUNK):
        band_end = min(band_begin + SUB_CHUNK, rows)
        reference_row = (band_begin + band_end) // 2
        reference = gate[reference_row]
        left_factor = _factor(
            gate[band_begin:band_end] - reference,
            use_exp2,
            -126,
            120,
        )
        right_factor = _factor(
            reference - gate[:band_end],
            use_exp2,
            -126,
            120,
        )
        q_plus = _bf16_value(
            q_hat[band_begin:band_end] * left_factor,
            compute_dtype,
        )
        k_plus = _bf16_value(
            k_hat[band_begin:band_end] * left_factor,
            compute_dtype,
        )
        k_minus = _bf16_value(k_hat[:band_end] * right_factor, compute_dtype)
        raw_qk = q_plus @ k_minus.transpose(0, 1)
        raw_kk = k_plus @ k_minus.transpose(0, 1)
        for local_row, row in enumerate(range(band_begin, band_end)):
            aqk[row, : row + 1] = raw_qk[local_row, : row + 1] * float(scale)
            if row:
                lkk[row, :row] = raw_kk[local_row, :row] * beta[row]
    return _bf16_value(aqk, compute_dtype), lkk


def _block_inverse(
    lkk: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    rows = lkk.shape[0]
    padded_rows = 32 if rows <= 32 else 64
    padded = torch.zeros(
        (padded_rows, padded_rows), dtype=compute_dtype, device=lkk.device
    )
    padded[:rows, :rows] = lkk[:, :rows]
    eye32 = torch.eye(32, dtype=compute_dtype, device=lkk.device)
    x0 = torch.linalg.solve_triangular(
        padded[:32, :32] + eye32,
        eye32,
        upper=False,
    )
    inverse = torch.zeros_like(padded)
    inverse[:32, :32] = x0
    if padded_rows == 64:
        x1 = torch.linalg.solve_triangular(
            padded[32:64, 32:64] + eye32,
            eye32,
            upper=False,
        )
        inverse[32:64, 32:64] = x1
        inverse[32:64, :32] = -x1 @ (padded[32:64, :32] @ x0)
    output = torch.zeros(
        (rows, CHUNK_SIZE), dtype=compute_dtype, device=lkk.device
    )
    output[:, :padded_rows] = _bf16_value(inverse[:rows], compute_dtype)
    return output


def _reference(
    inputs: PreparedInputs,
    spec: dict[str, Any],
) -> tuple[Optional[torch.Tensor], ...]:
    layout = str(spec["layout"])
    q = _layout_to_bsnd(inputs.q, layout).to(torch.float64)
    k = _layout_to_bsnd(inputs.k, layout).to(torch.float64)
    v = _layout_to_bsnd(inputs.v, layout).to(torch.float64)
    compute_dtype = q.dtype
    batch, tokens, key_heads, _ = q.shape
    value_heads = v.shape[2]
    group_size = value_heads // key_heads

    q_norm, k_norm, q_rstd, k_rstd = _normalize_qk(
        q,
        k,
        float(spec["epsilon"]),
        _as_bool(spec["use_qk_l2norm_in_kernel"]),
    )
    q_hat = _bf16_value(q_norm, compute_dtype)
    k_hat = _bf16_value(k_norm, compute_dtype)
    gate_step, beta_eff = _gate_and_beta(inputs, spec, compute_dtype)
    gate = torch.zeros_like(gate_step)
    spans = _chunk_spans(batch, tokens, inputs.cu_seqlens)
    gate_scale = 1.0 / math.log(2.0) if _as_bool(spec["use_exp2"]) else 1.0
    for batch_id, begin, end in spans:
        gate[batch_id, begin:end] = (
            torch.cumsum(gate_step[batch_id, begin:end], dim=0) * gate_scale
        )

    vector_shape = (batch, value_heads, tokens, HEAD_DIM)
    matrix_shape = (batch, value_heads, tokens, CHUNK_SIZE)
    gk = gate.permute(0, 2, 1, 3).contiguous()
    aqk = torch.zeros(matrix_shape, dtype=compute_dtype, device=q.device)
    akk = torch.zeros_like(aqk)
    w = torch.zeros(vector_shape, dtype=compute_dtype, device=q.device)
    u = torch.zeros_like(w)
    qg = torch.zeros_like(w)
    kg = torch.zeros_like(w)
    qg_scaled = torch.zeros_like(w)

    use_exp2 = _as_bool(spec["use_exp2"])
    for batch_id, begin, end in spans:
        rows = end - begin
        for value_head in range(value_heads):
            key_head = value_head // group_size
            q_block = q_hat[batch_id, begin:end, key_head]
            k_block = k_hat[batch_id, begin:end, key_head]
            v_block = v[batch_id, begin:end, value_head]
            gate_block = gate[batch_id, begin:end, value_head]
            beta_block = beta_eff[batch_id, begin:end, value_head]

            aqk_block, lkk = _s4_scores(
                q_block,
                k_block,
                gate_block,
                beta_block,
                float(spec["scale"]),
                use_exp2,
                compute_dtype,
            )
            akk_block = _block_inverse(lkk, compute_dtype)
            exp_gate = _factor(gate_block, use_exp2, -80, 80)
            qg_block = _bf16_value(q_block * exp_gate, compute_dtype)
            qg_scaled_block = _bf16_value(
                qg_block * float(spec["scale"]), compute_dtype
            )
            last_gate = gate_block[-1]
            kg_block = _bf16_value(
                k_block
                * _factor(last_gate - gate_block, use_exp2, -80, 80),
                compute_dtype,
            )

            # K_beta_g 与 qg_scaled 都必须消费已经 BF16 舍入的中间值。
            k_positive = _bf16_value(k_block * exp_gate, compute_dtype)
            k_beta_g = _bf16_value(
                k_positive * beta_block.unsqueeze(-1), compute_dtype
            )
            v_beta = _bf16_value(
                v_block * beta_block.unsqueeze(-1), compute_dtype
            )
            padded_rows = 32 if rows <= 32 else 64
            akk_operand = torch.zeros(
                (padded_rows, padded_rows), dtype=compute_dtype, device=q.device
            )
            akk_operand[:rows] = akk_block[:, :padded_rows]
            k_operand = torch.zeros(
                (padded_rows, HEAD_DIM), dtype=compute_dtype, device=q.device
            )
            v_operand = torch.zeros_like(k_operand)
            k_operand[:rows] = k_beta_g
            v_operand[:rows] = v_beta
            w_block = _bf16_value(akk_operand @ k_operand, compute_dtype)[:rows]
            u_block = _bf16_value(akk_operand @ v_operand, compute_dtype)[:rows]

            aqk[batch_id, value_head, begin:end] = aqk_block
            akk[batch_id, value_head, begin:end] = akk_block
            w[batch_id, value_head, begin:end] = w_block
            u[batch_id, value_head, begin:end] = u_block
            qg[batch_id, value_head, begin:end] = qg_block
            kg[batch_id, value_head, begin:end] = kg_block
            qg_scaled[batch_id, value_head, begin:end] = qg_scaled_block

    q_hat_out = q_hat.permute(0, 2, 1, 3).contiguous()
    k_hat_out = k_hat.permute(0, 2, 1, 3).contiguous()
    q_rstd_out = q_rstd.permute(0, 2, 1).contiguous()
    k_rstd_out = k_rstd.permute(0, 2, 1).contiguous()
    beta_eff_out = beta_eff.permute(0, 2, 1).contiguous()
    outputs = (
        gk,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        qg_scaled,
        q_hat_out,
        k_hat_out,
        q_rstd_out,
        k_rstd_out,
        beta_eff_out,
    )
    if layout in {"TND", "NTD"}:
        outputs = tuple(output.squeeze(0) for output in outputs)
    output_masks = {
        "none": (
            True, True, False, True, True, False, True, True,
            False, False, False, False, False,
        ),
        "recompute": (
            True, True, True, True, True, False, True, True,
            True, True, True, True, True,
        ),
        "save": (True,) * 13,
    }
    backward_mode = str(spec.get("backward_mode", "save"))
    try:
        output_mask = output_masks[backward_mode]
    except KeyError as exc:
        raise ValueError(
            "backward_mode 必须是 none、recompute 或 save。"
        ) from exc
    return tuple(
        output if enabled else None
        for output, enabled in zip(outputs, output_mask)
    )


def run_cpu(spec: dict[str, Any], inputs: PreparedInputs):
    return _reference(inputs, spec)


def run_npu(spec: dict[str, Any], inputs: PreparedInputs):
    from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

    return chunk_kda_fwd_prepare(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        float(spec["scale"]),
        layout=str(spec["layout"]),
        chunk_size=int(spec["chunk_size"]),
        epsilon=float(spec["epsilon"]),
        use_qk_l2norm_in_kernel=_as_bool(
            spec["use_qk_l2norm_in_kernel"]
        ),
        use_gate_in_kernel=_as_bool(spec["use_gate_in_kernel"]),
        use_beta_sigmoid_in_kernel=_as_bool(
            spec["use_beta_sigmoid_in_kernel"]
        ),
        allow_neg_eigval=_as_bool(spec["allow_neg_eigval"]),
        safe_gate=_as_bool(spec["safe_gate"]),
        lower_bound=float(spec["lower_bound"]),
        use_exp2=_as_bool(spec["use_exp2"]),
        a_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        cu_seqlens=inputs.cu_seqlens,
        chunk_indices=inputs.chunk_indices,
        backward_mode=str(spec.get("backward_mode", "save")),
    )


@register("executor_chunk_kda_fwd_prepare")
class FunctionApi(BaseApi):
    """只通过稳定 ctypes 入口执行 Prepare。"""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.spec: Optional[dict[str, Any]] = None
        self.inputs: Optional[PreparedInputs] = None

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, OP_NAME)
        self.inputs = build_inputs(
            self.spec,
            _marker_device(input_data),
            high_precision=self.device == "cpu",
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        del with_output
        if self.spec is None or self.inputs is None:
            self.init_by_input_data(input_data)
        if self.spec is None or self.inputs is None:
            raise RuntimeError(f"{OP_NAME}: ATK input initialization failed")
        if self.device == "cpu":
            outputs = run_cpu(self.spec, self.inputs)
        elif self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(self.spec, self.inputs)
        else:
            raise RuntimeError(
                f"{OP_NAME} only supports CPU golden and NPU DUT nodes, "
                f"got {self.device!r}"
            )
        return _finite_tuple(outputs, golden=self.device == "cpu")

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        if self.spec is None:
            raise RuntimeError(f"{OP_NAME}: case spec is unavailable")
        return {
            "case_key": str(self.spec["case_key"]),
            "layout": str(self.spec["layout"]),
            "B": int(self.spec["B"]),
            "HK": int(self.spec["HK"]),
            "HV": int(self.spec["HV"]),
            "T": int(self.spec["T"]),
            "gate_dtype": str(self.spec["gate_dtype"]),
            "beta_dtype": str(self.spec["beta_dtype"]),
        }
