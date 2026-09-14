"""chunk_kda_fwd_prepare 的 ATK executor 与独立 CPU 标杆。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

OP_NAME = "chunk_kda_fwd_prepare"
CHUNK_SIZE = 64
HEAD_DIM = 128
SUB_CHUNK = 16
REFERENCE_BATCH_SIZE = max(
    1, int(os.environ.get("KDA_PREPARE_ATK_REFERENCE_BATCH_SIZE", "256"))
)
OUTPUT_NAMES = (
    "gk",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "qg_scaled",
    "q_hat",
    "k_hat",
    "q_rstd",
    "k_rstd",
    "beta_eff",
)
OUTPUT_MASKS = {
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
OUTPUT_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.float32,
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _optional_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value) and value.numel() == 0:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value


def _normalize_int_list(value, name: str) -> Optional[tuple[int, ...]]:
    if value is None or (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and value[0] in (None, "null")
    ):
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an integer list or null")
    normalized = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be an integer")
        try:
            converted = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be an integer") from exc
        if isinstance(item, float) and not item.is_integer():
            raise ValueError(f"{name}[{index}] must be an integer")
        normalized.append(converted)
    return tuple(normalized)


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


def _derive_spec(values: dict[str, Any], case_id: int) -> dict[str, Any]:
    required = ("q", "k", "v", "g", "beta")
    if not all(torch.is_tensor(values.get(name)) for name in required):
        raise TypeError("q, k, v, g and beta must be direct tensor inputs")
    q = values["q"]
    v = values["v"]
    layout = str(values["layout"])
    if layout == "BNSD":
        batch, key_heads, tokens, key_dim = q.shape
        value_heads, value_dim = v.shape[1], v.shape[3]
    elif layout == "BSND":
        batch, tokens, key_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
    elif layout == "NTD":
        key_heads, tokens, key_dim = q.shape
        batch, value_heads, value_dim = 1, v.shape[0], v.shape[2]
    elif layout == "TND":
        tokens, key_heads, key_dim = q.shape
        batch, value_heads, value_dim = 1, v.shape[1], v.shape[2]
    else:
        raise ValueError(f"unsupported layout: {layout!r}")
    return {
        "case_id": case_id,
        "B": int(batch),
        "HK": int(key_heads),
        "HV": int(value_heads),
        "T": int(tokens),
        "K": int(key_dim),
        "V": int(value_dim),
        "layout": layout,
        "chunk_size": int(values["chunk_size"]),
        "scale": float(values["scale"]),
        "epsilon": float(values["epsilon"]),
        "use_qk_l2norm_in_kernel": _as_bool(
            values["use_qk_l2norm_in_kernel"]
        ),
        "use_gate_in_kernel": _as_bool(values["use_gate_in_kernel"]),
        "use_beta_sigmoid_in_kernel": _as_bool(
            values["use_beta_sigmoid_in_kernel"]
        ),
        "allow_neg_eigval": _as_bool(values["allow_neg_eigval"]),
        "safe_gate": _as_bool(values["safe_gate"]),
        "lower_bound": float(values["lower_bound"]),
        "use_exp2": _as_bool(values["use_exp2"]),
        "backward_mode": str(values["backward_mode"]),
    }


def _direct_inputs(
    values: dict[str, Any], *, high_precision: bool
) -> PreparedInputs:
    def convert(value):
        value = _optional_tensor(value)
        if value is not None and high_precision and value.is_floating_point():
            return value.to(torch.float64)
        return value

    return PreparedInputs(
        q=convert(values["q"]),
        k=convert(values["k"]),
        v=convert(values["v"]),
        g=convert(values["g"]),
        beta=convert(values["beta"]),
        a_log=convert(values.get("a_log")),
        dt_bias=convert(values.get("dt_bias")),
        cu_seqlens=_normalize_int_list(values.get("cu_seqlens"), "cu_seqlens"),
        chunk_indices=_normalize_int_list(
            values.get("chunk_indices"), "chunk_indices"
        ),
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


def _s4_scores_batched(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    use_exp2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, rows, _ = q_hat.shape
    aqk = torch.zeros(
        (batch_size, rows, CHUNK_SIZE), dtype=q_hat.dtype, device=q_hat.device
    )
    lkk = torch.zeros_like(aqk)
    for band_begin in range(0, rows, SUB_CHUNK):
        band_end = min(band_begin + SUB_CHUNK, rows)
        reference_row = (band_begin + band_end) // 2
        reference = gate[:, reference_row : reference_row + 1]
        left_factor = _factor(
            gate[:, band_begin:band_end] - reference,
            use_exp2,
            -126,
            120,
        )
        right_factor = _factor(
            reference - gate[:, :band_end],
            use_exp2,
            -126,
            120,
        )
        q_plus = _bf16_value(q_hat[:, band_begin:band_end] * left_factor, q_hat.dtype)
        k_plus = _bf16_value(k_hat[:, band_begin:band_end] * left_factor, q_hat.dtype)
        k_minus = _bf16_value(k_hat[:, :band_end] * right_factor, q_hat.dtype)
        raw_qk = torch.bmm(q_plus, k_minus.transpose(1, 2))
        raw_kk = torch.bmm(k_plus, k_minus.transpose(1, 2))

        row_indices = torch.arange(
            band_begin, band_end, dtype=torch.int64, device=q_hat.device
        )
        column_indices = torch.arange(
            band_end, dtype=torch.int64, device=q_hat.device
        )
        causal = column_indices.unsqueeze(0) <= row_indices.unsqueeze(1)
        strict = column_indices.unsqueeze(0) < row_indices.unsqueeze(1)
        aqk[:, band_begin:band_end, :band_end] = torch.where(
            causal.unsqueeze(0), raw_qk * float(scale), 0.0
        )
        weighted_kk = raw_kk * beta[:, band_begin:band_end].unsqueeze(-1)
        lkk[:, band_begin:band_end, :band_end] = torch.where(
            strict.unsqueeze(0), weighted_kk, 0.0
        )
    return _bf16_value(aqk, q_hat.dtype), lkk


def _block_inverse_batched(lkk: torch.Tensor) -> torch.Tensor:
    batch_size, rows, _ = lkk.shape
    padded_rows = 32 if rows <= 32 else 64
    padded = torch.zeros(
        (batch_size, padded_rows, padded_rows),
        dtype=lkk.dtype,
        device=lkk.device,
    )
    padded[:, :rows, :rows] = lkk[:, :, :rows]
    eye32 = torch.eye(32, dtype=lkk.dtype, device=lkk.device).expand(
        batch_size, -1, -1
    )
    x0 = torch.linalg.solve_triangular(
        padded[:, :32, :32] + eye32,
        eye32,
        upper=False,
    )
    inverse = torch.zeros_like(padded)
    inverse[:, :32, :32] = x0
    if padded_rows == 64:
        x1 = torch.linalg.solve_triangular(
            padded[:, 32:64, 32:64] + eye32,
            eye32,
            upper=False,
        )
        inverse[:, 32:64, 32:64] = x1
        inverse[:, 32:64, :32] = -torch.bmm(
            x1,
            torch.bmm(padded[:, 32:64, :32], x0),
        )
    output = torch.zeros(
        (batch_size, rows, CHUNK_SIZE), dtype=lkk.dtype, device=lkk.device
    )
    output[:, :, :padded_rows] = _bf16_value(inverse[:, :rows], lkk.dtype)
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

    backward_mode = str(spec.get("backward_mode", "save"))
    try:
        output_mask = OUTPUT_MASKS[backward_mode]
    except KeyError as exc:
        raise ValueError(
            "backward_mode 必须是 none、recompute 或 save。"
        ) from exc

    vector_shape = (batch, value_heads, tokens, HEAD_DIM)
    matrix_shape = (batch, value_heads, tokens, CHUNK_SIZE)
    gk = gate.permute(0, 2, 1, 3).contiguous()
    aqk = torch.zeros(matrix_shape, dtype=compute_dtype, device=q.device)
    akk = torch.zeros_like(aqk) if output_mask[2] else None
    w = torch.zeros(vector_shape, dtype=compute_dtype, device=q.device)
    u = torch.zeros_like(w)
    qg = torch.zeros_like(w) if output_mask[5] else None
    kg = torch.zeros_like(w)
    qg_scaled = torch.zeros_like(w)

    use_exp2 = _as_bool(spec["use_exp2"])
    records_by_rows: dict[int, list[tuple[int, int, int, int]]] = {}
    for batch_id, begin, end in spans:
        rows = end - begin
        records_by_rows.setdefault(rows, []).extend(
            (batch_id, begin, end, value_head)
            for value_head in range(value_heads)
        )

    for rows, records in records_by_rows.items():
        padded_rows = 32 if rows <= 32 else 64
        for offset in range(0, len(records), REFERENCE_BATCH_SIZE):
            batch_records = records[offset : offset + REFERENCE_BATCH_SIZE]
            q_block = torch.stack(
                [
                    q_hat[batch_id, begin:end, value_head // group_size]
                    for batch_id, begin, end, value_head in batch_records
                ]
            )
            k_block = torch.stack(
                [
                    k_hat[batch_id, begin:end, value_head // group_size]
                    for batch_id, begin, end, value_head in batch_records
                ]
            )
            v_block = torch.stack(
                [
                    v[batch_id, begin:end, value_head]
                    for batch_id, begin, end, value_head in batch_records
                ]
            )
            gate_block = torch.stack(
                [
                    gate[batch_id, begin:end, value_head]
                    for batch_id, begin, end, value_head in batch_records
                ]
            )
            beta_block = torch.stack(
                [
                    beta_eff[batch_id, begin:end, value_head]
                    for batch_id, begin, end, value_head in batch_records
                ]
            )

            aqk_block, lkk = _s4_scores_batched(
                q_block,
                k_block,
                gate_block,
                beta_block,
                float(spec["scale"]),
                use_exp2,
            )
            akk_block = _block_inverse_batched(lkk)
            exp_gate = _factor(gate_block, use_exp2, -80, 80)
            qg_block = _bf16_value(q_block * exp_gate, compute_dtype)
            qg_scaled_block = _bf16_value(
                qg_block * float(spec["scale"]), compute_dtype
            )
            last_gate = gate_block[:, -1:]
            kg_block = _bf16_value(
                k_block * _factor(last_gate - gate_block, use_exp2, -80, 80),
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
            akk_operand = torch.zeros(
                (len(batch_records), padded_rows, padded_rows),
                dtype=compute_dtype,
                device=q.device,
            )
            akk_operand[:, :rows] = akk_block[:, :, :padded_rows]
            k_operand = torch.zeros(
                (len(batch_records), padded_rows, HEAD_DIM),
                dtype=compute_dtype,
                device=q.device,
            )
            v_operand = torch.zeros_like(k_operand)
            k_operand[:, :rows] = k_beta_g
            v_operand[:, :rows] = v_beta
            w_block = _bf16_value(
                torch.bmm(akk_operand, k_operand), compute_dtype
            )[:, :rows]
            u_block = _bf16_value(
                torch.bmm(akk_operand, v_operand), compute_dtype
            )[:, :rows]

            for index, (batch_id, begin, end, value_head) in enumerate(
                batch_records
            ):
                aqk[batch_id, value_head, begin:end] = aqk_block[index]
                if akk is not None:
                    akk[batch_id, value_head, begin:end] = akk_block[index]
                w[batch_id, value_head, begin:end] = w_block[index]
                u[batch_id, value_head, begin:end] = u_block[index]
                if qg is not None:
                    qg[batch_id, value_head, begin:end] = qg_block[index]
                kg[batch_id, value_head, begin:end] = kg_block[index]
                qg_scaled[batch_id, value_head, begin:end] = qg_scaled_block[index]

    q_hat_out = (
        q_hat.permute(0, 2, 1, 3).contiguous() if output_mask[8] else None
    )
    k_hat_out = (
        k_hat.permute(0, 2, 1, 3).contiguous() if output_mask[9] else None
    )
    q_rstd_out = q_rstd.permute(0, 2, 1).contiguous() if output_mask[10] else None
    k_rstd_out = k_rstd.permute(0, 2, 1).contiguous() if output_mask[11] else None
    beta_eff_out = (
        beta_eff.permute(0, 2, 1).contiguous() if output_mask[12] else None
    )
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
        outputs = tuple(
            output.squeeze(0) if output is not None else None
            for output in outputs
        )
    return outputs


def run_cpu(spec: dict[str, Any], inputs: PreparedInputs):
    return _reference(inputs, spec)


def run_npu(spec: dict[str, Any], inputs: PreparedInputs):
    from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

    outputs = chunk_kda_fwd_prepare(
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
    return outputs


def _expected_output_shapes(spec: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    batch = int(spec["B"])
    key_heads = int(spec["HK"])
    value_heads = int(spec["HV"])
    tokens = int(spec["T"])
    packed = str(spec["layout"]) in {"NTD", "TND"}
    if packed:
        value_matrix = (value_heads, tokens, HEAD_DIM)
        value_block = (value_heads, tokens, CHUNK_SIZE)
        key_matrix = (key_heads, tokens, HEAD_DIM)
        key_scalar = (key_heads, tokens)
        value_scalar = (value_heads, tokens)
    else:
        value_matrix = (batch, value_heads, tokens, HEAD_DIM)
        value_block = (batch, value_heads, tokens, CHUNK_SIZE)
        key_matrix = (batch, key_heads, tokens, HEAD_DIM)
        key_scalar = (batch, key_heads, tokens)
        value_scalar = (batch, value_heads, tokens)
    return (
        value_matrix,
        value_block,
        value_block,
        value_matrix,
        value_matrix,
        value_matrix,
        value_matrix,
        value_matrix,
        key_matrix,
        key_matrix,
        key_scalar,
        key_scalar,
        value_scalar,
    )


def _validate_output_contract(
    spec: dict[str, Any],
    outputs,
    *,
    check_dtype: bool,
) -> None:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(
            f"{OP_NAME}: expected {len(OUTPUT_NAMES)} output slots, "
            f"got {type(outputs).__name__} with "
            f"{len(outputs) if isinstance(outputs, (tuple, list)) else 'unknown'}"
        )
    mode = str(spec.get("backward_mode", "save"))
    mask = OUTPUT_MASKS[mode]
    shapes = _expected_output_shapes(spec)
    for name, output, enabled, shape, dtype in zip(
        OUTPUT_NAMES, outputs, mask, shapes, OUTPUT_DTYPES
    ):
        if not enabled:
            if output is not None:
                raise RuntimeError(
                    f"{OP_NAME}: output {name} must be None in {mode} mode"
                )
            continue
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"{OP_NAME}: output {name} is required in {mode} mode"
            )
        if tuple(output.shape) != shape:
            raise RuntimeError(
                f"{OP_NAME}: output {name} shape {tuple(output.shape)} != {shape}"
            )
        if check_dtype and output.dtype != dtype:
            raise RuntimeError(
                f"{OP_NAME}: output {name} dtype {output.dtype} != {dtype}"
            )


@register("executor_chunk_kda_fwd_prepare")
class FunctionApi(BaseApi):
    """直接消费 ATK tensor，并通过稳定 ctypes 入口执行 Prepare。"""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.task_result = task_result
        self.spec: Optional[dict[str, Any]] = None
        self.inputs: Optional[PreparedInputs] = None
        case_config = getattr(task_result, "case_config", None)
        case_id = (
            case_config.get("id")
            if isinstance(case_config, dict)
            else getattr(case_config, "id", None)
        )
        self.runtime_case_id = 0 if case_id is None else int(case_id)

    def init_by_input_data(self, input_data: InputDataset):
        values = input_data.kwargs
        self.spec = _derive_spec(values, self.runtime_case_id)
        self.inputs = _direct_inputs(
            values, high_precision=self.device == "cpu"
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
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
        _validate_output_contract(
            self.spec,
            outputs,
            check_dtype=self.device in {"npu", "pyaclnn"},
        )
        if not with_output:
            return None
        if self.device in {"npu", "pyaclnn"}:
            torch.npu.synchronize()
        visible = []
        for output in outputs:
            if output is None or not isinstance(output, torch.Tensor):
                continue
            if output.is_floating_point() and not torch.isfinite(
                output.float()
            ).all().item():
                raise RuntimeError("output contains NaN or Inf")
            if self.device == "cpu" and output.dtype == torch.float64:
                output = output.to(torch.float32)
            visible.append(output)
        return tuple(visible)

    def export_custom_data(self, input_data: InputDataset):
        del input_data
        if self.spec is None:
            raise RuntimeError(f"{OP_NAME}: direct input spec is unavailable")
        case_config = getattr(self.task_result, "case_config", None)
        case_name = (
            case_config.get("name", OP_NAME)
            if isinstance(case_config, dict)
            else getattr(case_config, "name", OP_NAME)
        )
        return {
            "case_key": str(case_name),
            "layout": str(self.spec["layout"]),
            "B": int(self.spec["B"]),
            "HK": int(self.spec["HK"]),
            "HV": int(self.spec["HV"]),
            "T": int(self.spec["T"]),
            "backward_mode": str(self.spec["backward_mode"]),
        }
