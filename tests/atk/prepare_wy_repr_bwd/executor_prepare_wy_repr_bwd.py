"""ATK executor for prepare_wy_repr_bwd.

The CPU golden is embedded from fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/test_final_golden.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from fla_npu.ops import ascendc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _case_spec,
    _chunks,
    _finite_tuple,
    _gate,
    _marker_device,
    _orig_dtype,
    _rand,
    _randn,
)


OP_NAME = "prepare_wy_repr_bwd"


def build_inputs(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    data_dtype = _orig_dtype(dtype_name)
    seed = int(spec.get("seed", 20260817))
    B, HK, HV, T, K, V = (int(spec[x]) for x in ("B", "HK", "HV", "T", "K", "V"))
    chunk_size = int(spec["chunk_size"])
    return {
        "k": _randn((B, HK, T, K), dtype_name, data_dtype, device, seed + 1),
        "v": _randn((B, HV, T, V), dtype_name, data_dtype, device, seed + 2),
        "beta": _rand((B, HV, T), "fp32", torch.float32, device, seed + 3, 0.1, 0.9),
        "A": _randn((B, HV, T, chunk_size), dtype_name, data_dtype, device, seed + 4),
        "g": _gate((B, HV, T), torch.float32, device, seed + 5),
        "dw": _randn((B, HV, T, K), dtype_name, data_dtype, device, seed + 6),
        "du": _randn((B, HV, T, V), dtype_name, data_dtype, device, seed + 7),
        "chunk_size": chunk_size,
    }


def _compute_da_golden(inputs: dict[str, Any], high_precision: bool) -> torch.Tensor:
    """Port of test_da.py::compute_dA_cpu for the dense single-case route."""
    k, v = inputs["k"], inputs["v"]
    beta, A, g = inputs["beta"], inputs["A"], inputs["g"]
    dw, du = inputs["dw"], inputs["du"]
    B, HK, T, _ = k.shape
    HV = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    calc_dtype = torch.float64 if high_precision else torch.float32
    dA = torch.zeros(A.shape, dtype=calc_dtype, device=A.device)
    group_size = HV // HK

    for b in range(B):
        for hv in range(HV):
            hk = hv // group_size
            for start, end in _chunks(T, chunk_size):
                length = end - start
                a_chunk = A[b, hv, start:end, :length].to(calc_dtype)
                dw_chunk = dw[b, hv, start:end].to(calc_dtype)
                du_chunk = du[b, hv, start:end].to(calc_dtype)
                k_chunk = k[b, hk, start:end].to(calc_dtype)
                v_chunk = v[b, hv, start:end].to(calc_dtype)
                beta_chunk = beta[b, hv, start:end].to(calc_dtype)
                g_chunk = g[b, hv, start:end].to(calc_dtype)

                causal = torch.tril(
                    torch.ones((length, length), dtype=torch.bool, device=A.device),
                    diagonal=-1,
                )
                k_beta_g = k_chunk * (beta_chunk * torch.exp(g_chunk)).unsqueeze(-1)
                v_beta = v_chunk * beta_chunk.unsqueeze(-1)
                raw = torch.matmul(dw_chunk, k_beta_g.T) + torch.matmul(du_chunk, v_beta.T)
                masked = torch.where(causal, raw, torch.zeros_like(raw))
                transformed = torch.matmul(a_chunk.T, torch.matmul(masked, a_chunk.T))
                gate_ratio = torch.exp(g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0))
                result = torch.where(causal, -transformed * gate_ratio, torch.zeros_like(transformed))
                dA[b, hv, start:end, :length] = result.T

    return dA if high_precision else dA.to(A.dtype)


def _quantize(value: torch.Tensor, dtype: torch.dtype, calc_dtype: torch.dtype) -> torch.Tensor:
    return value.to(dtype).to(calc_dtype)


def _compute_full_golden(
    inputs: dict[str, Any],
    high_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of test.py compute_d{k,v,beta,g}_golden functions."""
    k, v = inputs["k"], inputs["v"]
    beta, A, dA, g = inputs["beta"], inputs["A"], inputs["dA"], inputs["g"]
    dw, du = inputs["dw"], inputs["du"]
    B, HK, T, K = k.shape
    HV, V = v.shape[1], v.shape[3]
    chunk_size = int(inputs["chunk_size"])
    calc_dtype = torch.float64 if high_precision else torch.float32

    dk = torch.zeros((B, HK, T, K), dtype=calc_dtype, device=k.device)
    dv = torch.zeros((B, HV, T, V), dtype=calc_dtype, device=v.device)
    dbeta = torch.zeros((B, HV, T), dtype=calc_dtype, device=beta.device)
    dg = torch.zeros((B, HV, T), dtype=calc_dtype, device=g.device)
    heads_per_kv = HV // HK

    for b in range(B):
        for hv in range(HV):
            hk = hv // heads_per_kv
            for start, end in _chunks(T, chunk_size):
                length = end - start
                a = A[b, hv, start:end, :length].to(calc_dtype)
                da = dA[b, hv, start:end, :length].to(calc_dtype)
                kc = k[b, hk, start:end].to(calc_dtype)
                vc = v[b, hv, start:end].to(calc_dtype)
                dwc = dw[b, hv, start:end].to(calc_dtype)
                duc = du[b, hv, start:end].to(calc_dtype)
                bc = beta[b, hv, start:end].to(calc_dtype)
                gc = g[b, hv, start:end].to(calc_dtype)
                exp_g = torch.exp(gc)

                tmp_w = torch.matmul(a.T, dwc)
                tmp_u = torch.matmul(a.T, duc)
                da_t_k = torch.matmul(da.T, kc)
                k_beta = kc * bc.unsqueeze(-1)

                term0 = _quantize(
                    torch.matmul(da, _quantize(k_beta, k.dtype, calc_dtype)),
                    k.dtype,
                    calc_dtype,
                )
                term1 = _quantize(da_t_k, k.dtype, calc_dtype) * bc.unsqueeze(-1)
                dk_chunk = _quantize(term0 + term1, k.dtype, calc_dtype)
                dk_chunk = dk_chunk + _quantize(tmp_w, k.dtype, calc_dtype) * (
                    bc * exp_g
                ).unsqueeze(-1)
                if not high_precision:
                    dk_chunk = _quantize(dk_chunk, k.dtype, calc_dtype)
                dk[b, hk, start:end] += dk_chunk

                if high_precision:
                    dv_chunk = tmp_u * bc.unsqueeze(-1)
                else:
                    dv_chunk = _quantize(
                        _quantize(tmp_u, v.dtype, calc_dtype) * bc.unsqueeze(-1),
                        v.dtype,
                        calc_dtype,
                    )
                dv[b, hv, start:end] = dv_chunk

                if high_precision:
                    dbeta_chunk = torch.sum(
                        _quantize(da_t_k, k.dtype, calc_dtype) * kc,
                        dim=-1,
                    )
                    dbeta_chunk = _quantize(dbeta_chunk, k.dtype, calc_dtype)
                    dbeta_chunk = _quantize(
                        dbeta_chunk
                        + torch.sum(
                            _quantize(tmp_w, k.dtype, calc_dtype) * kc * exp_g.unsqueeze(-1),
                            dim=-1,
                        ),
                        k.dtype,
                        calc_dtype,
                    )
                    dbeta_chunk = dbeta_chunk + torch.sum(
                        _quantize(tmp_u, k.dtype, calc_dtype) * vc,
                        dim=-1,
                    )
                else:
                    a_low = A[b, hv, start:end, :length]
                    da_low = dA[b, hv, start:end, :length]
                    k_low = k[b, hk, start:end]
                    v_low = v[b, hv, start:end]
                    dw_low = dw[b, hv, start:end]
                    du_low = du[b, hv, start:end]
                    exp_g_low = torch.exp(g[b, hv, start:end])
                    da_t_k_low = torch.matmul(da_low.T, k_low)
                    tmp_w_low = torch.matmul(a_low.T, dw_low)
                    tmp_u_low = torch.matmul(a_low.T, du_low)
                    dbeta_chunk = torch.sum(da_t_k_low.to(k.dtype) * k_low, dim=-1)
                    dbeta_chunk = dbeta_chunk.to(k.dtype) + torch.sum(
                        tmp_w_low.to(k.dtype) * k_low * exp_g_low.unsqueeze(-1), dim=-1
                    )
                    dbeta_chunk = dbeta_chunk.to(k.dtype) + torch.sum(
                        tmp_u_low.to(k.dtype) * v_low, dim=-1
                    )
                    dbeta_chunk = dbeta_chunk.to(calc_dtype)
                dbeta[b, hv, start:end] = dbeta_chunk

                if high_precision:
                    tmp_w_for_dg = _quantize(tmp_w, k.dtype, calc_dtype)
                    gram = torch.matmul(kc, kc.T)
                    da_gram = da.T * (gram * bc.unsqueeze(-1))
                    dg_chunk = torch.sum(
                        tmp_w_for_dg * kc * (bc * exp_g).unsqueeze(-1),
                        dim=-1,
                    )
                    dg_chunk = dg_chunk + torch.sum(da_gram, dim=1) - torch.sum(
                        da_gram, dim=0
                    )
                else:
                    a_low = A[b, hv, start:end, :length]
                    da_low = dA[b, hv, start:end, :length]
                    k_low = k[b, hk, start:end]
                    dw_low = dw[b, hv, start:end]
                    beta_low = beta[b, hv, start:end]
                    exp_g_low = torch.exp(g[b, hv, start:end])
                    tmp_w_low = torch.matmul(a_low.T, dw_low)
                    k_beta_g_low = k_low * (beta_low * exp_g_low).unsqueeze(-1)
                    gram_low = torch.matmul(k_low, k_low.T) * beta_low.unsqueeze(-1)
                    da_gram_low = da_low.T * gram_low
                    dg_chunk = torch.sum(
                        tmp_w_low.to(g.dtype) * k_beta_g_low.to(g.dtype), dim=-1
                    )
                    dg_chunk = dg_chunk.to(g.dtype) + (
                        torch.sum(da_gram_low, dim=1) - torch.sum(da_gram_low, dim=0)
                    ).to(g.dtype)
                    dg_chunk = dg_chunk.to(calc_dtype)
                dg[b, hv, start:end] = dg_chunk

    if high_precision:
        return dk, dv, dbeta, dg
    return dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype), dg.to(g.dtype)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    inputs = build_inputs(spec, torch.device("cpu"))
    chained_inputs = dict(inputs)
    chained_inputs["dA"] = _compute_da_golden(inputs, high_precision)
    return _compute_full_golden(chained_inputs, high_precision)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    inputs = build_inputs(spec, _marker_device(input_data))

    return ascendc.prepare_wy_repr_bwd(
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["A"],
        inputs["dw"],
        inputs["du"],
        inputs["g"],
        inputs["chunk_size"],
        cu_seqlens=None,
        chunk_indices=None,
    )


@register("executor_prepare_wy_repr_bwd")
class FunctionApi(BaseApi):
    """ATK execution entry."""

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        else:
            raise RuntimeError(f"{OP_NAME} supports only NPU DUT and CPU benchmark nodes")
        return _finite_tuple(outputs, golden=self.device == "cpu")
