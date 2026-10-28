from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from _ascendc_common_executor import (
    _case_spec,
    _finite_tuple,
    _marker_device,
    _rand,
    _randn,
)


OP_NAME = "chunk_kda_bwd"
DIM = 128
CHUNK_SIZE = 64


def build_inputs(spec, device, high_precision=False):
    batch, heads, tokens = (int(spec[name]) for name in ("B", "H", "T"))
    shape = (batch, heads, tokens, DIM)
    seed = int(spec["seed"])
    dtype_name = str(spec.get("dtype", "bf16"))
    beta_type = str(spec.get("beta_dtype", "bf16"))
    if dtype_name != "bf16" or beta_type not in {"bf16", "fp32"}:
        raise ValueError("public V2 acceptance uses BF16 tokens and BF16/FP32 beta")
    calc = torch.float64 if high_precision else torch.bfloat16
    beta_calc = torch.float64 if high_precision else (
        torch.bfloat16 if beta_type == "bf16" else torch.float32
    )
    fp = torch.float64 if high_precision else torch.float32
    cu = spec.get("cu_seqlens")
    if cu is not None and (batch != 1 or not cu or int(cu[-1]) != tokens):
        raise ValueError("packed cases require B=1 and cu_seqlens[-1]=T")
    values = {
        "q": _randn(shape, dtype_name, calc, device, seed + 1, 0.05),
        "k": _randn(shape, dtype_name, calc, device, seed + 2, 0.05),
        "v": _randn(shape, dtype_name, calc, device, seed + 3, 0.05),
        "raw_g": _randn(shape, "fp32", fp, device, seed + 4, 1.25),
        "beta": _rand(shape[:-1], beta_type, beta_calc, device, seed + 5, 0.1, 0.9),
        "A_log": _randn((heads,), "fp32", fp, device, seed + 6, 0.12),
        "d_o": _randn(shape, dtype_name, calc, device, seed + 7, 51.2),
    }
    if spec.get("dt_bias", True):
        values["dt_bias"] = (
            _randn((heads, DIM), "fp32", torch.float32, device, seed + 8, 1.65) - 3.0
        ).to(fp)
    return values


def _l2norm(value):
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + 1.0e-6)


@torch.enable_grad()
def run_cpu(values, spec=None):
    spec = spec or {}
    names = ["q", "k", "v", "beta", "raw_g", "A_log"]
    if "dt_bias" in values:
        names.append("dt_bias")
    leaves = {
        name: values[name].detach().double().requires_grad_(True)
        for name in names
    }
    if spec.get("norm", False):
        q = _l2norm(leaves["q"]).to(torch.bfloat16).double()
        k = _l2norm(leaves["k"]).to(torch.bfloat16).double()
    else:
        q, k = leaves["q"], leaves["k"]
    raw = leaves["raw_g"]
    if "dt_bias" in leaves:
        raw = raw + leaves["dt_bias"].view(1, raw.shape[1], 1, DIM)
    decay = leaves["A_log"].exp()[None, :, None, None]
    gate = float(spec.get("lower_bound", -5.0)) * torch.sigmoid(decay * raw)
    batch, heads, tokens, dim = q.shape
    state = torch.zeros(batch, heads, dim, dim, dtype=torch.float64, device=q.device)
    outputs = []
    starts = set((spec.get("cu_seqlens") or [0, tokens])[:-1])
    for token in range(tokens):
        if token in starts:
            state = torch.zeros_like(state)
        state = state * gate[:, :, token].exp().unsqueeze(-1)
        key = k[:, :, token]
        residual = leaves["v"][:, :, token] - torch.einsum("bhkv,bhk->bhv", state, key)
        delta = residual * leaves["beta"][:, :, token].unsqueeze(-1)
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, :, token]) / math.sqrt(dim))
    loss = (torch.stack(outputs, dim=2) * values["d_o"].double()).sum()
    gradients = torch.autograd.grad(loss, [leaves[name] for name in names])
    if spec.get("cu_seqlens"):
        gradients = tuple(
            gradient.squeeze(0) if index < 5 else gradient
            for index, gradient in enumerate(gradients)
        )
    return tuple(gradient.float() for gradient in gradients)


def _stage_input(value, packed, scalar=False):
    if not packed:
        return value.contiguous()
    value = value.squeeze(0)
    return (value.permute(1, 0) if scalar else value.permute(1, 0, 2)).contiguous()


def _prepare_forward(values, spec):
    from fla_npu.ops import ascendc

    packed = bool(spec.get("cu_seqlens"))
    cu = spec.get("cu_seqlens")
    saved = bool(spec.get("disable_recompute", True))
    norm = bool(spec.get("norm", False))
    layout = "TND" if packed else "BNSD"
    stage = {
        name: _stage_input(values[name], packed, scalar=name == "beta")
        for name in ("q", "k", "v", "raw_g", "beta")
    }
    dt_bias = values.get("dt_bias")
    prepared = ascendc.npu_chunk_kda_fwd_prepare(
        stage["q"], stage["k"], stage["v"], stage["raw_g"], stage["beta"],
        1.0 / math.sqrt(DIM), CHUNK_SIZE,
        layout=layout, cu_seqlens=cu, safe_gate=True, lower_bound=-5.0,
        use_gate_in_kernel=True, A_log=values["A_log"],
        dt_bias=None if dt_bias is None else dt_bias.reshape(-1),
        epsilon=1.0e-6, use_qk_l2norm_in_kernel=norm,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
        use_exp2=True, backward_mode="save" if saved else "recompute",
    )
    (gk, aqk, akk, w, u, qg, kg, qg_scaled, q_hat, k_hat,
     q_rstd, k_rstd, _) = prepared
    batch = (lambda value: value.unsqueeze(0)) if packed else (lambda value: value)
    h, v_new, _ = ascendc.npu_chunk_fwd_h(
        batch(kg), batch(w), batch(u), g=None, gk=batch(gk), initial_state=None,
        output_final_state=False, chunk_size=CHUNK_SIZE, save_new_value=True,
        cu_seqlens=cu, use_exp2=True, state_v_first=False,
    )
    attn_out = ascendc.npu_chunk_kda_fwd_finalize(
        qg_scaled, aqk, v_new, h,
        output_layout="NTD" if packed else "BNSD", state_v_first=False,
        cu_seqlens=cu,
    )
    if packed:
        h = h.squeeze(0)
        v_new = v_new.squeeze(0)
    torch.npu.synchronize()
    caches = dict(gk=gk, Aqk=aqk, Akk=akk, w=w, qg=qg, kg=kg,
                  v_new=v_new, h=h, q_hat=q_hat, k_hat=k_hat,
                  q_rstd=q_rstd, k_rstd=k_rstd)
    required = ["Aqk", "Akk", "attn_out"]
    if saved:
        required.extend(("gk", "w", "qg", "kg", "v_new", "h"))
    if norm:
        required.extend(("q_hat", "k_hat", "q_rstd", "k_rstd"))
    for name in required:
        value = attn_out if name == "attn_out" else caches[name]
        if value is None:
            raise RuntimeError(f"three-stage input preparation returned no {name}")
    _finite_tuple((attn_out,))
    return caches


@register("executor_chunk_kda_bwd")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, OP_NAME)
        device = torch.device("cpu") if self.device == "cpu" else _marker_device(input_data)
        self.values = build_inputs(self.spec, device, high_precision=self.device == "cpu")
        if self.device == "cpu":
            return
        from fla_npu.ops import ascendc

        self.operator = ascendc.chunk_kda_bwd
        v = self.values
        packed = bool(self.spec.get("cu_seqlens"))
        saved = bool(self.spec.get("disable_recompute", True))
        norm = bool(self.spec.get("norm", False))
        forward = _prepare_forward(v, self.spec)
        tokens = {
            name: v[name].squeeze(0).contiguous() if packed else v[name]
            for name in ("q", "k", "v", "raw_g", "beta", "d_o")
        }
        q = forward["q_hat"] if norm else tokens["q"]
        k = forward["k_hat"] if norm else tokens["k"]
        cache = {
            name: forward[name] if saved else None
            for name in ("gk", "w", "qg", "kg", "v_new", "h")
        }
        self.args = (
            q, k, tokens["v"], tokens["beta"], cache["gk"],
            forward["Aqk"], forward["Akk"], cache["w"], cache["qg"],
            cache["kg"], cache["v_new"], cache["h"], tokens["d_o"],
            1.0 / math.sqrt(DIM),
        )
        self.kwargs = dict(
            raw_g=tokens["raw_g"], A_log=v["A_log"],
            dt_bias=None if "dt_bias" not in v else v["dt_bias"].view(self.spec["H"], DIM),
            chunk_size=CHUNK_SIZE, safe_gate=True, lower_bound=-5.0,
            cu_seqlens=self.spec.get("cu_seqlens"), use_gate_in_kernel=True,
            disable_recompute=saved, use_exp2=True, state_v_first=False,
            implementation="optimized",
            q_rstd=forward["q_rstd"] if norm else None,
            k_rstd=forward["k_rstd"] if norm else None,
        )
        torch.npu.synchronize()

    def run_npu(self):
        outputs = self.operator(*self.args, **self.kwargs)
        return tuple(outputs[index] for index in (0, 1, 2, 3, 4, 6, 7)
                     if outputs[index] is not None)

    def __call__(self, input_data: InputDataset, with_output=False):
        if self.device == "cpu":
            return _finite_tuple(run_cpu(self.values, self.spec), golden=True)
        outputs = self.run_npu()
        return _finite_tuple(outputs) if with_output else outputs

    def export_custom_data(self, input_data: InputDataset):
        return self.spec
