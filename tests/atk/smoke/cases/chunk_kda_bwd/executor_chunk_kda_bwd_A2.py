"""ATK executor and independent CPU golden for A2/A3 ``chunk_kda_bwd``.

The golden follows the mathematical KDA recurrence rather than calling any
NPU sub-operator.  It models the public saved-cache and recompute modes, Q/K
normalization backward, beta sigmoid/2-sigmoid backward, dense/packed inputs,
and a single QKV layout with token-major gate inputs and gradients.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from _kda_profiles import case_profiles
from gen_chunk_kda_bwd_A2 import (  # noqa: E402
    ACCURACY_PROFILES,
    OP_NAME,
    validate_profiles,
)


torch.set_num_threads(max(1, int(os.environ.get("KDA_BWD_ATK_CPU_THREADS", "1"))))

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias")
PROFILE_PACKAGES = {
    "accuracy": ACCURACY_PROFILES,
}


def _profiles_for(spec: dict[str, Any]) -> list[dict]:
    package = str(spec.get("package", "accuracy"))
    if package not in PROFILE_PACKAGES:
        raise ValueError(f"unknown case package {package!r}")
    return PROFILE_PACKAGES[package]


def _case_spec(input_data, profiles=ACCURACY_PROFILES) -> dict[str, Any]:
    raw = input_data.kwargs.get("case_spec")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        import json

        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return profiles[int(input_data.kwargs["case_id"])]


def sequences(spec: dict[str, Any]):
    """Yield ``(batch, token_begin, token_end, global_chunk_begin)``."""

    if spec["mode"] == "dense":
        return [(batch, 0, int(spec["T"]), 0) for batch in range(int(spec["B"]))]
    result = []
    token_begin = 0
    chunk_begin = 0
    for length in spec["seqlens"]:
        length = int(length)
        if length:
            result.append((0, token_begin, token_begin + length, chunk_begin))
        token_begin += length
        chunk_begin += (length + 63) // 64
    return result


def _scores(q: torch.Tensor, k: torch.Tensor, cumulative: torch.Tensor):
    """Causal QK/KK scores using natural-log cumulative gates."""

    tokens = q.shape[0]
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()
    difference = cumulative[:, None, :] - cumulative[None, :, :]
    decay = torch.exp(difference.masked_fill(~causal[..., None], 0))
    decay = decay * causal[..., None]
    qk = (q[:, None, :] * k[None, :, :] * decay).sum(-1)
    kk = (k[:, None, :] * k[None, :, :] * decay).sum(-1)
    return qk, kk, decay


def _randn(
    generator: torch.Generator,
    shape,
    std: float,
    mean: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * std + mean).to(dtype)


def build_inputs(spec: dict[str, Any], *, exact_caches: bool = False) -> dict[str, Any]:
    """Build deterministic public inputs in canonical CPU head-major layout."""

    generator = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    shape = (int(spec["B"]), int(spec["H"]), int(spec["T"]), 128)
    x: dict[str, Any] = {
        name: _randn(generator, shape, float(spec["qk_std"]))
        for name in ("q", "k")
    }
    for name in ("q", "k"):
        rstd = torch.rsqrt(x[name].float().square().sum(-1) + 1e-6)
        x[f"{name}_rstd"] = rstd if spec["with_l2"] else None
        if spec["with_l2"]:
            x[name] = (x[name].float() * rstd[..., None]).to(torch.bfloat16)

    x["v"] = _randn(generator, shape, float(spec["v_std"]))
    x["d_o"] = _randn(generator, shape, float(spec["do_std"]))
    beta_low, beta_high = (float(value) for value in spec["beta_range"])
    x["beta"] = (
        torch.rand(shape[:-1], generator=generator) * (beta_high - beta_low) + beta_low
    ).to(DTYPES[str(spec["beta_dtype"])])
    x["raw_g"] = _randn(
        generator,
        shape,
        float(spec["raw_g_std"]),
        float(spec["raw_g_mean"]),
        DTYPES[str(spec["raw_g_dtype"])],
    )
    x["A_log"] = _randn(
        generator, (int(spec["H"]),), 0.2, -0.5, DTYPES[str(spec["a_log_dtype"])]
    )
    x["dt_bias"] = (
        _randn(generator, (int(spec["H"]), 128), 0.15, dtype=torch.float32)
        if spec["with_bias"] else None
    )

    cache_dtype = torch.float64 if exact_caches else torch.bfloat16
    calc_dtype = torch.float64 if exact_caches else torch.float32
    if spec["mode"] == "varlen":
        chunk_count = sum((int(length) + 63) // 64 for length in spec["seqlens"])
    else:
        chunk_count = (int(spec["T"]) + 63) // 64
    for name in ("w", "qg", "kg", "v_new"):
        x[name] = torch.zeros(shape, dtype=cache_dtype)
    x["gk"] = torch.zeros(shape, dtype=calc_dtype)
    for name in ("Aqk", "Akk"):
        x[name] = torch.zeros((*shape[:-1], 64), dtype=cache_dtype)
    x["h"] = torch.zeros(
        (int(spec["B"]), chunk_count, int(spec["H"]), 128, 128),
        dtype=cache_dtype,
    )

    for batch, token_begin, token_end, chunk_begin in sequences(spec):
        for head in range(int(spec["H"])):
            state = torch.zeros((128, 128), dtype=calc_dtype)
            for chunk, start in enumerate(range(token_begin, token_end, 64), chunk_begin):
                end = min(start + 64, token_end)
                index = (batch, head, slice(start, end))
                q, k, v, beta, raw_g = (
                    x[name][index].to(calc_dtype)
                    for name in ("q", "k", "v", "beta", "raw_g")
                )
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc_dtype)
                eig = x["A_log"][head].to(calc_dtype).exp()
                gate = float(spec["lower_bound"]) * torch.sigmoid(eig * (raw_g + bias))
                cumulative = gate.cumsum(0)
                qk, kk, _ = _scores(q, k, cumulative)
                identity = torch.eye(end - start, dtype=calc_dtype)
                inverse = torch.linalg.solve_triangular(
                    identity + (beta[:, None] * kk).tril(-1), identity, upper=False
                )
                # These tensors are public forward caches.  The normal path
                # intentionally stores their BF16 values before the backward.
                inverse = inverse.to(cache_dtype).to(calc_dtype)
                w = (inverse @ (beta[:, None] * k * cumulative.exp())).to(cache_dtype).to(calc_dtype)
                u = (inverse @ (beta[:, None] * v)).to(cache_dtype).to(calc_dtype)
                qg = (q * cumulative.exp()).to(cache_dtype).to(calc_dtype)
                kg = (k * (cumulative[-1] - cumulative).exp()).to(cache_dtype).to(calc_dtype)
                x["h"][batch, chunk, head] = state.to(cache_dtype)
                v_new = (u - w @ state.to(cache_dtype).to(calc_dtype)).to(cache_dtype).to(calc_dtype)
                x["Aqk"][index][..., : end - start] = (qk * float(spec["scale"])).to(cache_dtype)
                x["Akk"][index][..., : end - start] = inverse
                x["gk"][index] = cumulative / math.log(2)
                for name, value in (("w", w), ("qg", qg), ("kg", kg), ("v_new", v_new)):
                    x[name][index] = value
                state = cumulative[-1].exp()[:, None] * state + kg.T @ v_new
    return x


def recompute_cpu_caches(
    inputs: dict[str, Any], spec: dict[str, Any], high_precision: bool
) -> dict[str, Any]:
    """Recreate gk/w/qg/kg/v_new/h without using DUT outputs."""

    x = dict(inputs)
    calc_dtype = torch.float64 if high_precision else torch.float32
    cache_dtype = torch.float64 if high_precision else torch.bfloat16
    for name in ("w", "qg", "kg", "v_new", "h"):
        x[name] = torch.empty_like(inputs[name], dtype=cache_dtype)
    x["gk"] = torch.empty_like(inputs["gk"], dtype=calc_dtype)

    def quantize(value: torch.Tensor) -> torch.Tensor:
        return value.to(cache_dtype).to(calc_dtype)

    for batch, token_begin, token_end, chunk_begin in sequences(spec):
        for head in range(int(spec["H"])):
            state = torch.zeros((128, 128), dtype=calc_dtype)
            for chunk, start in enumerate(range(token_begin, token_end, 64), chunk_begin):
                end = min(start + 64, token_end)
                index = (batch, head, slice(start, end))
                q, k, v, beta, raw_g = (
                    x[name][index].to(calc_dtype)
                    for name in ("q", "k", "v", "beta", "raw_g")
                )
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc_dtype)
                eig = x["A_log"][head].to(calc_dtype).exp()
                cumulative = (
                    float(spec["lower_bound"]) * torch.sigmoid(eig * (raw_g + bias))
                ).cumsum(0)
                inverse = x["Akk"][index][..., : end - start].to(calc_dtype)
                w = quantize(inverse @ quantize(beta[:, None] * k * cumulative.exp()))
                u = quantize(inverse @ quantize(beta[:, None] * v))
                qg = quantize(q * cumulative.exp())
                kg = quantize(k * (cumulative[-1] - cumulative).exp())
                x["h"][batch, chunk, head] = state.to(cache_dtype)
                v_new = quantize(u - w @ quantize(state))
                for name, value in (("w", w), ("qg", qg), ("kg", kg), ("v_new", v_new)):
                    x[name][index] = value
                x["gk"][index] = cumulative / math.log(2)
                state = cumulative[-1].exp()[:, None] * state + kg.T @ v_new
    return x


def _beta_backward_factor(beta: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "plain":
        return torch.ones_like(beta)
    if mode == "sigmoid":
        return beta * (1 - beta)
    if mode == "two_sigmoid":
        return beta * (1 - beta / 2)
    raise ValueError(f"unknown beta mode {mode!r}")


def _format_token_output(value: torch.Tensor, spec: dict[str, Any], *, gate=False) -> torch.Tensor:
    token_major = gate or spec["layout"] in {"BSND", "TND"}
    if token_major:
        value = value.transpose(1, 2).contiguous()
    if spec["mode"] == "varlen":
        value = value.squeeze(0)
    return value


def cpu_reference(
    inputs: dict[str, Any], spec: dict[str, Any], high_precision: bool = True
):
    """Analytic full backward through Prepare, Dhu and Finalize semantics."""

    if not high_precision:
        return cpu_benchmark(inputs, spec)

    x = inputs
    if not spec["disable_recompute"]:
        x = recompute_cpu_caches(inputs, spec, high_precision)
    calc_dtype = torch.float64 if high_precision else torch.float32
    out = {
        name: torch.zeros_like(x[source], dtype=calc_dtype)
        for name, source in (
            ("dq", "q"), ("dk", "k"), ("dv", "v"),
            ("db", "beta"), ("dg", "raw_g"), ("dA", "A_log"),
        )
    }
    out["dh0"] = None
    out["dbias"] = (
        None if x["dt_bias"] is None else torch.zeros_like(x["dt_bias"], dtype=calc_dtype)
    )

    def stage(value: torch.Tensor) -> torch.Tensor:
        return value if high_precision else value.to(torch.bfloat16).float()

    for batch, token_begin, token_end, chunk_begin in sequences(spec):
        for head in range(int(spec["H"])):
            dh = torch.zeros((128, 128), dtype=calc_dtype)
            spans = list(enumerate(range(token_begin, token_end, 64), chunk_begin))
            for chunk, start in reversed(spans):
                end = min(start + 64, token_end)
                index = (batch, head, slice(start, end))
                q, k, v, beta, raw_g, d_o, v_new, w, qg, kg = (
                    x[name][index].to(calc_dtype)
                    for name in (
                        "q", "k", "v", "beta", "raw_g", "d_o",
                        "v_new", "w", "qg", "kg",
                    )
                )
                cumulative = x["gk"][index].to(calc_dtype) * math.log(2)
                h = x["h"][batch, chunk, head].to(calc_dtype)
                aqk = x["Aqk"][index][..., : end - start].to(calc_dtype)
                inverse = x["Akk"][index][..., : end - start].to(calc_dtype)
                _, kk, decay = _scores(q, k, cumulative)

                # Prepare and reverse state scan.
                d_aqk = (d_o @ v_new.T).tril() * float(spec["scale"])
                dv_local = stage(aqk.T @ d_o)
                ds = stage(dv_local + kg @ dh)
                dh_previous = stage(
                    cumulative[-1].exp()[:, None] * dh
                    + float(spec["scale"]) * qg.T @ d_o
                    - w.T @ ds
                )

                # WY inverse and state contributions.
                dw = -(ds @ h.T)
                du_source = inverse.T @ ds
                dw_source = inverse.T @ dw
                d_inverse = (
                    ds @ (beta[:, None] * v).T
                    + dw @ (beta[:, None] * k * cumulative.exp()).T
                )
                d_lower = (-(inverse.T @ d_inverse @ inverse.T)).tril(-1)
                dv = beta[:, None] * du_source
                db = (
                    (du_source * v + dw_source * k * cumulative.exp()).sum(-1)
                    + (d_lower * kk).sum(-1)
                )
                dk = beta[:, None] * dw_source * cumulative.exp()
                dg_cumulative = dk * k
                dq = float(spec["scale"]) * (d_o @ h.T) * cumulative.exp()
                dg_cumulative += dq * q
                dk_state = (v_new @ dh.T) * (cumulative[-1] - cumulative).exp()
                dk += dk_state
                dg_cumulative -= dk_state * k
                dg_cumulative[-1] += (
                    (dk_state * k).sum(0) + (dh * h).sum(-1) * cumulative[-1].exp()
                )

                # Intra-chunk QK and strict-lower KK score derivatives.
                pair_qk = d_aqk[..., None] * decay
                dq += (pair_qk * k[None, :, :]).sum(1)
                dk += (pair_qk * q[:, None, :]).sum(0)
                gate_qk = pair_qk * q[:, None, :] * k[None, :, :]
                pair_kk = (d_lower * beta[:, None])[..., None] * decay
                dk += (
                    (pair_kk * k[None, :, :]).sum(1)
                    + (pair_kk * k[:, None, :]).sum(0)
                )
                gate_kk = pair_kk * k[:, None, :] * k[None, :, :]
                pair_gate = gate_qk + gate_kk
                dg_cumulative += pair_gate.sum(1) - pair_gate.sum(0)

                # gk is log2 storage; gate_grad is dL/d natural-log gate.
                gate_grad = dg_cumulative.flip(0).cumsum(0).flip(0)
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc_dtype)
                raw_bias = raw_g + bias
                eig = x["A_log"][head].to(calc_dtype).exp()
                sigmoid = torch.sigmoid(eig * raw_bias)
                d_raw = (
                    gate_grad * float(spec["lower_bound"]) * eig * sigmoid * (1 - sigmoid)
                )
                out["dA"][head] += (d_raw * raw_bias).sum()
                if out["dbias"] is not None:
                    out["dbias"][head] += d_raw.sum(0)

                if spec["with_l2"]:
                    dq = (
                        dq - q * (dq * q).sum(-1, keepdim=True)
                    ) * x["q_rstd"][index].to(calc_dtype)[..., None]
                    dk = (
                        dk - k * (dk * k).sum(-1, keepdim=True)
                    ) * x["k_rstd"][index].to(calc_dtype)[..., None]

                db = db * _beta_backward_factor(beta, str(spec["beta_mode"]))
                for name, value in (
                    ("dq", dq), ("dk", dk), ("dv", dv),
                    ("db", db), ("dg", d_raw),
                ):
                    out[name][index] = value
                dh = dh_previous

    if not high_precision:
        # Low-precision output casting is applied by the public executor.
        out["dv"] = out["dv"].to(torch.bfloat16)
    for name in ("dq", "dk", "dv", "db", "dg"):
        out[name] = _format_token_output(out[name], spec, gate=name in {"db", "dg"})
    return tuple(out[name] for name in OUTPUT_NAMES)


def _external_input_token(value: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    token_major = spec["layout"] in {"BSND", "TND"}
    if token_major:
        value = value.transpose(1, 2).contiguous()
    if spec["mode"] == "varlen":
        value = value.squeeze(0)
    return value


def npu_inputs(
    cpu_inputs: dict[str, Any], spec: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Convert canonical CPU inputs to the exact public physical layouts."""

    packed = spec["mode"] == "varlen"
    external = {"v"}
    if not spec["with_l2"]:
        external.update(("q", "k"))
    absent_recompute_caches = {"gk", "w", "qg", "kg", "v_new", "h"}
    result: dict[str, Any] = {}
    for name, value in cpu_inputs.items():
        if not spec["disable_recompute"] and name in absent_recompute_caches:
            result[name] = None
            continue
        if value is None:
            result[name] = None
            continue
        if name in {"d_o", "raw_g", "beta"}:
            if name == "beta" and spec["use_beta_sigmoid_in_kernel"]:
                result["beta_eff"] = (value.squeeze(0) if packed else value).float().contiguous().to(device)
                divisor = 2 if spec["allow_neg_eigval"] else 1
                value = torch.logit(value.float() / divisor).to(DTYPES[spec["beta_dtype"]])
            value = value.transpose(1, 2).contiguous()
            if packed:
                value = value.squeeze(0)
        elif name in external:
            value = _external_input_token(value, spec)
        elif packed and name not in {"A_log", "dt_bias"}:
            value = value.squeeze(0)
        result[name] = value.contiguous().to(device)

    cu_seqlens = None
    chunk_indices = None
    if packed:
        cu_seqlens = [0]
        explicit_indices = []
        for sequence, length in enumerate(spec["seqlens"]):
            cu_seqlens.append(cu_seqlens[-1] + int(length))
            for chunk in range((int(length) + 63) // 64):
                explicit_indices.extend((sequence, chunk))
        if spec["metadata"] == "explicit":
            chunk_indices = explicit_indices

    result.update(
        scale=float(spec["scale"]),
        chunk_size=64,
        safe_gate=True,
        use_gate_in_kernel=True,
        lower_bound=float(spec["lower_bound"]),
        disable_recompute=bool(spec["disable_recompute"]),
        use_exp2=True,
        state_v_first=False,
        implementation=str(spec["implementation"]),
        layout=str(spec["layout"]),
        use_beta_sigmoid_in_kernel=bool(spec["use_beta_sigmoid_in_kernel"]),
        allow_neg_eigval=bool(spec["allow_neg_eigval"]),
        initial_state=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return result


def _finite_outputs(outputs, *, probe_device: bool = True):
    if len(outputs) != 8 or outputs[5] is not None:
        raise RuntimeError("chunk_kda_bwd output contract drift: expected 8 slots and dh0=None")
    visible = []
    for name, value in zip(OUTPUT_NAMES, outputs):
        if value is None:
            if name not in {"dh0", "dbias"}:
                raise RuntimeError(f"missing required output {name}")
            continue
        probe = value.detach()
        if probe_device and probe.device.type != "cpu":
            probe = probe.cpu()
        if probe.is_floating_point() and not torch.isfinite(probe.float()).all().item():
            raise FloatingPointError(f"{OP_NAME}: non-finite output {name}")
        visible.append(value)
    return tuple(visible)


def _cast_cpu_outputs_to_abi(outputs, spec):
    """Keep FP64 golden math, but expose the same dtypes as the public ABI."""

    result = []
    for name, value in zip(OUTPUT_NAMES, outputs):
        if value is None:
            result.append(None)
        elif name in {"dq", "dk", "dv"}:
            result.append(round_bfloat16(value) if value.dtype == torch.float64 else value.to(torch.bfloat16))
        elif name in {"db", "dg"}:
            result.append(value.to(DTYPES[spec["beta_dtype" if name == "db" else "raw_g_dtype"]]))
        else:
            result.append(value.to(torch.float32))
    return tuple(result)


def run_cpu(spec: dict[str, Any]):
    return cpu_reference(build_inputs(spec), spec, high_precision=True)


def run_npu(spec: dict[str, Any], marker: torch.Tensor):
    cpu = build_inputs(spec)
    values = npu_inputs(cpu, spec, marker.device)
    from fla_npu.ops.ascendc import chunk_kda_bwd

    return chunk_kda_bwd(**values)


def _autograd_beta(spec: dict[str, Any], beta_effective: torch.Tensor):
    mode = str(spec["beta_mode"])
    if mode == "plain":
        leaf = beta_effective.detach().clone().requires_grad_()
        return leaf, leaf
    if mode == "sigmoid":
        leaf = torch.logit(beta_effective).detach().clone().requires_grad_()
        return leaf, torch.sigmoid(leaf)
    if mode == "two_sigmoid":
        leaf = torch.logit(beta_effective / 2).detach().clone().requires_grad_()
        return leaf, 2 * torch.sigmoid(leaf)
    raise ValueError(mode)


def self_test() -> None:
    """Validate the analytic golden against token recurrence autograd."""

    validate_profiles()
    for beta_mode in ("plain", "sigmoid", "two_sigmoid"):
        for length in (1, 7, 65):
            spec = dict(
                ACCURACY_PROFILES[0],
                B=1,
                H=1,
                T=length,
                mode="dense",
                layout="BNSD",
                with_l2=False,
                with_bias=True,
                disable_recompute=True,
                beta_mode=beta_mode,
                use_beta_sigmoid_in_kernel=beta_mode != "plain",
                allow_neg_eigval=beta_mode == "two_sigmoid",
                beta_range=[0.15, 1.85] if beta_mode == "two_sigmoid" else [0.1, 0.9],
                scale=0.125,
                lower_bound=-0.5,
            )
            x = build_inputs(spec, exact_caches=True)
            leaves = {
                name: x[name].double().detach().requires_grad_()
                for name in ("q", "k", "v", "raw_g", "A_log", "dt_bias")
            }
            beta_leaf, beta = _autograd_beta(spec, x["beta"].double())
            state = torch.zeros(128, 128, dtype=torch.float64)
            loss = torch.zeros((), dtype=torch.float64)
            for token in range(length):
                q = leaves["q"][0, 0, token]
                k = leaves["k"][0, 0, token]
                v = leaves["v"][0, 0, token]
                raw_g = leaves["raw_g"][0, 0, token]
                beta_value = beta[0, 0, token]
                gate = float(spec["lower_bound"]) * torch.sigmoid(
                    leaves["A_log"][0].exp() * (raw_g + leaves["dt_bias"][0])
                )
                decayed = gate.exp()[:, None] * state
                delta = beta_value * (v - k @ decayed)
                state = decayed + k[:, None] * delta[None, :]
                loss = loss + (
                    (q @ state) * float(spec["scale"]) * x["d_o"][0, 0, token].double()
                ).sum()
            expected = torch.autograd.grad(
                loss,
                (
                    leaves["q"], leaves["k"], leaves["v"], beta_leaf,
                    leaves["raw_g"], leaves["A_log"], leaves["dt_bias"],
                ),
            )
            for disable_recompute in (True, False):
                mode_spec = dict(spec, disable_recompute=disable_recompute)
                actual = cpu_reference(x, mode_spec, high_precision=True)
                actual_visible = (
                    actual[0], actual[1], actual[2], actual[3],
                    actual[4], actual[6], actual[7],
                )
                for name, got, want in zip(
                    ("dq", "dk", "dv", "db", "dg", "dA", "dbias"),
                    actual_visible,
                    tuple(v.transpose(1, 2).contiguous() if i in (3, 4) else v for i, v in enumerate(expected)),
                ):
                    torch.testing.assert_close(
                        got, want, rtol=1e-8, atol=1e-10,
                        msg=(
                            f"T={length} beta_mode={beta_mode} "
                            f"disable_recompute={disable_recompute} {name}"
                        ),
                    )

    # Only QKV gradients follow layout; gate gradients stay token-major.
    base = dict(ACCURACY_PROFILES[0], H=2, T=65, with_l2=True, with_bias=True, layout="BNSD")
    x = build_inputs(base)
    reference = cpu_reference(x, base, high_precision=True)
    for layout in ("BNSD", "BSND"):
        actual = cpu_reference(x, dict(base, layout=layout), high_precision=True)
        for index, (got, want) in enumerate(zip(actual, reference)):
            if got is None:
                assert want is None
                continue
            if layout == "BSND" and index < 3:
                want = want.transpose(1, 2).contiguous()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    print("PASS: analytic golden, beta modes, 64/65 boundary and layout formatting")


try:
    from atk.configs.dataset_config import InputDataset
    from atk.configs.results_config import TaskResult
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.base_api import BaseApi
except ModuleNotFoundError as exc:
    if exc.name not in {"atk", "atk.configs", "atk.tasks"}:
        raise
else:
    @register("executor_chunk_kda_bwd_A2")
    class FunctionApi(BaseApi):
        def __init__(self, task_result: TaskResult):
            super().__init__(task_result)
            self.spec = None
            self.prepared = None
            self.prepared_id = None

        def init_by_input_data(self, input_data: InputDataset):
            profiles = case_profiles(self.task_result, Path(__file__).with_name("chunk_kda_bwd_10000.json"), ACCURACY_PROFILES)
            spec = _case_spec(input_data, profiles)
            if profiles is ACCURACY_PROFILES:
                profiles = _profiles_for(spec)
            elif (spec["K"], spec["V"], spec["implementation"]) != (128, 128, "optimized"):
                raise ValueError("A2/A3 backward V2 requires K=V=128 and implementation=optimized")
            case_id = int(spec["case_id"])
            if not 0 <= case_id < len(profiles) or profiles[case_id] != spec:
                raise ValueError(f"case_spec does not match frozen {spec['package']} profile")
            cpu = build_inputs(spec)
            if self.device in {"npu", "pyaclnn"}:
                self.prepared = npu_inputs(
                    cpu, spec, input_data.kwargs["low_precision_marker"].device
                )
            elif self.device == "cpu":
                self.prepared = cpu
            else:
                raise RuntimeError(
                    f"{OP_NAME} supports only NPU DUT and CPU golden nodes, "
                    f"got {self.device!r}"
                )
            self.spec = spec
            self.prepared_id = int(spec["case_id"])

        def __call__(self, input_data: InputDataset, with_output: bool = False):
            del with_output
            runtime_id = int(input_data.kwargs["case_id"])
            if self.prepared is None or self.prepared_id != runtime_id:
                self.init_by_input_data(input_data)
            if self.device in {"npu", "pyaclnn"}:
                from fla_npu.ops.ascendc import chunk_kda_bwd

                outputs = chunk_kda_bwd(**self.prepared)
                return _finite_outputs(outputs, probe_device=True)
            if self.device == "cpu":
                high = bool(self.task_result.is_benchmark_task)
                print(f"DUAL_ROLE case={runtime_id} golden={high}", flush=True)
                outputs = cpu_reference(self.prepared, self.spec, high_precision=high)
                if not high:
                    outputs = _cast_cpu_outputs_to_abi(outputs, self.spec)
                return _finite_outputs(outputs, probe_device=False)
            raise AssertionError("unreachable device branch")


def cpu_benchmark(inputs, spec):
    """CPU-only FP32 arithmetic with the published BF16 stage boundaries."""
    x = inputs if spec["disable_recompute"] else recompute_cpu_caches(inputs, spec, False)
    def bf(value):
        return value.bfloat16().float()
    daqk = torch.zeros_like(x["Aqk"], dtype=torch.float32)
    dqraw = torch.zeros_like(x["q"], dtype=torch.float32)
    dvscan = torch.zeros_like(x["v"], dtype=torch.bfloat16)
    dhout = torch.zeros_like(x["h"], dtype=torch.bfloat16)
    chunks = []
    for batch, begin, end, chunk_begin in sequences(spec):
        spans = list(enumerate(range(begin, end, 64), chunk_begin))
        chunks.extend((batch, local, start, min(64, end-start)) for local, (_, start) in enumerate(spans))
        for head in range(spec["H"]):
            state = torch.zeros((128,128), dtype=torch.float32)
            for chunk, start in reversed(spans):
                stop = min(start+64,end)
                ix = (batch,head,slice(start,stop))
                do = x["d_o"][ix].float()
                aq = x["Aqk"][ix][..., :stop-start].float()
                daqk[ix][..., :stop-start] = (do @ x["v_new"][ix].float().T).tril()*spec["scale"]
                dqraw[ix] = do @ x["h"][batch,chunk,head].float().T
                dvlocal = bf(aq.T @ do)
                # Dhu outputs BF16 dh, while its persistent state remains FP32.
                dh = bf(state)
                dhout[batch,chunk,head] = dh
                dvstate = bf(x["kg"][ix].float() @ dh)
                ds = bf(dvlocal + dvstate)
                dvscan[ix] = ds
                termq = bf(x["qg"][ix].float().T @ do)
                termw = bf(x["w"][ix].float().T @ ds)
                decay = (x["gk"][batch,head,stop-1].float()*math.log(2)).exp()
                state = state*decay[:,None] + (termq*spec["scale"] - termw)
    packed = spec["mode"] == "varlen"
    data = {k:(v.squeeze(0) if packed and k not in ("A_log","dt_bias") else v)
            for k,v in x.items() if v is not None}
    data["a_log"] = data.pop("A_log").float()
    data["akk"] = data.pop("Akk")
    data["beta"] = data["beta"].float()
    data["raw_g"] = data["raw_g"].float()
    data.setdefault("dt_bias", torch.zeros((spec["H"],128),dtype=torch.float32))
    for key,value in (("dAqk",daqk),("dq_raw",dqraw),("dh",dhout),("dv_scan",dvscan)):
        data[key] = value.squeeze(0) if packed else value
    fs = dict(B=spec["B"],H=spec["H"],T=spec["T"],lengths=spec.get("seqlens") if packed else None,
              rank=3 if packed else 4,norm=spec["with_l2"],
              beta_mode={"plain":0,"sigmoid":1,"two_sigmoid":2}[spec["beta_mode"]],
              scale=spec["scale"],lower_bound=spec["lower_bound"],output_layout=spec["layout"])
    result = list(_finalize_reference(data,fs,chunks,False))
    result[0] = result[0].float()
    result[1] = result[1].float()
    return (*result[:5],None,result[5],result[6] if x["dt_bias"] is not None else None)


FINALIZE_NAMES=("dq","dk","dv","db","dg","da_log","ddt_bias")

def round_bfloat16(value):
    """Round FP64 once, including subnormals; torch's cast can pass through FP32."""
    _, exponent = torch.frexp(value)
    step = (exponent - 8).clamp(min=-133)
    rounded = torch.ldexp(torch.ldexp(value, -step).round(), step)
    return rounded.bfloat16()

def _finalize_reference(data, spec, chunks, high_precision=True):
    """FP64 mathematical golden / FP32 benchmark with documented BF16 handoffs.

    The exponential difference is factored per CPU output tile. FP64 can
    handle an entire chunk; FP32 uses eight rows to keep positive exponents
    bounded. There is no BF16 residual split or device intermediate reuse.
    """
    dtype = torch.float64 if high_precision else torch.float32
    variable = spec['lengths'] is not None
    heads = spec['H']
    shape = data['q'].shape
    # Round the FP64 golden directly to the interface dtype. An intervening
    # FP32 store can turn a near-midpoint value into a BF16 tie.
    out = {n: torch.empty(shape, dtype=torch.bfloat16) for n in ('dq', 'dk', 'dv')}
    out['dg'] = torch.empty(shape, dtype=torch.float32)
    out['db'] = torch.empty(shape[:-1], dtype=data['beta'].dtype)
    da = torch.zeros(heads, dtype=dtype)
    bias = torch.zeros(heads, 128, dtype=dtype)

    # The arch22 Finalize WY path introduced by #1190 keeps these intermediates in FP32.

    for ci, (seq, local, first, rows) in enumerate(chunks):
        for head in range(heads):
            region = (head, slice(first, first+rows)) if variable else (seq, head, slice(first, first+rows))
            q, k, v, vn, ds, g, beta, aq, qr = [data[n][region].to(dtype) for n in
                ('q', 'k', 'v', 'v_new', 'dv_scan', 'gk', 'beta', 'dAqk', 'dq_raw')]
            a = data['akk'][region][:, :rows].to(dtype)
            hh = (data['h'][ci, head] if variable else data['h'][seq, local, head]).to(dtype)
            dh = (data['dh'][ci, head] if variable else data['dh'][seq, local, head]).to(dtype)
            e = torch.exp2(g)
            dw = ds @ hh.T
            ke = k * e
            dvb = a.T @ ds
            dkg = a.T @ dw
            zb = torch.tril((ds @ v.T - dw @ ke.T) * beta[None, :], -1)
            ak = -(a.T @ (zb @ a.T)).tril(-1)
            state = (vn @ dh.T) * torch.exp2(g[-1] - g)
            dq = spec['scale'] * qr * e
            dk = state - beta[:, None] * e * dkg
            db = (v * dvb).sum(-1) - (k * e * dkg).sum(-1)
            dg = q * dq - k * state - beta[:, None] * k * e * dkg
            dg[-1] += (k * state).sum(0) + torch.exp2(g[-1]) * (hh * dh).sum(-1)
            aq_lower = aq[:, :rows].tril(-1)
            local_q, left, right = (torch.empty_like(k) for _ in range(3))
            width = rows if high_precision else 8
            for start in range(0, rows, width):
                end = min(start + width, rows)
                center = (g[start] + g[end - 1]) * .5
                factor = torch.exp2(g[start:end] - center)
                neg = k[:end] * torch.exp2(center - g[:end])
                pos = torch.exp2(g[start:] - center)
                local_q[start:end] = (aq_lower[start:end, :end] @ neg) * factor
                left[start:end] = (ak[start:end, :end] @ neg) * factor
                right[start:end] = (aq_lower[start:, start:end].T @ (q[start:] * pos) +
                                    ak[start:, start:end].T @ (beta[start:, None] * k[start:] * pos)) / factor
            diag = aq[:, :rows].diag()[:, None]
            dq = dq + local_q + diag * k
            dk = dk + beta[:, None] * left + right + diag * q
            db = db + (left * k).sum(-1)
            dg = dg + q * local_q + k * (beta[:, None] * left - right)
            scan = dg.flip(0).cumsum(0).flip(0)
            x = data['raw_g'][region].to(dtype) + data['dt_bias'][head].to(dtype)
            ea = data['a_log'][head].to(dtype).exp()
            z = (-(x * ea).abs()).exp()
            dg = scan * (spec['lower_bound'] * ea * z / (1 + z).square())
            if spec['norm']:
                dq = (dq - q * (dq * q).sum(-1, keepdim=True)) * data['q_rstd'][region].to(dtype)[:, None]
                dk = (dk - k * (dk * k).sum(-1, keepdim=True)) * data['k_rstd'][region].to(dtype)[:, None]
            if spec['beta_mode']:
                db *= beta * (1 - beta / (2 if spec['beta_mode'] == 2 else 1))
            for n, value in zip(('dq', 'dk', 'dv', 'db', 'dg'), (dq, dk, beta[:, None]*dvb, db, dg)):
                out[n][region] = (round_bfloat16(value) if high_precision and out[n].dtype == torch.bfloat16
                                   else value.to(out[n].dtype))
            da[head] += (dg * x).sum()
            bias[head] += dg.sum(0)
    for n in ('dq', 'dk', 'dv'):
        if spec['output_layout'] in ('BSND', 'TND'):
            out[n] = out[n].transpose(-3, -2).contiguous()
    out['dg'] = out['dg'].transpose(-3, -2).contiguous()
    out['db'] = out['db'].transpose(-2, -1).contiguous()
    out['da_log'], out['ddt_bias'] = da.float(), bias.float()
    return interface_outputs(tuple(out[n] for n in FINALIZE_NAMES), spec)

def interface_outputs(outputs, spec):
    source_rank = 3 if spec['lengths'] is not None else 4
    if source_rank == spec['rank']:
        return outputs
    return tuple((x.unsqueeze(0) if spec['rank'] == 4 else x.squeeze(0)) if i < 5 else x
                 for i, x in enumerate(outputs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
