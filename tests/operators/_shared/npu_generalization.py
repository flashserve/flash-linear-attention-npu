"""Execute canonical JSON generalization cases through stable Ascend C APIs."""

from __future__ import annotations

import os
from typing import Any, Dict


Case = Dict[str, Any]


def _dtype(torch, name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported manifest dtype: {name}") from exc


def _tensor(torch, shape, dtype, device, *, positive=False, scale=0.1):
    shape = tuple(int(value) for value in shape)
    if positive:
        value = torch.rand(shape, dtype=torch.float32)
    else:
        value = torch.randn(shape, dtype=torch.float32) * scale
    return value.to(dtype=dtype, device=device).contiguous()


def _zeros(torch, shape, dtype, device):
    return torch.zeros(tuple(int(value) for value in shape), dtype=dtype, device=device)


def _meta(case: Case, name: str):
    return case.get("optional_inputs", {}).get(name)


def _shape(case: Case, name: str, default=None):
    value = case["shape"].get(name, default)
    if value is None:
        raise ValueError(f"{case['id']}: missing shape symbol {name}")
    return int(value)


def _chunk_count(case: Case) -> int:
    if "N_c" in case["shape"]:
        return _shape(case, "N_c")
    time = _shape(case, "T")
    chunk = int(case["attrs"].get("chunk_size", case["shape"].get("C", 64)))
    return (time + chunk - 1) // chunk


def _run_chunk_bwd_dqkwg(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V"))
    C, Nc = int(case["attrs"]["chunk_size"]), _chunk_count(case)
    data = _dtype(torch, d["qkv"])
    gate = _dtype(torch, d["g"])
    state = _dtype(torch, d["state"])
    q = _tensor(torch, (B, Hk, T, K), data, device)
    k = _tensor(torch, (B, Hk, T, K), data, device)
    v = _tensor(torch, (B, Hv, T, V), data, device)
    g = -_tensor(torch, (B, Hv, T), gate, device, positive=True, scale=0.02)
    h = _tensor(torch, (B, Hv, Nc, K, V), state, device, scale=0.02)
    dox = _tensor(torch, (B, Hv, T, V), data, device)
    dh = _tensor(torch, (B, Hv, Nc, K, V), state, device, scale=0.02)
    dv = _tensor(torch, (B, Hv, T, V), data, device)
    out = ops.chunk_bwd_dqkwg(
        q, k, v, g, h, dox, dh, dv, C,
        cu_seqlens=_meta(case, "cu_seqlens"),
        chunk_indices=_meta(case, "chunk_indices"),
        scale=float(case["attrs"]["scale"]),
        use_exp2=case["attrs"].get("use_exp2", False),
        transpose_state_layout=case["attrs"].get("transpose_state_layout", False),
    )
    return out, ((B, Hk, T, K), (B, Hk, T, K), (B, Hv, T, K), (B, Hv, T))


def _run_chunk_bwd_dv_local(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V"))
    data = _dtype(torch, d["qkdo"])
    q = _tensor(torch, (B, Hk, T, K), data, device)
    k = _tensor(torch, (B, Hk, T, K), data, device)
    d_o = _tensor(torch, (B, Hv, T, V), data, device)
    g = -_tensor(torch, (B, Hv, T), _dtype(torch, d["g"]), device, positive=True, scale=0.02)
    out = ops.chunk_bwd_dv_local(
        q, k, d_o, g, float(case["attrs"]["scale"]), int(case["attrs"]["chunk_size"]),
        cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"),
    )
    return out, ((B, Hv, T, V),)


def _run_chunk_gated_delta_rule_bwd_dhu(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V"))
    C, Nc = int(case["attrs"]["chunk_size"]), _chunk_count(case)
    data = _dtype(torch, d["data"])
    q = _tensor(torch, (B, Hk, T, K), data, device)
    k = _tensor(torch, (B, Hk, T, K), data, device)
    w = _tensor(torch, (B, Hv, T, K), data, device)
    d_o = _tensor(torch, (B, Hv, T, V), data, device)
    dv = _tensor(torch, (B, Hv, T, V), data, device)
    g = -_tensor(torch, (B, Hv, T), _dtype(torch, d["g"]), device, positive=True, scale=0.02)
    out = ops.chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv, float(case["attrs"]["scale"]), C, g=g,
        cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"),
        use_exp2=case["attrs"].get("use_exp2", False),
        transpose_state_layout=case["attrs"].get("transpose_state_layout", False),
    )
    return out, ((B, Hv, Nc, K, V), (B, Hv, T, V))


def _prepare_wy_inputs(torch, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V"))
    C = int(case["attrs"]["chunk_size"])
    data = _dtype(torch, d["data"])
    scalar = _dtype(torch, d["beta_g"])
    k = _tensor(torch, (B, Hk, T, K), data, device)
    v = _tensor(torch, (B, Hv, T, V), data, device)
    beta = _tensor(torch, (B, Hv, T), scalar, device, positive=True)
    A = _tensor(torch, (B, Hv, T, C), data, device, scale=0.02)
    dw = _tensor(torch, (B, Hv, T, K), data, device)
    du = _tensor(torch, (B, Hv, T, V), data, device)
    g = -_tensor(torch, (B, Hv, T), scalar, device, positive=True, scale=0.02)
    return (k, v, beta, A, dw, du, g), (B, Hk, Hv, T, K, V, C, data)


def _run_prepare_wy_repr_bwd_da(torch, ops, case, device):
    inputs, dims = _prepare_wy_inputs(torch, case, device)
    B, _, Hv, T, _, _, C, _ = dims
    out = ops.prepare_wy_repr_bwd_da(
        *inputs, chunk_size=C, cu_seqlens=_meta(case, "cu_seqlens"),
        chunk_indices=_meta(case, "chunk_indices"),
    )
    return out, ((B, Hv, T, C),)


def _run_prepare_wy_repr_bwd_full(torch, ops, case, device):
    inputs, dims = _prepare_wy_inputs(torch, case, device)
    k, v, beta, A, dw, du, g = inputs
    B, Hk, Hv, T, K, V, C, data = dims
    dA = _tensor(torch, (B, Hv, T, C), data, device)
    out = ops.prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, du, g, C,
        cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"),
    )
    return out, ((B, Hk, T, K), (B, Hv, T, V), (B, Hv, T), (B, Hv, T))


def _run_chunk_fwd_o(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V, Nc = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V", "N_c"))
    data = _dtype(torch, d["qkvh"])
    q = _tensor(torch, (B, Hk, T, K), data, device)
    k = _tensor(torch, (B, Hk, T, K), data, device)
    v = _tensor(torch, (B, Hv, T, V), data, device)
    h = _tensor(torch, (B, Hv, Nc, K, V), data, device, scale=0.02)
    g = -_tensor(torch, (B, Hv, T), _dtype(torch, d["g"]), device, positive=True, scale=0.02)
    out = ops.chunk_fwd_o(
        q, k, v, h, float(case["attrs"]["scale"]), g=g,
        cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"),
        chunk_size=int(case["attrs"]["chunk_size"]),
    )
    return out, ((B, Hv, T, V),)


def _run_chunk_gated_delta_rule_fwd_h(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V, Nc = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V", "N_c"))
    data = _dtype(torch, d["kwu"])
    k = _tensor(torch, (B, Hk, T, K), data, device)
    w = _tensor(torch, (B, Hv, T, K), data, device)
    u = _tensor(torch, (B, Hv, T, V), data, device)
    g = None
    gk = None
    if _meta(case, "g") is not None:
        g = -_tensor(torch, (B, Hv, T), _dtype(torch, d["g"]), device, positive=True, scale=0.02)
    if _meta(case, "gk") is not None:
        gk = -_tensor(torch, (B, Hv, T, K), _dtype(torch, d["gk"]), device, positive=True, scale=0.02)
    seq_num = len(_meta(case, "cu_seqlens")) - 1 if _meta(case, "cu_seqlens") else B
    initial_state = None
    if _meta(case, "initial_state") is not None:
        initial_state = _zeros(torch, (seq_num, Hv, K, V), _dtype(torch, d["state"]), device)
    out = ops.chunk_gated_delta_rule_fwd_h(
        k, w, u, g, gk=gk, initial_state=initial_state,
        output_final_state=case["attrs"].get("output_final_state", False),
        chunk_size=int(case["attrs"]["chunk_size"]),
        save_new_value=case["attrs"].get("save_new_value", True),
        cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"),
        use_exp2=case["attrs"].get("use_exp2", False),
        transpose_state_layout=case["attrs"].get("transpose_state_layout", False),
    )
    return out, ((B, Hv, Nc, K, V), (B, Hv, T, V), (seq_num, Hv, K, V))


def _run_recompute_wu_fwd(torch, ops, case, device):
    s, d = case["shape"], case["dtype"]
    B, Hk, Hv, T, K, V = (int(s[k]) for k in ("B", "H_k", "H_v", "T", "K", "V"))
    C = int(case["attrs"]["chunk_size"])
    data = _dtype(torch, d["kvA"])
    scalar = _dtype(torch, d["beta_g"])
    k = _tensor(torch, (B, Hk, T, K), data, device)
    v = _tensor(torch, (B, Hv, T, V), data, device)
    beta = _tensor(torch, (B, Hv, T), scalar, device, positive=True)
    A = _tensor(torch, (B, Hv, T, C), data, device, scale=0.02)
    g = -_tensor(torch, (B, Hv, T), scalar, device, positive=True, scale=0.02)
    out = ops.recompute_wu_fwd(
        k, v, beta, A, C, g=g, cu_seqlens=_meta(case, "cu_seqlens"),
        chunk_indices=_meta(case, "chunk_indices"),
    )
    return out, ((B, Hv, T, K), (B, Hv, T, V))


def _run_causal_conv1d(torch, ops, case, device):
    s, d, attrs = case["shape"], case["dtype"], case["attrs"]
    B, T, D, W = (int(s[k]) for k in ("B", "T", "D", "W"))
    dtype = _dtype(torch, d["x_weight_state"])
    rank2 = case["layout"] == "TH"
    x_shape = (T, D) if rank2 else (B, T, D)
    x = _tensor(torch, x_shape, dtype, device)
    weight = _tensor(torch, (W, D), dtype, device)
    states = _zeros(torch, (_shape(case, "D_s"), _shape(case, "L_s"), D), dtype, device)
    bias = _tensor(torch, (D,), dtype, device) if _meta(case, "bias") is not None else None
    out = ops.causal_conv1d(
        x, weight, bias, states,
        query_start_loc=_meta(case, "query_start_loc"), cache_indices=_meta(case, "cache_indices"),
        initial_state_mode=_meta(case, "initial_state_mode"),
        num_accepted_tokens=_meta(case, "num_accepted_tokens"),
        activation_mode=int(attrs["activation_mode"]), pad_slot_id=int(attrs["pad_slot_id"]),
        run_mode=int(attrs["run_mode"]), head_num=int(attrs["head_num"]),
    )
    if int(attrs["run_mode"]) == 0 and int(attrs["head_num"]) > 0:
        expected_y = (B, int(attrs["head_num"]), T, D // int(attrs["head_num"]))
    else:
        expected_y = x_shape
    return (out, states), (expected_y, tuple(states.shape))


def _run_causal_conv1d_bwd(torch, ops, case, device):
    s, d, attrs = case["shape"], case["dtype"], case["attrs"]
    B, T, D, W = (int(s[k]) for k in ("B", "T", "D", "W"))
    Hv, V = _shape(case, "H_v"), _shape(case, "V")
    dtype = _dtype(torch, d["floating"])
    layout = attrs["input_layout"]
    x_shape = (T, D) if layout in {"NTD", "TND"} else (B, T, D)
    if layout == "BNSD":
        dy_shape = (B, Hv, T, V)
    elif layout == "NTD":
        dy_shape = (Hv, T, V)
    else:
        dy_shape = x_shape
    x = _tensor(torch, x_shape, dtype, device)
    dy = _tensor(torch, dy_shape, dtype, device)
    y = _tensor(torch, dy_shape, dtype, device) if _meta(case, "y") is not None else None
    weight = _tensor(torch, (W, D), dtype, device)
    initial = _tensor(torch, (B, W, D), dtype, device) if _meta(case, "initial_state") is not None else None
    dht = _tensor(torch, (B, W, D), dtype, device) if _meta(case, "dht") is not None else None
    out = ops.causal_conv1d_bwd(
        x, y, weight, dy, initial, dht, query_start_loc=_meta(case, "query_start_loc"),
        activation=int(attrs["activation"]), input_layout=layout,
    )
    return out, (x_shape, (W, D), (D,), (B, W, D))


def _run_chunk_local_cumsum(torch, ops, case, device):
    B, Hv, T, P = (_shape(case, key) for key in ("B", "H_v", "T", "P"))
    input_shape = (B, Hv, T) if P == 1 else (B, Hv, T, P)
    g = _tensor(torch, input_shape, torch.float32, device)
    cu = _meta(case, "cu_seqlens")
    indices = _meta(case, "chunk_indices_out")
    attrs = case["attrs"]
    out = ops.chunk_local_cumsum(
        g, int(attrs["chunk_size"]), cu_seqlens=cu, chunk_indices_out=indices,
        reverse=attrs["reverse"], scale=float(attrs["scale"]), head_first=attrs["head_first"],
        output_dtype=attrs["output_dtype"],
    )
    return out, (input_shape,)


def _run_chunk_scaled_dot_kkt(torch, ops, case, device):
    B, Hk, Hv, T, K = (_shape(case, key) for key in ("B", "H_k", "H_v", "T", "K"))
    C = int(case["attrs"]["chunk_size"])
    k = _tensor(torch, (B, Hk, T, K), _dtype(torch, case["dtype"]["k"]), device)
    g = -_tensor(torch, (B, Hv, T), torch.float32, device, positive=True, scale=0.02)
    beta = _tensor(torch, (B, Hv, T), torch.float32, device, positive=True)
    out = ops.chunk_scaled_dot_kkt(
        k, g, beta, cu_seqlens=_meta(case, "cu_seqlens"),
        chunk_indices=_meta(case, "chunk_indices"), chunk_size=C,
    )
    return out, ((B, Hk, T, C),)


def _run_solve_tri(torch, ops, case, device):
    B, T, Hv, C = (_shape(case, key, 1) for key in ("B", "T", "H_v", "C"))
    layout = case["attrs"]["layout"]
    if layout == "bhtd":
        shape = (B, Hv, T, C)
        row_axis = 2
    elif layout == "bsnd":
        shape = (B, T, Hv, C)
        row_axis = 1
    else:
        shape = (T, Hv, C)
        row_axis = 0
    x = _tensor(torch, shape, _dtype(torch, case["dtype"]["x"]), device, scale=0.02)
    rows = torch.arange(T, device=device).remainder(C)
    cols = torch.arange(C, device=device)
    mask = cols.unsqueeze(0) < rows.unsqueeze(1)
    mask_shape = [1] * len(shape)
    mask_shape[row_axis] = T
    mask_shape[-1] = C
    x = (x * mask.reshape(mask_shape)).contiguous()
    out = ops.solve_tri(
        x, cu_seqlens=_meta(case, "cu_seqlens"), chunk_indices=_meta(case, "chunk_indices"), layout=layout,
    )
    return out, (shape,)


def _run_kda_gate_cumsum(torch, ops, case, device):
    s, attrs = case["shape"], case["attrs"]
    B, T, Hv, K = int(s.get("B", 1)), int(s["T"]), int(s["H_v"]), int(s["K"])
    layout = attrs["layout"]
    if layout == "NTD":
        shape = (Hv, T, K)
    elif layout == "TND":
        shape = (T, Hv, K)
    elif layout == "BNSD":
        shape = (B, Hv, T, K)
    else:
        shape = (B, T, Hv, K)
    g = _tensor(torch, shape, _dtype(torch, case["dtype"]["g"]), device)
    A_log = None
    dt_bias = None
    if _meta(case, "A_log") is not None:
        A_log = _tensor(torch, (Hv,), torch.float32, device)
    if _meta(case, "dt_bias") is not None:
        dt_bias = _tensor(torch, (Hv, K), torch.float32, device)
    out = ops.kda_gate_cumsum(
        g, int(attrs["chunk_size"]), A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=_meta(case, "cu_seqlens"), use_gate_in_kernel=attrs["use_gate_in_kernel"],
        safe_gate=attrs["safe_gate"], lower_bound=float(attrs["lower_bound"]), layout=layout,
    )
    return out, (shape,)


def _run_kda_layout_swap12(torch, ops, case, device):
    s = case["shape"]
    if case["layout"] == "ND-rank3":
        shape = (int(s["T"]), int(s["H_v"]), int(s["K"]))
        expected = (shape[1], shape[0], shape[2])
    else:
        shape = (int(s["B"]), int(s["N_c"]), int(s["H_v"]), int(s["K"]), int(s["V"]))
        expected = (shape[0], shape[2], shape[1], shape[3], shape[4])
    x = _tensor(torch, shape, _dtype(torch, case["dtype"]["x_y"]), device)
    dependency = _zeros(torch, (1,), x.dtype, device) if _meta(case, "dependency") is not None else None
    out = ops.kda_layout_swap12(x, dependency=dependency)
    return out, (expected,)


def _run_chunk_kda_fwd(torch, ops, case, device):
    s, attrs = case["shape"], case["attrs"]
    B, T, Hk, Hv, K, V = (int(s.get(k, 1)) for k in ("B", "T", "H_k", "H_v", "K", "V"))
    layout = attrs["layout"]
    if layout == "BSND":
        q_shape, v_shape = (B, T, Hk, K), (B, T, Hv, V)
        gk_shape, beta_shape = (B, T, Hv, K), (B, T, Hv)
    elif layout == "BNSD":
        q_shape, v_shape = (B, Hk, T, K), (B, Hv, T, V)
        gk_shape, beta_shape = (B, Hv, T, K), (B, Hv, T)
    elif layout == "NTD":
        q_shape, v_shape = (Hk, T, K), (Hv, T, V)
        gk_shape, beta_shape = (Hv, T, K), (Hv, T)
    else:
        q_shape, v_shape = (T, Hk, K), (T, Hv, V)
        gk_shape, beta_shape = (T, Hv, K), (T, Hv)
    data = _dtype(torch, case["dtype"]["q_k_v"])
    q = _tensor(torch, q_shape, data, device)
    k = _tensor(torch, q_shape, data, device)
    v = _tensor(torch, v_shape, data, device)
    gk = -_tensor(torch, gk_shape, torch.float32, device, positive=True, scale=0.02)
    beta = _tensor(torch, beta_shape, torch.float32, device, positive=True)
    cu = _meta(case, "cu_seqlens")
    seq_num = len(cu) - 1 if cu else B
    initial = None
    if _meta(case, "initial_state") is not None:
        initial = _zeros(torch, (seq_num, Hv, K, V), torch.float32, device)
    out = ops.chunk_kda_fwd(
        q, k, v, gk, beta, float(attrs["scale"]), int(attrs["chunk_size"]), layout=layout,
        initial_state=initial, output_final_state=attrs["output_final_state"], cu_seqlens=cu,
        chunk_indices=_meta(case, "chunk_indices"), return_intermediate=attrs["return_intermediate"],
        safe_gate=attrs["safe_gate"], transpose_state_layout=attrs["transpose_state_layout"],
    )
    return out, (v_shape, (seq_num, Hv, K, V))


def _sequence_lengths(total: int, batch: int, accepted):
    minimum = list(accepted) if accepted is not None else [1] * batch
    if len(minimum) != batch or sum(minimum) > total:
        raise ValueError("invalid recurrent batch/accepted-token manifest")
    lengths = minimum[:]
    for index in range(total - sum(lengths)):
        lengths[index % batch] += 1
    return lengths


def _run_recurrent_gated_delta_rule(torch, ops, case, device):
    s = case["shape"]
    B, T, Hk, Hv, K, V, Ds = (int(s[k]) for k in ("B", "T", "H_k", "H_v", "K", "V", "D_s"))
    data = _dtype(torch, case["dtype"]["qkv_beta_out"])
    q = _tensor(torch, (T, Hk, K), data, device)
    k = _tensor(torch, (T, Hk, K), data, device)
    v = _tensor(torch, (T, Hv, V), data, device)
    beta = _tensor(torch, (T, Hv), data, device, positive=True)
    state = _zeros(torch, (Ds, Hv, V, K), torch.float32, device)
    accepted = _meta(case, "num_accepted_tokens")
    lengths = _sequence_lengths(T, B, accepted)
    actual = torch.tensor([0] + lengths, dtype=torch.int32, device=device)
    indices = torch.arange(T, dtype=torch.int32, device=device).remainder(Ds)
    g = None
    gk = None
    if _meta(case, "g") is not None:
        g = -_tensor(torch, (T, Hv), torch.float32, device, positive=True, scale=0.02)
    if _meta(case, "gk") is not None:
        gk = -_tensor(torch, (T, Hv, K), torch.float32, device, positive=True, scale=0.02)
    accepted_tensor = None if accepted is None else torch.tensor(accepted, dtype=torch.int32, device=device)
    out = ops.recurrent_gated_delta_rule(
        q, k, v, beta, state, actual, indices, g=g, gk=gk,
        num_accepted_tokens=accepted_tensor, scale_value=float(case["attrs"]["scale_value"]),
    )
    return out, ((T, Hv, V), (Ds, Hv, V, K))


RUNNERS = {
    "causal_conv1d": _run_causal_conv1d,
    "causal_conv1d_bwd": _run_causal_conv1d_bwd,
    "chunk_bwd_dqkwg": _run_chunk_bwd_dqkwg,
    "chunk_bwd_dv_local": _run_chunk_bwd_dv_local,
    "chunk_fwd_o": _run_chunk_fwd_o,
    "chunk_gated_delta_rule_bwd_dhu": _run_chunk_gated_delta_rule_bwd_dhu,
    "chunk_gated_delta_rule_fwd_h": _run_chunk_gated_delta_rule_fwd_h,
    "chunk_kda_fwd": _run_chunk_kda_fwd,
    "chunk_local_cumsum": _run_chunk_local_cumsum,
    "chunk_scaled_dot_kkt": _run_chunk_scaled_dot_kkt,
    "kda_gate_cumsum": _run_kda_gate_cumsum,
    "kda_layout_swap12": _run_kda_layout_swap12,
    "prepare_wy_repr_bwd_da": _run_prepare_wy_repr_bwd_da,
    "prepare_wy_repr_bwd_full": _run_prepare_wy_repr_bwd_full,
    "recompute_wu_fwd": _run_recompute_wu_fwd,
    "recurrent_gated_delta_rule": _run_recurrent_gated_delta_rule,
    "solve_tri": _run_solve_tri,
}


def _flatten_outputs(value):
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten_outputs(item))
        return result
    return [value]


def run_generalization_cases(op_name: str, cases: list[Case]) -> None:
    """Run every selected positive case and enforce shape/finiteness contracts."""

    if op_name not in RUNNERS:
        raise KeyError(f"missing NPU generalization runner for {op_name}")
    try:
        import torch
        import torch_npu  # noqa: F401
        from fla_npu.ops import ascendc as ascendc_ops
    except Exception as exc:  # pragma: no cover - device environment dependent
        raise RuntimeError("NPU generalization tests require torch_npu and fla_npu") from exc

    device_id = int(os.environ.get("TEST_DEVICE_ID", 0))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    requested_case_id = os.environ.get("FLA_NPU_CASE_ID")
    if requested_case_id:
        cases = [case for case in cases if case["id"] == requested_case_id]
        if len(cases) != 1:
            raise AssertionError(f"{op_name}: unknown or duplicate case id {requested_case_id}")
    for case in cases:
        if case.get("expect", {}).get("return_code") != "ACLNN_SUCCESS":
            raise ValueError(f"{case['id']}: generalization runner accepts positive cases only")
        torch.manual_seed(int(case["seed"]))
        try:
            outputs, expected_shapes = RUNNERS[op_name](torch, ascendc_ops, case, device)
        except Exception as exc:
            raise AssertionError(f"{op_name}/{case['id']}: {exc}") from exc
        torch.npu.synchronize()
        tensors = _flatten_outputs(outputs)
        if len(tensors) < len(expected_shapes):
            raise AssertionError(
                f"{case['id']}: expected at least {len(expected_shapes)} tensor outputs, got {len(tensors)}"
            )
        for index, expected in enumerate(expected_shapes):
            actual = tuple(int(value) for value in tensors[index].shape)
            if actual != tuple(expected):
                raise AssertionError(f"{case['id']}: output[{index}] shape {actual} != {tuple(expected)}")
        for index, tensor in enumerate(tensors):
            if tensor.is_floating_point() and tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
                raise AssertionError(f"{case['id']}: output[{index}] contains NaN or Inf")
        print(f"[PASS] {op_name}/{case['id']} on {os.environ.get('FLA_NPU_SOC', '<unspecified-soc>')}")
