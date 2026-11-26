import ctypes
import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chunk_kda_fwd_fused"))
from executor_chunk_kda_fwd import ChunkKdaFwdApi as ForwardReference, _apply_output_policy, _public_kwargs, _spans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chunk_kda_fwd_prepare"))
from executor_chunk_kda_fwd_prepare_a2 import PreparedInputs, run_cpu as prepare_cpu


def _run_v2(inputs, spec):
    from fla_npu.ops.ascendc import _aclnn_ctypes as acl

    q = inputs.q
    if (q.dtype, spec["K"], spec["V"], spec["chunk_size"]) != (torch.bfloat16, 128, 128, 64):
        raise ValueError("Forward V2 requires BF16 Q/K/V, K=V=128 and chunk_size=64")
    cu = inputs.cu_seqlens
    if cu and any(a >= b for a, b in zip(cu, cu[1:])):
        raise ValueError("Forward V2 requires strictly increasing cu_seqlens")
    indices = inputs.chunk_indices or acl._kda_build_chunk_indices(cu, 64)
    batch, heads, tokens = spec["B"], spec["HV"], spec["T"]
    packed = spec["layout"] in ("TND", "NTD")
    prefix = () if packed else (batch,)
    chunks = acl._kda_total_chunks(batch, tokens, 64, cu, indices)
    head_shape = (*prefix, heads, tokens, 128)
    matrix_shape = (*prefix, heads, tokens, 64)
    mask = acl.kda_fwd_optional_output_mask(
        output_final_state=spec["output_final_state"], use_gate_in_kernel=spec["use_gate_in_kernel"],
        disable_recompute=spec["disable_recompute"], return_intermediate_states=spec["return_intermediate_states"],
    )
    seqs = len(cu) - 1 if cu else batch
    shapes = ((*prefix, tokens, heads, 128), (seqs, heads, 128, 128), head_shape,
              matrix_shape, matrix_shape, head_shape, head_shape, head_shape, head_shape,
              head_shape, (*prefix, chunks, heads, 128, 128))
    outputs = tuple(torch.empty(shape, device=q.device, dtype=torch.float32 if i in (1, 2) else q.dtype)
                    if mask[i] else None for i, shape in enumerate(shapes))
    layout = ctypes.create_string_buffer(spec["layout"].encode())
    def build(ctx):
        args = [ctx.tensor(getattr(inputs, key), key) for key in
                ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state")]
        args.extend([ctx.int_array(cu), ctx.int_array(indices), ctypes.cast(layout, ctypes.c_char_p),
                     ctypes.c_double(spec["scale"]), ctypes.c_int64(64), ctypes.c_bool(spec["safe_gate"]),
                     ctypes.c_double(spec["lower_bound"]), ctypes.c_bool(spec["use_gate_in_kernel"]),
                     ctypes.c_bool(spec["state_v_first"]), ctypes.c_double(spec["epsilon"]),
                     ctypes.c_bool(spec["use_qk_l2norm_in_kernel"]), ctypes.c_bool(spec["use_beta_sigmoid_in_kernel"]),
                     ctypes.c_bool(spec["allow_neg_eigval"]), ctypes.c_bool(spec["use_exp2"])])
        names = ("attn_out", "final_state", "gk", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h")
        args.extend(ctx.tensor(t, name) for t, name in zip(outputs, names))
        args.extend(ctx.tensor(None, name) for name in ("q_hat", "k_hat", "q_rstd", "k_rstd", "beta_eff"))
        return args

    acl._call_aclnn("aclnnChunkKdaFwdV2", build, outputs)
    return (*outputs, inputs.initial_state)


def _run_public(inputs, spec):
    from fla_npu.ops.ascendc import chunk_kda_fwd

    if (inputs.q.dtype, spec["K"], spec["V"], spec["chunk_size"]) != (torch.bfloat16, 128, 128, 64):
        raise ValueError("Forward V2 requires BF16 Q/K/V, K=V=128 and chunk_size=64")
    kwargs = _public_kwargs(inputs, spec)
    kwargs.update(epsilon=spec.get("epsilon", 1e-6),
                  use_qk_l2norm_in_kernel=spec.get("use_qk_l2norm_in_kernel", False),
                  use_beta_sigmoid_in_kernel=spec.get("use_beta_sigmoid_in_kernel", False),
                  allow_neg_eigval=spec.get("allow_neg_eigval", False),
                  use_exp2=spec.get("use_exp2", True))
    return chunk_kda_fwd(inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta,
                         spec["scale"], spec["chunk_size"], **kwargs)


def _forward_reference(inputs, spec, role):
    prepared = PreparedInputs(
        q=inputs.q, k=inputs.k, v=inputs.v, g=inputs.g, beta=inputs.beta,
        a_log=inputs.A_log, dt_bias=inputs.dt_bias,
        cu_seqlens=inputs.cu_seqlens, chunk_indices=inputs.chunk_indices,
    )
    prepare_spec = dict(spec, HK=spec["H"], backward_mode="save")
    for name, default in (("epsilon", 1e-6), ("use_qk_l2norm_in_kernel", False),
                          ("use_beta_sigmoid_in_kernel", False), ("allow_neg_eigval", False),
                          ("use_exp2", True)):
        prepare_spec.setdefault(name, default)
    stage = prepare_cpu(prepare_spec, prepared, role=role)
    compute_dtype = torch.float64 if role == "truth_fp64" else torch.float32
    exp = torch.exp2 if prepare_spec["use_exp2"] else torch.exp
    gk, aqk, akk, w, u, qg, kg, scaled = stage[:8]
    packed = spec["layout"] in ("TND", "NTD")
    gate, scores, weights, values, keys, queries = (
        t.unsqueeze(0) if packed else t for t in (gk, aqk, w, u, kg, scaled)
    )
    batch, heads, tokens, dim = weights.shape
    spans = _spans(batch, tokens, spec["chunk_size"], inputs.cu_seqlens)
    chunks = max(span[2] for span in spans) + 1
    sequences = len(inputs.cu_seqlens) - 1 if inputs.cu_seqlens else batch
    if inputs.initial_state is None:
        state = torch.zeros((sequences, heads, dim, dim), dtype=compute_dtype)
    else:
        state = inputs.initial_state.to(compute_dtype).clone()
        if spec["state_v_first"]:
            state = state.transpose(-1, -2)
    history = weights.new_empty((batch, chunks, heads, dim, dim))
    v_new = torch.empty_like(values)
    output = torch.empty_like(values)
    for b, seq, chunk, start, end in spans:
        region = (b, slice(None), slice(start, end))
        previous = state[seq].to(weights.dtype)
        history[b, chunk] = previous
        updated = values[region].to(compute_dtype) - weights[region].to(compute_dtype) @ previous.to(compute_dtype)
        v_new[region] = updated
        rounded = v_new[region].to(compute_dtype)
        last_gate = gate[b, :, end - 1].to(compute_dtype)
        state[seq] = exp(last_gate).unsqueeze(-1) * state[seq] + keys[region].to(compute_dtype).transpose(-1, -2) @ rounded
        output[region] = queries[region].to(compute_dtype) @ previous.to(compute_dtype) + scores[region][..., :end-start].to(compute_dtype) @ rounded

    if spec["state_v_first"]:
        state = state.transpose(-1, -2)
        history = history.transpose(-1, -2)
    output = output.transpose(1, 2).contiguous()
    if packed:
        output, v_new, history = (t.squeeze(0) for t in (output, v_new, history))
    return _apply_output_policy((output, state, gk, aqk, akk, w, u, qg, kg, v_new,
                                 history, inputs.initial_state), spec)


def _run_cpu_control(inputs, spec, role="control_fp32_bf16"):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _forward_reference(inputs, spec, role)
    finally:
        torch.set_num_threads(threads)


@register("executor_chunk_kda_fwd_v2")
class FunctionApi(ForwardReference):
    @staticmethod
    def _run_fp64(inputs, spec):
        return _run_cpu_control(inputs, spec, role="truth_fp64")

    _run_npu = staticmethod(_run_v2)
    _run_cpu_control = staticmethod(_run_cpu_control)

@register("executor_chunk_kda_fwd_public")
class PublicForwardApi(FunctionApi):
    """Test the public chunk64 dispatcher with the switches specified by the case."""

    _run_npu = staticmethod(_run_public)
