import sys
from pathlib import Path

import torch
from atk.tasks.api_execute import register

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chunk_kda_fwd_fused"))
from executor_chunk_kda_fwd import ChunkKdaFwdApi as ForwardReference, _apply_output_policy, _public_kwargs, _spans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chunk_kda_fwd_prepare"))
from executor_chunk_kda_fwd_prepare_a2 import PreparedInputs, run_cpu as prepare_cpu


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

    @staticmethod
    def _run_npu(inputs, spec):
        cu = inputs.cu_seqlens
        if cu and any(left >= right for left, right in zip(cu, cu[1:])):
            raise ValueError("Forward V2 requires strictly increasing cu_seqlens")
        outputs = _run_public(inputs, spec)
        # This ATK registration exposes the original 11 outputs and caller state.
        return (*outputs[:11], inputs.initial_state)

    _run_cpu_control = staticmethod(_run_cpu_control)

@register("executor_chunk_kda_fwd_public")
class PublicForwardApi(FunctionApi):
    """Test the public chunk64 dispatcher with the switches specified by the case."""

    _run_npu = staticmethod(_run_public)
