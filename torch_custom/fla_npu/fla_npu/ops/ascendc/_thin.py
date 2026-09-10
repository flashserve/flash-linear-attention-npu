# Thin C++ launcher adapters for the hot FLA NPU operators.
#
# Enabled per-process with FLA_NPU_THIN_LAUNCHER=1 after the extension has been
# built (FLA_NPU_BUILD_THIN=1). Signatures mirror the ctypes wrappers in
# _aclnn_ctypes.py so the existing public API and mutation contracts are kept.
from __future__ import annotations

import os


def _extension() -> "module":
    import fla_npu._C_thin as ext

    lib = os.environ.get("FLA_NPU_OP_API_LIB", "")
    if lib:
        try:
            ext.init(lib)
        except Exception:
            pass
    return ext


def _current_stream_ptr() -> int:
    """Return the raw aclrtStream of the *current* stream of the calling thread.

    ``torch_npu._C._npu_getCurrentRawStream`` returns the stream pointer
    directly (~1us) instead of building a ``Stream`` object through
    ``torch.npu.current_stream()`` (~24us).  Unlike a process-global cache it
    cannot leak one thread's stream into another: servers such as vLLM run the
    operators from multiple worker threads with independent NPU streams, and a
    shared cache there enqueues kernels on the wrong stream (illegal address /
    broken ordering).  Older torch_npu builds without the raw accessor fall back
    to the object path.
    """

    import torch

    try:
        import torch_npu

        raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
        if raw_stream is not None:
            return int(raw_stream(torch.npu.current_device()))
    except Exception:
        pass
    return int(torch.npu.current_stream().npu_stream)


def npu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    *,
    beta,
    scale=1.0,
    actual_seq_lengths,
    ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
):
    if g is None and gk is None:
        raise RuntimeError(
            "npu_recurrent_gated_delta_rule: either g or gk must be provided.")
    ext = _extension()
    return ext.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        float(scale),
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        _current_stream_ptr(),
    )


def npu_causal_conv1d(
    x,
    weight,
    bias=None,
    conv_states=None,
    *,
    query_start_loc=None,
    cache_indices=None,
    initial_state_mode=None,
    num_accepted_tokens=None,
    activation_mode=0,
    pad_slot_id=-1,
    run_mode=0,
    head_num=0,
):
    ext = _extension()

    def as_list(value):
        return [] if value is None else [int(v) for v in value]

    return ext.npu_causal_conv1d(
        x,
        weight,
        bias,
        conv_states,
        as_list(query_start_loc),
        as_list(cache_indices),
        as_list(initial_state_mode),
        as_list(num_accepted_tokens),
        int(activation_mode),
        int(pad_slot_id),
        int(run_mode),
        int(head_num),
        _current_stream_ptr(),
    )


def npu_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
    out=None,
    *,
    conv_state_indices_cpu=None,
    num_accepted_tokens_cpu=None,
    query_start_loc_cpu=None,
):
    """PR #390 update form backed by the thin launcher."""

    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        raise NotImplementedError(
            "npu_causal_conv1d_update thin path: block_idx_last_scheduled_token "
            "and initial_state_idx are not supported")
    if validate_data:
        raise NotImplementedError(
            "npu_causal_conv1d_update thin path: validate_data is not supported")
    if activation is None:
        act = "none"
    elif activation in ("silu", "swish"):
        act = activation
    else:
        raise ValueError(
            f"activation must be None, 'silu', or 'swish', got {activation!r}")

    ext = _extension()

    def as_list(value):
        return [] if value is None else [int(v) for v in value]

    result = ext.npu_causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        act,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        as_list(conv_state_indices_cpu),
        as_list(num_accepted_tokens_cpu),
        as_list(query_start_loc_cpu),
        int(max_query_len),
        int(null_block_id),
        out,
        _current_stream_ptr(),
    )
    if out is None:
        x.copy_(result)
        return x
    return out
