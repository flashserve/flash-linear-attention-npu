"""Stable-ABI backend (Phase 1).

Loads ``libfla_npu_stable.so`` (a plain shared object registered through
``STABLE_TORCH_LIBRARY``) and exposes the same Python call shape as the ctypes
reference.  This module contains no ABI-sensitive code: the only tensor objects
crossing the boundary are handled by torch's own dispatcher.

The mutation contract (version bump / requires_grad rejection) is *not*
provided by the dispatcher for these ops -- measured in Phase 1 -- so the
dispatch layer keeps applying ``_wrap_mutable_direct_op`` on top of whatever
backend it picks.
"""

from __future__ import annotations

import os


_LIB_ENV = "FLA_NPU_STABLE_LIB"
# Lowest torch whose stable runtime symbols the launcher was verified against.
# Keep in sync with STABLE_ABI_MIN_TORCH in scripts/build_wheel.py.
_MIN_TORCH = "2.7.1"
_loaded_path: str | None = None
_OP_CACHE: dict[str, object] = {}
# Cached objects for the hot path.  `torch`/`torch_npu` are plain module
# handles and the raw-stream accessor is a plain function: caching *those* is
# safe.  The stream itself is never cached -- that is what corrupted the vLLM
# run earlier, where a process-global stream pointer followed a different
# thread.
_torch = None
_torch_npu = None
_raw_stream_fn = None
# int[] argument cache: value tuple -> host int64 tensor (see _host_ints).
_INT_CACHE: dict[tuple, object] = {}
_INT_CACHE_MAX = 64

# The stable value conversions have no std::string support, so string enum
# arguments travel as int codes.  Every layout argument uses the same order --
# BSND, BNSD, TND, NTD -- which is what `stable/layout_math.h` assumes;
# tools/op_abi_parity.py checks these tables against the adapters' name tables.
_LAYOUT_CODES = {"BSND": 0, "BNSD": 1, "TND": 2, "NTD": 3}

_ENUM = {
    "npu_causal_conv1d_bwd": {"input_layout": _LAYOUT_CODES},
    "npu_chunk_kda_bwd_intra": {"layout": {"BSND": 0, "BNSD": 1, "TND": 2}},
    "npu_chunk_kda_fwd": {"layout": _LAYOUT_CODES},
    # The recurrent KDA kernel only implements the two spellings the reference
    # accepts, so this op's table is the (BSND, TND) subset.
    "npu_recurrent_kda": {"layout": {"BSND": 0, "TND": 1}},
    "npu_chunk_local_cumsum": {"output_dtype": {"float32": 0,
                                                "bfloat16": 1}},
    "npu_solve_tri": {"layout": {"bsnd": 0, "bnsd": 1, "tnd": 2, "ntd": 3}},
}


def _lib_path() -> str:
    path = os.environ.get(_LIB_ENV)
    if path:
        return path
    # Wheels that ship the ABI-free launcher place it next to this module.
    bundled = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "libfla_npu_stable.so")
    if os.path.exists(bundled):
        return bundled
    raise RuntimeError(
        f"{_LIB_ENV} is not set and no bundled libfla_npu_stable.so was found")


def _current_stream_ptr() -> int:
    """Raw NPU stream of the calling thread (no cached stream pointer)."""

    global _raw_stream_fn
    torch = _modules()[0]
    if _raw_stream_fn is not None:
        return int(_raw_stream_fn(torch.npu.current_device()))
    try:
        torch_npu = _modules()[1]
        if not torch_npu:
            raise AttributeError("torch_npu is not importable")
        _raw_stream_fn = getattr(torch_npu._C, "_npu_getCurrentRawStream")
    except Exception:
        _raw_stream_fn = False  # look the slow way from now on
    if not _raw_stream_fn:
        return int(torch.npu.current_stream().npu_stream)
    return int(_raw_stream_fn(torch.npu.current_device()))


def _modules():
    """(torch, torch_npu) once imported; kept out of the per-call path."""

    global _torch, _torch_npu
    if _torch is None:
        import torch as _t

        _torch = _t
    if _torch_npu is None:
        try:
            import torch_npu as _tn

            _torch_npu = _tn
        except Exception:
            _torch_npu = False
    return _torch, _torch_npu


def load() -> None:
    """dlopen the stable library through torch (no-op when already loaded)."""

    global _loaded_path
    # Hot path: once a library is loaded, re-resolving it means an environment
    # lookup plus a filesystem stat on every single operator call (~47us
    # measured).  Only an explicitly different FLA_NPU_STABLE_LIB re-resolves.
    if _loaded_path is not None:
        requested = os.environ.get(_LIB_ENV)
        if not requested or requested == _loaded_path:
            return
    path = _lib_path()
    if _loaded_path == path:
        return
    torch = _modules()[0]
    try:
        torch.ops.load_library(path)
    except Exception as exc:  # symbol resolution happens here, not at dlopen
        # The launcher resolves aoti_torch_* at load; an older torch fails with
        # "undefined symbol" a long way from the cause, so say what is wrong.
        raise RuntimeError(
            f"fla_npu: cannot load the Stable-ABI launcher {path} against "
            f"torch {torch.__version__}. It needs torch >= {_MIN_TORCH} (the "
            f"aoti_torch_* runtime symbols it resolves were added over 2.7.x). "
            f"Original error: {exc}") from exc
    _check_build_stamp(path)
    _loaded_path = path


def _check_build_stamp(path: str) -> None:
    """Refuse a library built from different adapter sources than this glue.

    ``build_stable.py`` stamps the library with the hash of the sources it
    compiled and writes the same value into ``_stable_hash``; a wrapper whose
    matching adapter was not rebuilt otherwise shows up as a dispatcher error
    deep inside a call or -- when only a stack index moved -- as a wrong stream,
    which is much harder to read.  A library predating the stamp reports
    ``unknown`` and is accepted, and a tree without the generated module (a
    source checkout that never built one) skips the check.
    """

    try:
        from . import _stable_hash

        expected = _stable_hash.SOURCE_HASH
    except Exception:
        return
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        lib.fla_npu_stable_source_hash.restype = ctypes.c_char_p
        actual = lib.fla_npu_stable_source_hash().decode("utf-8", "replace")
    except Exception:
        return
    if actual in ("unknown", expected):
        return
    raise RuntimeError(
        f"{path} was built from different adapter sources than this package "
        f"(library {actual}, package {expected}). Rebuild the launcher: "
        f"python csrc/build_stable.py --out {path} --no-debug-probe")


def available() -> bool:
    try:
        load()
        import torch

        return hasattr(torch.ops.fla_npu_stable, "npu_recurrent_gated_delta_rule")
    except Exception:
        return False


# The conv1d family used to be re-exported from the ctypes module with only its
# launch handed to an internal op, which meant every call kept paying for the
# reference marshalling.  It now has three real adapters (see the hand-written
# wrappers at the end of this module and csrc/src/stable_conv1d.cpp).
def _op(name: str):
    """Cached torch.ops handle: the attribute chain is not free per call."""

    op = _OP_CACHE.get(name)
    if op is None:
        load()
        import torch

        op = getattr(torch.ops.fla_npu_stable, name)
        _OP_CACHE[name] = op
    return op


def _bound_op(name: str):
    """Op handle for a hot path: one dict lookup, resolved on the first call.

    ``_op`` is already cached, but it is still a Python call per invocation.
    The hot wrappers below read the cache directly and only fall back to
    ``_op`` (which loads the library) the first time, so the per-call work is
    the dispatch itself plus the stream lookup.
    """

    op = _OP_CACHE.get(name)
    if op is None:
        op = _op(name)
    return op


def stream_probe(device_index: int) -> tuple[int, int]:
    """Return (raw backend stream ptr, stable Stream::id()) for comparison."""

    load()
    import torch

    raw, stream_id = _op("_stream_probe")(int(device_index))
    return int(raw), int(stream_id)


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
    """Recurrent GDN forward; mutates ``state`` in place like the other paths."""

    # The reference refuses a call that supplies neither gate; the kernel does
    # not.  Measured on 910B3: without this check the launcher returned the
    # ungated result instead of raising -- exactly the failure mode this path
    # must not have, i.e. silently different numbers rather than an error.  Same
    # shape as the use_exp2 refusal repeated in npu_chunk_gated_delta_rule_bwd.
    if g is None and gk is None:
        raise RuntimeError(
            "npu_recurrent_gated_delta_rule: either g or gk must be provided.")

    # Hot path: `_bound_op` reads the already-populated cache directly (the
    # first call is what loads the library), so a decode step does not pay for
    # the load check, the module imports or an extra Python call per operator.
    return _bound_op("npu_recurrent_gated_delta_rule")(
        query,
        key,
        value,
        state,
        beta,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        float(scale),
        _current_stream_ptr(),
    )


def npu_recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
):
    """Recurrent KDA forward via the Stable-ABI launcher.

    ``layout`` is mapped to an int code (BSND=0, TND=1) because the stable
    argument conversions do not carry strings.  Mutates ``initial_state`` in
    place when ``inplace_final_state`` is true, exactly like the ctypes path.
    """

    import torch
    import torch_npu

    layout_code = _char_code("npu_recurrent_kda", "layout", layout)

    if not inplace_final_state:
        # ctypes drives the same kernel with a scratch state and returns it,
        # leaving the caller's tensor untouched.  The stable launcher only
        # exposes the inplace form (handing the caller's handle back as a second
        # output trips over shared ownership), so build the scratch here -- which
        # is also what keeps the mutation contract honest: the caller's tensor is
        # genuinely not written, and the dispatch layer's MUTATION_FLAGS entry
        # already skips the version bump for this case.
        if initial_state is None:
            raise RuntimeError(
                "npu_recurrent_kda: inplace_final_state=False requires "
                "initial_state (no shape to build the scratch from)")
        scratch = torch.empty_like(initial_state)
        out, _ = npu_recurrent_kda(
            q, k, v, g, beta, scratch, cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices, A_log=A_log, dt_bias=dt_bias,
            num_accepted_tokens=num_accepted_tokens, layout=layout, scale=scale,
            output_final_state=False, inplace_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval, safe_gate=safe_gate,
            lower_bound=lower_bound, state_v_first=state_v_first)
        return out, (scratch if output_final_state else None)

    scale_value = (128.0 ** -0.5) if scale is None else float(scale)
    lower = -5.0 if lower_bound is None else float(lower_bound)
    stream = _current_stream_ptr()
    out, final_state = _op("npu_recurrent_kda")(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        A_log,
        dt_bias,
        num_accepted_tokens,
        layout_code,
        scale_value,
        bool(output_final_state),
        bool(inplace_final_state),
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval),
        bool(safe_gate),
        lower,
        bool(state_v_first),
        stream,
    )
    if not output_final_state:
        return out, None
    if inplace_final_state and final_state is None:
        # The launcher returns the inplace result implicitly (the kernel writes
        # the caller's tensor); the stable ABI cannot hand that handle back as a
        # second output without a double ownership release, so mirror ctypes
        # here, which also returns the caller's object.
        final_state = initial_state
    return out, final_state


# ---------------------------------------------------------------------------
# Generic plumbing for the generated wrappers.
# ---------------------------------------------------------------------------
def _host_ints(values):
    """int[] arguments travel as host int64 tensors (no list support in the
    stable conversions).

    Decode-time calls reuse the same length list over and over (a batch of
    identical sequences), and building a tensor costs ~18us, so the result is
    cached by value.  Only list/tuple inputs are cached: a tensor is passed
    through, and anything else is converted without caching.
    """

    if values is None:
        return None
    import torch

    if not isinstance(values, (list, tuple)):
        return torch.tensor(list(values), dtype=torch.int64, device="cpu")
    key = tuple(values)
    cached = _INT_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.tensor(list(values), dtype=torch.int64, device="cpu")
    if len(_INT_CACHE) >= _INT_CACHE_MAX:
        _INT_CACHE.clear()
    _INT_CACHE[key] = tensor
    return tensor


def _char_code(op_name: str, argument: str, value):
    """Map a string argument to the int code the stable schema carries."""

    table = _ENUM[op_name][argument]
    if value is None:
        return 0
    if not isinstance(value, str):
        return value
    try:
        return table[value]
    except KeyError:
        raise RuntimeError(
            f"{op_name}: {argument} must be one of "
            f"{sorted(table)}, got {value!r}") from None




# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------
#
# Every wrapper here has the same shape: a real signature (so a positional call
# does no argument binding at run time) that maps the public argument names
# onto the adapter's schema and nothing else.  Validation stays with the
# operator: an illegal input either reaches aclnn and comes back as a status,
# or is caught by the C++ adapter.  FLA_NPU_STABLE_VALIDATE=1 routes
# such a call through the ctypes reference instead, which validates in Python
# and reports a precise message.
#
# These wrappers are appended after the generated import so a hand-written one
# always wins, which is what makes migrating an operator a one-file change on
# each side.


def npu_fast_gelu_custom(self):
    """GELU with the operator's own approximation; mirrors the ctypes shape."""

    return _op("npu_fast_gelu_custom")(self, _current_stream_ptr())


def npu_fast_gelu_custom_backward(grad, self):
    """Backward of :func:`npu_fast_gelu_custom`."""

    return _op("npu_fast_gelu_custom_backward")(
        grad, self, _current_stream_ptr())


def npu_kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None,
                        cu_seqlens=None, use_gate_in_kernel=False,
                        safe_gate=False, lower_bound=None):
    """KDA gate with the log-cumsum folded in.

    The schema can only carry real values, so the optional-argument defaults the
    ctypes reference applies are applied here too (`lower_bound` defaults to
    -5.0 there, and passing None straight through is not representable).
    """

    return _op("npu_kda_gate_cumsum")(
        g,
        A_log,
        dt_bias,
        _host_ints(cu_seqlens),
        chunk_size,
        False if use_gate_in_kernel is None else bool(use_gate_in_kernel),
        False if safe_gate is None else bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        _current_stream_ptr(),
    )


def npu_chunk_kda_bwd_intra(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, *,
                            cu_seqlens=None, chunk_indices=None, chunk_size=64,
                            safe_gate=True, layout="BSND"):
    """Safe-gate KDA intra-chunk backward.

    All three layouts go to the kernel: the aclnn entry point takes the layout
    as a string and the kernel reads the tensor as ND, so BSND needs no
    transposed copy (the ctypes reference does the same thing).  The previous
    shape/flag guard only kept the dense-BNSD case on this path and sent
    everything else through the ctypes reference, which cost as much as the
    reference for the whole operator; validation is now the kernel's job, and
    FLA_NPU_STABLE_VALIDATE=1 routes a call through the reference when a precise
    Python-side message matters.
    """

    # `safe_gate=False` is reserved but not implemented, and the adapter passes
    # the value straight through, so it has to be refused here as well.
    if not safe_gate:
        raise RuntimeError(
            "npu_chunk_kda_bwd_intra: safe_gate=False is reserved but not "
            "supported in v1.")
    return _op("npu_chunk_kda_bwd_intra")(
        q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        chunk_size,
        safe_gate,
        _char_code("npu_chunk_kda_bwd_intra", "layout", str(layout)),
        _current_stream_ptr(),
    )


def npu_chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None,
                           A=None, cu_seqlens=None, chunk_indices=None):
    """Local dv contribution of the chunked GDN backward."""

    return _op("npu_chunk_bwd_dv_local")(
        q, k, d_o, g, g_gamma, A,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        scale, chunk_size, _current_stream_ptr(),
    )


def npu_chunk_local_cumsum(g, chunk_size, *, cu_seqlens=None,
                           chunk_indices_out=None, reverse=False, scale=1.0,
                           head_first=True, output_dtype="float32"):
    """Per-chunk cumulative sum of ``g``."""

    return _op("npu_chunk_local_cumsum")(
        g,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices_out),
        chunk_size,
        reverse,
        scale,
        head_first,
        _char_code("npu_chunk_local_cumsum", "output_dtype", output_dtype),
        _current_stream_ptr(),
    )


def npu_chunk_scaled_dot_kkt(k, g, beta, *, cu_seqlens=None,
                             chunk_indices=None, chunk_size=64):
    """Chunked scaled dot product used to build the WY representation."""

    return _op("npu_chunk_scaled_dot_kkt")(
        k, g, beta,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_chunk_bwd_dqkwg(q, k, v, g, h, dox, dh, dv, chunk_size, *,
                        cu_seqlens=None, chunk_indices=None, w=None,
                        g_gamma=None, scale=None, use_exp2=None,
                        transpose_state_layout=None):
    """dq / dk / dw / dg of one chunk.

    The three trailing flags are optional in the published signature; the
    ctypes reference supplies ``scale=1.0`` and ``False`` for the booleans, so
    do the same here rather than passing None into a scalar slot.
    """

    return _op("npu_chunk_bwd_dqkwg")(
        q, k, v, g, h, dox, dh, dv,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        w, g_gamma,
        1.0 if scale is None else float(scale),
        chunk_size,
        False if use_exp2 is None else bool(use_exp2),
        False if transpose_state_layout is None
        else bool(transpose_state_layout),
        _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd_da(k, v, beta, A, dw, du, g, *, chunk_size,
                               cu_seqlens=None, chunk_indices=None):
    """dA only, for backends that already have the other gradients."""

    return _op("npu_prepare_wy_repr_bwd_da")(
        k, v, beta, A, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size, *,
                                 cu_seqlens=None, chunk_indices=None):
    """dk / dv / dbeta / dg, taking dA as an input."""

    return _op("npu_prepare_wy_repr_bwd_full")(
        k, v, beta, A, dA, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_prepare_wy_repr_bwd(k, v, beta, A, dw, du, g, chunk_size, *,
                            cu_seqlens=None, chunk_indices=None):
    """dk / dv / dbeta / dg; produces dA internally."""

    return _op("npu_prepare_wy_repr_bwd")(
        k, v, beta, A, dw, du, g,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_recompute_w_u_fwd(k, v, beta, A, chunk_size, *, g=None, gk=None,
                          cu_seqlens=None, chunk_indices=None):
    """Recompute w and u for the backward pass."""

    return _op("npu_recompute_w_u_fwd")(
        k, v, beta, A, g, gk,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size, _current_stream_ptr(),
    )


def npu_causal_conv1d_bwd(x, y, weight, dy, initial_state=None, dht=None, *,
                          query_start_loc=None, activation=0,
                          input_layout="BSND"):
    """dx / dweight / dbias / d(initial_state) of the causal conv1d.

    ``input_layout`` selects whether the state gradient carries one row per
    batch entry or one per segment, which the adapter derives from the layout
    name and the query_start_loc values.
    """

    return _op("npu_causal_conv1d_bwd")(
        x, y, weight, dy, initial_state, dht,
        _host_ints(query_start_loc),
        activation,
        _char_code("npu_causal_conv1d_bwd", "input_layout", str(input_layout)),
        _current_stream_ptr(),
    )


def npu_chunk_fwd_o(q, k, v, h, scale, *, g=None, g_gamma=None,
                    cu_seqlens=None, chunk_indices=None, chunk_size=None,
                    transpose_state_layout=False):
    """Output of one chunked attention pass.

    ``g_gamma`` is part of the published signature and ignored, exactly as the
    ctypes reference ignores it, and so is ``transpose_state_layout``: this
    branch's aclnn entry point takes neither, so there is nothing to forward.
    ``chunk_size`` defaults to 64, matching the reference.
    """

    del g_gamma, transpose_state_layout
    return _op("npu_chunk_fwd_o")(
        q, k, v, h, g,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        scale,
        64 if chunk_size is None else chunk_size,
        _current_stream_ptr(),
    )




# ---------------------------------------------------------------------------
# conv1d family
# ---------------------------------------------------------------------------
#
# One aclnn entry point, three published entry points; the run mode is baked
# into which adapter is called rather than travelling as an argument.  What
# stays here is the part the adapter cannot express: refusing the scheduling
# parameters the operator does not implement, translating the activation name,
# and staging a non-dense conv_state (the kernel reads it with the dense
# strides, so a paged view would be read and written in the wrong place).

_PAD_SLOT_ID = -1
_NULL_BLOCK_ID = 0
_CONV1D_ACTIVATION_CODES = {"none": 0, "silu": 1, "swish": 2}


def _conv1d_activation_code(activation):
    code = _CONV1D_ACTIVATION_CODES.get(
        "none" if activation is None else str(activation))
    if code is None:
        raise ValueError(
            f"activation must be None, 'silu', or 'swish', got {activation!r}")
    return code


def _reject_conv1d_scheduling(**values):
    """The operator implements neither block-cache nor APC scheduling."""

    enabled = [name for name, value in values.items() if value is not None]
    if enabled:
        raise NotImplementedError(
            "CausalConv1d APC/block-cache scheduling is not supported by the "
            "Ascend operator: " + ", ".join(enabled))


def _dense_conv_state(conv_state):
    """(argument to pass, tensor to copy back into) for a paged conv_state.

    A contiguous view is handed over as-is, storage offset included: the
    operator addresses the state with the stride fallback and still honours the
    descriptor's offset, so it reads and writes the right rows.  Measured
    bit-exact against the staging path on a state whose storage offset is 96
    elements (see `scenario_conv1d_update_offset_state`), and that state is what
    took the staging copy off the hot path.  Only a state whose rows are not
    dense has to be staged.
    """

    if conv_state.is_contiguous():
        return conv_state, None
    return conv_state.contiguous(), conv_state


def npu_causal_conv1d_fn(x, weight, bias, conv_states=None,
                         query_start_loc=None, cache_indices=None,
                         has_initial_state=None, activation="silu",
                         pad_slot_id=_PAD_SLOT_ID,
                         null_block_id=_NULL_BLOCK_ID,
                         block_idx_first_scheduled_token=None,
                         block_idx_last_scheduled_token=None,
                         initial_state_idx=None, num_computed_tokens=None,
                         block_size_to_align=0, metadata=None,
                         validate_data=False, *, query_start_loc_cpu=None,
                         cache_indices_cpu=None, has_initial_state_cpu=None,
                         head_num=0):
    """Prefill: convolve ``x`` and roll its tail into ``conv_states``."""

    _reject_conv1d_scheduling(
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed_tokens, metadata=metadata)
    if block_size_to_align not in (0, None):
        raise NotImplementedError(
            "CausalConv1d block_size_to_align is not supported by the Ascend "
            "operator")
    state_arg, restore = (None, None) if conv_states is None else (
        _dense_conv_state(conv_states))
    result = _op("npu_causal_conv1d_fn")(
        x, weight, bias, state_arg,
        query_start_loc, cache_indices, has_initial_state,
        _host_ints(query_start_loc_cpu), _host_ints(cache_indices_cpu),
        _host_ints(has_initial_state_cpu),
        _conv1d_activation_code(activation),
        _PAD_SLOT_ID if pad_slot_id is None else pad_slot_id,
        _NULL_BLOCK_ID if null_block_id is None else null_block_id,
        head_num, _current_stream_ptr(),
    )
    if restore is not None:
        restore.copy_(state_arg)
    return result


def npu_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None,
                             conv_state_indices=None,
                             num_accepted_tokens=None, query_start_loc=None,
                             max_query_len=-1,
                             null_block_id=_NULL_BLOCK_ID,
                             block_idx_last_scheduled_token=None,
                             initial_state_idx=None, validate_data=False,
                             out=None, *, conv_state_indices_cpu=None,
                             num_accepted_tokens_cpu=None,
                             query_start_loc_cpu=None):
    """Decode: one token per sequence, mutating ``conv_state`` in place."""

    _reject_conv1d_scheduling(
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx)
    state_arg, restore = _dense_conv_state(conv_state)
    result = _bound_op("npu_causal_conv1d_update")(
        x, state_arg, weight, bias, _conv1d_activation_code(activation),
        conv_state_indices, num_accepted_tokens, query_start_loc,
        max_query_len,
        _NULL_BLOCK_ID if null_block_id is None else null_block_id,
        _host_ints(conv_state_indices_cpu),
        _host_ints(num_accepted_tokens_cpu),
        _host_ints(query_start_loc_cpu), out, _current_stream_ptr(),
    )
    if restore is not None:
        restore.copy_(state_arg)
    if out is not None:
        # The operator wrote into the caller's buffer: it is the result, and the
        # copy the reference needs (aclnn always allocates its own output) is
        # exactly what this path avoids.
        return out
    x.copy_(result)
    return x


def npu_causal_conv1d(x, weight, bias=None, conv_states=None, *,
                      query_start_loc=None, cache_indices=None,
                      initial_state_mode=None, num_accepted_tokens=None,
                      activation_mode=0, pad_slot_id=-1, run_mode=0,
                      head_num=0):
    """Deprecated host-metadata compatibility interface."""

    import warnings

    warnings.warn(
        "fla_npu.ops.ascendc.npu_causal_conv1d is a deprecated compatibility "
        "API and will be removed in 2027/02. Use causal_conv1d_fn or "
        "causal_conv1d_update instead.",
        FutureWarning,
        stacklevel=4,
    )
    activation_mode = int(activation_mode)
    if activation_mode not in (0, 1):
        raise ValueError(
            f"activation_mode only supports 0/1, got {activation_mode}")
    state_arg, restore = (None, None) if conv_states is None else (
        _dense_conv_state(conv_states))
    result = _op("npu_causal_conv1d")(
        x, weight, bias, state_arg,
        _host_ints(query_start_loc), _host_ints(cache_indices),
        _host_ints(initial_state_mode), _host_ints(num_accepted_tokens),
        _CONV1D_ACTIVATION_CODES["silu" if activation_mode == 1 else "none"],
        pad_slot_id, run_mode, head_num, _current_stream_ptr(),
    )
    if restore is not None:
        restore.copy_(state_arg)
    return result


# ---------------------------------------------------------------------------
# chunked forward-h and backward-dhu
# ---------------------------------------------------------------------------


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    """Fill in the chunk_indices a varlen caller left out.

    The operator takes both forms; deriving the canonical sequence-major list
    here keeps the call shape the reference accepts without the caller having to
    build it.
    """

    indices = []
    for seq in range(len(cu_seqlens) - 1):
        length = cu_seqlens[seq + 1] - cu_seqlens[seq]
        for local in range((length + chunk_size - 1) // chunk_size):
            indices.extend((seq, local))
    return indices




def npu_chunk_gated_delta_rule_fwd_h(k, w, u, g=None, *, gk=None,
                                     initial_state=None,
                                     output_final_state=False, chunk_size=None,
                                     cu_seqlens=None, chunk_indices=None,
                                     state_v_first=False):
    """chunk_fwd_h without the GDN recompute flags."""

    chunk_size = 64 if chunk_size is None else chunk_size
    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    return _op("npu_chunk_gated_delta_rule_fwd_h")(
        k, w, u, g, gk, initial_state,
        output_final_state, chunk_size,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        state_v_first, _current_stream_ptr(),
    )


def npu_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv, scale, chunk_size, *, g=None, gK=None, h0=None,
        dht=None, cu_seqlens=None, chunk_indices=None, use_exp2=False,
        transpose_state_layout=False):
    """dh / dh0 / dv, where dh0 is only produced when h0 was supplied.

    ``transpose_state_layout`` is part of the published signature and ignored,
    exactly as the ctypes reference ignores it: this branch's entry point has no
    stateVFirst flag, so dh0 always carries the (K, V) tail.  ``use_exp2``
    follows the reference's own default: ``None`` means "true when gK was
    supplied".
    """

    del transpose_state_layout
    use_exp2 = bool(gK is not None) if use_exp2 is None else bool(use_exp2)
    return _op("npu_chunk_gated_delta_rule_bwd_dhu")(
        q, k, w, d_o, dv, g, gK, h0, dht,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        scale, chunk_size, use_exp2,
        _current_stream_ptr(),
    )


def npu_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size=64, *,
                      layout="BSND", initial_state=None,
                      output_final_state=False, cu_seqlens=None,
                      chunk_indices=None, safe_gate=False, lower_bound=None,
                      use_gate_in_kernel=False, A_log=None, dt_bias=None,
                      disable_recompute=False,
                      return_intermediate_states=False, state_v_first=False):
    """KDA chunked forward, returning the saved tensors the backward needs.

    ``disable_recompute`` is what makes `w`/`u`/`qg`/`kg`/`v_new` real outputs
    rather than null handles, and `chunk_indices` defaults to the canonical
    sequence-major list the kernel expects.  The trailing value is the caller's
    own ``initial_state``, which the operator updates in place (the reference
    API returns it the same way).
    """

    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    result = _op("npu_chunk_kda_fwd")(
        q, k, v, g, beta, A_log, dt_bias, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_chunk_kda_fwd", "layout", layout),
        float(scale), chunk_size,
        bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        bool(use_gate_in_kernel), bool(state_v_first),
        bool(output_final_state), bool(disable_recompute),
        bool(return_intermediate_states),
        _current_stream_ptr(),
    )
    return (*result, initial_state)




def npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd"):
    """Solve the chunked lower-triangular system.

    The reference densifies `x` before the launch, so the same thing happens
    here: the kernel reads it as a contiguous block.

    ``layout='tnd'`` is refused rather than forwarded.  Measured on 910B3 with
    the OPP in this tree: the kernel kills the process for that spelling, with
    and without cu_seqlens, so letting it through would turn an illegal input
    into a crash on the stable path -- exactly the class of input the reference
    rejects in Python.

    ``ntd`` crashes the same way (re-measured: five of six shapes segfault, the
    sixth is rejected 161001), and it is deliberately left unguarded to stay
    behaviour-identical with the reference; refusing it is the operator owner's
    call.  See the inventory's known limits for the measurement.
    """

    layout = str(layout)
    if layout == "tnd":
        raise RuntimeError(
            "npu_solve_tri: layout='tnd' is refused because the operator "
            "crashes the process for that spelling on this OPP (verified on "
            "both the ctypes and the Stable-ABI path). Use layout='bsnd' or "
            "'bnsd'.")
    return _op("npu_solve_tri")(
        x.contiguous(), _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_solve_tri", "layout", layout),
        _current_stream_ptr(),
    )



