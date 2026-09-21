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
import sys


_LIB_ENV = "FLA_NPU_STABLE_LIB"
# Lowest torch whose stable runtime symbols the launcher was verified against.
# Keep in sync with STABLE_ABI_MIN_TORCH in scripts/build_wheel.py.
_MIN_TORCH = "2.7.1"
# KDA chunked forward 的 K/V 只交付两档且必须同档：K=V=64 或 K=V=128。
# 混合档（K=64/V=128 等）与其它取值都不支持，判据与 ctypes 参考
# （``_aclnn_ctypes._KDA_FWD_SUPPORTED_KV_DIMS``）及 aclnn L2 校验保持一致。
_KDA_FWD_SUPPORTED_KV_DIMS = frozenset({64, 128})
_loaded_path: str | None = None
# ctypes handle of the loaded launcher; see _launcher_lib.
_lib_handle = None
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
# Whether the loaded launcher hands its launches to torch_npu's task queue
# (see _enqueues_launch): that is what makes the non-flushing stream read legal.
_launcher_enqueues: bool | None = None
# None until the escape hatch has been read (see _nowait_allowed).
_nowait_ok: bool | None = None
# Values of FLA_NPU_STABLE_STREAM that mean "read the stream without draining
# the task queue" and "drain it first".  The older spellings stay accepted: the
# field harnesses and the released wheels set them, and both questions they
# encoded -- does the launcher queue the launch, may the read skip the drain --
# have one answer each in this design.
_NOWAIT_STREAM_VALUES = ("", "nowait", "launcher")
_BARRIER_STREAM_VALUES = ("accessor", "python", "python-tensor")
# The two stream accessors, resolved on first use (see _nowait_stream_fn and
# _barrier_stream_fn).
_nowait_stream = None
# torch.autograd.graph.increment_version, resolved on first use (see
# _bump_state): the hot wrappers bump a mutated state in their own frame.
_INCREMENT_VERSION = None

# The stable value conversions have no std::string support, so string enum
# arguments travel as int codes.  Every layout argument uses the same order --
# BSND, BNSD, TND, NTD -- which is what `stable/layout_math.h` assumes;
# tools/op_abi_parity.py checks these tables against the adapters' name tables.
_LAYOUT_CODES = {"BSND": 0, "BNSD": 1, "TND": 2, "NTD": 3}

_ENUM = {
    "npu_causal_conv1d_bwd": {"input_layout": _LAYOUT_CODES},
    "npu_chunk_fwd_o": {"output_layout": _LAYOUT_CODES},
    "npu_chunk_gated_delta_rule_fwd": {"layout": _LAYOUT_CODES},
    "npu_chunk_kda_bwd_intra": {"layout": {"BSND": 0, "BNSD": 1, "TND": 2}},
    "npu_chunk_kda_fwd": {"layout": _LAYOUT_CODES},
    "npu_chunk_kda_fwd_finalize": {"output_layout": _LAYOUT_CODES},
    "npu_chunk_kda_fwd_prepare": {"layout": _LAYOUT_CODES},
    # The recurrent KDA kernel only implements the two spellings the reference
    # accepts, so this op's table is the (BSND, TND) subset.
    "npu_recurrent_kda": {"layout": {"BSND": 0, "TND": 1}},
    "npu_chunk_gated_delta_rule_bwd": {"layout": _LAYOUT_CODES},
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


def _capability(name: str) -> bool:
    """Ask the loaded launcher a yes/no question, once.

    Asked through ctypes because the answer decides what every later call puts
    in its ``stream`` slot, and asked *of the library* so that this glue and the
    launcher cannot disagree about it.
    """

    try:
        # A wrapper may ask before it has touched an op handle
        # (`npu_recurrent_kda` builds its argument list first), so the library
        # has to be loaded before it can be interrogated.
        load()
        import ctypes

        lib = ctypes.CDLL(_loaded_path)
        probe = getattr(lib, name, None)
        if probe is None:
            return False
        probe.restype = ctypes.c_int32
        return bool(probe())
    except Exception:
        return False


def _enqueues_launch() -> bool:
    """Whether the loaded launcher hands its launches to torch_npu's queue.

    When it does, ordering comes from the queue itself, so the stream may be
    read through the non-flushing accessor -- which is the whole point: on a
    busy vLLM worker the flushing one waits for the model's pending launches to
    be submitted, and that is ~1 ms per call.
    """

    global _launcher_enqueues
    if _launcher_enqueues is None:
        _launcher_enqueues = _capability(
            "fla_npu_stable_queue_enqueue_available")
    return _launcher_enqueues


def _nowait_stream_fn():
    """torch_npu's non-flushing raw-stream accessor, or None when absent.

    ``_npu_getCurrentRawStream`` drains torch_npu's task queue before handing
    the stream back; this spelling skips that, and is therefore only usable
    when the launch itself is queue-ordered (see _enqueues_launch).
    """

    global _nowait_stream
    if _nowait_stream is None:
        try:
            torch_npu = _modules()[1]
            if not torch_npu:
                raise AttributeError("torch_npu is not importable")
            _nowait_stream = getattr(torch_npu._C,
                                     "_npu_getCurrentRawStreamNoWait")
        except Exception:
            _nowait_stream = False
    return _nowait_stream or None


def _launcher_lib():
    """The loaded launcher as a ctypes handle, or None when there is none.

    Cached: the readback below is only used by the multi-stream regression, but
    re-opening the library on every call would dwarf what it measures.
    """

    global _lib_handle
    if _lib_handle is None:
        try:
            import ctypes

            load()
            _lib_handle = ctypes.CDLL(_loaded_path)
        except Exception:
            _lib_handle = False
    return _lib_handle or None


def _barrier_stream_fn():
    """torch_npu's raw-stream accessor, or None when it is not reachable.

    This is the spelling that drains torch_npu's task queue before it hands the
    stream over -- the one a *direct* submission has to use, because it is what
    keeps the launch behind everything the host enqueued before it.
    """

    global _raw_stream_fn
    if _raw_stream_fn is None:
        try:
            torch_npu = _modules()[1]
            if not torch_npu:
                raise AttributeError("torch_npu is not importable")
            _raw_stream_fn = getattr(torch_npu._C, "_npu_getCurrentRawStream")
        except Exception:
            _raw_stream_fn = False  # look the slow way from now on
    return _raw_stream_fn or None


def _nowait_allowed() -> bool:
    """Whether the stream may be read without draining the task queue.

    Only a queue-ordered launch may skip that drain (see _current_stream_ptr).
    ``FLA_NPU_STABLE_STREAM=accessor`` (or the older ``python`` and
    ``python-tensor``) is the field escape hatch that turns the skip off:
    slower on a busy worker, but it puts the draining accessor back on the
    path.  Read once, because it decides what every later call passes in its
    ``stream`` slot.
    """

    global _nowait_ok
    if _nowait_ok is None:
        requested = (os.environ.get("FLA_NPU_STABLE_STREAM")
                     or "").strip().lower()
        if requested in _NOWAIT_STREAM_VALUES:
            _nowait_ok = True
        elif requested in _BARRIER_STREAM_VALUES:
            _nowait_ok = False
        else:
            print(f"fla_npu: unknown FLA_NPU_STABLE_STREAM={requested!r}; "
                  "reading the stream without draining the task queue",
                  file=sys.stderr)
            _nowait_ok = True
    return _nowait_ok


def _last_launch_stream() -> int | None:
    """The stream the launcher used for the last call on this thread.

    ``None`` when no launcher is loaded or it predates the readback symbol.  The
    value is thread-local on the C++ side, so it answers the only question that
    matters under vLLM: did *this* worker's call go to *this* worker's stream.
    """

    lib = _launcher_lib()
    if lib is None:
        return None
    try:
        readback = lib.fla_npu_stable_last_launch_stream
    except AttributeError:
        return None
    import ctypes

    readback.restype = ctypes.c_int64
    return int(readback())


def _current_stream_ptr() -> int:
    """The launcher's ``stream`` argument (never a cached stream pointer).

    Read on every call: a process-global stream pointer is what sent kernels to
    another thread's stream in the vLLM run.

    Which accessor is legal depends on how the launcher submits.  A queued
    launch is ordered by torch_npu's task queue, so its stream may come from the
    non-flushing accessor -- the pairing torch_npu documents for a queue
    dispatch, and the one its own inductor codegen uses.  The other accessor
    drains the queue first, which is what a direct submission needs and what
    costs ~1 ms per call on a busy vLLM worker, against ~2 us for the plain
    read.  The launcher, not this file, decides which one is used.
    """

    torch = _modules()[0]
    device = torch.npu.current_device()
    if _enqueues_launch() and _nowait_allowed():
        nowait = _nowait_stream_fn()
        if nowait is not None:
            return int(nowait(device))
    barrier = _barrier_stream_fn()
    if barrier is not None:
        return int(barrier(device))
    return int(torch.npu.current_stream().npu_stream)


# --- the in-place contract, applied in the wrapper's own frame --------------
#
# `fla_npu.ops.ascendc` used to add a second Python wrapper around the operators
# that write a state argument in place, to refuse a state that requires grad and
# to bump its version counter afterwards.  That wrapper is two frames, a
# resolved-mutation plan and a rebuilt tensor list on every call, and the host
# bench charges ~8us per call for it on a decode path that walks these wrappers
# some thirty times a step.  The four hot operators therefore apply the contract
# in their own body and the dispatch layer skips its wrapper for them --
# `_fla_npu_inplace_contract` is the marker it reads (see
# __init__._get_direct_op).  Which arguments are mutable is still declared once,
# in __init__.MUTATED_ARGUMENTS: that table is what the ctypes path and
# tests/.../regression_mutation_contract.py read.


def _increment_version():
    """The cheapest version-counter bump this torch exposes, resolved once.

    ``torch.autograd.graph.increment_version`` only normalises its argument and
    forwards to ``torch._C._increment_version``; the public entry point costs
    about twice as much per call (measured on 2.10 in the container), and the
    wrappers below run once per layer per decode step.  The public name stays as
    the fallback for a build that does not expose the private callable.
    """

    global _INCREMENT_VERSION
    if _INCREMENT_VERSION is None:
        torch = _modules()[0]
        _INCREMENT_VERSION = (getattr(torch._C, "_increment_version", None)
                              or torch.autograd.graph.increment_version)
    return _INCREMENT_VERSION


def _refuse_grad_state(name: str, state) -> None:
    """Raise the same error the generic mutation wrapper raises."""

    if state is not None and state.requires_grad:
        raise RuntimeError(
            f"{name} mutates state tensors in place. Mutable state tensors "
            "must not require gradients; use a functional state API for "
            "training.")


def _bump_state(state) -> None:
    """Bump a state argument's version counter after it was written in place."""

    if state is not None:
        _increment_version()((state,))


# --- hot-path globals -------------------------------------------------------
#
# The four operators a decode step walks (recurrent GDN/KDA, conv1d fn/update)
# are written without helper calls further down.  Inside the enqueue loop one
# Python call in the wrapper costs ~0.7us of host time -- several times what the
# same call costs in a tight loop, see probes/wrapper_ablation.py -- so every
# statement that can be resolved once is resolved once here.
#
# The stream slot is *not* cached here: `_current_stream_ptr` is asked on every
# call (it answers with the launcher's sentinel when the launcher resolves the
# stream itself), which is the property tests/stable_abi's stream interleaving
# regression spies on -- caching a stream pointer is what corrupted the vLLM run
# earlier.
#
# `_INC_VERSION` is the resolved version-counter bump callable.
_INC_VERSION = None


def _init_hot() -> None:
    """Resolve `_INC_VERSION`; `load` calls this so the hot bodies only read."""

    global _INC_VERSION
    if _INC_VERSION is None:
        _INC_VERSION = _increment_version()


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
    _init_hot()
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
        # The launcher resolves aoti_torch_* at load; a missing one otherwise
        # surfaces as a bare "undefined symbol" a long way from the cause.
        raise RuntimeError(
            f"fla_npu: cannot load the Stable-ABI launcher {path} against "
            f"torch {torch.__version__}. {_load_failure_hint(path, exc)} "
            f"Original error: {exc}") from exc
    _check_build_stamp(path)
    _loaded_path = path


def _load_failure_hint(path: str, exc: Exception) -> str:
    """Name the missing runtime symbol instead of blaming the torch floor.

    A launcher built against a torch newer than its declared floor fails with
    "undefined symbol: <aoti_torch_...>", which is a different problem from an
    old runtime: the fix is to rebuild, not to upgrade torch.
    """

    marker = "undefined symbol: "
    detail = str(exc)
    if marker in detail:
        symbol = detail.split(marker, 1)[1].split()[0].strip("'\"")
        return (
            f"The runtime symbol {symbol!r} does not exist in this torch, so "
            f"this launcher was built against a newer one than the "
            f">= {_MIN_TORCH} it declares; rebuild it with "
            f"`python csrc/build_stable.py --out {path} --no-debug-probe`.")
    return (
        f"It needs torch >= {_MIN_TORCH} (the aoti_torch_* runtime symbols it "
        f"resolves were added over 2.7.x).")


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
    """Return (raw backend stream ptr, stable Stream::id()) for comparison.

    Both values are -3 when the runtime has no stream shims (torch < 2.9), -1
    when no stream handle came back, and -2 when the shim refused to report an
    id.  -3 is informational: every operator still runs through the launcher.
    """

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

    # Hot path: a decode step walks this operator some thirty times, so it is
    # written without helper calls and reads the op cache directly (the first
    # call is what loads the library).  See the note above `_INC_VERSION`.
    if state is not None and state.requires_grad:
        _refuse_grad_state("npu_recurrent_gated_delta_rule", state)
    stream = _current_stream_ptr()
    op = _OP_CACHE.get("npu_recurrent_gated_delta_rule")
    if op is None:
        op = _op("npu_recurrent_gated_delta_rule")
    result = op(
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
        stream,
    )
    if state is not None:
        bump = _INC_VERSION
        if bump is None:
            bump = _increment_version()
        bump((state,))
    return result


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

    layout_code = _char_code("npu_recurrent_kda", "layout", layout)

    # The contract only covers the in-place form: with
    # `inplace_final_state=False` the caller's tensor is not written (the
    # recursion below runs the kernel on a scratch state instead), which is the
    # same split the dispatch layer's MUTATION_FLAGS entry declares.
    if (inplace_final_state and initial_state is not None
            and initial_state.requires_grad):
        _refuse_grad_state("npu_recurrent_kda", initial_state)
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
        import torch

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
    # Hot path, same shape as npu_recurrent_gated_delta_rule above.
    stream = _current_stream_ptr()
    op = _OP_CACHE.get("npu_recurrent_kda")
    if op is None:
        op = _op("npu_recurrent_kda")
    out, final_state = op(
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
        final_state = None
    elif inplace_final_state and final_state is None:
        # The launcher returns the inplace result implicitly (the kernel writes
        # the caller's tensor); the stable ABI cannot hand that handle back as a
        # second output without a double ownership release, so mirror ctypes
        # here, which also returns the caller's object.
        final_state = initial_state
    if initial_state is not None:
        bump = _INC_VERSION
        if bump is None:
            bump = _increment_version()
        bump((initial_state,))
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
                    transpose_state_layout=False, use_exp2=False,
                    output_layout="BNSD"):
    """Output of one chunked attention pass.

    ``g_gamma`` is part of the published signature and ignored, exactly as the
    ctypes reference ignores it.  ``chunk_size`` defaults to 64 and ``use_exp2``
    to False, matching the reference's defaults.
    """

    return _op("npu_chunk_fwd_o")(
        q, k, v, h, g,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        scale,
        64 if chunk_size is None else chunk_size,
        False if use_exp2 is None else bool(use_exp2),
        bool(transpose_state_layout),
        _char_code("npu_chunk_fwd_o", "output_layout", output_layout),
        _current_stream_ptr(),
    )


def npu_chunk_gdn_bwd_intra(q, k, v, g, beta, A, d_o, scale, chunk_size, *,
                            cu_seqlens=None, chunk_indices=None, use_exp2=True):
    """dq / dk / dv of one chunk."""

    return _op("npu_chunk_gdn_bwd_intra")(
        q, k, v, g, beta, A, d_o,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        scale, chunk_size, use_exp2, _current_stream_ptr(),
    )


# ---------------------------------------------------------------------------
# conv1d family
# ---------------------------------------------------------------------------
#
# One aclnn entry point, three published entry points; the run mode is baked
# into which adapter is called rather than travelling as an argument.  What
# stays here is the part the adapter cannot express: refusing the scheduling
# parameters the operator does not implement and translating the activation
# name.  The conv_state crosses the boundary as the descriptor's own view --
# strides and storage offset included -- and what the operator does with that
# view is the operator's business, not the adapter's.

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

    if (block_idx_first_scheduled_token is not None
            or block_idx_last_scheduled_token is not None
            or initial_state_idx is not None
            or num_computed_tokens is not None
            or metadata is not None):
        _reject_conv1d_scheduling(
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens, metadata=metadata)
    if block_size_to_align not in (0, None):
        raise NotImplementedError(
            "CausalConv1d block_size_to_align is not supported by the Ascend "
            "operator")
    # Hot path, same shape as the decode wrappers: resolve what can be resolved
    # once and call no helper on the way in.  See the note above `_INC_VERSION`.
    if conv_states is not None and conv_states.requires_grad:
        _refuse_grad_state("npu_causal_conv1d_fn", conv_states)
    code = _CONV1D_ACTIVATION_CODES.get(
        "none" if activation is None else str(activation))
    if code is None:
        code = _conv1d_activation_code(activation)
    stream = _current_stream_ptr()
    op = _OP_CACHE.get("npu_causal_conv1d_fn")
    if op is None:
        op = _op("npu_causal_conv1d_fn")
    result = op(
        x, weight, bias, conv_states,
        query_start_loc, cache_indices, has_initial_state,
        None if query_start_loc_cpu is None else _host_ints(
            query_start_loc_cpu),
        None if cache_indices_cpu is None else _host_ints(cache_indices_cpu),
        None if has_initial_state_cpu is None else _host_ints(
            has_initial_state_cpu),
        code,
        _PAD_SLOT_ID if pad_slot_id is None else pad_slot_id,
        _NULL_BLOCK_ID if null_block_id is None else null_block_id,
        head_num, stream,
    )
    if conv_states is not None:
        bump = _INC_VERSION
        if bump is None:
            bump = _increment_version()
        bump((conv_states,))
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

    if (block_idx_last_scheduled_token is not None
            or initial_state_idx is not None):
        _reject_conv1d_scheduling(
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx)
    if conv_state is not None and conv_state.requires_grad:
        _refuse_grad_state("npu_causal_conv1d_update", conv_state)
    code = _CONV1D_ACTIVATION_CODES.get(
        "none" if activation is None else str(activation))
    if code is None:
        code = _conv1d_activation_code(activation)
    stream = _current_stream_ptr()
    op = _OP_CACHE.get("npu_causal_conv1d_update")
    if op is None:
        op = _op("npu_causal_conv1d_update")
    result = op(
        x, conv_state, weight, bias, code,
        conv_state_indices, num_accepted_tokens, query_start_loc,
        max_query_len,
        _NULL_BLOCK_ID if null_block_id is None else null_block_id,
        None if conv_state_indices_cpu is None else _host_ints(
            conv_state_indices_cpu),
        None if num_accepted_tokens_cpu is None else _host_ints(
            num_accepted_tokens_cpu),
        None if query_start_loc_cpu is None else _host_ints(
            query_start_loc_cpu),
        out, stream,
    )
    if conv_state is not None:
        bump = _INC_VERSION
        if bump is None:
            bump = _increment_version()
        bump((conv_state,))
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
    result = _op("npu_causal_conv1d")(
        x, weight, bias, conv_states,
        _host_ints(query_start_loc), _host_ints(cache_indices),
        _host_ints(initial_state_mode), _host_ints(num_accepted_tokens),
        _CONV1D_ACTIVATION_CODES["silu" if activation_mode == 1 else "none"],
        pad_slot_id, run_mode, head_num, _current_stream_ptr(),
    )
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


def npu_chunk_fwd_h(k, w, u, *, g=None, gk=None, initial_state=None,
                    output_final_state=False, chunk_size=64,
                    save_new_value=True, cu_seqlens=None, chunk_indices=None,
                    use_exp2=False, state_v_first=False):
    """Chunk-local states h, the recomputed v, and optionally the final state."""

    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    return _op("npu_chunk_fwd_h")(
        k, w, u, g, gk, initial_state,
        output_final_state, chunk_size, save_new_value,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        use_exp2, state_v_first, _current_stream_ptr(),
    )


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
    """dh / dh0 / dv, where dh0 is only produced when h0 was supplied."""

    return _op("npu_chunk_gated_delta_rule_bwd_dhu")(
        q, k, w, d_o, dv, g, gK, h0, dht,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        scale, chunk_size, use_exp2, transpose_state_layout,
        _current_stream_ptr(),
    )


# ---------------------------------------------------------------------------
# kda backward recompute
# ---------------------------------------------------------------------------


def npu_chunk_kda_bwd_recompute(q, k, v, g, beta, a, chunk_size, *,
                                A_log=None, dt_bias=None, cu_seqlens=None,
                                chunk_indices=None, use_gate_in_kernel=True,
                                use_exp2=True, lower_bound=-5.0):
    """Recompute the KDA saved tensors (`gk`, `w`, `u`, `qg`, `kg`).

    The declared order starts with the gate cumsum because that is the tensor
    the caller almost always wants; aclnn takes the recomputed tensors first,
    so the adapter reorders the results before returning them.
    """

    return _op("npu_chunk_kda_bwd_recompute")(
        q, k, v, g, beta, a, A_log, dt_bias,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        chunk_size,
        True if use_exp2 is None else bool(use_exp2),
        -5.0 if lower_bound is None else float(lower_bound),
        bool(use_gate_in_kernel),
        _current_stream_ptr(),
    )


def npu_chunk_kda_fwd(q, k, v, g, beta, scale, chunk_size=64, *,
                      layout="BSND", initial_state=None,
                      output_final_state=False, cu_seqlens=None,
                      chunk_indices=None, safe_gate=False, lower_bound=None,
                      use_gate_in_kernel=False, A_log=None, dt_bias=None,
                      disable_recompute=False,
                      return_intermediate_states=False, state_v_first=False,
                      epsilon=1e-6, use_qk_l2norm_in_kernel=False,
                      use_beta_sigmoid_in_kernel=False,
                      allow_neg_eigval=False, use_exp2=True,
                      # 反向 L2 norm 保存值出口：传入自己的张量才导出，
                      # 不传（None）就是空槽，接口不报错、老行为不变。
                      q_hat_out=None, k_hat_out=None, q_rstd_out=None,
                      k_rstd_out=None, beta_eff_out=None):
    """KDA chunked forward, returning the saved tensors the backward needs.

    ``disable_recompute`` is what makes `w`/`u`/`qg`/`kg`/`v_new` real outputs
    rather than null handles, and `chunk_indices` defaults to the canonical
    sequence-major list the kernel expects.  The trailing value is the caller's
    own ``initial_state``, which the operator updates in place (the reference
    API returns it the same way).

    ``epsilon`` / ``use_qk_l2norm_in_kernel`` / ``use_beta_sigmoid_in_kernel``
    / ``allow_neg_eigval`` / ``use_exp2`` 是三算子组合入口才有的归一化与 gate
    语义开关，默认值即历史语义；取非默认值时调用必须落在组合场景
    （bfloat16 q/k/v、K=V=128、chunk_size=64），否则按参考实现拒绝。
    """

    epsilon = 1e-6 if epsilon is None else float(epsilon)
    if not epsilon > 0.0:
        raise RuntimeError(
            "npu_chunk_kda_fwd: epsilon must be a positive finite number.")
    import torch

    # K/V 档位：只支持 K=V=64 与 K=V=128，混合档与其它取值都拒绝。单列一条
    # 判据而不是交给 L2，是为了让两条后端在 Python 侧就给出同一句话。
    key_dim = int(k.shape[-1])
    value_dim = int(v.shape[-1])
    if key_dim != value_dim or key_dim not in _KDA_FWD_SUPPORTED_KV_DIMS:
        raise RuntimeError(
            "npu_chunk_kda_fwd: K/V must both be 64 or both be 128 "
            "(mixed K/V and other dims are not supported), "
            f"but got K={key_dim}, V={value_dim}.")
    use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)
    use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    allow_neg_eigval = bool(allow_neg_eigval)
    use_exp2 = True if use_exp2 is None else bool(use_exp2)
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise RuntimeError(
            "npu_chunk_kda_fwd: allow_neg_eigval=True requires "
            "use_beta_sigmoid_in_kernel=True.")
    if (use_qk_l2norm_in_kernel or use_beta_sigmoid_in_kernel
            or allow_neg_eigval or not use_exp2):
        if (q.dtype != torch.bfloat16 or key_dim != 128 or value_dim != 128
                or int(chunk_size) != 64):
            raise RuntimeError(
                "npu_chunk_kda_fwd: non-default gate/L2norm switches require "
                "the three-stage scenario (bfloat16 q/k/v, K=V=128, "
                "chunk_size=64).")

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
        epsilon, use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel,
        allow_neg_eigval, use_exp2,
        bool(output_final_state), bool(disable_recompute),
        bool(return_intermediate_states),
        q_hat_out, k_hat_out, q_rstd_out, k_rstd_out, beta_eff_out,
        _current_stream_ptr(),
    )
    return (*result, initial_state)


def npu_chunk_kda_fwd_prepare(q, k, v, g, beta, scale, chunk_size=64, *,
                              layout="BSND", cu_seqlens=None, chunk_indices=None,
                              safe_gate=False, lower_bound=None,
                              use_gate_in_kernel=False, A_log=None, dt_bias=None,
                              epsilon=1e-6, use_qk_l2norm_in_kernel=False,
                              use_beta_sigmoid_in_kernel=False,
                              allow_neg_eigval=False, use_exp2=True,
                              backward_mode="save"):
    """三算子组合里 Prepare 段的公共入口（13 个输出槽全部可选）。

    反向需要的 L2 norm 保存值（q_hat/k_hat/q_rstd/k_rstd/beta_eff）按需在这里
    导出；不传输出槽时接口只做校验并返回 13 个 None，老路径不受影响。
    """
    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    # 与 ctypes 后端同一口径：L2 的档位契约要求 gk/aqk/w/u/kg/qg_scaled 六个槽
    # 必选，aux 只在 recompute/save 档出现。调用方省略的前置槽在这里补齐，
    # 这样"不给/只给部分输出槽"都不会报错。
    import torch

    rank3 = layout in ("TND", "NTD")
    sequence_major = layout in ("BSND", "TND")
    if rank3:
        seq_len, qk_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[1], v.shape[2]
        batch = 1
    elif sequence_major:
        batch, seq_len, qk_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
    else:
        batch, qk_heads, seq_len, key_dim = q.shape
        value_heads, value_dim = v.shape[1], v.shape[3]
    key_shape = ((value_heads, seq_len, key_dim) if rank3
                 else (batch, value_heads, seq_len, key_dim))
    value_shape = ((value_heads, seq_len, value_dim) if rank3
                   else (batch, value_heads, seq_len, value_dim))
    matrix_shape = ((value_heads, seq_len, chunk_size) if rank3
                    else (batch, value_heads, seq_len, chunk_size))
    qk_head_shape = ((qk_heads, seq_len, key_dim) if rank3
                     else (batch, qk_heads, seq_len, key_dim))
    qk_scalar_shape = (qk_heads, seq_len) if rank3 else (batch, qk_heads, seq_len)
    value_scalar_shape = (value_heads, seq_len) if rank3 \
        else (batch, value_heads, seq_len)

    def _alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=q.device)

    # 档位 → 需要产出的槽位（与算子文档的 backward_mode 表一致）；未选中的槽
    # 直接传 None（空槽），不参与公开 GM 写回。
    shapes = {
        "gk": (key_shape, torch.float32),
        "Aqk": (matrix_shape, q.dtype),
        "Akk": (matrix_shape, q.dtype),
        "w": (key_shape, q.dtype),
        "u": (value_shape, q.dtype),
        "qg": (key_shape, q.dtype),
        "kg": (key_shape, q.dtype),
        "qg_scaled": (key_shape, q.dtype),
        "q_hat": (qk_head_shape, q.dtype),
        "k_hat": (qk_head_shape, q.dtype),
        "q_rstd": (qk_scalar_shape, torch.float32),
        "k_rstd": (qk_scalar_shape, torch.float32),
        "beta_eff": (value_scalar_shape, torch.float32),
    }
    mode_slots = {
        "none": ("gk", "Aqk", "w", "u", "kg", "qg_scaled"),
        "forward": ("gk", "Aqk", "Akk", "w", "u", "kg", "qg_scaled"),
        "recompute": ("gk", "Aqk", "Akk", "w", "u", "kg", "qg_scaled",
                      "q_hat", "k_hat", "q_rstd", "k_rstd", "beta_eff"),
        "save": tuple(shapes),
    }
    backward_mode = str(backward_mode).lower()
    if backward_mode not in mode_slots:
        raise RuntimeError(
            "npu_chunk_kda_fwd_prepare: backward_mode must be none/forward/recompute/save.")
    selected = set(mode_slots[backward_mode])
    slots = {
        name: (_alloc(shape, dtype) if name in selected else None)
        for name, (shape, dtype) in shapes.items()
    }
    gk_out = slots["gk"]
    aqk_out = slots["Aqk"]
    akk_out = slots["Akk"]
    w_out = slots["w"]
    u_out = slots["u"]
    qg_out = slots["qg"]
    kg_out = slots["kg"]
    qg_scaled_out = slots["qg_scaled"]
    q_hat_out = slots["q_hat"]
    k_hat_out = slots["k_hat"]
    q_rstd_out = slots["q_rstd"]
    k_rstd_out = slots["k_rstd"]
    beta_eff_out = slots["beta_eff"]
    return _op("npu_chunk_kda_fwd_prepare")(
        q, k, v, g, beta, A_log, dt_bias,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_chunk_kda_fwd_prepare", "layout", layout),
        float(scale), chunk_size,
        1e-6 if epsilon is None else float(epsilon),
        bool(use_qk_l2norm_in_kernel), bool(use_gate_in_kernel),
        bool(use_beta_sigmoid_in_kernel), bool(allow_neg_eigval),
        bool(safe_gate),
        -5.0 if lower_bound is None else float(lower_bound),
        True if use_exp2 is None else bool(use_exp2),
        gk_out, aqk_out, akk_out, w_out, u_out, qg_out, kg_out,
        qg_scaled_out, q_hat_out, k_hat_out, q_rstd_out, k_rstd_out,
        beta_eff_out,
        _current_stream_ptr(),
    )


def npu_chunk_kda_fwd_finalize(qg_scaled, aqk, v_new, h, *,
                               output_layout="BSND", state_v_first=False,
                               cu_seqlens=None, chunk_indices=None):
    """KDA chunked forward 的 finalize 段（唯一的公开输出是 ``attn_out``）。

    ``output_layout`` 只描述输出的拼写：BSND/TND 是 sequence-major，
    BNSD/NTD 是 head-major；四个输入始终是 head-major（packed 拼写没有 batch
    轴）。``cu_seqlens`` 传了就按 canonical 顺序生成 ``chunk_indices``。
    """

    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, 64)
    return _op("npu_chunk_kda_fwd_finalize")(
        qg_scaled, aqk, v_new, h,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_chunk_kda_fwd_finalize", "output_layout",
                   output_layout),
        bool(state_v_first),
        _current_stream_ptr(),
    )


def npu_chunk_gated_delta_rule_fwd(q, k, v, g, beta, *, initial_state=None,
                                   output_final_state=False, chunk_size=64,
                                   cu_seqlens=None, chunk_indices=None,
                                   scale=None, use_exp2=False,
                                   use_qk_l2norm_in_kernel=False,
                                   use_gate_in_kernel=False,
                                   use_beta_sigmoid_in_kernel=False,
                                   allow_neg_eigval=False,
                                   disable_recompute=True,
                                   return_intermediate_states=False,
                                   state_v_first=False, a_log=None,
                                   dt_bias=None, layout="BNSD"):
    """Fused GDN forward: `o` plus the intermediates the backward consumes.

    The tuple is fixed at ten slots -- `(o, final_state, g_cumsum, A, beta_eff,
    h, q_hat, k_hat, q_rstd, k_rstd)` -- carrying `None` in the slots the flags
    switch off, exactly like the reference.  `q_hat`/`k_hat` are the caller's
    own q/k when the kernel is not asked to normalise them.
    """

    # Gate-in-kernel is unsupported by this kernel build, and the reference
    # refuses the flag together with a_log/dt_bias before the launch.  Dropping
    # them here instead would turn a rejected call into a silently different
    # one, so the refusal is repeated.
    if use_gate_in_kernel:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd: use_gate_in_kernel=True is not "
            "supported.")
    if a_log is not None or dt_bias is not None:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd: a_log and dt_bias must be None "
            "while gate-in-kernel is unsupported.")
    if scale is None:
        scale = float(k.shape[-1]) ** -0.5
    return _op("npu_chunk_gated_delta_rule_fwd")(
        q, k, v, g, beta, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_chunk_gated_delta_rule_fwd", "layout", layout),
        float(scale), chunk_size, bool(use_exp2),
        bool(use_qk_l2norm_in_kernel),
        bool(use_beta_sigmoid_in_kernel), bool(allow_neg_eigval),
        bool(disable_recompute), bool(output_final_state),
        bool(return_intermediate_states), bool(state_v_first),
        _current_stream_ptr(),
    )


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


def npu_chunk_gated_delta_rule_fwd_prepare(
        q, k, v, g, beta, chunk_size=64, *, use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False, use_exp2=False, a_log=None, dt_bias=None,
        cu_seqlens=None, chunk_indices=None, output_a=True):
    """Phase-6 prefill preparation: normalized q/k, beta and the WY tensors.

    `beta_out` has to be a real tensor either way, so when the kernel is not
    asked to sigmoid it the reference returns a float32 copy of `beta` -- the
    same conversion happens here.
    """

    # Gate-in-kernel is the only spelling left that the reference refuses; with
    # ``use_qk_l2norm_in_kernel`` off the hats are the caller's own q/k and both
    # rstd slots stay null, which the adapter below already handles.
    if use_gate_in_kernel:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_fwd_prepare: use_gate_in_kernel "
            "currently only supports False.")
    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    (q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, a) = _op(
        "npu_chunk_gated_delta_rule_fwd_prepare")(
            q, k, v, g, beta,
            a_log if use_gate_in_kernel else None,
            dt_bias if use_gate_in_kernel else None,
            _host_ints(cu_seqlens), _host_ints(chunk_indices),
            chunk_size, bool(use_qk_l2norm_in_kernel),
            bool(use_gate_in_kernel), bool(use_beta_sigmoid_in_kernel),
            bool(allow_neg_eigval), bool(use_exp2), bool(output_a),
            _current_stream_ptr())
    if beta_out is None:
        import torch

        beta_out = beta.to(dtype=torch.float32)
    return q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, a


def npu_chunk_gated_delta_rule_bwd_finalize(
        q, k, v, v_new, do, du, g, beta, h, dh, a, *, q_rstd=None,
        k_rstd=None, beta_raw=None, cu_seqlens=None, chunk_indices=None,
        scale=None, chunk_size=64, use_qk_l2_norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, use_gate_in_kernel=False,
        state_v_first=False, use_exp2=True):
    """The Ascend950-only backward finalize, returning dq/dk/dv/dbeta/dg.

    The kernel exists only for Ascend950, and calling it anywhere else has no
    defined result, so the device is checked here rather than letting the
    launch decide.
    """

    import torch

    device_index = q.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    device_name = torch.npu.get_device_name(device_index)
    if not device_name.startswith("Ascend950"):
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_bwd_finalize only supports Ascend "
            f"950, got {device_name}.")
    if use_gate_in_kernel:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_bwd_finalize: use_gate_in_kernel only "
            "supports False.")
    if scale is None:
        scale = 1.0 / (128.0 ** 0.5)
    return _op("npu_chunk_gated_delta_rule_bwd_finalize")(
        q, k, v, v_new, do, du, g, beta, h, dh, a,
        q_rstd, k_rstd, beta_raw,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        float(scale), chunk_size, bool(use_qk_l2_norm_in_kernel),
        bool(use_beta_sigmoid_in_kernel), bool(use_gate_in_kernel),
        bool(state_v_first), bool(use_exp2),
        _current_stream_ptr(),
    )


def npu_chunk_gated_delta_rule_bwd(
        q, k, v, g, beta, A, d_o, scale, *, chunk_size=64, cu_seqlens=None,
        chunk_indices=None, initial_state=None, dht=None, q_rstd=None,
        k_rstd=None, beta_raw=None, use_exp2=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
        return_intermediate_states=False, state_v_first=False, a_log=None,
        dt_bias=None, layout="BNSD"):
    """The composite GDN backward, in one aclnn call.

    Returns `(dq, dk, dv, d_beta, d_g, dh0, d_a_log, d_dt_bias)`; the last two
    are reserved slots the operator does not fill yet and come back as None,
    exactly like the reference.  `return_intermediate_states` is accepted for
    ABI compatibility and has no effect, again like the reference -- and so is
    `allow_neg_eigval`, which the reference keeps in its public signature for
    the same reason without forwarding it to the kernel.
    """

    # Gate-in-kernel is the one spelling the composite backward still does not
    # implement, and it receives the value, so the reference's refusal has to
    # be repeated here.
    if use_gate_in_kernel:
        raise RuntimeError(
            "npu_chunk_gated_delta_rule_bwd: use_gate_in_kernel=True is not "
            "supported.")
    if cu_seqlens and not chunk_indices:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    return _op("npu_chunk_gated_delta_rule_bwd")(
        q, k, v, g, beta, A, d_o, initial_state, dht, q_rstd, k_rstd,
        beta_raw, a_log, dt_bias,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_chunk_gated_delta_rule_bwd", "layout", layout),
        float(scale), chunk_size, bool(use_exp2), bool(use_gate_in_kernel),
        bool(use_qk_l2norm_in_kernel), bool(use_beta_sigmoid_in_kernel),
        bool(state_v_first),
        _current_stream_ptr(),
    )


def _kda_bwd_single_launch(q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h,
                           d_o, raw_g, A_log, dt_bias, scale, chunk_size,
                           safe_gate, lower_bound, use_gate_in_kernel,
                           cu_seqlens=None, chunk_indices=None):
    """One fused call on the tensors it is given, dense or packed.

    Packed (rank-3) inputs must carry the varlen metadata: the aclnn entry
    rejects rank-3 `q` without `cu_seqlens`/`chunk_indices`, and the fused
    tiling derives the chunk count from them.  Dense inputs pass both as None.
    """

    return _op("npu_chunk_kda_bwd")(
        q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, raw_g, A_log,
        dt_bias,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        float(scale), int(chunk_size), bool(safe_gate),
        bool(use_gate_in_kernel), float(lower_bound),
        True,  # disable_recompute is the only supported spelling
        True,  # use_exp2 likewise
        False,  # state_v_first likewise
        False, None, None,  # legacy implementation, no norm backward
        _current_stream_ptr(),
    )


def _kda_bwd_optimized_launch(args):
    """Share input policy with the reference, but launch V2 through Stable ABI."""
    from ._kda_policy import _prepare_kda_bwd_optimized

    bias_shape = None if args["dt_bias"] is None else args["dt_bias"].shape
    a = _prepare_kda_bwd_optimized(args)
    outputs = _op("npu_chunk_kda_bwd")(
        a["q"], a["k"], a["v"], a["beta"], a["gk"], a["Aqk"], a["Akk"],
        a["w"], a["qg"], a["kg"], a["v_new"], a["h"], a["d_o"],
        a["raw_g"], a["A_log"], a["dt_bias"],
        _host_ints(a["cu_seqlens"]), _host_ints(a["chunk_indices"]),
        float(a["scale"]), int(a["chunk_size"]), a["safe_gate"],
        a["use_gate_in_kernel"], a["lower_bound"], a["disable_recompute"],
        a["use_exp2"], a["state_v_first"], True, a["q_rstd"], a["k_rstd"],
        _current_stream_ptr(),
    )
    if bias_shape is not None:
        outputs = (*outputs[:-1], outputs[-1].view(bias_shape))
    return outputs


def npu_chunk_kda_bwd(q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o,
                      scale, *, raw_g=None, A_log=None, dt_bias=None,
                      initial_state=None, dht=None, cu_seqlens=None,
                      chunk_indices=None, chunk_size=64, safe_gate=True,
                      lower_bound=-5.0, use_gate_in_kernel=False,
                      disable_recompute=True, use_exp2=True,
                      state_v_first=False, implementation="auto",
                      q_rstd=None, k_rstd=None):
    """Fused KDA backward, returning `(dq, dk, dv, db, dg, dh0, dA, dbias)`.

    Three shape-specific workarounds sit on top of the single launch, and they
    are the reason this wrapper is longer than the others:

    * a packed sequence whose last chunk is short (or a packed V=256 call on
      Atlas A2) is split into independent dense calls, because the fused
      pipeline produces stale or non-finite values for those;
    * a single packed sequence with a short tail is instead padded inside its
      last chunk and the token gradients sliced back;
    * on A2 an odd head count gets a duplicated partner head, because the
      Intra pipeline processes heads in pairs.

    All three are the reference's own strategy; the only difference here is
    that the dense call goes to the launcher instead of to ctypes.
    """

    import torch

    from ._runtime import optional_bool as _optional_bool
    from ._kda_policy import _select_kda_bwd_optimized

    if _select_kda_bwd_optimized(
            implementation, q_rstd, k_rstd,
            _optional_bool(disable_recompute, True)):
        return _kda_bwd_optimized_launch(locals())

    chunk_size = int(chunk_size)
    # The flags below are *reserved but not implemented* by this operator.  The
    # reference refuses them in Python, and the adapter hard-codes the supported
    # values, so accepting them here would silently ignore the caller's request
    # instead of reporting it.
    if not disable_recompute:
        raise RuntimeError(
            "npu_chunk_kda_bwd: disable_recompute=false is reserved but not "
            "supported.")
    if not use_exp2:
        raise RuntimeError(
            "npu_chunk_kda_bwd: use_exp2=false is reserved but not supported.")
    if state_v_first:
        raise RuntimeError(
            "npu_chunk_kda_bwd: state_v_first=true is reserved but not "
            "supported.")
    if initial_state is not None or dht is not None:
        raise RuntimeError(
            "npu_chunk_kda_bwd: initial_state and dht are not supported by the "
            "current fused backward.")
    if not safe_gate:
        raise RuntimeError(
            "npu_chunk_kda_bwd: safe_gate=False is reserved but not supported.")
    cu = None if cu_seqlens is None else tuple(int(x) for x in cu_seqlens)
    is_varlen = cu is not None
    if is_varlen:
        heads, seqlen = int(q.shape[0]), int(q.shape[1])
        value_dim = int(v.shape[2])
    else:
        batch, heads, seqlen, _ = q.shape
        value_dim = int(v.shape[3])
    indices = (None if chunk_indices is None
               else tuple(int(x) for x in chunk_indices))
    if indices is None and cu is not None:
        indices = _canonical_chunk_indices(cu, chunk_size)
    use_gate_in_kernel = bool(use_gate_in_kernel)
    safe_gate = bool(safe_gate)
    lower_bound = -5.0 if lower_bound is None else float(lower_bound)
    scale = float(scale)
    launch = _kda_bwd_single_launch

    use_dense_varlen_fallback = is_varlen and value_dim == 256
    has_varlen_tail = is_varlen and any(
        (end - begin) % chunk_size != 0 for begin, end in zip(cu, cu[1:]))
    is_a2_device = False
    if use_dense_varlen_fallback or heads % 2 != 0 or has_varlen_tail:
        device_index = q.device.index
        if device_index is None:
            device_index = torch.npu.current_device()
        device_name = str(torch.npu.get_device_name(device_index))
        is_a2_device = device_name.startswith("Ascend910B")

    # 短尾 packed 序列统一拆成逐序列 dense 调用：fused packed 流水线对其会产生
    # 陈旧/非有限值，拆开后每条序列走已验证的 dense 路径（A2 上 V=256 同理）。
    if (is_a2_device and use_dense_varlen_fallback) or has_varlen_tail:
        sequence_results = []
        chunk_begin = 0
        for token_begin, token_end in zip(cu, cu[1:]):
            sequence_length = token_end - token_begin
            if sequence_length == 0:
                continue
            sequence_chunks = (sequence_length + chunk_size - 1) // chunk_size

            def dense_slice(tensor):
                if tensor is None:
                    return None
                return tensor.narrow(1, token_begin, sequence_length) \
                    .unsqueeze(0).contiguous()

            sequence_results.append(npu_chunk_kda_bwd(
                dense_slice(q), dense_slice(k), dense_slice(v),
                dense_slice(beta), dense_slice(gk), dense_slice(Aqk),
                dense_slice(Akk), dense_slice(w), dense_slice(qg),
                dense_slice(kg), dense_slice(v_new),
                h.narrow(0, chunk_begin, sequence_chunks).unsqueeze(0)
                 .contiguous(),
                dense_slice(d_o), scale,
                raw_g=dense_slice(raw_g), A_log=A_log, dt_bias=dt_bias,
                initial_state=None, dht=None, cu_seqlens=None,
                chunk_indices=None, chunk_size=chunk_size, safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                disable_recompute=True, use_exp2=True, state_v_first=False))
            chunk_begin += sequence_chunks

        # Token gradients are concatenated along the token axis; the scalar
        # gradients (dA, dbias) are summed, because each sequence contributed
        # an independent reduction.
        restored = []
        for output_index in range(8):
            values = [result[output_index] for result in sequence_results]
            if values[0] is None:
                restored.append(None)
            elif output_index < 5:
                restored.append(torch.cat(
                    [value.squeeze(0) for value in values], dim=1).contiguous())
            else:
                total = values[0]
                for value in values[1:]:
                    total = total + value
                restored.append(total)
        return tuple(restored)

    # A single packed sequence with a short tail is padded inside its last
    # chunk: zero-gradient rows change neither the math nor the chunk layout,
    # and the token gradients are sliced back afterwards.
    original_seqlen = seqlen
    original_heads = heads
    padded_tail = seqlen % chunk_size != 0 and (cu is None or len(cu) == 2)
    token_dim = 1 if is_varlen else 2
    if padded_tail:
        padded_seqlen = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size
        pad_rows = padded_seqlen - seqlen

        def pad_rows_of(tensor, repeat_last=False):
            if tensor is None:
                return None
            pad_shape = list(tensor.shape)
            pad_shape[token_dim] = pad_rows
            if repeat_last:
                tail = tensor.narrow(token_dim, seqlen - 1, 1) \
                    .expand(*pad_shape).clone()
            else:
                tail = tensor.new_zeros(pad_shape)
            return torch.cat((tensor, tail), dim=token_dim).contiguous()

        q, k, v = (pad_rows_of(tensor) for tensor in (q, k, v))
        beta = pad_rows_of(beta)
        # `gk` is cumulative: repeating its last value keeps every padded
        # contraction finite while contributing no gradient.
        gk = pad_rows_of(gk, repeat_last=True)
        Aqk, Akk = (pad_rows_of(tensor) for tensor in (Aqk, Akk))
        w, qg, kg, v_new, d_o = (pad_rows_of(tensor)
                                 for tensor in (w, qg, kg, v_new, d_o))
        raw_g = pad_rows_of(raw_g)
        seqlen = padded_seqlen
        cu = None if cu is None else (0, padded_seqlen)
        indices = _canonical_chunk_indices(cu, chunk_size)

    # A2's fused Intra pipeline processes heads in pairs; a lone final head can
    # keep a stale correction from the previous launch.
    padded_head = bool(heads % 2 != 0 and is_a2_device)
    if padded_head:
        head_dim = 0 if is_varlen else 1

        def duplicate_head(tensor, dim):
            if tensor is None:
                return None
            return torch.cat(
                (tensor, tensor.narrow(dim, heads - 1, 1).clone()),
                dim=dim).contiguous()

        (q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, d_o, raw_g) = (
            duplicate_head(tensor, head_dim)
            for tensor in (q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, d_o,
                           raw_g))
        h = duplicate_head(h, 1 if is_varlen else 2)
        A_log = duplicate_head(A_log, 0)
        dt_bias = duplicate_head(dt_bias, 0)
        heads += 1

    result = launch(q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o,
                    raw_g, A_log, dt_bias, scale, chunk_size, safe_gate,
                    lower_bound, use_gate_in_kernel,
                    cu_seqlens=cu, chunk_indices=indices)
    restored = []
    for index, value in enumerate(result):
        if value is None:
            restored.append(None)
            continue
        if padded_tail and index < 5:
            value = value.narrow(token_dim, 0, original_seqlen)
        if padded_head:
            if index < 5:
                value = value.narrow(0 if is_varlen else 1, 0, original_heads)
            elif index in (6, 7):
                value = value.narrow(0, 0, original_heads)
        restored.append(value.contiguous()
                        if padded_tail or padded_head else value)
    return tuple(restored)


# ---------------------------------------------------------------------------
# Which wrappers apply the in-place contract themselves
# ---------------------------------------------------------------------------
# These four are the ones a decode step calls with a state argument.  Each
# refuses a grad-requiring state and bumps the version counter inside its own
# frame, which is what __init__._get_direct_op reads this marker for: adding its
# generic mutation wrapper on top would bump the counter twice for one call, so
# exactly one side declares the contract for a given backend.
for _contract_op in (
        "npu_causal_conv1d_fn",
        "npu_causal_conv1d_update",
        "npu_recurrent_gated_delta_rule",
        "npu_recurrent_kda"):
    globals()[_contract_op]._fla_npu_inplace_contract = True
del _contract_op
