#!/usr/bin/env python3
"""Multi-stream / multi-thread regression for the stable launcher (T4).

vLLM runs several worker threads, each on its own NPU stream, and one decode
step alternates the two operators this launcher serves: the recurrent GDR and
the causal-conv1d update.  A launcher that caches the current stream in a
process-global passes every single-stream test and breaks exactly this mix --
the kernel is enqueued on whichever stream another thread happens to own, so the
ordering between the two operators is lost.  That is the failure mode that took
down the 512-token request.

Why the obvious probe is not used
---------------------------------

Recording an event pair on the calling stream and requiring a non-zero elapsed
time does *not* work on this hardware.  Measured on 910B3:

    empty event pair, fresh stream : 0.07 - 0.18 ms
    op pair on the correct stream  : ~0.15 ms
    op forced onto another stream  : 0.145 ms

The three are indistinguishable, so `elapsed > 0.01` holds for any input -- the
timing-based control this test used to carry was vacuous for that reason.

What is checked instead
-----------------------

Two values per operator call, both read from the calling thread:

* what the wrapper passed (``_stable._current_stream_ptr()``) -- either the
  launcher-resolve sentinel or this thread's own raw stream, never another
  thread's;
* what the launcher launched on (``_stable._last_resolved_stream()``) -- it has
  to be this thread's own raw stream.  The launcher resolves the stream in C++
  now, so this readback of a per-thread value it stores on every call is the
  only place the decision is observable.

``negative_control`` hands the wrapper a foreign stream on purpose and requires
the readback to report *that* stream.  That is what makes the check evidence
rather than decoration: a readback that simply echoed the calling thread would
fail it.

usage::

    FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
        python tests/stable_abi/test_stable_stream_interleaving.py [--threads 8] [--rounds 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

STABLE_LIB = os.environ.get("FLA_NPU_STABLE_LIB", "")

# This file is collected by name (`test_*.py`), so a machine without the NPU
# stack must end up with a *skip*, not a collection error: the checks below need
# a device and a launcher, and reporting that as a failure would be wrong.
_SKIP_REASON: str | None
try:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    from fla_npu.ops.ascendc import _stable  # noqa: E402
except Exception as exc:  # noqa: BLE001 - the reason is reported, not raised
    _SKIP_REASON = f"{type(exc).__name__}: {exc}"
else:
    _SKIP_REASON = None

if _SKIP_REASON is not None:
    try:
        import pytest
    except ImportError:
        pass
    else:
        pytest.skip(f"needs the NPU stack ({_SKIP_REASON})",
                    allow_module_level=True)

# Per-thread recording: a worker installs a list, every stream read made by that
# thread appends to it, and the worker reads it back after its calls.
_RECORD = threading.local()


def raw_stream_ptr() -> int:
    """The raw pointer of the calling thread's current stream."""

    return int(torch_npu._C._npu_getCurrentRawStream(0))


def make_templates(batch: int, nk: int = 8, nv: int = 16, dim: int = 128):
    """Read-only inputs, built once so every thread sees identical values."""

    torch.manual_seed(20260914)
    normalize = lambda t: torch.nn.functional.normalize(t, p=2, dim=-1)  # noqa: E731
    return dict(
        query=normalize(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16),
        key=normalize(torch.randn(batch, nk, dim, device="npu")).to(torch.bfloat16),
        value=torch.randn(batch, nv, dim, dtype=torch.bfloat16, device="npu"),
        beta=torch.rand(batch, nv, dtype=torch.bfloat16, device="npu"),
        g=torch.rand(batch, nv, dtype=torch.float32, device="npu"),
        scale=dim ** -0.5,
        actual_seq_lengths=torch.tensor([0] + [1] * batch, dtype=torch.int32,
                                        device="npu"),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device="npu"),
        x=(torch.arange(2 * 16, dtype=torch.float32) + 1.0).reshape(2, 16).to(
            torch.bfloat16).npu(),
        weight=(torch.arange(4 * 16, dtype=torch.float32) + 101.0).reshape(
            4, 16).to(torch.bfloat16).npu(),
        bias=(torch.arange(16, dtype=torch.float32) + 201.0).to(
            torch.bfloat16).npu(),
        # Block id 0 is the null block: a sequence that addresses it is skipped
        # and its output row is never written, so the ids start at 1.
        conv_indices=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
        batch=batch,
        nv=nv,
        dim=dim,
    )


def make_paged_state(batch: int, nv: int, dim: int, gap: int = 16384,
                     offset: int = 12288):
    """A non-contiguous state, the spelling the original bug reproduced with."""

    block_stride = nv * dim * dim + gap
    raw = torch.empty((batch + 1) * block_stride * 4, dtype=torch.int8,
                      device="npu")
    state = torch.as_strided(
        raw.view(torch.float32),
        size=(batch + 1, nv, dim, dim),
        stride=(block_stride, dim * dim, dim, 1),
        storage_offset=offset)
    state.zero_()
    return state, raw


def make_case(templates):
    """One iteration's inputs.  The mutable ones are fresh every time."""

    state, raw = make_paged_state(templates["batch"], templates["nv"],
                                  templates["dim"])
    conv_state = (torch.arange(3 * 3 * 16, dtype=torch.float32) + 301.0).reshape(
        3, 3, 16).to(torch.bfloat16).npu()
    return dict(
        templates,
        state=state,
        _state_storage=raw,
        conv_state=conv_state,
        conv_x=templates["x"].clone(),
    )


def run_pair(case, observed=None):
    """The decode-step mix: recurrent GDR then the conv1d update.

    ``observed`` collects the stream the launcher used for each call, read back
    immediately so both operators are attributed to this thread.
    """

    recurrent = _stable.npu_recurrent_gated_delta_rule(
        case["query"], case["key"], case["value"], case["state"],
        beta=case["beta"], scale=case["scale"],
        actual_seq_lengths=case["actual_seq_lengths"],
        ssm_state_indices=case["ssm_state_indices"],
        num_accepted_tokens=None, g=case["g"])
    if observed is not None:
        observed.append(_stable._last_resolved_stream())
    conv = _stable.npu_causal_conv1d_update(
        case["conv_x"], case["conv_state"], case["weight"], case["bias"],
        activation="silu", conv_state_indices=case["conv_indices"])
    if observed is not None:
        observed.append(_stable._last_resolved_stream())
    return recurrent, conv


def install_spy(real):
    """Record the value of every stream read made while a bucket is installed."""

    def spy():
        value = real()
        bucket = getattr(_RECORD, "seen", None)
        if bucket is not None:
            bucket.append(value)
        return value

    return spy


def worker(index, templates, golden, rounds, barrier, errors, notes) -> None:
    try:
        stream = torch.npu.Stream()
        observed: list[int | None] = []
        with torch.npu.stream(stream):
            expected = raw_stream_ptr()
            _RECORD.seen = []
            barrier.wait(timeout=180)
            for _ in range(rounds):
                case = make_case(templates)
                recurrent, conv = run_pair(case, observed)
                torch.npu.synchronize()
                for label, got, want in (("recurrent", recurrent, golden[0]),
                                         ("conv1d", conv, golden[1])):
                    diff = float((got.float() - want.float()).abs().max().item())
                    if diff != 0.0:
                        raise AssertionError(
                            f"{label} parity diff={diff} on a separate stream")
            passed = list(_RECORD.seen)
            _RECORD.seen = None
        if len(observed) != 2 * rounds or len(passed) != 2 * rounds:
            raise AssertionError(
                f"expected {2 * rounds} stream reads for {rounds} pairs, saw "
                f"{len(observed)} launched and {len(passed)} passed")
        if any(value is None for value in observed):
            raise AssertionError(
                "the loaded launcher does not export "
                "fla_npu_stable_last_resolved_stream, so a stream it resolves "
                "cannot be observed")
        wrong = sorted({value for value in passed
                        if value not in (expected, _stable._STREAM_SENTINEL)})
        if wrong:
            raise AssertionError(
                f"a call passed stream {wrong} instead of this thread's "
                f"{expected}")
        wrong = sorted({value for value in observed if value != expected})
        if wrong:
            raise AssertionError(
                f"a call launched on stream {wrong} instead of this thread's "
                f"{expected}")
        notes.append(f"thread {index}: {rounds} pairs, {len(passed)} calls, all "
                     f"on its own stream {expected}")
    except Exception as exc:  # noqa: BLE001
        errors.append((index, repr(exc)))


def negative_control(templates) -> tuple[bool, int | None, int]:
    """Hand the wrapper a foreign stream and require the readback to show it.

    This is the control that gives the readback its teeth: it proves the value
    reports the stream the launch actually used instead of echoing the calling
    thread.  A launcher that cached one thread's pointer and replayed it for the
    next caller -- the shape of the failure this test exists for -- fails here.
    """

    frozen = raw_stream_ptr()
    other = torch.npu.Stream()
    original = _stable._current_stream_ptr
    try:
        _stable._current_stream_ptr = lambda: frozen
        with torch.npu.stream(other):
            expected = raw_stream_ptr()
            run_pair(make_case(templates))
            torch.npu.synchronize()
            saw = _stable._last_resolved_stream()
    finally:
        _stable._current_stream_ptr = original
    return saw == frozen and saw != expected, saw, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if _SKIP_REASON is not None:
        print(f"SKIP test_stable_stream_interleaving: needs the NPU stack "
              f"({_SKIP_REASON})")
        return 0
    torch.npu.set_device(0)
    if not STABLE_LIB:
        print("note: FLA_NPU_STABLE_LIB is unset; relying on the bundled "
              "libfla_npu_stable.so next to the package")
    if not _stable.available():
        # A script run without the launcher is a missing prerequisite, not a
        # failed check: say so and exit clean, the way the rest of the suite
        # records "not run here".
        print("SKIP test_stable_stream_interleaving: no Stable-ABI launcher "
              "(set FLA_NPU_STABLE_LIB to a built libfla_npu_stable.so, or "
              "install a wheel that bundles one)")
        return 0

    # Built on this thread: the tensors must be identical for every worker, and
    # the device RNG is a single global sequence.
    templates = make_templates(batch=8)
    golden = run_pair(make_case(templates))
    torch.npu.synchronize()

    errors: list[tuple[int, str]] = []
    notes: list[str] = []
    barrier = threading.Barrier(args.threads + 1, timeout=180)
    real = _stable._current_stream_ptr
    # The spy stays installed for the negative control too: that check has to
    # observe the stream the call reported, so it needs the same recording.
    _stable._current_stream_ptr = install_spy(real)
    status = 0
    try:
        threads = [threading.Thread(
            target=worker,
            args=(i, templates, golden, args.rounds, barrier, errors, notes))
            for i in range(args.threads)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        for note in sorted(notes):
            print(f"  {note}")
        for index, message in errors:
            print(f"FAIL thread {index}: {message}")
        if errors:
            status = 1
        else:
            print(f"PASS {args.threads} threads x own stream, interleaved "
                  f"recurrent+conv1d, every call on its own stream, parity 0.0")
            detected, saw, expected = negative_control(templates)
            if detected:
                print(f"PASS negative control: a forced stream is reported "
                      f"(call on {expected} reported {saw})")
                print("ALL PASS: stable multi-stream interleaving")
            else:
                print(f"FAIL negative control: a call forced onto {expected} "
                      f"reported {saw} -- the readback does not follow the "
                      f"launch")
                status = 1
    finally:
        _stable._current_stream_ptr = real
    return status


if __name__ == "__main__":
    sys.exit(main())
