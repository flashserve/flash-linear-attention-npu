"""vLLM-pattern multi-thread / multi-stream regression (Recurrent + Conv1d).

Mirrors the failing production shape reported from the vLLM side: several
worker threads, each with its own NPU stream, alternating
``recurrent_gated_delta_rule`` and ``causal_conv1d_update`` thin calls.

Before the stream fix the process-global cache in ``_thin.py`` stored the
pointer of whichever thread called ``set_stream`` last, so other threads
enqueued kernels on the wrong stream, breaking ordering and eventually hitting
an illegal device address (single-request crash around token 257).  These
tests assert that every call lands on the calling thread's own stream and stays
bit-identical to the ctypes path.
"""
from __future__ import annotations

import threading
import unittest

import pytest
import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def _have_thin_ops() -> bool:
    try:
        from fla_npu.ops import ascendc as ops

        return (
            hasattr(ops, "recurrent_gated_delta_rule")
            and hasattr(ops, "causal_conv1d_update")
        )
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _npu_available(), reason="NPU is required"),
    pytest.mark.skipif(
        not _have_thin_ops(),
        reason="thin launcher with conv1d update (PR #390 branch) is required"),
]


def _make_recurrent_case(batch=32):
    Hk, Hv, D = 8, 16, 128

    def norm(t):
        return torch.nn.functional.normalize(t, p=2, dim=-1)

    query = norm(torch.randn(batch, Hk, D, device="npu")).to(torch.bfloat16)
    key = norm(torch.randn(batch, Hk, D, device="npu")).to(torch.bfloat16)
    value = torch.randn(batch, Hv, D, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(batch, Hv, dtype=torch.bfloat16, device="npu")
    g = torch.rand(batch, Hv, dtype=torch.float32, device="npu")
    asl = torch.tensor([0] + [1] * batch, dtype=torch.int32, device="npu")
    ssi = torch.arange(batch, dtype=torch.int32, device="npu")
    inner = Hv * D * D
    block_stride = inner + 16384

    def make_state():
        raw = torch.empty(
            (batch + 1) * block_stride * 4, dtype=torch.int8, device="npu")
        typed = raw.view(torch.float32)
        state = torch.as_strided(
            typed,
            size=(batch + 1, Hv, D, D),
            stride=(block_stride, D * D, D, 1),
            storage_offset=12288,
        )
        state.zero_()
        return state, raw

    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "g": g,
        "scale": D ** -0.5,
        "actual_seq_lengths": asl,
        "ssm_state_indices": ssi,
        "make_state": make_state,
    }


def _make_conv_case(batch=8, channels=4096, conv_width=6, kernel_size=4,
                    gap=565248):
    inner = conv_width * channels
    block_stride = inner + gap

    def make_state():
        raw = torch.empty(
            batch * block_stride * 2, dtype=torch.int8, device="npu")
        typed = raw.view(torch.bfloat16)
        state = torch.as_strided(
            typed,
            size=(batch, conv_width, channels),
            stride=(block_stride, channels, 1),
            storage_offset=0,
        )
        state.zero_()
        return state, raw

    x = torch.rand(batch, channels, dtype=torch.bfloat16, device="npu")
    weight = torch.rand(
        kernel_size, channels, dtype=torch.bfloat16, device="npu")
    return {
        "x": x,
        "weight": weight,
        "query_start_loc": torch.arange(
            0, batch + 1, dtype=torch.int32, device="npu"),
        "state_indices": torch.arange(
            1, batch + 1, dtype=torch.int32, device="npu"),
        "make_state": make_state,
    }


def _call_recurrent_ctypes(case, state):
    from fla_npu.ops.ascendc import _aclnn_ctypes

    return _aclnn_ctypes.npu_recurrent_gated_delta_rule(
        case["query"], case["key"], case["value"], state,
        beta=case["beta"], g=case["g"], scale=case["scale"],
        actual_seq_lengths=case["actual_seq_lengths"],
        ssm_state_indices=case["ssm_state_indices"])


def _call_recurrent_public(case, state):
    from fla_npu.ops import ascendc as ops

    return ops.recurrent_gated_delta_rule(
        case["query"], case["key"], case["value"], state,
        beta=case["beta"], g=case["g"], scale=case["scale"],
        actual_seq_lengths=case["actual_seq_lengths"],
        ssm_state_indices=case["ssm_state_indices"])


def _call_conv_ctypes(case, state, out):
    from fla_npu.ops.ascendc import _aclnn_ctypes

    return _aclnn_ctypes.npu_causal_conv1d_update(
        case["x"], state, case["weight"], bias=None, activation="silu",
        conv_state_indices=case["state_indices"], num_accepted_tokens=None,
        query_start_loc=case["query_start_loc"], max_query_len=1,
        null_block_id=0, out=out)


def _call_conv_public(case, state, out):
    from fla_npu.ops import ascendc as ops

    return ops.causal_conv1d_update(
        case["x"], state, case["weight"], bias=None, activation="silu",
        conv_state_indices=case["state_indices"], num_accepted_tokens=None,
        query_start_loc=case["query_start_loc"], max_query_len=1,
        null_block_id=0, out=out)


def _event_gap_ms(call):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end), result


class TestThinMultiThreadVllmPattern(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        torch.manual_seed(20260910)
        cls.rec = _make_recurrent_case()
        cls.conv = _make_conv_case()

        rec_state, _ = cls.rec["make_state"]()
        cls.rec_out_golden = _call_recurrent_ctypes(cls.rec, rec_state)
        conv_state, _ = cls.conv["make_state"]()
        cls.conv_out_golden = _call_conv_ctypes(
            cls.conv, conv_state, torch.empty_like(cls.conv["x"]))
        torch.npu.synchronize()

    def tearDown(self):
        torch.npu.synchronize()

    def _assert_parity(self, out, golden, what):
        diff = float((out.float() - golden.float()).abs().max().item())
        self.assertEqual(diff, 0.0, f"{what}: parity diff={diff}")

    def _run_iteration(self, index, rec_state, rec_raw, conv_state, conv_raw):
        """One recurrent + one conv update on the calling thread's stream."""
        from fla_npu.ops.ascendc import _thin

        self.assertEqual(
            _thin._current_stream_ptr(),
            int(torch.npu.current_stream().npu_stream),
            f"thread {index}: stream pointer does not match the thread stream")

        # The operators may touch rows outside the as_strided view (state
        # indices address the raw buffer), so reset the whole backing storage
        # to keep every iteration's expected result identical to the golden.
        rec_raw.zero_()
        conv_raw.zero_()

        gap, rec_out = _event_gap_ms(
            lambda: _call_recurrent_public(self.rec, rec_state))
        self.assertGreater(
            gap, 0.01,
            f"thread {index}: recurrent op did not land on its own stream "
            f"(gap={gap:.4f} ms)")
        self._assert_parity(rec_out, self.rec_out_golden,
                            f"thread {index} recurrent")

        conv_out = torch.empty_like(self.conv["x"])
        gap, _ = _event_gap_ms(
            lambda: _call_conv_public(self.conv, conv_state, conv_out))
        self.assertGreater(
            gap, 0.01,
            f"thread {index}: conv1d update did not land on its own stream "
            f"(gap={gap:.4f} ms)")
        self._assert_parity(conv_out, self.conv_out_golden,
                            f"thread {index} conv1d update")

    def test_threads_alternate_recurrent_and_conv1d_update(self):
        """vLLM worker shape: N threads, N streams, alternating thin calls."""
        n_threads, iterations = 4, 4
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker(index):
            try:
                stream = torch.npu.Stream()
                rec_state, rec_raw = self.rec["make_state"]()
                conv_state, conv_raw = self.conv["make_state"]()
                with torch.npu.stream(stream):
                    # Every worker owns its stream before the first launch;
                    # the old global cache kept only the last one written.
                    barrier.wait(timeout=60)
                    for _ in range(iterations):
                        self._run_iteration(
                            index, rec_state, rec_raw, conv_state, conv_raw)
            except Exception as exc:  # noqa: BLE001 - reported via errors list
                errors.append((index, repr(exc)))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        torch.npu.synchronize()
        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_repeated_calls_with_stream_switching(self):
        """Launcher-level soak: 24 repeats, rotating over three streams."""
        streams = [torch.npu.current_stream(), torch.npu.Stream(),
                   torch.npu.Stream()]
        rec_state, rec_raw = self.rec["make_state"]()
        conv_state, conv_raw = self.conv["make_state"]()
        for index in range(24):
            with torch.npu.stream(streams[index % len(streams)]):
                self._run_iteration(
                    index, rec_state, rec_raw, conv_state, conv_raw)


if __name__ == "__main__":
    unittest.main()
