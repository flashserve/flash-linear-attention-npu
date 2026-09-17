"""Multi-stream / multi-thread regression for the thin launcher stream lookup.

The thin Python adapter resolves the current stream on every call through
``torch_npu._C._npu_getCurrentRawStream`` (no process-global cache, so one
worker thread can never leak its stream into another).  These tests make sure:
1. the resolved pointer always follows ``torch.npu.set_stream`` / ``torch.npu.stream``;
2. there is no process-global cached stream left behind;
3. after a switch the thin op really lands on the *current* stream
   (verified with stream-local events: the op must sit between the markers);
4. results are bit-identical to the ctypes golden path on every stream;
5. interleaving default -> A -> default -> B -> default stays correct;
6. several threads with independent streams (vLLM-style workers) each enqueue
   on their own stream and keep bit-identical results.
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


def _have_thin() -> bool:
    try:
        import fla_npu._C_thin  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _npu_available(), reason="NPU is required"),
    pytest.mark.skipif(not _have_thin(), reason="thin launcher extension not built"),
]


def _make_inputs(batch=32):
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


def _call_public(inputs, state):
    from fla_npu.ops import ascendc as ops

    return ops.recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state,
        beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"])


def _call_ctypes(inputs, state):
    from fla_npu.ops.ascendc import _aclnn_ctypes

    return _aclnn_ctypes.npu_recurrent_gated_delta_rule(
        inputs["query"], inputs["key"], inputs["value"], state,
        beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
        actual_seq_lengths=inputs["actual_seq_lengths"],
        ssm_state_indices=inputs["ssm_state_indices"])


class TestThinStreamInterleaving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1024)
        torch.npu.set_device(0)
        cls.inputs = _make_inputs()
        cls.default = torch.npu.current_stream()
        cls.s1 = torch.npu.Stream()
        cls.s2 = torch.npu.Stream()
        # Golden results from the ctypes path on the default stream.
        cls.state_g, _ = cls.inputs["make_state"]()
        cls.out_g = _call_ctypes(cls.inputs, cls.state_g)
        torch.npu.synchronize()

    def tearDown(self):
        torch.npu.set_stream(self.default)
        torch.npu.synchronize()

    def test_current_stream_ptr_follows_set_stream(self):
        from fla_npu.ops.ascendc import _thin

        for stream in (self.s1, self.default, self.s2, self.default):
            with self.subTest(stream=stream):
                torch.npu.set_stream(stream)
                self.assertEqual(
                    _thin._current_stream_ptr(),
                    int(torch.npu.current_stream().npu_stream))

    def test_no_process_global_stream_cache(self):
        from fla_npu.ops.ascendc import _thin

        # Regression guard for the vLLM crash: a process-global cached stream
        # pointer leaks one thread's stream into another.
        self.assertFalse(hasattr(_thin, "_CURRENT_STREAM_PTR"))
        self.assertFalse(hasattr(_thin, "_ensure_stream_tracking"))
        self.assertFalse(hasattr(_thin, "_STREAM_PATCHED"))

    def _assert_op_on_stream(self, stream):
        torch.npu.set_stream(stream)
        state, _ = self.inputs["make_state"]()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        out = _call_public(self.inputs, state)
        end.record()
        torch.npu.synchronize()
        elapsed = start.elapsed_time(end)
        # If the op were enqueued on a different stream, nothing would sit
        # between the two markers and the interval would be near zero.
        self.assertGreater(
            elapsed, 0.01,
            f"thin op did not land on the selected stream "
            f"(elapsed={elapsed:.4f} ms)")
        self.assertEqual(
            float((out.float() - self.out_g.float()).abs().max().item()), 0.0)
        self.assertEqual(
            float((state.float() - self.state_g.float()).abs().max().item()),
            0.0)

    def test_op_lands_on_each_stream(self):
        for stream in (self.default, self.s1, self.s2):
            with self.subTest(stream=stream):
                self._assert_op_on_stream(stream)

    def test_interleaved_switch_default_a_default_b_default(self):
        for stream in (self.default, self.s1, self.default,
                       self.s2, self.default):
            with self.subTest(stream=stream):
                self._assert_op_on_stream(stream)

    def test_parallel_streams_independent(self):
        results = {}
        torch.npu.set_stream(self.s1)
        state1, _ = self.inputs["make_state"]()
        out1 = _call_public(self.inputs, state1)
        torch.npu.set_stream(self.s2)
        state2, _ = self.inputs["make_state"]()
        out2 = _call_public(self.inputs, state2)
        torch.npu.synchronize()
        results["s1"] = (out1, state1)
        results["s2"] = (out2, state2)
        for name, (out, state) in results.items():
            with self.subTest(stream=name):
                self.assertEqual(
                    float((out.float() - self.out_g.float()).abs().max().item()),
                    0.0)
                self.assertEqual(
                    float((state.float() - self.state_g.float()).abs().max().item()),
                    0.0)

    def test_threads_use_their_own_streams(self):
        """vLLM-style workers: N threads, N streams, no cross-thread leak."""
        from fla_npu.ops.ascendc import _thin

        n_threads = 4
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker(index):
            try:
                stream = torch.npu.Stream()
                with torch.npu.stream(stream):
                    # Every worker switches to its own stream before any
                    # launcher call; the old global cache kept whichever
                    # stream was written last and misrouted all the others.
                    barrier.wait(timeout=60)
                    for _ in range(3):
                        self.assertEqual(
                            _thin._current_stream_ptr(),
                            int(torch.npu.current_stream().npu_stream))
                        state, _ = self.inputs["make_state"]()
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        out = _call_public(self.inputs, state)
                        end.record()
                        torch.npu.synchronize()
                        elapsed = start.elapsed_time(end)
                        # A misplaced enqueue leaves nothing between the two
                        # markers recorded on this thread's own stream.
                        if not elapsed > 0.01:
                            raise AssertionError(
                                f"thread {index}: op did not land on its own "
                                f"stream (elapsed={elapsed:.4f} ms)")
                        diff = float(
                            (out.float() - self.out_g.float()).abs().max().item())
                        if diff != 0.0:
                            raise AssertionError(
                                f"thread {index}: parity diff={diff}")
            except Exception as exc:  # noqa: BLE001 - reported via errors list
                errors.append((index, repr(exc)))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        torch.npu.synchronize()
        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))


if __name__ == "__main__":
    unittest.main()
