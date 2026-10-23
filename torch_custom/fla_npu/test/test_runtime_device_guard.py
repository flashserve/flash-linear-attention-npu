"""Stable Python stream selection across devices and threads.

The removed Python ACL descriptor/workspace executor has no remaining behavior
to test. Device guards, workspace ownership and descriptor destruction now live
in the launcher; real launch/lifetime regressions are in tests/stable_abi.
These tests exercise the remaining Python responsibility: never cache a stream
pointer and only skip the queue barrier when the launcher promises ordering.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import threading
import types
import unittest
from unittest import mock


STABLE_PATH = Path(__file__).resolve().parents[1] / "fla_npu/ops/ascendc/_stable.py"


class StableStreamSelectionTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("fla_npu_test_stable", STABLE_PATH)
        self.stable = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.stable)
        self.state = threading.local()
        self.state.device = 0
        self.state.stream = 101
        npu = types.SimpleNamespace(
            current_device=lambda: self.state.device,
            current_stream=lambda: types.SimpleNamespace(npu_stream=self.state.stream),
        )
        self.modules = mock.patch.object(self.stable, "_modules", return_value=(types.SimpleNamespace(npu=npu), None))
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_queue_ordered_launch_reads_current_device_on_every_call(self):
        seen = []

        def nowait(device):
            seen.append(device)
            return self.state.stream

        with mock.patch.object(self.stable, "_enqueues_launch", return_value=True), \
             mock.patch.object(self.stable, "_nowait_allowed", return_value=True), \
             mock.patch.object(self.stable, "_nowait_stream_fn", return_value=nowait), \
             mock.patch.object(self.stable, "_barrier_stream_fn") as barrier:
            self.assertEqual(self.stable._current_stream_ptr(), 101)
            self.state.device, self.state.stream = 2, 202
            self.assertEqual(self.stable._current_stream_ptr(), 202)
            barrier.assert_not_called()
        self.assertEqual(seen, [0, 2])

    def test_direct_launch_and_explicit_barrier_never_use_nowait_accessor(self):
        for queued, allowed in ((False, True), (True, False), (False, False)):
            with self.subTest(queued=queued, allowed=allowed), \
                 mock.patch.object(self.stable, "_enqueues_launch", return_value=queued), \
                 mock.patch.object(self.stable, "_nowait_allowed", return_value=allowed), \
                 mock.patch.object(self.stable, "_nowait_stream_fn") as nowait, \
                 mock.patch.object(self.stable, "_barrier_stream_fn", return_value=lambda device: self.state.stream + device):
                self.state.device, self.state.stream = 3, 300
                self.assertEqual(self.stable._current_stream_ptr(), 303)
                nowait.assert_not_called()

    def test_missing_nowait_accessor_uses_barrier(self):
        barrier = mock.Mock(return_value=404)
        with mock.patch.object(self.stable, "_enqueues_launch", return_value=True), \
             mock.patch.object(self.stable, "_nowait_allowed", return_value=True), \
             mock.patch.object(self.stable, "_nowait_stream_fn", return_value=None), \
             mock.patch.object(self.stable, "_barrier_stream_fn", return_value=barrier):
            self.assertEqual(self.stable._current_stream_ptr(), 404)
            barrier.assert_called_once_with(0)

    def test_missing_raw_accessor_reads_public_stream_each_time(self):
        with mock.patch.object(self.stable, "_enqueues_launch", return_value=False), \
             mock.patch.object(self.stable, "_barrier_stream_fn", return_value=None):
            self.assertEqual(self.stable._current_stream_ptr(), 101)
            self.state.stream = 505
            self.assertEqual(self.stable._current_stream_ptr(), 505)

    def test_thread_interleaving_keeps_thread_device_and_stream(self):
        barrier = threading.Barrier(4)
        results = []
        errors = []

        def worker(index):
            try:
                self.state.device = index
                for step in range(5):
                    self.state.stream = 1000 * (index + 1) + step
                    barrier.wait(timeout=5)
                    actual = self.stable._current_stream_ptr()
                    results.append((index, step, actual))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(self.stable, "_enqueues_launch", return_value=True), \
             mock.patch.object(self.stable, "_nowait_allowed", return_value=True), \
             mock.patch.object(self.stable, "_nowait_stream_fn", return_value=lambda device: self.state.stream if device == self.state.device else -1):
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertFalse(errors, errors)
        self.assertEqual(len(results), 20)
        for index, step, actual in results:
            self.assertEqual(actual, 1000 * (index + 1) + step)


if __name__ == "__main__":
    unittest.main()
