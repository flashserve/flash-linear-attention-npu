"""Stable pre-process contract: Python policy, bindings and device validation.

Set FLA_NPU_TEST_NPU=1 for the original shape/output cases now owned by the C++
adapter. Those cases cannot be proven by mocking a removed Python allocator.
"""
from __future__ import annotations
import importlib.util
import inspect
import os
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "fla_npu/ops/ascendc/_stable.py"
spec = importlib.util.spec_from_file_location("preprocess_stable", PATH)
STABLE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STABLE)


class PreProcessStableTest(unittest.TestCase):
    def invoke(self, **kwargs):
        args = [object(), object(), object()]
        launch = mock.Mock(return_value=object())
        with mock.patch.object(STABLE, "_op", return_value=launch), \
                mock.patch.object(STABLE, "_host_ints", side_effect=lambda x: x), \
                mock.patch.object(STABLE, "_current_stream_ptr", return_value=123):
            result = STABLE.npu_pre_process_fwd_kernel_merged(*args, **kwargs)
        return args, launch, result

    def test_subinterval_and_multisegment_metadata_forwarded(self):
        for metadata in ([40, 512], [0, 88, 188, 512]):
            gate = object()
            args, launch, result = self.invoke(gk=gate, cu_seqlens=metadata)
            launch.assert_called_once_with(*args, None, gate, None, None, metadata, 64, 123)
            self.assertIs(result, launch.return_value)

    def test_defaults_and_explicit_none(self):
        self.assertEqual(inspect.signature(STABLE.npu_pre_process_fwd_kernel_merged).parameters["chunk_size"].default, 64)
        for chunk_size in (None, 64):
            _, launch, _ = self.invoke(g=object(), cu_seqlens=[0, 64], chunk_size=chunk_size)
            self.assertEqual(launch.call_args.args[-2], 64)

    def test_invalid_python_contract_never_launches(self):
        cases = [({}, ValueError, "requires cu_seqlens"),
                 ({"cu_seqlens": [0, 64]}, ValueError, "exactly one"),
                 ({"cu_seqlens": [0, 64], "g": object(), "gk": object()}, ValueError, "exactly one"),
                 ({"cu_seqlens": [0, 64], "g": object(), "bg": object()}, NotImplementedError, "bg"),
                 ({"cu_seqlens": [0, 64], "gk": object(), "v": object()}, NotImplementedError, "v"),
                 ({"cu_seqlens": [0, 64], "g": object(), "chunk_size": 32}, ValueError, "chunk_size")]
        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs), mock.patch.object(STABLE, "_op") as launch:
                with self.assertRaisesRegex(error, message):
                    STABLE.npu_pre_process_fwd_kernel_merged(object(), object(), object(), **kwargs)
                launch.assert_not_called()


@unittest.skipUnless(os.environ.get("FLA_NPU_TEST_NPU") == "1", "requires built Stable launcher and NPU")
class PreProcessDeviceContractTest(unittest.TestCase):
    def test_original_shape_and_error_contracts(self):
        import torch
        import torch_npu
        from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged
        torch.npu.set_device(int(os.getenv("TEST_DEVICE_ID", "0")))
        def tensors(batch=1):
            return [torch.zeros((batch, 2, 512, 128), dtype=torch.bfloat16, device="npu") for _ in range(3)]
        k, w, u = tensors()
        gk = torch.zeros_like(k, dtype=torch.float32)
        for cu in ([40, 512], [0, 88, 188, 512]):
            hm = pre_process_fwd_kernel_merged(k, w, u, gk=gk, cu_seqlens=cu)
            torch.npu.synchronize()
            self.assertEqual(tuple(hm.shape), (len(cu)-1, 2, 128, 256))
            self.assertEqual(hm.dtype, torch.float32)
        for cu in ([-1, 512], [40, 513], [0, 300, 200, 512]):
            with self.subTest(cu=cu), self.assertRaises((RuntimeError, ValueError)):
                pre_process_fwd_kernel_merged(k, w, u, gk=gk, cu_seqlens=cu)
        k, w, u = tensors(2)
        with self.assertRaises((RuntimeError, ValueError)):
            pre_process_fwd_kernel_merged(k, w, u, gk=torch.zeros_like(k, dtype=torch.float32), cu_seqlens=[0, 512])


if __name__ == "__main__":
    unittest.main()
