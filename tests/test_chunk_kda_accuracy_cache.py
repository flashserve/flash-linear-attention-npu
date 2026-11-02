"""CPU-only regression tests for the ChunkKDA Example/ST golden cache.

Run with: python -m unittest discover -s tests -p test_chunk_kda_accuracy_cache.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def _load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "chunk_kda.py"
    spec = importlib.util.spec_from_file_location("chunk_kda_cache_example", path)
    module = importlib.util.module_from_spec(spec)
    ascendc = types.ModuleType("fla_npu.ops.ascendc")
    ascendc.chunk_kda_fwd = mock.Mock()
    ascendc.chunk_kda_bwd = mock.Mock()
    stubs = {
        "torch_npu": None,
        "fla_npu": types.ModuleType("fla_npu"),
        "fla_npu.ops": types.ModuleType("fla_npu.ops"),
        "fla_npu.ops.ascendc": ascendc,
        spec.name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


EXAMPLE = _load_example()


class ChunkKdaAccuracyCacheTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.cache_dir = Path(self.tempdir.name) / "golden"
        args = EXAMPLE.parse_args([
            "--batch", "1", "--tokens", "4", "--query-heads", "1",
            "--value-heads", "1", "--key-dim", "2", "--value-dim", "3",
        ])
        self.inputs = EXAMPLE.make_inputs(args, torch.float32, torch.device("cpu"))
        self.params = {
            "scale": 0.5,
            "lower_bound": -5.0,
            "cu_seqlens": None,
            "qk_l2norm": True,
        }

    def golden(self, *, inputs=None, **params):
        return EXAMPLE._load_or_create_accuracy_golden(
            cache_dir=self.cache_dir,
            inputs=self.inputs if inputs is None else inputs,
            **(self.params | params),
        )

    def assert_golden_equal(self, actual, expected):
        self.assertEqual(set(actual), set(EXAMPLE.ACCURACY_TENSORS))
        for name in EXAMPLE.ACCURACY_TENSORS:
            with self.subTest(tensor=name):
                self.assertTrue(torch.equal(actual[name], expected[name]))
                self.assertEqual(actual[name].dtype, torch.float32)
                self.assertEqual(actual[name].device.type, "cpu")
                self.assertFalse(actual[name].requires_grad)
                self.assertIsNone(actual[name].grad_fn)

    def test_cold_reference_and_hot_cache_preserve_all_outputs_and_gradients(self):
        ref_o, _, leaves = EXAMPLE.recurrent_reference(
            self.inputs["q"], self.inputs["k"], self.inputs["v"], self.inputs["g"],
            self.inputs["beta"], self.inputs["A_log"], self.inputs["dt_bias"],
            **self.params,
        )
        gradients = torch.autograd.grad(
            ref_o.sum(), [leaves[name] for name in ("q", "k", "v", "beta", "raw_gate")],
        )
        expected = dict(zip(EXAMPLE.ACCURACY_TENSORS, (ref_o.detach(), *gradients)))
        cold = self.golden()
        self.assert_golden_equal(cold, expected)
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 1)
        with mock.patch.object(torch.autograd, "grad", side_effect=AssertionError("CPU reference reran")):
            hot = self.golden()
        self.assert_golden_equal(hot, expected)

    def test_every_reference_input_changes_the_cache_key(self):
        self.golden()
        for index, name in enumerate(self.inputs, start=2):
            with self.subTest(input=name):
                inputs = {key: value.clone() for key, value in self.inputs.items()}
                inputs[name].view(-1)[0] += 0.125
                self.golden(inputs=inputs)
                self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), index)

    def test_reference_parameters_change_the_cache_key(self):
        self.golden()
        changes = [
            {"scale": 0.75}, {"lower_bound": -4.0},
            {"cu_seqlens": (0, 2, 4)}, {"qk_l2norm": False},
        ]
        for index, change in enumerate(changes, start=2):
            with self.subTest(change=change):
                self.golden(**change)
                self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), index)

    def test_input_dtype_and_shape_change_the_cache_key(self):
        self.golden()
        inputs = {name: value.to(torch.bfloat16) for name, value in self.inputs.items()}
        self.golden(inputs=inputs)
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 2)
        # The same contiguous bytes with B/T rearranged represent a different input.
        inputs = {name: (value.reshape(2, 2, *value.shape[2:]) if value.ndim > 1 else value)
                  for name, value in self.inputs.items()}
        self.golden(inputs=inputs)
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 3)

    def test_noncontiguous_inputs_reuse_the_same_logical_values(self):
        expected = self.golden()
        inputs = dict(self.inputs)
        padded = torch.empty(*self.inputs["q"].shape[:-1], self.inputs["q"].shape[-1] * 2)
        inputs["q"] = padded[..., ::2]
        inputs["q"].copy_(self.inputs["q"])
        self.assertFalse(inputs["q"].is_contiguous())
        with mock.patch.object(torch.autograd, "grad", side_effect=AssertionError("CPU reference reran")):
            actual = self.golden(inputs=inputs)
        self.assert_golden_equal(actual, expected)
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 1)

    def test_torch_and_reference_source_versions_change_the_cache_key(self):
        self.golden()
        with mock.patch.object(torch, "__version__", str(torch.__version__) + "+cache-test"):
            self.golden()
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 2)
        original_getsource = EXAMPLE.inspect.getsource
        for index, target in enumerate(
                (EXAMPLE.recurrent_reference, EXAMPLE._run_cpu_accuracy_reference), start=3):
            def changed_source(obj):
                source = original_getsource(obj)
                return source + "\n# changed reference version" if obj is target else source

            with self.subTest(function=target.__name__), mock.patch.object(
                    EXAMPLE.inspect, "getsource", side_effect=changed_source):
                self.golden()
            self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), index)

    def test_unreadable_cache_is_regenerated(self):
        expected = self.golden()
        cache_path, = self.cache_dir.glob("*.pt")
        cache_path.write_bytes(b"incomplete torch save")
        self.assert_golden_equal(self.golden(), expected)
        with mock.patch.object(torch.autograd, "grad", side_effect=AssertionError("CPU reference reran")):
            self.assert_golden_equal(self.golden(), expected)

    def test_incomplete_or_invalid_cached_tensors_are_regenerated(self):
        expected = self.golden()
        cache_path, = self.cache_dir.glob("*.pt")
        for corruption in ("missing", "shape", "dtype", "nonfinite"):
            with self.subTest(corruption=corruption):
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                if corruption == "missing":
                    del payload["tensors"]["dq"]
                elif corruption == "shape":
                    payload["tensors"]["dq"] = torch.zeros(1)
                elif corruption == "dtype":
                    payload["tensors"]["dq"] = payload["tensors"]["dq"].double()
                else:
                    payload["tensors"]["dq"].view(-1)[0] = float("nan")
                torch.save(payload, cache_path)
                with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                    actual = self.golden()
                self.assertEqual(grad.call_count, 1)
                self.assert_golden_equal(actual, expected)

    def test_force_regenerate_bypasses_an_existing_cache(self):
        expected = self.golden()
        with mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            actual = self.golden(force_regenerate=True)
        self.assertEqual(grad.call_count, 1)
        self.assert_golden_equal(actual, expected)
        self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 1)

    def test_cache_read_permission_error_falls_back_to_real_reference(self):
        expected = self.golden()
        with mock.patch.object(torch, "load", side_effect=PermissionError("denied")), \
                mock.patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
            actual = self.golden()
        self.assertEqual(grad.call_count, 1)
        self.assert_golden_equal(actual, expected)

    def test_cache_write_failures_return_reference_without_partial_files(self):
        expected = EXAMPLE._run_cpu_accuracy_reference(self.inputs, **self.params)

        def failed_save(payload, path):
            Path(path).write_bytes(b"partial save")
            raise RuntimeError("torch save failed")

        for target, name, failure in (
                (torch, "save", failed_save),
                (EXAMPLE.os, "replace", OSError("rename failed"))):
            with self.subTest(operation=name):
                self.cache_dir = Path(self.tempdir.name) / name
                with mock.patch.object(target, name, side_effect=failure):
                    actual = self.golden()
                self.assert_golden_equal(actual, expected)
                self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_default_directory_reuses_the_persistent_ci_golden_mount(self):
        with mock.patch.dict(os.environ, {"GDR_ACCURACY_CACHE_DIR": "/persistent/golden"}):
            args = EXAMPLE.parse_args([])
        self.assertEqual(Path(args.accuracy_cache_dir), Path("/persistent/golden/chunk_kda"))
        with mock.patch.dict(os.environ, {}, clear=True):
            args = EXAMPLE.parse_args([])
        self.assertEqual(Path(args.accuracy_cache_dir), Path("third_party/gdr_accuracy_golden/chunk_kda"))
        args = EXAMPLE.parse_args(["--accuracy-cache-dir", "/custom", "--accuracy-force-regenerate"])
        self.assertEqual(Path(args.accuracy_cache_dir), Path("/custom"))
        self.assertTrue(args.accuracy_force_regenerate)


if __name__ == "__main__":
    unittest.main()
