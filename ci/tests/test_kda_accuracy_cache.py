"""Cache lifecycle contracts; no NPU or PyTorch installation required."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("o", "dq", "dk", "dv", "dbeta", "dg")


class Tensor:
    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


def save(payload, stream):
    stream.write(json.dumps(payload, default=lambda tensor: vars(tensor)).encode())


def load(path, *, map_location, weights_only):
    assert map_location == "cpu" and weights_only is True
    payload = json.loads(path.read_text())
    payload["tensors"] = {key: Tensor(**value) for key, value in payload["tensors"].items()}
    return payload


class KdaAccuracyCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.args = types.SimpleNamespace(
            accuracy_cache_dir=self.directory.name, accuracy_force_regenerate=False,
            batch=1, tokens=4, query_heads=2, value_heads=2, key_dim=8, value_dim=8,
            chunk_size=64, dtype="bf16", seed=42, qk_l2norm=True, lower_bound=-5.0,
            layout="BSND",
        )
        self.inputs = {key: Tensor((1, 4, 2, 8)) for key in ("q", "k", "v", "g")}
        self.inputs["beta"] = Tensor((1, 4, 2))
        golden = {name: Tensor(self.inputs[source].shape)
                  for name, source in zip(NAMES, ("v", "q", "k", "v", "beta", "g"))}
        self.compute = Mock(return_value=golden)
        self.torch = types.SimpleNamespace(
            __version__="test", float32="float32", is_tensor=lambda value: isinstance(value, Tensor),
            set_num_threads=Mock(), save=save, load=load,
        )
        self.ns = dict(torch=self.torch, hashlib=hashlib, json=json, os=os, Path=Path,
                       pickle=pickle, tempfile=tempfile, ACCURACY_TENSORS=NAMES,
                       ACCURACY_REFERENCE_VERSION=1, _run_cpu_accuracy_reference=self.compute)
        tree = ast.parse((ROOT / "examples/chunk_kda.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_accuracy_config", "_load_or_create_accuracy_golden")]
        self.assertEqual(len(functions), 2)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "chunk_kda.py", "exec"), self.ns)

    def run_cache(self, *, scale=0.5, cu=None):
        return self.ns["_load_or_create_accuracy_golden"](
            self.args, self.inputs, scale=scale, cu_seqlens=cu)

    def test_second_run_reuses_all_outputs_without_computing_reference(self):
        self.run_cache()
        result = self.run_cache()
        self.compute.assert_called_once()
        self.assertEqual(set(result), set(NAMES))
        self.assertEqual(self.torch.set_num_threads.call_count, 2)
        self.assertEqual(len(list(Path(self.directory.name).glob("*.pt"))), 1)

    def test_reference_inputs_and_versions_invalidate_cache(self):
        self.run_cache()
        for name, value in (
            ("batch", 2), ("tokens", 8), ("query_heads", 4), ("value_heads", 4),
            ("key_dim", 16), ("value_dim", 16), ("chunk_size", 128), ("dtype", "fp16"),
            ("seed", 43), ("qk_l2norm", False), ("lower_bound", -4.0), ("layout", "BNSD"),
        ):
            with self.subTest(name=name), patch.object(self.args, name, value):
                before = self.compute.call_count
                self.run_cache()
                self.assertEqual(self.compute.call_count, before + 1)
        for kwargs in ({"scale": 1.0}, {"cu": (0, 2, 4)}):
            before = self.compute.call_count
            self.run_cache(**kwargs)
            self.assertEqual(self.compute.call_count, before + 1)
        before = self.compute.call_count
        self.ns["ACCURACY_REFERENCE_VERSION"] = 2
        self.run_cache()
        self.assertEqual(self.compute.call_count, before + 1)
        with patch.object(self.torch, "__version__", "new-version"):
            self.run_cache()
        self.assertEqual(self.compute.call_count, before + 2)

    def test_force_regenerate_recomputes_existing_entry(self):
        self.run_cache()
        self.args.accuracy_force_regenerate = True
        self.run_cache()
        self.assertEqual(self.compute.call_count, 2)

    def test_invalid_files_metadata_and_tensor_shapes_are_regenerated(self):
        self.run_cache()
        path = next(Path(self.directory.name).glob("*.pt"))
        original = path.read_bytes()
        for defect in ("truncated", "config", "missing", "shape", "dtype"):
            with self.subTest(defect=defect):
                payload = json.loads(original)
                if defect == "config":
                    payload["config"]["seed"] = -1
                elif defect == "missing":
                    del payload["tensors"]["dg"]
                elif defect == "shape":
                    payload["tensors"]["o"]["shape"] = [1]
                elif defect == "dtype":
                    payload["tensors"]["o"]["dtype"] = "float16"
                path.write_text("truncated" if defect == "truncated" else json.dumps(payload))
                before = self.compute.call_count
                self.run_cache()
                self.assertEqual(self.compute.call_count, before + 1)

    def test_failed_write_preserves_previous_entry_and_removes_temporary_file(self):
        self.run_cache()
        path = next(Path(self.directory.name).glob("*.pt"))
        original = path.read_bytes()
        self.args.accuracy_force_regenerate = True
        with patch.object(self.torch, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_cache()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
