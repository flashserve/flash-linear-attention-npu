"""Wheel/OPP mismatches fail before importing torch or loading an ELF."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experimental_fixtures import ABI_HELPER, write_abi

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "experimental/_python/fla_npu_experimental"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class WheelContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.package.mkdir()
        (self.package / "__init__.py").write_bytes(b"# package source\n")
        self.vendor = self.root / "vendor"
        library = self.vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"fake host library")
        write_abi(self.vendor)
        self.launcher = self.package / "libfla_npu_experimental_stable.so"
        self.launcher.write_bytes(b"fake launcher")
        self.ops = [{"name": "probe", "op_type": "Probe"}]
        (self.vendor / "experimental-build-manifest.json").write_text(json.dumps({"operators": self.ops}), encoding="utf-8")
        build = {"operators": self.ops, "launcher_sha256": hashlib.sha256(self.launcher.read_bytes()).hexdigest(),
                 "python_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.package.glob("*.py")}}
        (self.package / "_build.json").write_text(json.dumps(build), encoding="utf-8")
        parent = types.ModuleType("experimental_fixture")
        loader = load("fixture_loader", PYTHON / "_loader.py")
        parent._loader = loader
        modules = patch.dict(sys.modules, {"experimental_fixture": parent,
            "experimental_fixture._op_abi": load("fixture_abi", ABI_HELPER)})
        modules.start()
        self.addCleanup(modules.stop)
        self.runtime = load("experimental_fixture._stable", PYTHON / "_stable.py")

    def check(self):
        return self.runtime._package_contract(self.package, self.vendor)

    def test_matching_snapshot_without_loading_runtime(self):
        before = set(sys.modules)
        build, library = self.check()
        self.assertEqual(library, self.launcher.resolve())
        self.assertEqual(build["operators"], self.ops)
        self.assertFalse({"torch", "torch_npu", "fla_npu"} & (set(sys.modules) - before))

    def test_python_replaced_after_build(self):
        (self.package / "__init__.py").write_bytes(b"# replaced\n")
        with self.assertRaisesRegex(RuntimeError, "Python source"):
            self.check()

    def test_launcher_replaced_after_build(self):
        self.launcher.write_bytes(b"different launcher")
        with self.assertRaisesRegex(RuntimeError, "launcher hash"):
            self.check()

    def test_host_library_replaced_after_build(self):
        (self.vendor / "op_api/lib/libcust_opapi.so").write_bytes(b"different opapi")
        with self.assertRaisesRegex(RuntimeError, "ABI manifest"):
            self.check()

    def test_op_selection_differs_from_launcher(self):
        (self.vendor / "experimental-build-manifest.json").write_text('{"operators": []}', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "operator snapshots differ"):
            self.check()
