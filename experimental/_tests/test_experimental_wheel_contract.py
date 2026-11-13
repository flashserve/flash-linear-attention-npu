"""First-load checks and the shared layer protocol under experimental paths."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from experimental_fixtures import ABI_HELPER, write_abi
from tests.opp_abi_fixtures import write_elf_exports

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
        parent = types.ModuleType("experimental_fixture")
        parent.__path__ = [str(PYTHON)]
        self.manager = load("fixture_layers", ROOT / "torch_custom/fla_npu/fla_npu_wheel_overlay.py")
        self.loader = load("fixture_loader", PYTHON / "_loader.py")
        modules = patch.dict(sys.modules, {"experimental_fixture": parent,
            "experimental_fixture._loader": self.loader,
            "experimental_fixture._wheel_overlay": self.manager,
            "experimental_fixture._op_abi": load("fixture_abi", ABI_HELPER)})
        modules.start()
        self.addCleanup(modules.stop)
        self.layout = load("experimental_fixture._layers", PYTHON / "_layers.py")
        parent._layers = self.layout
        sys.modules["experimental_fixture._layers"] = self.layout
        self.runtime = load("experimental_fixture._stable", PYTHON / "_stable.py")
        self.package = self.make_package(self.root / self.manager.PACKAGE)
        self.vendor = self.package / "opp/vendors" / self.manager.BASE
        self.launcher = self.package / self.manager.LAUNCHER
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_package(self, path):
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("# fixture\n")
        shutil.copy2(ABI_HELPER, path / "_op_abi.py")
        vendor = path / "opp/vendors" / self.manager.BASE
        write_abi(vendor)
        launcher = path / self.manager.LAUNCHER
        launcher.write_bytes(b"fake launcher")
        stamp = "fixture-source"
        (path / "_build.json").write_text(json.dumps({"operators": [{"name": "probe"}],
            "source_hash": stamp, "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest()}))
        (path / self.manager.STAMP).write_text(f"SOURCE_HASH = {stamp!r}\nLAYER_PROTOCOL = 1\n")
        abi = self.manager.abi_module(path)
        contracts = abi.validated_contracts(vendor, require_manifest=True)
        (path / self.manager.MANIFEST).write_text(json.dumps({"protocol": 1, "tier": "a2",
            "source_hash": stamp, "exports": ["npu_probe"],
            "operators": {"npu_probe": {"requires": contracts}}}))
        return path

    def wheel(self, source):
        files = {self.manager.PACKAGE + "/" + p.relative_to(source).as_posix(): p.read_bytes()
                 for p in source.rglob("*") if p.is_file()}
        files["fixture-1.dist-info/WHEEL"] = b"Wheel-Version: 1.0\nTag: py3-none-any\n"
        files["fixture-1.dist-info/METADATA"] = b"Metadata-Version: 2.1\nName: fixture\nVersion: 1\n"
        record = io.StringIO()
        writer = csv.writer(record)
        for name, data in files.items():
            writer.writerow([name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(), len(data)])
        files["fixture-1.dist-info/RECORD"] = record.getvalue().encode()
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return wheel

    def check(self):
        return self.runtime._package_contract(self.package, [str(self.vendor / self.manager.LIBRARY)])

    def test_matching_contract_without_loading_runtime(self):
        before = set(sys.modules)
        build, library, active = self.check()
        self.assertEqual(library, self.launcher.resolve())
        self.assertEqual(build["operators"], [{"name": "probe"}])
        self.assertEqual(active, {"npu_probe"})
        self.assertFalse({"torch", "torch_npu", "fla_npu"} & (set(sys.modules) - before))

    def test_launcher_replaced_after_build(self):
        self.launcher.write_bytes(b"different launcher")
        with self.assertRaisesRegex(RuntimeError, "launcher hash"):
            self.check()

    def test_host_library_replaced_after_build(self):
        (self.vendor / self.manager.LIBRARY).write_bytes(b"different opapi")
        with self.assertRaisesRegex(RuntimeError, "ABI manifest"):
            self.check()

    def test_abi_mismatch_is_rejected(self):
        path = self.package / self.manager.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["operators"]["npu_probe"]["requires"]["aclnnProbeGetWorkspaceSize"][0] = "bool"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "ACLNN contract mismatch"):
            self.check()

    def set_contracts(self, package, specs):
        vendor = package / "opp/vendors" / self.manager.BASE
        header = vendor / "op_api/include/aclnnop/aclnn_probe.h"
        symbols, declarations = set(), []
        for name, attrs in specs.items():
            symbols.update({"aclnn" + name, "aclnn" + name + "GetWorkspaceSize"})
            declarations.extend([
                f"aclnnStatus aclnn{name}GetWorkspaceSize(const aclTensor* input, {attrs}"
                "uint64_t* workspaceSize, aclOpExecutor** executor);",
                f"aclnnStatus aclnn{name}(void* workspace, uint64_t workspaceSize, "
                "aclOpExecutor* executor, aclrtStream stream);"])
        header.write_text("\n".join(declarations), encoding="utf-8")
        write_elf_exports(vendor / self.manager.LIBRARY, symbols)
        abi = self.manager.abi_module(package)
        with patch.object(abi, "exported_symbols", return_value=symbols):
            abi.write_manifest(vendor)
        contracts = abi.validated_contracts(vendor, require_manifest=True)
        path = package / self.manager.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["operators"] = {"npu_" + name.lower(): {"requires": {
            s: v for s, v in contracts.items() if s in {"aclnn" + name, "aclnn" + name + "GetWorkspaceSize"}}}
            for name in specs}
        manifest["exports"] = sorted(manifest["operators"])
        path.write_text(json.dumps(manifest))
        build_path = package / "_build.json"
        build = json.loads(build_path.read_text())
        build["operators"] = [{"name": name.lower()} for name in specs]
        build_path.write_text(json.dumps(build))

    def test_partial_wheel_update_keeps_unchanged_base_operator(self):
        self.set_contracts(self.package, {"Probe": "", "Keep": ""})
        update = self.make_package(self.root / "update")
        self.set_contracts(update, {"Probe": "int64_t flag, "})
        self.assertEqual(self.manager.install(self.package, self.wheel(update))["state"], "active")
        plan = self.manager.runtime_plan(self.package)
        layer = Path(plan["layer"]) / self.manager.PACKAGE
        libraries = plan["bindings"]["base"]
        with patch.object(self.loader, "_PLAN", plan):
            self.assertEqual(self.runtime._package_contract(layer, libraries)[2], {"npu_probe"})
            self.assertEqual(self.runtime._package_contract(self.package, libraries)[2], {"npu_keep"})
            # Still reject a mismatch in an operator that remains owned by base.
            path = self.package / self.manager.MANIFEST
            original = path.read_text()
            manifest = json.loads(original)
            manifest["operators"]["npu_keep"]["requires"]["aclnnKeepGetWorkspaceSize"][0] = "bool"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "aclnnKeepGetWorkspaceSize"):
                self.runtime._package_contract(self.package, libraries)
            path.write_text(original)
        # A run-only update cannot remove the old adapter's ABI requirements.
        with patch.object(self.loader, "_PLAN", {**plan, "layer": None, "exports": []}):
            with self.assertRaisesRegex(RuntimeError, "aclnnProbeGetWorkspaceSize"):
                self.runtime._package_contract(self.package, libraries)

    def test_wheel_and_opp_components_reuse_common_install_and_restore(self):
        update = self.make_package(self.root / "update")
        before = {p.relative_to(self.package): p.read_bytes() for p in self.package.rglob("*") if p.is_file()}
        result = self.manager.install(self.package, self.wheel(update))
        self.assertEqual(result["state"], "active")
        plan = self.manager.runtime_plan(self.package)
        self.assertEqual(plan["exports"], ["npu_probe"])
        self.assertEqual(len(plan["vendors"]), 2)
        first_adapter = plan["components"]["adapter"]
        result = self.manager.install(self.package, update / "opp/vendors" / self.manager.BASE)
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["components"]["adapter"], first_adapter)
        self.assertNotEqual(result["components"]["opp"], first_adapter)
        with patch.dict(os.environ, FLA_NPU_OVERLAY_VENDOR=""):
            self.assertIsNone(self.manager.runtime_plan(self.package))
        self.manager.remove(self.package)
        self.assertIsNone(self.manager.runtime_plan(self.package))
        for path, data in before.items():
            self.assertEqual((self.package / path).read_bytes(), data)

    def test_runtime_validation_does_not_read_disk_after_configure(self):
        self.loader._CONFIGURATION = (self.root, self.vendor, self.vendor / self.manager.LIBRARY)
        self.loader._ENVIRONMENT = self.loader._environment()
        with patch.object(Path, "read_text", side_effect=AssertionError("per-call file read")), \
             patch.object(Path, "stat", side_effect=AssertionError("per-call stat")), \
             patch.object(self.loader, "_mapped_libraries", side_effect=AssertionError("per-call maps scan")):
            self.loader.validate_active_environment()
