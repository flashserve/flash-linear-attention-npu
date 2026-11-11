from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "torch_custom/fla_npu"
sys.path.insert(0, str(SETUP))
import fla_npu_wheel_overlay as layers
from tests.opp_abi_fixtures import write_elf_exports


def contract(kind="int64", symbol="aclnnDemo"):
    return {symbol + "GetWorkspaceSize": ["tensor", kind, "uint64*", "executor**"],
            symbol: ["void*", "uint64", "executor*", "stream"]}


def vendor(package, definitions):
    path = package / "opp/vendors" / layers.BASE
    header = path / "op_api/include/aclnnop/aclnn_demo.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    exports = set()
    for symbol, kind in definitions.items():
        spelling = {"int64": "int64_t", "bool": "bool"}[kind]
        text += (f"aclnnStatus {symbol}GetWorkspaceSize(const aclTensor *x, {spelling} mode, "
                 f"uint64_t *workspaceSize, aclOpExecutor **executor);\n"
                 f"aclnnStatus {symbol}(void *workspace, uint64_t workspaceSize, "
                 "aclOpExecutor *executor, aclrtStream stream);\n")
        exports.update((symbol, symbol + "GetWorkspaceSize"))
    header.write_text(text, encoding="utf-8")
    write_elf_exports(path / layers.LIBRARY, exports)
    abi = layers.abi_module(package)
    with mock.patch.object(abi, "exported_symbols", return_value=exports):
        abi.write_manifest(path)


def package(path, *, kind="int64", exports=None, opp=None):
    (path / "ops/ascendc").mkdir(parents=True)
    (path / "ops/ascendc/_stable_hash.py").write_text('SOURCE_HASH = "' + "a" * 32 + '"\nLAYER_PROTOCOL = 1\n')
    (path / "libfla_npu_stable.so").write_bytes(b"test launcher")
    shutil.copyfile(SETUP / "fla_npu/_op_abi.py", path / "_op_abi.py")
    data = {"protocol": 1, "tier": "a2", "source_hash": "a" * 32,
            "exports": exports or ["npu_a", "npu_b"],
            "operators": {"npu_a": {"requires": contract(kind)},
                          "npu_b": {"requires": contract(symbol="aclnnOther")}}}
    (path / layers.MANIFEST).write_text(json.dumps(data))
    if opp:
        vendor(path, opp)
    return path


def wheel(path, source, *, scheme="", corrupt=False):
    data = {}
    for item in source.rglob("*"):
        if item.is_file():
            name = scheme + "fla_npu/" + item.relative_to(source).as_posix()
            data[name] = item.read_bytes()
    data["demo-1.dist-info/WHEEL"] = b"Wheel-Version: 1.0\nTag: py3-none-any\n"
    data["demo-1.dist-info/METADATA"] = b"Metadata-Version: 2.1\nName: demo\nVersion: 1\n"
    stream = io.StringIO()
    writer = csv.writer(stream)
    for name, payload in data.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(payload)))
    data["demo-1.dist-info/RECORD"] = stream.getvalue().encode()
    if corrupt:
        data[scheme + "fla_npu/libfla_npu_stable.so"] = b"corrupt"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in data.items():
            archive.writestr(name, payload)
    return path


class WheelOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def base(self, full=True):
        return package(self.root / "base", opp={"aclnnDemo": "int64", "aclnnOther": "int64"} if full else None)

    def update(self, *, full=True, kind="bool", scheme="", corrupt=False):
        src = package(self.root / uuid_name(), kind=kind, exports=["npu_a"], opp={"aclnnDemo": kind} if full else None)
        return wheel(self.root / (uuid_name() + ".whl"), src, scheme=scheme, corrupt=corrupt)

    def test_full_wheel_changes_abi_and_preserves_base_b_and_files(self):
        base = self.base()
        before = {p.relative_to(base): p.read_bytes() for p in base.rglob("*") if p.is_file()}
        result = layers.install(base, self.update())
        self.assertEqual(result["state"], "active")
        plan = layers.runtime_plan(base)
        self.assertEqual(plan["exports"], ["npu_a"])
        self.assertEqual(len(plan["vendors"]), 2)
        for path, data in before.items():
            self.assertEqual((base / path).read_bytes(), data)
        layers.remove(base)
        self.assertIsNone(layers.active(base))
        for path, data in before.items():
            self.assertEqual((base / path).read_bytes(), data)

    def test_skeleton_overlay_reuses_compatible_base(self):
        result = layers.install(self.base(), self.update(full=False, kind="int64"))
        self.assertEqual(result["state"], "active")

    def test_unmodified_base_contract_is_not_a_new_install_gate(self):
        base = self.base()
        data = layers.manifest(base)
        data["operators"]["npu_b"]["requires"] = contract("bool", symbol="aclnnOther")
        (base / layers.MANIFEST).write_text(json.dumps(data))
        result = layers.install(base, self.update(full=False, kind="int64"))
        self.assertEqual(result["state"], "active")

    def test_skeleton_missing_matching_opp_is_pending(self):
        base = self.base()
        result = layers.install(base, self.update(full=False))
        self.assertEqual(result["state"], "pending")
        self.assertTrue(result["errors"])
        self.assertIsNone(layers.active(base))

    def test_skeleton_base_full_overlay(self):
        result = layers.install(self.base(False), self.update())
        self.assertEqual(result["state"], "active")

    def test_skeleton_base_skeleton_overlay(self):
        result = layers.install(self.base(False), self.update(full=False))
        self.assertEqual(result["state"], "pending")

    def test_attach_matching_opp_and_activate_pending(self):
        base = self.base()
        update = self.update(full=False)
        self.assertEqual(layers.install(base, update)["state"], "pending")
        opp_package = package(self.root / "opp", kind="bool", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        result = layers.install(base, update, force=True, opp_source=opp_package / "opp/vendors")
        self.assertEqual(result["state"], "active")

    def test_pending_replacement_preserves_active_layer(self):
        base = self.base()
        layers.install(base, self.update())
        old = layers.active(base)
        layers.install(base, self.update(full=False), force=True)
        self.assertEqual(layers.active(base), old)

    def test_corrupt_replacement_preserves_active_layer(self):
        base = self.base()
        layers.install(base, self.update())
        old = layers.active(base)
        with self.assertRaisesRegex(RuntimeError, "RECORD mismatch"):
            layers.install(base, self.update(corrupt=True), force=True)
        self.assertEqual(layers.active(base), old)

    def test_explicit_disable(self):
        base = self.base()
        layers.install(base, self.update())
        with mock.patch.dict(os.environ, {"FLA_NPU_OVERLAY_VENDOR": ""}):
            self.assertIsNone(layers.active(base))
        self.assertIsNotNone(layers.active(base))

    def test_both_wheel_payload_schemes(self):
        base = self.base()
        for scheme in ("demo-1.data/purelib/", "demo-1.data/platlib/"):
            with self.subTest(scheme=scheme):
                self.assertEqual(layers.install(base, self.update(scheme=scheme))["state"], "active")
                layers.remove(base)

    def test_shared_dependency_requires_matching_base_adapter(self):
        base = self.base()
        data = layers.manifest(base)
        data["operators"]["npu_b"]["requires"].update(contract())
        (base / layers.MANIFEST).write_text(json.dumps(data))
        result = layers.install(base, self.update())
        self.assertEqual(result["state"], "pending")
        self.assertTrue(any("base/npu_b" in e for e in result["errors"]))

    def test_unclaimed_opp_does_not_silently_override_b(self):
        base = self.base()
        src = package(self.root / "overwide", exports=["npu_a"], opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        result = layers.install(base, wheel(self.root / "overwide.whl", src))
        self.assertEqual(result["state"], "pending")
        self.assertTrue(any("outside the overlay scope" in e for e in result["errors"]))

    def test_traversal_is_rejected(self):
        update = self.update()
        with zipfile.ZipFile(update, "a") as archive:
            archive.writestr("../escape.py", "oops")
        with self.assertRaisesRegex(RuntimeError, "Unsafe wheel"):
            layers.install(self.base(), update)

    def test_wrong_tier_is_rejected(self):
        base = self.base()
        data = layers.manifest(base)
        data["tier"] = "a5"
        (base / layers.MANIFEST).write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "different NPU"):
            layers.install(base, self.update())

    def test_broken_active_can_be_removed_without_runtime_import(self):
        base = self.base()
        layers.install(base, self.update())
        (layers.active(base) / "fla_npu/layer_manifest.json").write_text("broken")
        self.assertTrue(layers.status(base)[0]["errors"])
        layers.remove(base)
        self.assertIsNone(layers.active(base))

    def test_atomic_pointer_failure_keeps_previous_install(self):
        base = self.base()
        layers.install(base, self.update())
        old = layers.active(base)
        original = layers._atomic_json
        def fail(path, value):
            if path.name == "state.json":
                raise OSError("injected disk failure")
            return original(path, value)
        with mock.patch.object(layers, "_atomic_json", side_effect=fail):
            with self.assertRaises(OSError):
                layers.install(base, self.update(), force=True)
        self.assertEqual(layers.active(base), old)

    def test_interrupt_after_pointer_commit_keeps_referenced_payload(self):
        base = self.base()
        original = layers._atomic_json
        def interrupt(path, value):
            original(path, value)
            if path.name == "state.json":
                raise KeyboardInterrupt()
        with mock.patch.object(layers, "_atomic_json", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                layers.install(base, self.update())
        self.assertTrue(layers.active(base).is_dir())
        self.assertFalse(layers.runtime_plan(base)["errors"])

    def test_public_signature_alias_and_uncovered_function_routing(self):
        base = self.base()
        layers.install(base, self.update(full=False, kind="int64"))
        plan = layers.runtime_plan(base)
        payload = Path(plan["layer"]) / "fla_npu/ops/ascendc/__init__.py"
        payload.write_text('''_ASCENDC_OPS = ("npu_a",)
BACKWARD_OPS = {}
MUTATED_ARGUMENTS = {"npu_a": ("x",)}
MUTATION_FLAGS = {}
MUTATION_PREDICATES = {}
def npu_a(x, *, gain=2):
    return x * gain
a = npu_a
''')
        root_module = types.ModuleType("fla_npu")
        root_module._WHEEL_LAYER_PLAN = plan
        def untouched(x):
            return x + 1
        namespace = {"_ASCENDC_OPS": ("npu_a", "npu_b"), "__all__": ["npu_a", "a", "npu_b", "b"],
                     "npu_b": untouched, "b": untouched, "BACKWARD_OPS": {},
                     "MUTATED_ARGUMENTS": {}, "MUTATION_FLAGS": {}, "MUTATION_PREDICATES": {}}
        spec = importlib.util.spec_from_file_location("layer_runtime_test", SETUP / "fla_npu/_wheel_layer_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        prefix = "_fla_npu_layer_" + Path(plan["layer"]).name
        try:
            with mock.patch.dict(sys.modules, {"fla_npu": root_module}):
                runtime.install_exports(namespace)
                self.assertEqual(namespace["a"](3, gain=4), 12)
                self.assertEqual(str(inspect.signature(namespace["npu_a"])), "(x, *, gain=2)")
                self.assertIs(namespace["a"], namespace["npu_a"])
                self.assertIs(namespace["npu_b"], untouched)
                self.assertEqual(namespace["b"](3), 4)
                self.assertEqual(namespace["MUTATED_ARGUMENTS"]["npu_a"], ("x",))
        finally:
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + "."):
                    del sys.modules[name]

    def test_shell_selection_matches_active_and_disabled_layers(self):
        base = self.base()
        layers.install(base, self.update())
        active_vendor = str(Path(layers.runtime_plan(base)["vendors"][0]))
        self.assertIn(active_vendor, layers.shell_environment(base))
        with mock.patch.dict(os.environ, {"FLA_NPU_OVERLAY_VENDOR": ""}):
            self.assertNotIn(active_vendor, layers.shell_environment(base))

    def test_damaged_state_is_reported_and_removable(self):
        base = self.base()
        layers.install(base, self.update())
        (layers.storage(base) / "state.json").write_text("broken")
        self.assertEqual(layers.status(base)[0]["state"], "damaged")
        layers.remove(base)
        self.assertIsNone(layers.active(base))

    def test_invalid_state_identity_is_reported_and_removable(self):
        base = self.base()
        layers.install(base, self.update())
        for invalid in ([], {"active": "../outside"}):
            (layers.storage(base) / "state.json").write_text(json.dumps(invalid))
            self.assertEqual(layers.status(base)[0]["state"], "damaged")
            layers.remove(base)

    def test_windows_drive_member_is_rejected(self):
        for member in ("fla_npu/C:/escape.py", "C:/fla_npu/escape.py"):
            with self.assertRaisesRegex(RuntimeError, "Unsafe wheel member"):
                layers._payload_name(member)

    def test_source_manifest_covers_all_public_entries(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_layer_manifest import requirements
        operators = requirements()
        self.assertEqual(len(operators), 38)
        for name, description in operators.items():
            self.assertTrue(description["dispatch"], name)
            self.assertFalse(any("?" in signature for signature in description["requires"].values()), name)

    def test_installed_cli_wheel_install_status_remove_without_root_import(self):
        site = self.root / "cli"
        base = package(site / "fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        (base / "__init__.py").write_text('raise RuntimeError("must not import runtime")')
        shutil.copyfile(SETUP / "fla_npu/install_opp.py", base / "install_opp.py")
        for name in ("fla_npu_opp_manage.py", "fla_npu_wheel_overlay.py"):
            shutil.copyfile(SETUP / name, site / name)
        env = dict(os.environ, PYTHONPATH=str(site))
        for args in (("--overlay", "--source", str(self.update())), ("--status",), ("--remove-overlay",)):
            result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", *args],
                                    env=env, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(layers.active(base))
        # An empty wheel state must not swallow subsequent run-overlay removal.
        legacy = package(self.root / "legacy", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        for args in (("--overlay", "--source", str(legacy / "opp/vendors")), ("--remove-overlay",)):
            result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", *args],
                                    env=env, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((base / "opp/vendors" / layers.NAME).exists())


def uuid_name():
    import uuid
    return uuid.uuid4().hex


if __name__ == "__main__":
    unittest.main()
