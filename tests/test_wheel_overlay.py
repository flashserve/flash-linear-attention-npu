from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
import runpy
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
        layers.install(base, self.update(full=False, kind="int64"), force=True)
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

    def test_opp_component_can_update_b_while_reusing_compatible_base_adapter(self):
        base = self.base()
        src = package(self.root / "overwide", exports=["npu_a"], opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        result = layers.install(base, wheel(self.root / "overwide.whl", src))
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["affected_base_operators"], ["npu_b"])

    def test_independent_component_install_orders_match_full_wheel(self):
        for order in ("adapter_first", "opp_first", "full"):
            with self.subTest(order=order):
                base = package(self.root / order, opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
                source = package(self.root / (order + "-opp"), kind="bool",
                                 exports=["npu_a"], opp={"aclnnDemo": "bool"})
                opp = source / "opp/vendors"
                standalone = self.update(full=False)
                if order == "full":
                    result = layers.install(base, wheel(self.root / "full.whl", source))
                else:
                    first, second = ((standalone, opp) if order == "adapter_first" else (opp, standalone))
                    waiting = layers.install(base, first)
                    self.assertEqual(waiting["state"], "pending")
                    self.assertIsNone(layers.active(base))
                    self.assertEqual(layers.environment_vendors(base), [layers._vendor(base)])
                    result = layers.install(base, second)
                self.assertEqual(result["state"], "active")
                plan = layers.runtime_plan(base)
                self.assertEqual(plan["exports"], ["npu_a"])
                self.assertEqual(plan["affected_base_operators"], [])
                self.assertEqual(layers.manifest(Path(plan["layer"]) / "fla_npu")["operators"]["npu_a"]["requires"], contract("bool"))
                self.assertEqual(layers.abi_module(base).validated_contracts(Path(plan["vendors"][0]), require_manifest=True),
                                 {symbol: tuple(types) for symbol, types in contract("bool").items()})
                self.assertEqual(Path(plan["vendors"][1]), layers._vendor(base))

    def test_standalone_preserves_opp_and_last_adapter_wins(self):
        base = self.base()
        first = layers.install(base, self.update())
        second = layers.install(base, self.update(full=False))
        self.assertEqual(second["state"], "active")
        self.assertNotEqual(first["components"]["adapter"], second["components"]["adapter"])
        self.assertEqual(first["components"]["opp"], second["components"]["opp"])
        self.assertEqual(len(layers.runtime_plan(base)["vendors"]), 2)

    def test_run_preserves_adapter_and_last_opp_wins(self):
        base = self.base()
        first = layers.install(base, self.update())
        opp = package(self.root / "new-opp", kind="bool", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        second = layers.install(base, opp / "opp/vendors")
        self.assertEqual(second["state"], "active")
        self.assertEqual(first["components"]["adapter"], second["components"]["adapter"])
        self.assertNotEqual(first["components"]["opp"], second["components"]["opp"])

    def test_full_wheel_replaces_both_components(self):
        base = self.base()
        first = layers.install(base, self.update())
        second = layers.install(base, self.update(kind="int64"))
        self.assertEqual(second["state"], "active")
        for kind in ("adapter", "opp"):
            self.assertNotEqual(first["components"][kind], second["components"][kind])
        self.assertEqual(second["components"]["adapter"], second["components"]["opp"])

    def test_pending_components_accumulate_without_falling_back_to_active_selection(self):
        base = self.base()
        initial = layers.install(base, self.update())
        old = layers.active(base)
        pending = layers.install(base, self.update(full=False, kind="int64"))
        self.assertEqual(pending["state"], "pending")
        self.assertEqual(layers.active(base), old)
        self.assertIn(initial["components"]["opp"], layers.shell_environment(base))
        opp = package(self.root / "matching-opp", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        completed = layers.install(base, opp / "opp/vendors")
        self.assertEqual(completed["state"], "active")
        self.assertEqual(completed["components"]["adapter"], pending["components"]["adapter"])
        self.assertIsNone(layers.state(base)["pending"])

    def test_newest_pending_adapter_replaces_earlier_request(self):
        base = self.base(False)
        layers.install(base, self.update(full=False))
        newest = layers.install(base, self.update(full=False, kind="int64"))
        opp = package(self.root / "latest-opp", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        result = layers.install(base, opp / "opp/vendors")
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["components"]["adapter"], newest["components"]["adapter"])

    def test_opp_only_update_keeps_all_base_wrappers(self):
        base = self.base()
        opp = package(self.root / "opp-only", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        result = layers.install(base, opp / "opp/vendors")
        self.assertEqual(result["state"], "active")
        self.assertIsNone(result["components"]["adapter"])
        self.assertEqual(result["operators"], [])
        self.assertEqual(result["affected_base_operators"], ["npu_a"])
        self.assertIsNone(layers.runtime_plan(base)["layer"])

    def test_shared_opp_with_matching_base_adapter_is_allowed(self):
        base = self.base()
        data = layers.manifest(base)
        data["operators"]["npu_b"]["requires"].update(contract())
        (base / layers.MANIFEST).write_text(json.dumps(data))
        result = layers.install(base, self.update(kind="int64"))
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["affected_base_operators"], ["npu_b"])

    def test_legacy_run_overlay_is_adopted_then_removed(self):
        base = self.base()
        legacy = package(self.root / "legacy-run", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        old_vendor = base / "opp/vendors" / layers.NAME
        shutil.copytree(layers._vendor(legacy), old_vendor)
        result = layers.install(base, self.update(full=False, kind="int64"))
        self.assertEqual(result["state"], "active")
        self.assertEqual(len(layers.runtime_plan(base)["vendors"]), 2)
        with mock.patch.dict(os.environ, {"FLA_NPU_OVERLAY_VENDOR": ""}):
            self.assertEqual(layers.environment_vendors(base), [layers._vendor(base)])
        layers.remove(base)
        self.assertFalse(old_vendor.exists())
        self.assertEqual(layers.environment_vendors(base), [layers._vendor(base)])

    def test_original_whole_wheel_state_can_receive_opp_update(self):
        base = self.base()
        first = layers.install(base, self.update())
        snapshot = Path(first["path"])
        (snapshot / "components.json").unlink()
        metadata = json.loads((snapshot / "installation.json").read_text())
        metadata.pop("component_protocol")
        (snapshot / "installation.json").write_text(json.dumps(metadata))
        (layers.storage(base) / "state.json").write_text(json.dumps({"active": snapshot.name, "pending": None}))
        opp = package(self.root / "old-state-opp", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        result = layers.install(base, opp / "opp/vendors")
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["components"]["adapter"], snapshot.name)

    def test_invalid_or_missing_component_is_reported_and_removable(self):
        base = self.base()
        for invalid in ("../outside", "b" * 32):
            result = layers.install(base, self.update())
            snapshot = Path(result["path"])
            (snapshot / "components.json").write_text(json.dumps({"version": 1, "adapter": invalid, "opp": None}))
            self.assertTrue(layers.status(base)[0]["errors"])
            with self.assertRaises(RuntimeError):
                layers.runtime_plan(base)
            layers.remove(base)

    def test_opp_missing_after_activation_is_not_silently_replaced_by_base(self):
        base = self.base()
        result = layers.install(base, self.update())
        (Path(result["path"]) / "fla_npu/opp/vendors" / layers.BASE / layers.LIBRARY).unlink()
        with self.assertRaisesRegex(RuntimeError, "OPP is missing"):
            layers.environment_vendors(base)
        layers.remove(base)

    def test_missing_composition_does_not_drop_inherited_opp(self):
        base = self.base()
        layers.install(base, self.update())
        result = layers.install(base, self.update(full=False))
        (Path(result["path"]) / "components.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "component selections are missing"):
            layers.environment_vendors(base)
        self.assertTrue(layers.status(base)[0]["errors"])
        layers.remove(base)

    def test_explicit_opp_source_overrides_wheel_opp(self):
        base = self.base()
        opp = package(self.root / "explicit-opp", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        result = layers.install(base, self.update(kind="int64"), opp_source=opp / "opp/vendors")
        self.assertEqual(result["state"], "pending")
        selected = result["components"]["opp"]
        completed = layers.install(base, self.update(full=False))
        self.assertEqual(completed["state"], "active")
        self.assertEqual(completed["components"]["opp"], selected)

    def test_pth_and_shell_follow_composed_opp_not_adapter_payload(self):
        base = package(self.root / "boot-site/fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        layers.install(base, self.update())
        layers.install(base, self.update(full=False))
        expected = layers.environment_vendors(base)
        self.assertEqual(len(expected), 2)
        bootstrap = base.parent / "fla_npu_opp_env.py"
        shutil.copyfile(SETUP / "fla_npu_opp_env.py", bootstrap)
        env = runpy.run_path(str(bootstrap))
        self.assertEqual(env["_vendor_dirs"](), expected)
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], os.pathsep.join(map(str, expected)))
        self.assertIn(str(expected[0]), layers.shell_environment(base))
        with mock.patch.dict(os.environ, {"FLA_NPU_OVERLAY_VENDOR": ""}):
            self.assertEqual(layers.environment_vendors(base), [layers._vendor(base)])
            self.assertNotIn(str(expected[0]), layers.shell_environment(base))

    @unittest.skipUnless(os.name == "posix", "run extraction requires bash")
    def test_run_file_can_complete_pending_adapter(self):
        import shlex
        base = self.base()
        layers.install(base, self.update(full=False))
        source = package(self.root / "run-payload", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        run = self.root / "update.run"
        run.write_text('#!/bin/bash\nset -eu\nfor arg in "$@"; do\n'
                       'case "$arg" in --install-path=*) destination="${arg#*=}";; esac\ndone\n'
                       'mkdir -p "$destination/vendors"\ncp -R '
                       + shlex.quote(str(layers._vendor(source))) + ' "$destination/vendors/"\n')
        self.assertEqual(layers.install(base, run)["state"], "active")

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

    def test_cli_pending_is_success_and_separate_opp_completes_it(self):
        site = self.root / "cli-components"
        base = package(site / "fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        (base / "__init__.py").write_text('raise RuntimeError("must not import runtime")')
        shutil.copyfile(SETUP / "fla_npu/install_opp.py", base / "install_opp.py")
        for name in ("fla_npu_opp_manage.py", "fla_npu_wheel_overlay.py"):
            shutil.copyfile(SETUP / name, site / name)
        env = dict(os.environ, PYTHONPATH=str(site))
        opp = package(self.root / "cli-opp", exports=["npu_a"], opp={"aclnnDemo": "bool"})
        for source, expected in ((self.update(full=False), "pending"), (opp / "opp/vendors", "active")):
            result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", "--overlay", "--source", str(source)],
                                    env=env, cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"state": "' + expected + '"', result.stdout)
        self.assertFalse(layers.runtime_plan(base)["errors"])

    def test_cli_remove_preserves_base_vendor_configuration_bytes(self):
        site = self.root / "cli-remove"
        base = package(site / "fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        config = base / "opp/vendors/config.ini"
        original = b"# original package configuration\nload_priority=fla_npu_transformer\n"
        config.write_bytes(original)
        shutil.copyfile(SETUP / "fla_npu/install_opp.py", base / "install_opp.py")
        for name in ("fla_npu_opp_manage.py", "fla_npu_wheel_overlay.py"):
            shutil.copyfile(SETUP / name, site / name)
        env = dict(os.environ, PYTHONPATH=str(site))
        layers.install(base, self.update())
        result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", "--remove-overlay"],
                                env=env, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(config.read_bytes(), original)

    def summary_cli_base(self, full=True):
        site = self.root / "summary-site"
        base = package(site / "fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"} if full else None)
        (base / "__init__.py").write_text('raise RuntimeError("must not import runtime")')
        shutil.copyfile(SETUP / "fla_npu/install_opp.py", base / "install_opp.py")
        for name in ("fla_npu_opp_manage.py", "fla_npu_wheel_overlay.py"):
            shutil.copyfile(SETUP / name, site / name)
        return base

    def summary_cli_install(self, base, source, **environment):
        result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", "--overlay", "--source", str(source)],
                                env=dict(os.environ, PYTHONPATH=str(base.parent), **environment),
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_install_summary_reports_paths_and_pending_without_claiming_activation(self):
        base = self.summary_cli_base()
        output = self.summary_cli_install(base, self.update())
        original_opp = layers.component_paths(base, layers.active(base))[1]
        self.assertIn("Installation successful: mode=overlay, artifact=wheel", output)
        self.assertIn(f"base OPP: {base / 'opp/vendors' / layers.BASE}", output)
        self.assertIn("previous overlay OPP (active): none", output)
        self.assertIn(f"installed OPP: {original_opp}", output)
        self.assertIn("selection for a new process: overlay", output)

        output = self.summary_cli_install(base, self.update(full=False, kind="int64"))
        self.assertIn(f"previous overlay OPP (active): {original_opp}", output)
        self.assertIn("installed OPP: none (no OPP supplied; previous selection retained)", output)
        self.assertIn("combination: pending (installed, not enabled; previous active combination unchanged)", output)
        self.assertIn("selection for a new process: overlay", output)
        self.assertEqual(layers.component_paths(base, layers.active(base))[1], original_opp)

        opp = package(self.root / "run-update", exports=["npu_a"], opp={"aclnnDemo": "int64"})
        output = self.summary_cli_install(base, opp / "opp/vendors")
        new_opp = layers.component_paths(base, layers.active(base))[1]
        self.assertIn("Installation successful: mode=overlay, artifact=run", output)
        self.assertIn(f"previous overlay OPP (pending): {original_opp}", output)
        self.assertIn(f"installed OPP: {new_opp}", output)
        self.assertNotEqual(new_opp, original_opp)
        self.assertIn("combination: active", output)

        output = self.summary_cli_install(base, self.update(full=False, kind="int64"))
        self.assertIn(f"previous overlay OPP (active): {new_opp}", output)
        self.assertIn("installed OPP: none (no OPP supplied; previous selection retained)", output)
        self.assertEqual(layers.component_paths(base, layers.active(base))[1], new_opp)
        output = self.summary_cli_install(base, self.update(), FLA_NPU_OVERLAY_VENDOR="")
        self.assertIn("combination: active", output)
        self.assertIn("selection for a new process: base", output)

    def test_install_summary_without_base_opp_and_without_active_overlay(self):
        base = self.summary_cli_base(full=False)
        output = self.summary_cli_install(base, self.update(full=False))
        self.assertIn("base OPP: none", output)
        self.assertIn("combination: pending (installed, not enabled", output)
        self.assertIn("selection for a new process: base", output)
        output = self.summary_cli_install(base, self.update())
        self.assertIn("base OPP: none", output)
        self.assertIn("combination: active", output)
        self.assertIn(f"installed OPP: {layers.component_paths(base, layers.active(base))[1]}", output)

    def test_failed_install_does_not_print_success_summary(self):
        base = self.summary_cli_base()
        result = subprocess.run([sys.executable, "-m", "fla_npu_opp_manage", "--overlay", "--source",
                                 str(self.update(corrupt=True))],
                                env=dict(os.environ, PYTHONPATH=str(base.parent)), cwd=self.root,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RECORD", result.stderr)
        self.assertNotIn("Installation successful", result.stdout)
        self.assertIsNone(layers.active(base))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "run installer requires Linux/bash")
    def test_run_installer_summary_for_managed_and_external_opp(self):
        base = self.summary_cli_base()
        stage = self.root / "payload"
        source = package(stage / "packages", opp={"aclnnDemo": "int64"})
        shutil.move(str(source / "opp/vendors"), str(stage / "packages/vendors"))
        installer = stage / "install.sh"
        installer.write_text((ROOT / "cmake/scripts/custom/install.sh").read_text().replace(
            "vendor_name=customize", "vendor_name=fla_npu_transformer"))
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)
        env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.defpath, PYTHONPATH=str(base.parent))
        previous = None
        for args in ((), ("--install",), ("--full",)):
            result = subprocess.run(["bash", str(installer), *args], cwd=stage, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            current = layers.component_paths(base, layers.active(base))[1]
            self.assertIn("Installation successful: mode=overlay, artifact=run", result.stdout)
            self.assertIn(f"previous overlay OPP (active): {previous or 'none'}", result.stdout)
            self.assertIn(f"installed OPP: {current}", result.stdout)
            previous = current
        target = self.root / "external-opp"
        for previous in ("none", str(target / "vendors" / layers.BASE)):
            result = subprocess.run(["bash", str(installer), f"--install-path={target}"], cwd=stage, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("mode=CANN (external OPP; not wheel base/overlay), artifact=run", result.stdout)
            self.assertIn(f"previous OPP: {previous}", result.stdout)
            self.assertIn(f"installed OPP: {target / 'vendors' / layers.BASE}", result.stdout)


def uuid_name():
    import uuid
    return uuid.uuid4().hex


if __name__ == "__main__":
    unittest.main()
