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

    def test_unmodified_base_contract_is_not_a_new_install_gate(self):
        base = self.base()
        data = layers.manifest(base)
        data["operators"]["npu_b"]["requires"] = contract("bool", symbol="aclnnOther")
        (base / layers.MANIFEST).write_text(json.dumps(data))
        result = layers.install(base, self.update(kind="int64"))
        self.assertEqual(result["state"], "active")

    def test_skeleton_base_full_overlay(self):
        result = layers.install(self.base(False), self.update())
        self.assertEqual(result["state"], "active")

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
        with self.assertRaisesRegex(RuntimeError, "base/npu_b"):
            layers.install(base, self.update())
        self.assertIsNone(layers.active(base))

    def test_opp_component_can_update_b_while_reusing_compatible_base_adapter(self):
        base = self.base()
        src = package(self.root / "overwide", exports=["npu_a"], opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        result = layers.install(base, wheel(self.root / "overwide.whl", src))
        self.assertEqual(result["state"], "active")
        self.assertEqual(result["affected_base_operators"], ["npu_b"])

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
        result = layers.install(base, self.update(kind="int64"))
        self.assertEqual(result["state"], "active")
        self.assertEqual(len(layers.runtime_plan(base)["vendors"]), 3)
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
        result = layers.install(base, self.update())
        (Path(result["path"]) / "components.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "component selections are missing"):
            layers.environment_vendors(base)
        self.assertTrue(layers.status(base)[0]["errors"])
        layers.remove(base)

    def test_pth_and_shell_follow_composed_opp_not_adapter_payload(self):
        base = package(self.root / "boot-site/fla_npu", opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        layers.install(base, self.update())
        run = package(self.root / "run", opp={"aclnnDemo": "bool"})
        layers.install(base, run / "opp/vendors")
        expected = layers.environment_vendors(base)
        self.assertEqual(len(expected), 3)
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
    def test_run_file_updates_active_adapter(self):
        import shlex
        base = self.base()
        layers.install(base, self.update())
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
        layers.install(base, self.update(kind="int64"))
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

    def test_cli_rejects_standalone_and_external_opp(self):
        base = self.summary_cli_base()
        wheel = self.update(full=False, kind="int64")
        for extra, reason in (([], "requires an embedded OPP"), (["--opp-source", str(layers._vendor(base))], "--opp-source")):
            result = subprocess.run(
                [sys.executable, "-m", "fla_npu_opp_manage", "--overlay", "--source", str(wheel), *extra],
                env=dict(os.environ, PYTHONPATH=str(base.parent)), cwd=self.root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(reason, result.stderr)
            self.assertIsNone(layers.active(base))
        self.assertIn("combination: active", self.summary_cli_install(base, self.update()))

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

    def test_install_summary_reports_committed_wheel_and_run(self):
        base = self.summary_cli_base()
        output = self.summary_cli_install(base, self.update())
        original_opp = layers.component_paths(base, layers.active(base))[1]
        self.assertIn("Installation successful: mode=overlay, artifact=wheel", output)
        self.assertIn(f"base OPP: {base / 'opp/vendors' / layers.BASE}", output)
        self.assertIn("previous overlay OPP (active): none", output)
        self.assertIn(f"installed OPP: {original_opp}", output)
        self.assertIn("selection for a new process: overlay", output)

        opp = package(self.root / "run-update", opp={"aclnnDemo": "bool"})
        output = self.summary_cli_install(base, opp / "opp/vendors")
        self.assertIn(f"previous overlay OPP (active): {original_opp}", output)
        self.assertIn("mode=overlay, artifact=run", output)
        output = self.summary_cli_install(base, self.update(), FLA_NPU_OVERLAY_VENDOR="")
        self.assertIn("combination: active", output)
        self.assertIn("selection for a new process: base", output)

    def test_install_summary_without_base_opp_and_without_active_overlay(self):
        base = self.summary_cli_base(full=False)
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


class MultiOppTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.base = package(self.root / "site/fla_npu",
                            opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        self.a = package(self.root / "a", opp={"aclnnDemo": "int64"})
        self.b = package(self.root / "b", opp={"aclnnOther": "int64"})

    def install(self, source):
        return layers.install(self.base, source / "opp/vendors")

    def check_order(self, *results):
        expected = [str(Path(r["path"]) / "fla_npu/opp/vendors" / layers.BASE) for r in results]
        expected.append(str(layers._vendor(self.base)))
        plan = layers.runtime_plan(self.base)
        self.assertEqual(plan["vendors"], expected)
        self.assertEqual(list(map(str, layers.environment_vendors(self.base))), expected)
        for owner in ("base", "overlay"):
            self.assertEqual(plan["bindings"][owner], [str(Path(p) / layers.LIBRARY) for p in expected])
        self.assertEqual(layers.status(self.base)[0]["vendors"], expected)
        return plan

    def test_disjoint_runs_preserve_both_orders_and_remove_restores_base(self):
        before = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        for first_source, second_source in ((self.a, self.b), (self.b, self.a)):
            with self.subTest(first=first_source.name):
                first = self.install(first_source)
                second = self.install(second_source)
                plan = self.check_order(second, first)
                self.assertEqual(plan["affected_base_operators"], ["npu_a", "npu_b"])
                self.assertIsNone(plan["layer"])
                layers.remove(self.base)
                self.assertIsNone(layers.runtime_plan(self.base))
                self.assertEqual(layers.environment_vendors(self.base), [layers._vendor(self.base)])
                self.assertFalse(any(p.is_dir() for p in layers.storage(self.base).iterdir()))
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_newest_duplicate_wins_and_other_operator_is_retained(self):
        first = self.install(self.a)
        second = self.install(self.b)
        third = self.install(self.a)
        plan = self.check_order(third, second, first)
        abi = layers.abi_module(self.base)
        contracts = {p: abi.validated_contracts(Path(p), require_manifest=True) for p in plan["vendors"]}
        for symbol, index in (("aclnnDemo", 0), ("aclnnOther", 1)):
            self.assertEqual(next(p for p in plan["vendors"] if symbol in contracts[p]), plan["vendors"][index])
        self.assertEqual(third["installation"]["opp_order"], plan["vendors"])

    def test_rejected_standalone_preserves_full_opp_chain(self):
        first = self.install(self.a)
        src = package(self.root / "full", exports=["npu_b"], opp={"aclnnOther": "int64"})
        second = layers.install(self.base, wheel(self.root / "full.whl", src))
        skeleton = package(self.root / "no-opp", exports=["npu_b"])
        with self.assertRaisesRegex(RuntimeError, "requires an embedded OPP"):
            layers.install(self.base, wheel(self.root / "standalone.whl", skeleton))
        self.check_order(second, first)

    def test_newest_incompatible_symbol_does_not_fall_through(self):
        first = self.install(self.a)
        second = self.install(self.b)
        bad = package(self.root / "bad", opp={"aclnnDemo": "bool"})
        with self.assertRaisesRegex(RuntimeError, "base/npu_a: aclnnDemoGetWorkspaceSize mismatches OPP"):
            self.install(bad)
        self.check_order(second, first)
        third = self.install(self.b)
        self.check_order(third, second, first)

    def test_version_one_record_migrates_without_dropping_old_opp(self):
        first = self.install(self.a)
        snapshot = Path(first["path"])
        (snapshot / "components.json").write_text(json.dumps({"version": 1, "adapter": None, "opp": snapshot.name}))
        second = self.install(self.b)
        self.check_order(second, first)
        self.assertEqual(json.loads((Path(second["path"]) / "components.json").read_text())["version"], 2)

    def test_missing_retained_opp_is_reported_and_removal_recovers(self):
        first = self.install(self.a)
        self.install(self.b)
        (Path(first["path"]) / "fla_npu/opp/vendors" / layers.BASE / layers.LIBRARY).unlink()
        with self.assertRaisesRegex(RuntimeError, "OPP is missing"):
            layers.environment_vendors(self.base)
        self.assertTrue(layers.status(self.base)[0]["errors"])
        layers.remove(self.base)
        self.assertEqual(layers.environment_vendors(self.base), [layers._vendor(self.base)])

    def test_invalid_chain_is_rejected(self):
        for invalid in (None, "bad", [None], ["../escape"], ["a" * 32, "a" * 32]):
            with self.subTest(opps=invalid):
                first = self.install(self.a)
                (Path(first["path"]) / "components.json").write_text(json.dumps(
                    {"version": 2, "adapter": None, "opps": invalid}))
                with self.assertRaises(RuntimeError):
                    layers.runtime_plan(self.base)
                layers.remove(self.base)

    def test_pth_manual_source_and_disabled_selection_agree(self):
        first = self.install(self.a)
        second = self.install(self.b)
        expected = self.check_order(second, first)["vendors"]
        bootstrap = self.base.parent / "fla_npu_opp_env.py"
        shutil.copyfile(SETUP / bootstrap.name, bootstrap)
        runpy.run_path(str(bootstrap))
        self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], os.pathsep.join(expected))
        libs = os.pathsep.join(str(Path(p) / layers.LIBRARY) for p in expected)
        # The .pth hook configures CANN only; import later binds the host list.
        shell = layers.shell_environment(self.base)
        self.assertIn(os.pathsep.join(expected), shell)
        self.assertIn(libs, shell)
        with mock.patch.dict(os.environ, {"FLA_NPU_OVERLAY_VENDOR": ""}):
            self.assertEqual(layers.environment_vendors(self.base), [layers._vendor(self.base)])
            self.assertNotIn(first["path"], layers.shell_environment(self.base))
            self.assertNotIn(second["path"], layers.shell_environment(self.base))

    def test_failed_pointer_commit_keeps_entire_previous_chain(self):
        first = self.install(self.a)
        second = self.install(self.b)
        before = set(layers.storage(self.base).iterdir())
        original = layers._atomic_json
        def fail(path, value):
            if path.name == "state.json":
                raise OSError("injected disk failure")
            original(path, value)
        with mock.patch.object(layers, "_atomic_json", side_effect=fail):
            with self.assertRaisesRegex(OSError, "injected disk failure"):
                self.install(self.a)
        self.check_order(second, first)
        self.assertEqual(set(layers.storage(self.base).iterdir()), before)



class MultiArtifactTests(unittest.TestCase):
    setUp = MultiOppTests.setUp

    def artifact(self, name, op, kind, *, abi="int64", gain=3):
        symbol = "aclnnDemo" if op == "npu_a" else "aclnnOther"
        src = package(self.root / name, exports=[op],
                      opp={symbol: abi})
        data = layers.manifest(src)
        data["operators"][op]["requires"] = contract(abi, symbol)
        (src / layers.MANIFEST).write_text(json.dumps(data))
        self.provider(src, {op: gain})
        return src / "opp/vendors" if kind == "run" else wheel(self.root / (name + ".whl"), src)

    def provider(self, src, gains):
        code = (f"_ASCENDC_OPS = {tuple(gains)!r}\n"
                "BACKWARD_OPS = {}\nMUTATION_FLAGS = {}\nMUTATION_PREDICATES = {}\n"
                "MUTATED_ARGUMENTS = {}\n_TORCH_NPU_COMPAT_OPS = ()\n")
        for op, gain in gains.items():
            alias = op.removeprefix("npu_")
            code += f"def {op}(x, *, gain={gain}):\n    return x * gain\n{alias} = {op}\n"
            code += f"MUTATED_ARGUMENTS[{op!r}] = ('{gain}',)\n"
        (src / "ops/ascendc/__init__.py").write_text(code)

    def route(self, plan):
        runtime = runpy.run_path(str(SETUP / "fla_npu/_wheel_layer_runtime.py"))
        root = types.ModuleType("fla_npu")
        root._WHEEL_LAYER_PLAN = plan
        old_a, old_b = lambda x: x, lambda x: x
        namespace = {"_ASCENDC_OPS": ("npu_a", "npu_b"), "__all__": ["npu_a", "a", "npu_b", "b"],
                     "npu_a": old_a, "a": old_a, "npu_b": old_b, "b": old_b,
                     "BACKWARD_OPS": {}, "MUTATION_FLAGS": {}, "MUTATION_PREDICATES": {},
                     "MUTATED_ARGUMENTS": {}, "_TORCH_NPU_COMPAT_OPS": ("npu_a", "npu_b")}
        def cleanup():
            for key in list(sys.modules):
                if key.startswith("_fla_npu_layer_"):
                    del sys.modules[key]
        self.addCleanup(cleanup)
        with mock.patch.dict(sys.modules, {"fla_npu": root}):
            runtime["install_exports"](namespace)
        return namespace

    def test_three_packages_in_all_mixed_orders(self):
        from itertools import permutations
        for last_kind in ("run", "full"):
            entries = [("npu_a", "full"), ("npu_a", "run"), ("npu_b", last_kind)]
            artifacts = [self.artifact(f"{last_kind}-{i}", op, kind, gain=i + 3)
                         for i, (op, kind) in enumerate(entries)]
            for order in permutations(range(3)):
                with self.subTest(last_kind=last_kind, order=order):
                    owners, opps, gains = {}, [], {}
                    for i in order:
                        op, kind = entries[i]
                        result = layers.install(self.base, artifacts[i])
                        path = Path(result["path"])
                        opps.insert(0, path.name)
                        if kind == "full":
                            owners[op] = str(path)
                            gains[op] = i + 3
                        plan = layers.runtime_plan(self.base)
                        self.assertEqual(plan["components"]["opps"], opps)
                        self.assertEqual({n: e["path"] for e in plan["adapter_layers"] for n in e["exports"]}, owners)
                        routed = self.route(plan)
                        for name in ("npu_a", "npu_b"):
                            self.assertEqual(routed[name](2), 2 * gains.get(name, 1))
                    layers.remove(self.base)

    def test_latest_declared_entry_wins_and_older_other_entry_keeps_metadata(self):
        src = package(self.root / "both", exports=["npu_a", "npu_b"], opp={"aclnnDemo": "int64", "aclnnOther": "int64"})
        self.provider(src, {"npu_a": 3, "npu_b": 4})
        first = layers.install(self.base, wheel(self.root / "both.whl", src))
        second = layers.install(self.base, self.artifact("new-b", "npu_b", "full", gain=5))
        plan = layers.runtime_plan(self.base)
        self.assertEqual(plan["adapter_layers"], [
            {"path": second["path"], "exports": ["npu_b"]},
            {"path": first["path"], "exports": ["npu_a"]}])
        namespace = self.route(plan)
        for op, gain in (("npu_a", 3), ("npu_b", 5)):
            self.assertEqual(namespace[op](2), 2 * gain)
            self.assertEqual(str(inspect.signature(namespace[op])), f"(x, *, gain={gain})")
            self.assertIs(namespace[op], namespace[op[4:]])
            self.assertEqual(namespace["MUTATED_ARGUMENTS"][op], (str(gain),))
        self.assertEqual(namespace["_TORCH_NPU_COMPAT_OPS"], ())

    def test_failed_middle_run_is_not_retried_by_later_wheel(self):
        first = layers.install(self.base, self.artifact("b1", "npu_b", "full"))
        run = self.artifact("a2", "npu_a", "run", abi="bool")
        retained = set(layers.storage(self.base).iterdir())
        with self.assertRaisesRegex(RuntimeError, "base/npu_a: aclnnDemoGetWorkspaceSize mismatches OPP"):
            layers.install(self.base, run)
        self.assertEqual(set(layers.storage(self.base).iterdir()), retained)
        self.assertEqual(layers.active(self.base), Path(first["path"]))
        third = layers.install(self.base, self.artifact("a1", "npu_a", "full", abi="bool"))
        chain = [Path(r["path"]).name for r in (third, first)]
        self.assertEqual(third["components"]["opps"], chain)
        retried = layers.install(self.base, run)
        self.assertEqual(retried["components"]["adapters"], chain)
        self.assertEqual(retried["components"]["opps"], [Path(retried["path"]).name, *chain])

    def test_run_cannot_break_retained_wheel_adapter(self):
        a = layers.install(self.base, self.artifact("a1", "npu_a", "full", abi="bool"))
        b = layers.install(self.base, self.artifact("b1", "npu_b", "full"))
        # This run matches the base adapter but would break the selected A1 adapter.
        with self.assertRaisesRegex(RuntimeError, "overlay/npu_a: aclnnDemoGetWorkspaceSize mismatches OPP"):
            layers.install(self.base, self.a / "opp/vendors")
        self.assertEqual(layers.runtime_plan(self.base)["adapter_layers"],
                         [{"path": b["path"], "exports": ["npu_b"]}, {"path": a["path"], "exports": ["npu_a"]}])

    def test_run_adds_opp_without_adding_a_python_entry(self):
        for symbols in ({"aclnnNew": "int64"}, {"aclnnNew": "int64", "aclnnOther": "int64"}):
            with self.subTest(symbols=symbols):
                src = package(self.root / f"run-{len(symbols)}", opp=symbols)
                layers.install(self.base, src / "opp/vendors")
                plan = layers.runtime_plan(self.base)
                self.assertEqual(plan["affected_base_operators"], ["npu_b"] if "aclnnOther" in symbols else [])
                self.assertFalse(plan["adapter_layers"])
                routed = self.route(plan)
                self.assertEqual(routed["npu_b"](2), 2)
                self.assertNotIn("npu_c", routed)
                layers.remove(self.base)

    def test_wheel_can_add_new_entry_alone_or_with_existing_entry(self):
        for names in (["npu_c"], ["npu_b", "npu_c"]):
            with self.subTest(names=names):
                symbols = {"aclnnNew": "int64"}
                if "npu_b" in names:
                    symbols["aclnnOther"] = "int64"
                src = package(self.root / "-".join(names), opp=symbols)
                data = layers.manifest(src)
                data["operators"]["npu_c"] = {"requires": contract(symbol="aclnnNew")}
                data["exports"] = names
                (src / layers.MANIFEST).write_text(json.dumps(data))
                self.provider(src, dict.fromkeys(names, 5))
                layers.install(self.base, wheel(self.root / "add.whl", src))
                routed = self.route(layers.runtime_plan(self.base))
                self.assertEqual(routed["npu_a"](2), 2)
                self.assertEqual(routed["npu_b"](2), 10 if "npu_b" in names else 2)
                self.assertEqual(routed["npu_c"](2), 10)
                layers.remove(self.base)

    def test_legacy_pending_never_becomes_active(self):
        for had_active in (False, True):
            with self.subTest(had_active=had_active):
                root = layers.storage(self.base)
                root.mkdir(parents=True, exist_ok=True)
                first = Path(layers.install(self.base, self.a / "opp/vendors")["path"]).name if had_active else None
                pending_id = "d" * 32
                (root / pending_id).mkdir()
                (root / "state.json").write_text(json.dumps({"version": 2, "active": first, "pending": pending_id}))
                self.assertEqual(len(layers.status(self.base)), int(had_active))
                next_layer = layers.install(self.base, self.b / "opp/vendors")
                previous = [first] if first else []
                self.assertEqual(next_layer["components"]["opps"], [Path(next_layer["path"]).name, *previous])
                self.assertNotIn("pending", json.loads((root / "state.json").read_text()))
                layers.remove(self.base)

    def test_single_adapter_records_migrate_without_losing_earlier_entry(self):
        for version in (1, 2):
            with self.subTest(version=version):
                first = layers.install(self.base, self.artifact("legacy-"+str(version), "npu_a", "full"))
                snapshot = Path(first["path"])
                # Recreate an adapter-only layer written by the previous manager.
                shutil.rmtree(snapshot / "fla_npu/opp")
                data = {"version": version, "adapter": snapshot.name}
                data.update({"opp": None} if version == 1 else {"opps": []})
                (snapshot / "components.json").write_text(json.dumps(data))
                second = layers.install(self.base, self.artifact("next-"+str(version), "npu_b", "full"))
                self.assertEqual(second["components"]["adapters"], [Path(second["path"]).name, snapshot.name])
                self.assertEqual(second["operators"], ["npu_a", "npu_b"])
                layers.remove(self.base)

    def test_missing_older_adapter_is_not_silently_replaced_by_base(self):
        first = layers.install(self.base, self.artifact("older-a", "npu_a", "full"))
        layers.install(self.base, self.artifact("newer-b", "npu_b", "full"))
        (Path(first["path"]) / "fla_npu" / layers.MANIFEST).unlink()
        with self.assertRaisesRegex(RuntimeError, "adapter is missing"):
            layers.runtime_plan(self.base)
        layers.remove(self.base)
        self.assertIsNone(layers.runtime_plan(self.base))

    def test_invalid_adapter_order_is_rejected(self):
        for i, invalid in enumerate((None, "bad", [None], ["../outside"], ["a"*32, "a"*32])):
            with self.subTest(adapters=invalid):
                result = layers.install(self.base, self.artifact("invalid-"+str(i), "npu_a", "full"))
                (Path(result["path"]) / "components.json").write_text(json.dumps(
                    {"version": 2, "adapters": invalid, "opps": []}))
                with self.assertRaises(RuntimeError):
                    layers.runtime_plan(self.base)
                layers.remove(self.base)


if __name__ == "__main__":
    unittest.main()
