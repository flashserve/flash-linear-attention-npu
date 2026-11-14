#!/usr/bin/env python3
"""The installed-OPP manifest check has to compare installed paths only.

``bdist_wheel`` moves the whole payload under ``<dist>-<version>.data/<scheme>/``
when a distribution reports ``root_is_pure == False`` without extension modules,
and pip maps that back onto the same site root.  The archive layout is therefore
not part of the comparison: the check has to fold the scheme directory away and
keep comparing the installed tree byte for byte.
"""

import runpy
import json
import os
import shlex
import shutil
import sys
import tempfile
import unittest
import zipfile
from functools import partial
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_install_workflows.py"
DIST_INFO = "flash_linear_attention_npu_a2-1.0.dist-info"

OPP_FILES = {
    "fla_npu/opp/vendors/config.ini": b"load_priority=fla_npu_transformer\n",
    "fla_npu/opp/vendors/fla_npu_transformer/README.txt": b"readme\n",
    "fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash": b"export A=1\n",
    "fla_npu/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so": b"\x7fELF",
}


def _write_wheel(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{DIST_INFO}/METADATA",
            "Metadata-Version: 2.1\nName: flash-linear-attention-npu-a2\nVersion: 1.0\n",
        )
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr(f"{DIST_INFO}/RECORD", f"{DIST_INFO}/RECORD,,\n")


def _data_layout_entries(entries):
    return {
        f"flash_linear_attention_npu_a2-1.0.data/purelib/{name}": payload
        for name, payload in entries.items()
    }


def _write_installed_tree(site_root, entries):
    for name, payload in entries.items():
        path = site_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


class InstalledOppManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = runpy.run_path(str(CHECKER))

    def _assert_layout_accepted(self, wheel_entries):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wheel = root / "flash_linear_attention_npu_a2-1.0-py3-none-any.whl"
            _write_wheel(wheel, wheel_entries)
            site_root = root / "site-packages"
            _write_installed_tree(site_root, OPP_FILES)
            package_dir = site_root / "fla_npu"

            self.assertEqual(
                self.checker["_manifest_from_wheel"](wheel),
                self.checker["_manifest_from_directory"](package_dir),
            )
            self.checker["_assert_manifest_matches_wheel"](package_dir, wheel)

    def _failure_for(self, wheel_entries, installed_entries):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wheel = root / "flash_linear_attention_npu_a2-1.0-py3-none-any.whl"
            _write_wheel(wheel, wheel_entries)
            site_root = root / "site-packages"
            _write_installed_tree(site_root, installed_entries)
            with self.assertRaises(AssertionError) as context:
                self.checker["_assert_manifest_matches_wheel"](
                    site_root / "fla_npu", wheel
                )
            return str(context.exception)

    def test_root_layout_matches_installed_tree(self):
        self._assert_layout_accepted(OPP_FILES)

    def test_data_purelib_layout_matches_installed_tree(self):
        self._assert_layout_accepted(_data_layout_entries(OPP_FILES))

    def test_data_platlib_layout_matches_installed_tree(self):
        self._assert_layout_accepted(
            {
                name.replace(".data/purelib/", ".data/platlib/"): payload
                for name, payload in _data_layout_entries(OPP_FILES).items()
            }
        )

    def test_dot_slash_prefixed_entries_match_installed_tree(self):
        self._assert_layout_accepted(
            {f"./{name}": payload for name, payload in OPP_FILES.items()}
        )

    def test_payload_outside_the_known_schemes_is_still_reported(self):
        wheel_entries = {
            name.replace(".data/purelib/", ".data/data/"): payload
            for name, payload in _data_layout_entries(OPP_FILES).items()
        }
        message = self._failure_for(wheel_entries, OPP_FILES)
        # Only ``purelib``/``platlib`` map onto the site root; any other scheme
        # keeps the payload outside the installed OPP tree and must not pass.
        self.assertIn("missing=[]", message)
        self.assertIn("unexpected=['fla_npu/opp/vendors/config.ini'", message)

    def test_missing_installed_opp_file_is_reported(self):
        installed = dict(OPP_FILES)
        installed.pop("fla_npu/opp/vendors/config.ini")
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("missing=['fla_npu/opp/vendors/config.ini']", message)

    def test_extra_installed_opp_file_is_still_reported(self):
        installed = dict(OPP_FILES)
        installed["fla_npu/opp/vendors/fla_npu_transformer/extra.bin"] = b"x"
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("unexpected=['fla_npu/opp/vendors/fla_npu_transformer/extra.bin']", message)

    def test_changed_installed_opp_file_is_still_reported(self):
        installed = dict(OPP_FILES)
        installed["fla_npu/opp/vendors/config.ini"] = b"load_priority=other\n"
        message = self._failure_for(_data_layout_entries(OPP_FILES), installed)
        self.assertIn("changed=['fla_npu/opp/vendors/config.ini']", message)


class SetEnvContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = runpy.run_path(str(CHECKER))

    def _paths(self, root):
        package_dir = root / "site-packages" / "fla_npu"
        vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
        set_env = vendor_dir / "bin" / "set_env.bash"
        set_env.parent.mkdir(parents=True)
        set_env.write_text("# test fixture\n", encoding="utf-8")
        op_api_lib = str(vendor_dir / "op_api" / "lib" / "libcust_opapi.so")
        return package_dir, vendor_dir, op_api_lib

    def _check_snapshots(self, package_dir, first, second):
        output = "\n".join([*first, *second]) + "\n"
        with mock.patch.object(
            self.checker["subprocess"], "check_output", return_value=output
        ):
            self.checker["_assert_set_env_idempotent"](
                package_dir, {}, package_dir.parent
            )

    def test_accepts_current_root_based_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir, vendor_dir, op_api_lib = self._paths(Path(temp_dir))
            snapshot = [
                f"{vendor_dir.parent.parent}:{vendor_dir}",
                "",
                str(vendor_dir.parent.parent),
                op_api_lib,
                "",
            ]
            self._check_snapshots(package_dir, snapshot, snapshot)

    def test_accepts_ordered_vendor_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir, vendor_dir, op_api_lib = self._paths(Path(temp_dir))
            snapshot = [
                str(vendor_dir),
                "",
                str(vendor_dir.parent.parent),
                op_api_lib,
                op_api_lib,
            ]
            self._check_snapshots(package_dir, snapshot, snapshot)

    def test_rejects_second_source_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir, vendor_dir, op_api_lib = self._paths(Path(temp_dir))
            first = [
                str(vendor_dir),
                "",
                str(vendor_dir.parent.parent),
                op_api_lib,
                op_api_lib,
            ]
            second = [*first[:-1], "/wrong/libcust_opapi.so"]
            with self.assertRaisesRegex(AssertionError, "not idempotent"):
                self._check_snapshots(package_dir, first, second)

    def test_accepts_overlay_then_base_and_rejects_reversed_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir, base, library = self._paths(Path(temp_dir))
            overlay = package_dir / "opp/wheel_layers/id/fla_npu/opp/vendors/fla_npu_transformer"
            snapshot = [os.pathsep.join(map(str, [overlay, base])), "",
                        str(base.parent.parent), library,
                        os.pathsep.join([str(overlay / "op_api/lib/libcust_opapi.so"), library])]
            for expected, passes in (([overlay, base], True), ([base, overlay], False)):
                with mock.patch.object(self.checker["subprocess"], "check_output",
                                       return_value="\n".join(snapshot * 2) + "\n"):
                    if passes:
                        self.checker["_assert_set_env_idempotent"](package_dir, {}, package_dir.parent, vendors=expected)
                    else:
                        with self.assertRaisesRegex(AssertionError, "unexpected environment"):
                            self.checker["_assert_set_env_idempotent"](package_dir, {}, package_dir.parent, vendors=expected)


class ManagedRunWorkflowTest(unittest.TestCase):
    def setUp(self):
        from tests.test_wheel_overlay import package, layers, SETUP
        self.layers = layers
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checker = runpy.run_path(str(CHECKER))
        self.base = package(self.root / "site/fla_npu")  # Real skeleton metadata.
        (self.base / "__init__.py").write_text("")
        shutil.copyfile(SETUP / "fla_npu_wheel_overlay.py", self.base.parent / "fla_npu_wheel_overlay.py")
        source = package(self.root / "source", opp={"aclnnDemo": "int64"})
        self.source_vendor = source / "opp/vendors" / layers.BASE
        self.info = self.base.parent / DIST_INFO
        self.info.mkdir()
        self.record = self.info / "RECORD"
        self.record.write_text("")  # Skeleton has no OPP payload yet.
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=str(self.base.parent), PYTHONDONTWRITEBYTECODE="1",
                        FLA_NPU_DISABLE_PTH="1")  # The fixture supplies metadata, not a startup hook.
        self.env.pop("FLA_NPU_OVERLAY_VENDOR", None)
        self.env.pop("ASCEND_CUSTOM_OPP_PATH", None)

    def test_active_skeleton_uses_layer_without_changing_base_record(self):
        before = self.checker["_base_snapshot"](self.base)
        self.layers.install(self.base, self.source_vendor)
        vendors = self.checker["_active_run_vendors"](Path(sys.executable), self.env, self.root)
        self.assertEqual(len(vendors), 1)
        self.checker["_assert_record_covers_opp"](self.base, managed=True)
        self.checker["_assert_base_unchanged"](self.base, before)
        with mock.patch.object(self.checker["subprocess"], "check_output", return_value=""):
            self.checker["_assert_opp_layout"](self.base, require_runtime=True, vendors=vendors)
        with self.assertRaisesRegex(AssertionError, "missing packaged op_api"):
            self.checker["_assert_opp_layout"](self.base, require_runtime=True)

    def test_pending_is_not_mistaken_for_successful_install(self):
        from tests.test_wheel_overlay import package
        source = package(self.root / "incompatible", opp={"aclnnDemo": "bool"})
        subprocess = self.checker["subprocess"]
        # Run the real child process; capture stderr to verify why it failed.
        capture_stderr = partial(subprocess.check_output, stderr=subprocess.PIPE)
        for previous_active in (False, True):
            with self.subTest(previous_active=previous_active):
                if previous_active:
                    result = self.layers.install(self.base, self.source_vendor)
                    self.assertEqual(result["state"], "active")
                active_before = self.layers.state(self.base).get("active")
                result = self.layers.install(self.base, source / "opp/vendors" / self.layers.BASE)
                self.assertEqual(result["state"], "pending")
                state = self.layers.state(self.base)
                self.assertEqual(state["pending"], Path(result["path"]).name)
                self.assertEqual(state.get("active"), active_before)
                with mock.patch.object(subprocess, "check_output", wraps=capture_stderr):
                    with self.assertRaises(subprocess.CalledProcessError) as error:
                        self.checker["_active_run_vendors"](Path(sys.executable), self.env, self.root)
                self.assertIn("Run installation must activate, not just return success",
                              error.exception.stderr)

    def test_repeat_compares_payload_not_snapshot_identity(self):
        first = self.layers.install(self.base, self.source_vendor)
        second = self.layers.install(self.base, self.source_vendor)
        self.assertNotEqual(first["path"], second["path"])
        selected = Path(second["path"]) / "fla_npu/opp/vendors" / self.layers.BASE
        self.assertEqual(self.checker["_payload_manifest"](selected),
                         self.checker["_payload_manifest"](self.source_vendor))
        self.layers.remove(self.base)
        self.checker["_assert_removed_overlay"](self.base)
        (self.base / "opp/wheel_layers/unexpected").write_text("leftover")
        with self.assertRaisesRegex(AssertionError, "left component payloads"):
            self.checker["_assert_removed_overlay"](self.base)

    def test_manager_exclusion_does_not_hide_base_damage_or_unknown_files(self):
        self.layers.install(self.base, self.source_vendor)
        before = self.checker["_base_snapshot"](self.base)
        (self.base / "libfla_npu_stable.so").write_bytes(b"damaged")
        with self.assertRaisesRegex(AssertionError, "changed the base"):
            self.checker["_assert_base_unchanged"](self.base, before)
        (self.base / "opp/unexpected.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(AssertionError, "does not cover"):
            self.checker["_assert_record_covers_opp"](self.base, managed=True)

    def test_record_cannot_absorb_manager_payload(self):
        self.layers.install(self.base, self.source_vendor)
        self.record.write_text("fla_npu/opp/.overlay.lock,,\n")
        with self.assertRaisesRegex(AssertionError, "must not be added"):
            self.checker["_assert_record_covers_opp"](self.base, managed=True)

    def test_record_still_rejects_deleted_base_payload(self):
        self.record.write_text("fla_npu/opp/missing.bin,sha256=x,1\n")
        with self.assertRaisesRegex(AssertionError, "references missing"):
            self.checker["_assert_record_covers_opp"](self.base, managed=True)

    def build_adapter_ci(self):
        source = (REPO_ROOT / "ci/run_checks.sh").read_text(encoding="utf-8")
        names = ["configure_installed_fla_npu_opp", "install_custom_opp_package", "build_pytorch_adapter"]
        env = self.env.copy()
        script = "set -euo pipefail\n"
        # A real Python without site-packages makes the no-wheel case
        # independent of whatever is installed on the test machine.
        bin_dir = self.root / "bin"
        bin_dir.mkdir(exist_ok=True)
        python = bin_dir / "python3"
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
        python.chmod(0o700)
        run = self.root / "fixture.run"
        run.write_text("#!/bin/sh\nexit 0\n")
        run.chmod(0o700)
        env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], CI_TEST_RUN_PACKAGE=str(run))
        script += '''change_scope=full
ci_mode=quick
CI_RUN_TORCH_TESTS=false
find_single_run_package() { printf '%s\\n' "$CI_TEST_RUN_PACKAGE"; }
check_example_python_deps() { :; }
build_torch_custom() { echo WHEEL_BUILD_REACHED; }
'''
        for name in names:
            body = source.split(name + "() {", 1)[1].split("\n}\n", 1)[0]
            script += name + "() {" + body + "\n}\n"
        script += "build_pytorch_adapter\n"
        script += 'python3 -c \'import json,os; print(json.dumps({k:os.environ.get(k) for k in ("ASCEND_CUSTOM_OPP_PATH","FLA_NPU_OP_API_LIBS")}))\'\n'
        return self.checker["subprocess"].run(["bash", "-c", script], env=env, cwd=self.root,
                                              capture_output=True, text=True)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell required")
    def test_ci_first_install_uses_cann_then_reaches_wheel_build(self):
        self.env["PYTHONPATH"] = ""
        self.env["ASCEND_OPP_PATH"] = str(self.source_vendor.parents[1])
        result = self.build_adapter_ci()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("before wheel installation: " + str(self.source_vendor / "op_api/lib"),
                      result.stdout)
        self.assertIn("WHEEL_BUILD_REACHED", result.stdout)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell required")
    def test_ci_configuration_accepts_skeleton_and_preserves_full_base_fallback(self):
        from tests.test_wheel_overlay import vendor
        for full in (False, True):
            with self.subTest(full=full):
                if full:
                    vendor(self.base, {"aclnnDemo": "int64", "aclnnOther": "int64"})
                self.layers.install(self.base, self.source_vendor)
                result = self.build_adapter_ci()
                self.assertEqual(result.returncode, 0, result.stderr)
                values = json.loads(result.stdout.splitlines()[-1])
                vendors = self.layers.runtime_plan(self.base)["vendors"]
                self.assertEqual(len(vendors), 2 if full else 1)
                self.assertEqual(values["ASCEND_CUSTOM_OPP_PATH"], os.pathsep.join(vendors))
                self.assertEqual(values["FLA_NPU_OP_API_LIBS"], os.pathsep.join(
                    str(Path(p) / self.layers.LIBRARY) for p in vendors))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell required")
    def test_ci_configuration_rejects_pending_or_disabled_update(self):
        from tests.test_wheel_overlay import package
        self.layers.install(self.base, self.source_vendor)
        # A valid CANN vendor must not mask an installed wheel's readiness error.
        self.env["ASCEND_OPP_PATH"] = str(self.source_vendor.parents[1])
        self.env["FLA_NPU_OVERLAY_VENDOR"] = ""
        result = self.build_adapter_ci()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Selected OPP is missing or disabled", result.stderr)
        self.assertNotIn("WHEEL_BUILD_REACHED", result.stdout)
        self.env.pop("FLA_NPU_OVERLAY_VENDOR")
        bad = package(self.root / "bad", opp={"aclnnDemo": "bool"})
        self.layers.install(self.base, bad / "opp/vendors" / self.layers.BASE)
        result = self.build_adapter_ci()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Requested OPP is pending; CI must not test an older active/base version",
                      result.stderr)
        self.assertNotIn("WHEEL_BUILD_REACHED", result.stdout)


if __name__ == "__main__":
    unittest.main()
