"""Exercise experimental install and stable-wheel rejection without a CANN install."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from experimental_fixtures import ABI_HELPER, write_abi


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER_DIR = REPO_ROOT / "experimental/cmake/scripts/experimental"
VENDOR = "fla_npu_experimental_transformer"
MANIFEST = "experimental-build-manifest.json"


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class ExperimentalPackageIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.package = self.root / "package"
        self.package.mkdir()
        for name in ("install.sh", "install_experimental.py", "help.info"):
            shutil.copyfile(INSTALLER_DIR / name, self.package / name)
        shutil.copy2(ABI_HELPER, self.package / "experimental_op_abi.py")
        self.vendor = self.package / "packages" / "vendors" / VENDOR
        lib = self.vendor / "op_api" / "lib" / "libcust_opapi.so"
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"experimental fixture library")
        write_abi(self.vendor)
        (self.vendor / MANIFEST).write_text(json.dumps(
            {"mode": "experimental", "vendor": VENDOR, "source_commit": "fixture"}))
        self.stable = self.root / "stable" / "opp"
        stable_lib = self.stable / "vendors" / "fla_npu_transformer" / "op_api" / "lib" / "libcust_opapi.so"
        stable_lib.parent.mkdir(parents=True)
        stable_lib.write_bytes(b"stable library must stay unchanged")
        (self.stable / "vendors" / "config.ini").write_text("load_priority=fla_npu_transformer\n")
        self.stable_before = _snapshot(self.stable)
        self.env = {name: value for name, value in os.environ.items()
                    if not name.startswith(("ASCEND_", "FLA_NPU_"))}
        self.env["ASCEND_OPP_PATH"] = str(self.stable)

    def tearDown(self):
        self.assertEqual(_snapshot(self.stable), self.stable_before)

    def _run(self, *args):
        result = subprocess.run(
            [sys.executable, "-S", str(self.package / "install_experimental.py"), *map(str, args)],
            capture_output=True, text=True, env=self.env,
        )
        return result

    def _link(self, target, link, directory=True):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")

    def test_installs_only_experimental_vendor_and_environment(self):
        destination = self.root / "isolated-opp"
        result = self._run(f"--install-path={destination}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in (destination / "vendors").iterdir()}, {VENDOR, "config.ini"})
        self.assertEqual((destination / "vendors" / VENDOR / MANIFEST).read_bytes(),
                         (self.vendor / MANIFEST).read_bytes())
        environment = (destination / "vendors" / VENDOR / "bin" / "set_env.bash").read_text()
        self.assertIn("FLA_NPU_EXPERIMENTAL_OPP_PATH=", environment)
        self.assertIn('export ASCEND_CUSTOM_OPP_PATH="${_fla_exp_vendor}"', environment)
        self.assertIn("export FLA_NPU_DISABLE_PTH=1", environment)
        self.assertNotIn("export FLA_NPU_OPP_PATH=", environment)
        self.assertNotIn("FLA_NPU_OP_API_LIB=", environment)

    @unittest.skipUnless(os.name == "posix", "POSIX run-package installer")
    def test_existing_empty_directory_is_supported(self):
        destination = self.root / "empty"
        destination.mkdir()
        result = self._run(f"--install-path={destination}")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_corrupted_library_and_missing_abi_manifest_are_rejected(self):
        library = self.vendor / "op_api/lib/libcust_opapi.so"
        library.write_bytes(b"tampered")
        destination = self.root / "isolated"
        self.assertNotEqual(self._run(f"--install-path={destination}").returncode, 0)
        self.assertFalse(destination.exists())
        (self.vendor / "abi_manifest.json").unlink()
        self.assertNotEqual(self._run(f"--install-path={destination}").returncode, 0)
        self.assertFalse(destination.exists())

    def test_default_and_wheel_modes_are_rejected(self):
        for args in ([], ["--full"], ["--install"], ["--cann"], ["--upgrade"],
                     [f"--install-path={self.root / 'new'}", "--full"],
                     [f"--install-path={self.root / 'new'}", "--install"]):
            with self.subTest(args=args):
                self.assertNotEqual(self._run(*args).returncode, 0)
        self.assertFalse((self.root / "new").exists())

    def test_existing_and_active_opp_roots_are_rejected(self):
        for path in (self.stable, self.stable / "empty", self.stable / "vendors" / "new"):
            with self.subTest(path=path):
                self.assertNotEqual(self._run(f"--install-path={path}").returncode, 0)
                self.assertNotIn("fla_npu_experimental", str(_snapshot(self.stable)))

    def test_inactive_opp_ancestor_is_rejected(self):
        self.env.pop("ASCEND_OPP_PATH")
        result = self._run(f"--install-path={self.stable / 'empty'}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing OPP", result.stderr)

    def test_empty_sibling_of_existing_opp_is_allowed(self):
        destination = self.stable.parent / "experimental-opp"
        result = self._run(f"--install-path={destination}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((destination / "vendors" / VENDOR / MANIFEST).is_file())

    def test_new_directory_inside_stable_python_package_is_rejected(self):
        stable_package = self.root / "site-packages" / "fla_npu"
        stable_package.mkdir(parents=True)
        (stable_package / "__init__.py").write_text("# stable package\n")
        destination = stable_package / "experimental-opp"
        before = _snapshot(stable_package)
        result = self._run(f"--install-path={destination}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stable package", result.stderr)
        self.assertEqual(_snapshot(stable_package), before)

    def test_nonempty_directory_is_rejected_without_modification(self):
        destination = self.root / "nonempty"
        destination.mkdir()
        (destination / "sentinel").write_text("do not overwrite")
        before = _snapshot(destination)
        self.assertNotEqual(self._run(f"--install-path={destination}").returncode, 0)
        self.assertEqual(_snapshot(destination), before)

    def test_relative_and_missing_parent_paths_are_rejected(self):
        for path in ("relative", self.root / "missing" / "child"):
            self.assertNotEqual(self._run(f"--install-path={path}").returncode, 0)
        self.assertFalse((self.root / "missing").exists())

    def test_target_symlink_and_ancestor_symlink_are_rejected(self):
        alias = self.root / "alias"
        self._link(self.stable, alias)
        for path in (alias, alias / "empty"):
            result = self._run(f"--install-path={path}")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("symlink", result.stderr)

    def test_escaping_payload_symlink_is_rejected(self):
        self._link(self.stable, self.vendor / "escape")
        destination = self.root / "isolated"
        result = self._run(f"--install-path={destination}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink escapes", result.stderr)
        self.assertFalse(destination.exists())

    def test_additional_stable_vendor_is_rejected(self):
        (self.vendor.parent / "fla_npu_transformer").mkdir()
        result = self._run(f"--install-path={self.root / 'isolated'}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected vendor", result.stderr)

    def test_stable_metadata_is_rejected(self):
        (self.vendor / MANIFEST).write_text(json.dumps({"mode": "stable"}))
        result = self._run(f"--install-path={self.root / 'isolated'}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mode=experimental", result.stderr)

    def test_repeat_install_is_rejected_and_original_is_preserved(self):
        destination = self.root / "isolated"
        self.assertEqual(self._run(f"--install-path={destination}").returncode, 0)
        before = _snapshot(destination)
        self.assertNotEqual(self._run(f"--install-path={destination}").returncode, 0)
        self.assertEqual(_snapshot(destination), before)

    def _makeself_context(self, name="fla_npu_experimental_linux-x86_64.run"):
        archive = self.root / name
        archive.write_text("fixture makeself archive")
        self.env["USER_PWD"] = str(self.root)
        return ["--" + str(archive), "--" + str(self.root)]

    def test_legacy_makeself_prefix_allows_isolated_install(self):
        prefix = self._makeself_context()
        for index, archive_arg in enumerate((prefix[0], "--fla_npu_experimental_linux-x86_64.run")):
            destination = self.root / f"wrapped-{index}"
            result = self._run(archive_arg, prefix[1], f"--install-path={destination}", "--quiet")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((destination / "vendors" / VENDOR / MANIFEST).is_file())

    def test_renamed_run_archive_allows_isolated_install(self):
        prefix = self._makeself_context("rwkv8-a5-20261023.run")
        destination = self.root / "renamed-archive-install"
        result = self._run(*prefix, f"--install-path={destination}", "--quiet")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((destination / "vendors" / VENDOR / MANIFEST).is_file())

    def test_existing_non_run_file_is_not_accepted_as_wrapper_metadata(self):
        prefix = self._makeself_context("not-an-archive.txt")
        destination = self.root / "invalid-wrapper"
        result = self._run(*prefix, f"--install-path={destination}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)
        self.assertFalse(destination.exists())

    def test_legacy_makeself_prefix_does_not_allow_overlay_or_unknown_options(self):
        prefix = self._makeself_context()
        destination = self.root / "wrapped"
        for option in ("--full", "--install", "--cann", "--upgrade", "--unknown-option"):
            with self.subTest(option=option):
                result = self._run(*prefix, f"--install-path={destination}", option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unrecognized arguments", result.stderr)
        self.assertFalse(destination.exists())

    def test_unverified_or_malformed_makeself_prefix_is_rejected(self):
        prefix = self._makeself_context()
        destination = self.root / "wrapped"
        invalid_prefixes = ([prefix[0]], prefix[::-1],
                            [prefix[0], "--" + str(self.package)],
                            ["--" + str(self.root / "missing.run"), prefix[1]],
                            [*prefix, *prefix])
        for invalid in invalid_prefixes:
            with self.subTest(prefix=invalid):
                self.assertNotEqual(self._run(*invalid, f"--install-path={destination}").returncode, 0)
        self.env.pop("USER_PWD")
        self.assertNotEqual(self._run(*prefix, f"--install-path={destination}").returncode, 0)
        self.assertFalse(destination.exists())

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires bash")
    def test_shell_entrypoint_accepts_actual_makeself_argument_shape(self):
        prefix = self._makeself_context()
        destination = self.root / "wrapped"
        result = subprocess.run(
            ["bash", str(self.package / "install.sh"), *prefix,
             "--quiet", f"--install-path={destination}"],
            capture_output=True, text=True, env=self.env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((destination / "vendors" / VENDOR / MANIFEST).is_file())

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires bash")
    def test_shell_entrypoint_rejects_overlay(self):
        result = subprocess.run(["bash", str(self.package / "install.sh"), "--full"],
                                capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--install-path", result.stderr)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires bash")
    def test_environment_is_idempotent_and_does_not_load_stable_vendor(self):
        destination = self.root / "isolated"
        self.assertEqual(self._run(f"--install-path={destination}").returncode, 0)
        environment = destination / "vendors" / VENDOR / "bin" / "set_env.bash"
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; source "$1"; printf "%s\\n%s\\n%s" '
             '"$ASCEND_CUSTOM_OPP_PATH" "$FLA_NPU_EXPERIMENTAL_OPP_PATH" "$FLA_NPU_DISABLE_PTH"',
             "fixture", str(environment)], capture_output=True, text=True,
            env={**self.env, "ASCEND_CUSTOM_OPP_PATH": str(self.stable)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(destination / "vendors" / VENDOR),
                                                     str(destination), "1"])




if __name__ == "__main__":
    unittest.main()
