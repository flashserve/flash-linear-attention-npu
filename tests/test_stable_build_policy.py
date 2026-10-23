"""Stable-only build invariants, independent of CANN and an NPU device."""

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import setuptools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stable_build_policy as policy


class StableBuildPolicyTest(unittest.TestCase):
    def test_removed_build_modes_fail_in_all_entry_points(self):
        for env in ({"FLA_NPU_BUILD_STABLE_ABI": value} for value in ("0", "false", "no", "off", "typo")):
            self._assert_rejected(env)
        for env in ({"FLA_NPU_BUILD_LEGACY_EXTENSION": "1"},
                    {"FLA_NPU_SKIP_TORCH_GEN": "0"}):
            self._assert_rejected(env)

    def _assert_rejected(self, env):
        for entry in (ROOT / "setup.py", ROOT / "torch_custom/fla_npu/setup.py",
                      ROOT / "scripts/build_wheel.py"):
            with self.subTest(entry=entry, env=env), mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.object(setuptools, "setup"), mock.patch("subprocess.run") as run:
                    with self.assertRaisesRegex(RuntimeError, next(iter(env))):
                        loaded = runpy.run_path(str(entry))
                        if entry.name == "build_wheel.py":
                            loaded["_prepare_abi_free_launcher"]()
                    run.assert_not_called()

    def test_setup_metadata_does_not_compile_or_import_torch(self):
        for relative in ("setup.py", "torch_custom/fla_npu/setup.py"):
            with self.subTest(relative=relative), mock.patch.dict(
                    os.environ, {"FLA_NPU_DISABLE_LOCAL_VERSION": "TRUE"}, clear=True), \
                    mock.patch.dict(sys.modules, {"torch": None, "torch_npu": None}):
                with mock.patch.object(setuptools, "setup") as setup, mock.patch("subprocess.run") as run:
                    runpy.run_path(str(ROOT / relative))
                    run.assert_not_called()
                kwargs = setup.call_args.kwargs
                self.assertTrue(set(policy.RUNTIME_REQUIREMENTS).issubset(kwargs["install_requires"]))
                self.assertEqual(kwargs["python_requires"], ">=3.9")
                distribution = kwargs["distclass"]()
                self.assertFalse(distribution.is_pure())
                self.assertFalse(distribution.has_ext_modules())

    def test_standalone_wheel_uses_platform_tag(self):
        with mock.patch.object(setuptools, "setup") as setup:
            loaded = runpy.run_path(str(ROOT / "torch_custom/fla_npu/setup.py"))
        kwargs = setup.call_args.kwargs
        with mock.patch.dict(os.environ, {"FLA_NPU_ARCH": "aarch64"}):
            wheel = loaded["StableBdistWheel"](kwargs["distclass"]({"name": "test", "version": "0"}))
            wheel.finalize_options()
            self.assertEqual(wheel.get_tag(), ("py3", "none", "manylinux_2_34_aarch64"))

    def test_stale_legacy_extension_is_rejected_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            old = package / "custom_aclnn_extension_lib.cpython-311-aarch64-linux-gnu.so"
            old.write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "stale legacy extension"):
                policy.reject_legacy_artifacts(package)
            self.assertEqual(old.read_bytes(), b"old")

    def test_missing_or_empty_launcher_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            for required in policy.LAUNCHER_FILES:
                path = package / required
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SOURCE_HASH = '0123456789abcdef0123456789abcdef'\n"
                                 if required.endswith("_stable_hash.py") else b"payload")
            policy.validate_launcher_payload(package)
            for required in policy.LAUNCHER_FILES:
                path = package / required
                for payload in (None, b""):
                    with self.subTest(required=required, payload=payload):
                        if payload is None:
                            path.unlink()
                        else:
                            path.write_bytes(payload)
                        with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                            policy.validate_launcher_payload(package)
                        path.write_bytes(b"SOURCE_HASH = '0123456789abcdef0123456789abcdef'\n"
                                         if required.endswith("_stable_hash.py") else b"payload")

    def test_stamp_must_be_one_literal_source_fingerprint(self):
        for content in (b"", b"# no stamp", b"SOURCE_HASH = 'unknown'", b"SOURCE_HASH = 'deadbeef'",
                        b"SOURCE_HASH = 0", b"SOURCE_HASH = open('secret').read()",
                        b"SOURCE_HASH = ('0' * 32)", b"SOURCE_HASH = 'not hex'", b"not python!", b"\xff"):
            with self.subTest(content=content):
                with self.assertRaisesRegex(RuntimeError, "Invalid Stable-ABI build stamp"):
                    policy.validate_build_stamp(content)
        valid = b"SOURCE_HASH = '0123456789abcdef0123456789abcdef'\n"
        self.assertEqual(policy.validate_build_stamp(valid), "0123456789abcdef0123456789abcdef")
        with self.assertRaisesRegex(RuntimeError, "Invalid Stable-ABI build stamp"):
            policy.validate_build_stamp(valid + valid)

    def test_missing_builder_cannot_generate_python_only_wheel(self):
        loaded = runpy.run_path(str(ROOT / "scripts/build_wheel.py"))
        with tempfile.TemporaryDirectory() as tmp:
            loaded["_prepare_abi_free_launcher"].__globals__["REPO_ROOT"] = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "builder is missing"):
                loaded["_prepare_abi_free_launcher"]()

    def test_removed_cli_options_fail_before_environment_checks(self):
        for script, option in (("check_npu_env.py", "--legacy-extension"),
                               ("check_npu_env.py", "--skip-torchnpugen"),
                               ("check_pypi_wheel.py", "--without-launcher")):
            with self.subTest(option=option):
                args = ["missing.whl"] if script == "check_pypi_wheel.py" else []
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args, option],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(option, result.stderr)

    def test_ci_does_not_require_removed_generator(self):
        for relative in ("ci/Dockerfile", "ci/Dockerfile.ascend950", "ci/run_checks.sh"):
            self.assertNotIn("torchnpugen", (ROOT / relative).read_text(encoding="utf-8"))

    def test_build_only_preflight_needs_torch_but_no_npu_or_codegen(self):
        loaded = runpy.run_path(str(ROOT / "scripts/check_npu_env.py"))
        checks = (
            "_check_cmake_version", "_check_gcc_version", "_check_make_exists",
            "_check_patch_exists", "_check_bisheng_exists", "_check_setuptools_version",
            "_check_build_system_deps", "_report_cann_requirement", "_report_validated_combo",
        )
        replacements = {name: mock.Mock() for name in checks}
        imported = mock.Mock(return_value=types.SimpleNamespace(__version__="2.7.1"))
        replacements.update(_import_module=imported, _detect_cann_version=lambda: "9.2.0")
        with mock.patch.dict(loaded["main"].__globals__, replacements), \
                mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": "/fake/cann"}, clear=True), \
                mock.patch.object(sys, "argv", ["check_npu_env.py", "--build-only"]), \
                mock.patch("shutil.which", return_value="/bin/tool"):
            self.assertEqual(loaded["main"](), 0)
        self.assertEqual([call.args[1] for call in imported.call_args_list], ["torch"])

    def test_standalone_wheel_stages_required_payload_and_metadata(self):
        # Exercise setuptools, not just its keyword arguments. A deterministic
        # compiler stub isolates staging from CANN and the host C++ toolchain.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standalone = root / "torch_custom/fla_npu"
            package = standalone / "fla_npu"
            (root / "scripts").mkdir()
            (root / "fla").mkdir()
            shutil.copy2(ROOT / "fla/__init__.py", root / "fla/__init__.py")
            for name in ("stable_build_policy.py", "fla_npu_artifacts.py", "npu_compat.py"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            for relative in ("", "ops", "ops/ascendc"):
                directory = package / relative
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "__init__.py").write_text("", encoding="utf-8")
            shutil.copy2(ROOT / "torch_custom/fla_npu/setup.py", standalone / "setup.py")
            (standalone / "csrc").mkdir()
            (standalone / "csrc/build_stable.py").write_text(
                "import sys\nfrom pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
                "out.write_bytes(b'compiler fixture')\n"
                "(out.parent / 'ops/ascendc/_stable_hash.py').write_text(\"SOURCE_HASH = '0123456789abcdef0123456789abcdef'\\n\")\n",
                encoding="utf-8",
            )
            env = dict(os.environ, FLA_NPU_DISABLE_LOCAL_VERSION="TRUE", FLA_NPU_ARCH="x86_64",
                       FLA_NPU_SOC="ascend950")
            result = subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=standalone,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheels = list((standalone / "dist").glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            self.assertIn("-py3-none-manylinux_2_34_x86_64.whl", wheels[0].name)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                for required in (*policy.LAUNCHER_FILES, "_build_meta.py", "_compat.py"):
                    self.assertTrue(any(name.endswith("fla_npu/" + required) for name in names), required)
                metadata = archive.read(next(name for name in names if name.endswith("/METADATA"))).decode()
                for requirement in policy.RUNTIME_REQUIREMENTS:
                    self.assertIn("Requires-Dist: " + requirement, metadata)
                self.assertFalse(any("custom_aclnn_extension_lib" in name for name in names))

    def test_skip_build_rejects_unvalidated_staging_in_both_setup_entries(self):
        # Run the actual setuptools command against stale build/lib contents.
        # --skip-build used to bypass both build_py validation implementations.
        for relative in ("setup.py", "torch_custom/fla_npu/setup.py"):
            with self.subTest(entry=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("setup.py", "torch_custom/fla_npu/setup.py", "fla/__init__.py",
                             "scripts/stable_build_policy.py", "scripts/fla_npu_artifacts.py",
                             "scripts/npu_compat.py", "scripts/check_build_ops.py", "README.md"):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / name, target)
                for name in ("torch_custom/fla_npu/fla_npu", "fla/ops/triton/triton_core"):
                    package = root / name
                    package.mkdir(parents=True, exist_ok=True)
                    (package / "__init__.py").write_text("", encoding="utf-8")
                entry = root / relative
                staged = entry.parent / "build/lib/fla_npu"
                staged.mkdir(parents=True)
                (staged / "__init__.py").write_text("", encoding="utf-8")
                legacy = staged / "custom_aclnn_extension_lib.so"
                legacy.write_bytes(b"stale legacy library")
                env = dict(os.environ, FLA_NPU_DISABLE_LOCAL_VERSION="TRUE",
                           FLA_NPU_SOC="ascend950", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
                result = subprocess.run(
                    [sys.executable, "setup.py", "bdist_wheel", "--skip-build"],
                    cwd=entry.parent, env=env, capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--skip-build is not supported", result.stdout + result.stderr)
                self.assertFalse(list(root.rglob("*.whl")))
                self.assertEqual(legacy.read_bytes(), b"stale legacy library")


if __name__ == "__main__":
    unittest.main()
