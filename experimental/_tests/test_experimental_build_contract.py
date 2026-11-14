"""Contract tests for the experimental operator discovery rules."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
from contextlib import redirect_stdout, redirect_stderr
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_build_ops = _load_module(
    "fla_check_build_ops", REPO_ROOT / "experimental/scripts/check_build_ops.py"
)


class ExperimentalBuildContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        operator = self.root / "experimental/ops" / "rwkv_recurrent"
        operator.mkdir(parents=True)
        (operator / "CMakeLists.txt").write_text(
            "add_subdirectory(op_host)\n", encoding="utf-8"
        )
        (operator / "op_host").mkdir()
        (operator / "op_host" / "CMakeLists.txt").write_text(
            "# test operator\n", encoding="utf-8"
        )
        for name in check_build_ops.RESERVED:
            directory = self.root / "experimental/ops" / name
            directory.mkdir()
            (directory / "CMakeLists.txt").write_text("# not an operator")
        reserved = self.root / "experimental" / "_common"
        reserved.mkdir()
        (reserved / "CMakeLists.txt").write_text(
            "# reserved shared input\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_experimental_discovery_uses_ops_directory_only(self):
        legacy = self.root / "experimental/legacy_probe"
        legacy.mkdir()
        (legacy / "CMakeLists.txt").write_text("# outside ops")
        self.assertEqual(
            check_build_ops.discover_supported_ops(self.root),
            ["rwkv_recurrent"],
        )

    @unittest.skipIf(os.name == "nt", "symlink creation requires Windows privileges")
    def test_discovery_rejects_redirected_ops_root(self):
        ops = self.root / "experimental/ops"
        relocated = self.root / "experimental/legacy_ops"
        ops.rename(relocated)
        ops.symlink_to(relocated, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ops root must not redirect"):
            check_build_ops.discover_supported_ops(self.root)

    def test_default_discovery_stays_on_the_stable_tree(self):
        stable_host = self.root / "fla" / "ops" / "ascendc" / "gdn" / "stable_probe" / "op_host"
        stable_host.mkdir(parents=True)
        (stable_host / "CMakeLists.txt").write_text("# stable\n", encoding="utf-8")
        self.assertEqual(check_build_ops.common.discover_supported_ops(self.root), ["stable_probe"])

    def test_flat_name_containing_op_host_is_preserved(self):
        entry = self.root / "experimental/ops" / "op_host_probe"
        entry.mkdir()
        (entry / "CMakeLists.txt").write_text("# operator\n", encoding="utf-8")
        self.assertIn("op_host_probe", check_build_ops.discover_supported_ops(self.root))

    @unittest.skipIf(os.name == "nt", "symlink creation requires Windows privileges")
    def test_discovery_rejects_operator_symlink_into_stable_tree(self):
        stable = self.root / "fla" / "stable_op"
        stable.mkdir(parents=True)
        (stable / "CMakeLists.txt").write_text("# stable\n", encoding="utf-8")
        (self.root / "experimental/ops" / "linked_op").symlink_to(stable, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            check_build_ops.discover_supported_ops(self.root)


class ExperimentalPackageTest(unittest.TestCase):
    def setUp(self):
        self.builder = _load_module("package_builder_test", REPO_ROOT / "experimental/_framework/build.py")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original_module = self.builder.module
        def module(name, path):
            return original_module(name, REPO_ROOT / path.relative_to(self.root))
        wrappers = {name: ("_" + name, name) for name in ("alpha", "beta")}
        for name, (wrapper, _) in wrappers.items():
            op = self.root / "experimental/ops" / name
            (op / "adapter").mkdir(parents=True)
            (op / "CMakeLists.txt").write_text("# fixture")
            (op / "adapter" / f"stable_{name}.cpp").write_text("// fixture")
            package = self.root / "experimental/_python/fla_npu_experimental"
            package.mkdir(parents=True, exist_ok=True)
            (package / (wrapper + ".py")).write_text("# fixture")
        for context in (patch.object(self.builder, "ROOT", self.root),
                        patch.object(self.builder, "module", side_effect=module),
                        patch.object(self.builder, "WRAPPERS", wrappers),
                        patch.dict(os.environ, {}, clear=True),
                        patch.object(os, "uname", return_value=SimpleNamespace(machine="fixture"), create=True),
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)
        self.run_file = self.root / "build_out/fla_npu_experimental_linux-fixture.run"
        self.manifest = self.root / "build/experimental-build-manifest.json"
        for path in (self.run_file, self.manifest):
            path.parent.mkdir(parents=True)
            path.write_text("stale")

    def test_selection_all_multiple_and_extra_argument_precedence(self):
        for ops in ("", "all", "ALL", "alpha,beta"):
            with self.subTest(ops=ops), patch.dict(os.environ, FLA_NPU_OPS=ops):
                self.assertEqual(self.builder.package_options()[:2], ("ascend910b", ["alpha", "beta"]))
        with patch.dict(os.environ, FLA_NPU_SOC="ascend910b", FLA_NPU_OPS="beta",
                        FLA_NPU_BUILD_ARGS="--ops=alpha -n beta,alpha --soc=ascend950 -j8"):
            soc, names, extra = self.builder.package_options()
            self.assertEqual((soc, names), ("ascend950", ["beta", "alpha"]))
            self.assertIn("-j8", extra)

    def test_bad_selection_soc_or_missing_adapter_fails_before_build(self):
        for env in ({"FLA_NPU_OPS": "formal_op"}, {"FLA_NPU_SOC": "ascend910b,ascend950"}):
            with self.subTest(env=env), patch.dict(os.environ, env), \
                 patch.object(self.builder.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    self.builder.build_package(self.root / "dist", "g++")
                run.assert_not_called()
        (self.root / "experimental/ops/alpha/adapter/stable_alpha.cpp").unlink()
        with patch.object(self.builder.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "adapter"):
                self.builder.build_package(self.root / "dist", "g++")
            run.assert_not_called()

    def test_missing_wrapper_fails_before_build(self):
        (self.root / "experimental/_python/fla_npu_experimental/_beta.py").unlink()
        with patch.object(self.builder.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "wrapper"):
                self.builder.build_package(self.root / "dist", "g++")
            run.assert_not_called()

    def test_output_in_build_cleanup_directory_is_rejected(self):
        for name in ("build", "build_out", "output"):
            with self.subTest(name=name), patch.object(self.builder.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "build.sh cleans"):
                    self.builder.build_package(self.root / name / "wheel", "g++")
                run.assert_not_called()

    def test_preflight_failure_does_not_compile_or_remove_previous_products(self):
        with patch.object(self.builder.subprocess, "run", side_effect=RuntimeError("missing dependency")) as run:
            with self.assertRaisesRegex(RuntimeError, "missing dependency"):
                self.builder.build_package(self.root / "dist", "g++")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.run_file.read_text(), "stale")

    def test_successful_build_uses_fresh_run_and_temporary_install(self):
        commands = []
        def run(command, **kwargs):
            commands.append(list(map(str, command)))
            if command[:2] == ["bash", "build.sh"]:
                self.assertFalse(self.run_file.exists())
                self.assertFalse(self.manifest.exists())
                self.assertIn("--experimental", command)
                self.assertIn("--ops=alpha,beta", command)
                self.run_file.write_text("fresh run")
                self.manifest.write_text("fresh manifest")
        install_paths = []
        def wheel(data, install, output, compiler):
            install_paths.append(install)
            self.assertTrue(install.parent.is_dir())
            self.assertEqual(output, self.root / "dist")
        # Missing ATK is only a warning, not a build dependency.
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError), \
             patch.object(self.builder.subprocess, "run", side_effect=run), \
             patch.object(self.builder, "selection", return_value={}), \
             patch.object(self.builder, "build_wheel", side_effect=wheel):
            self.builder.build_package(self.root / "dist", "g++")
        self.assertIn("--build-only", commands[0])
        self.assertTrue(commands[2][2].startswith("--install-path="))
        self.assertFalse(install_paths[0].parent.exists())

    def test_early_exit_build_cannot_reuse_stale_run_or_manifest(self):
        with patch.object(self.builder.subprocess, "run"), patch.object(self.builder, "build_wheel") as wheel:
            with self.assertRaisesRegex(RuntimeError, "did not produce"):
                self.builder.build_package(self.root / "dist", "g++")
            wheel.assert_not_called()


class ExperimentalWheelVersionTest(unittest.TestCase):
    def test_version_tracks_all_payloads_without_changing_launcher_stamp(self):
        builder = _load_module("version_builder_test", REPO_ROOT / "experimental/_framework/build.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "fla_npu_experimental"
            files = [package / "opp/op_kernel/kernel.bin", package / "libfla_npu_experimental_stable.so",
                     package / "_rwkv8.py", package / "layer_manifest.json",
                     root / "fla_npu_experimental_manage.py"]
            stamp = package / "_stable_hash.py"
            for path in [*files, stamp]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unchanged")
            original = builder.wheel_version(package)
            for path in files:
                with self.subTest(payload=path.name):
                    path.write_bytes(b"updated")
                    self.assertNotEqual(builder.wheel_version(package), original)
                    self.assertEqual(stamp.read_bytes(), b"unchanged")
                    path.write_bytes(b"unchanged")
            self.assertEqual(builder.wheel_version(package), original)
            # Renaming a payload file also changes the package identity.
            files[0].rename(files[0].with_name("renamed.bin"))
            self.assertNotEqual(builder.wheel_version(package), original)


if __name__ == "__main__":
    unittest.main()
