"""Execute the formal wheel's real artifact selection against experimental output."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class FormalPackageBoundary(unittest.TestCase):
    def test_experimental_output_cannot_replace_formal_wheel_payload(self):
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"_build_run_package", "_find_single_run_package"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "build_out"
            output.mkdir()
            (output / "fla_npu_linux-aarch64.run").write_bytes(b"stale formal package")

            def build(command, cwd):
                self.assertIn("--experimental", command)
                self.assertFalse(output.exists(), "previous formal artifact must be removed")
                output.mkdir()
                (output / "fla_npu_experimental_linux-aarch64.run").write_bytes(b"experimental output")

            scope = dict(REPO_ROOT=root, os=os, shlex=shlex, shutil=shutil,
                         get_soc=lambda: "ascend910b", _check_ops_filter=lambda _: None,
                         DEFAULT_VENDOR_NAME="fla_npu", _run=build)
            exec(compile(ast.Module(body=functions, type_ignores=[]), "setup.py", "exec"), scope)
            with patch.dict(os.environ, FLA_NPU_BUILD_ARGS="--experimental", FLA_NPU_OPS=""):
                with self.assertRaisesRegex(RuntimeError, "No fla_npu_linux"):
                    scope["_build_run_package"]()

    def test_formal_sources_and_ci_have_no_experimental_dispatch(self):
        paths = ["scripts/check_build_ops.py", "scripts/stable_build_policy.py",
                 "torch_custom/fla_npu/csrc/src/runtime.cpp", "torch_custom/fla_npu/csrc/build_stable.py"]
        for path in paths:
            self.assertNotIn("experimental", (ROOT / path).read_text(encoding="utf-8"), path)
        for path in [*(ROOT / ".github/workflows").glob("*.yml"), *(ROOT / "ci").glob("*.sh")]:
            self.assertNotIn("suite=experimental", path.read_text(encoding="utf-8"), str(path))


class OneClickBoundaryTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("one_click_builder", ROOT / "scripts/build_wheel.py")
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for context in (patch.object(self.builder, "REPO_ROOT", self.root),
                        patch.dict(os.environ, {}, clear=True),
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def invoke(self, *args):
        with patch.object(sys, "argv", ["build_wheel.py", "--no-progress", *args]):
            return self.builder.main()

    def test_default_uses_only_formal_pipeline(self):
        def run(command, env, progress, **kwargs):
            self.assertEqual(command[1:4], ["-m", "pip", "wheel"])
            self.assertNotIn("experimental", " ".join(command))
            (Path(command[-1]) / "flash_linear_attention_npu_a2-1-py3-none-any.whl").write_bytes(b"formal")
            return 0
        with patch.object(self.builder, "_prepare_abi_free_launcher") as prepare, \
             patch.object(self.builder, "_inject_runtime_pins") as pins, \
             patch.object(self.builder, "_run_with_progress", side_effect=run):
            self.assertEqual(self.invoke(), 0)
        prepare.assert_called_once()
        pins.assert_called_once()

    def test_experimental_dispatch_skips_formal_launcher_and_forwards_options(self):
        with patch.dict(os.environ, FLA_NPU_SOC="ascend910b", FLA_NPU_OPS="fused_recurrent_rwkv8",
                        FLA_NPU_BUILD_ARGS="-j8"):
            def run(command, env, progress, **kwargs):
                self.assertEqual(command[1:3], ["experimental/_framework/build.py", "package"])
                self.assertEqual(env["FLA_NPU_OPS"], "fused_recurrent_rwkv8")
                self.assertIn("-j8", env["FLA_NPU_BUILD_ARGS"])
                self.assertIn("-O2", env["FLA_NPU_BUILD_ARGS"])
                self.assertIn("--bisheng_flags=ccec_g,sanitizer,oom", env["FLA_NPU_BUILD_ARGS"])
                self.assertEqual(env["FLA_NPU_DISABLE_PTH"], "1")
                self.assertEqual(kwargs["log_path"].name, "fla-npu-experimental-build.log")
                (Path(command[-1]) / "flash_linear_attention_npu_experimental_a2-1-py3-none-any.whl").write_bytes(b"experimental")
                return 0
            output = self.root / "custom wheels"
            output.mkdir()
            stale = output / "flash_linear_attention_npu_a2-old.whl"
            stale.write_bytes(b"old formal")
            with patch.object(self.builder, "_prepare_abi_free_launcher") as prepare, \
                 patch.object(self.builder, "_inject_runtime_pins") as pins, \
                 patch.object(self.builder, "_run_with_progress", side_effect=run):
                self.assertEqual(self.invoke("--experimental", "--wheel-dir", str(output),
                                            "--build-args=-O2", "--debug", "--sanitizer", "--oom"), 0)
            prepare.assert_not_called()
            pins.assert_not_called()
            self.assertEqual(stale.read_bytes(), b"old formal")
            self.assertEqual(len(list(output.glob("*experimental*.whl"))), 1)

    def test_hidden_mode_switch_fails_before_any_build(self):
        for cli, env in ((["--build-args=--experimental"], {}),
                         ([], {"FLA_NPU_BUILD_ARGS": "--experimental"})):
            with self.subTest(cli=cli), patch.dict(os.environ, env), \
                 patch.object(self.builder, "_prepare_abi_free_launcher") as prepare, \
                 patch.object(self.builder, "_run_with_progress") as run:
                with self.assertRaises(SystemExit) as error:
                    self.invoke(*cli)
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()
                run.assert_not_called()

    def test_failure_or_wrong_product_does_not_publish_a_wheel(self):
        for returncode in (7, 0):
            def run(command, env, progress, **kwargs):
                (Path(command[-1]) / "flash_linear_attention_npu_a2-wrong.whl").write_bytes(b"formal")
                return returncode
            with self.subTest(returncode=returncode), \
                 patch.object(self.builder, "_run_with_progress", side_effect=run):
                with self.assertRaises((RuntimeError, self.builder.subprocess.CalledProcessError)):
                    self.invoke("--experimental")
            self.assertEqual(list((self.root / "dist").glob("*.whl")), [])

    def test_experimental_subprocess_log_and_exit_code_are_preserved(self):
        log = self.root / "build.log"
        progress = self.builder._BuildProgress(False, False)
        result = self.builder._run_with_progress(
            [sys.executable, "-c", "print('build diagnostic'); raise SystemExit(7)"],
            os.environ.copy(), progress, log_path=log)
        self.assertEqual(result, 7)
        self.assertIn("build diagnostic", log.read_text())
