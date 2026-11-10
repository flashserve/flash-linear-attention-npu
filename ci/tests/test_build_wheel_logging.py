"""Verify optional backend streaming without compiling NPU operators."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_wheel_logging", ROOT / "scripts/build_wheel.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BuildWheelLoggingTest(unittest.TestCase):
    def test_cli_overrides_environment_without_changing_build_arguments(self):
        cases = [
            ([], "", False), ([], "true", True), ([], "false", False),
            (["--verbose"], "false", True), (["--no-verbose"], "true", False),
        ]
        for flags, setting, verbose in cases:
            with self.subTest(flags=flags, setting=setting), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                with mock.patch.dict(os.environ, {"FLA_NPU_BUILD_VERBOSE": setting,
                                                  "FLA_NPU_BUILD_ARGS": "-j16"}), \
                     mock.patch.object(sys, "argv", ["build_wheel.py", "--wheel-dir", str(output), *flags]), \
                     mock.patch.object(BUILD, "_prepare_abi_free_launcher"), \
                     mock.patch.object(BUILD, "_run_with_progress", return_value=23) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError) as raised:
                        BUILD.main()
                self.assertEqual(raised.exception.returncode, 23)
                command, env, _ = run.call_args.args
                self.assertEqual("-v" in command, verbose)
                self.assertEqual(env["FLA_NPU_BUILD_ARGS"], "-j16")
                self.assertEqual(command[command.index("--log") + 1], str(output / "fla-npu-build.log"))
                self.assertEqual(run.call_args.kwargs["follow_path"],
                                 None if verbose else output / "fla-npu-build.log")

    def test_real_verbose_pip_preserves_early_backend_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
                encoding="utf-8",
            )
            (root / "backend.py").write_text(
                "from pathlib import Path\n"
                "def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):\n"
                "    directory = Path(metadata_directory) / 'ci_log_fixture-1.0.dist-info'\n"
                "    directory.mkdir()\n"
                "    (directory / 'METADATA').write_text('Metadata-Version: 2.1\\nName: ci-log-fixture\\nVersion: 1.0\\n')\n"
                "    return directory.name\n"
                "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
                "    print('fixture.cpp:7: error: ORIGINAL_BACKEND_ERROR', flush=True)\n"
                "    print('[ 20%] Built target fixture', flush=True)\n"
                "    for index in range(100): print('later build output', index, flush=True)\n"
                "    raise RuntimeError('fixture build failed')\n",
                encoding="utf-8",
            )
            log = root / "build.log"
            console = io.StringIO()
            progress = BUILD._BuildProgress(enabled=False, announce=False, stream=console)
            command = [sys.executable, "-m", "pip", "wheel", "-v", "--no-index",
                       "--disable-pip-version-check", "--no-build-isolation", "--no-deps",
                       "--log", str(log), ".", "-w", str(root / "dist")]
            with mock.patch.object(BUILD, "REPO_ROOT", root):
                result = BUILD._run_with_progress(command, dict(os.environ), progress)
            self.assertNotEqual(result, 0)
            self.assertIn("fixture.cpp:7: error: ORIGINAL_BACKEND_ERROR", console.getvalue())
            raw = log.read_text(encoding="utf-8")
            self.assertIn("ORIGINAL_BACKEND_ERROR", raw)
            self.assertNotIn("ORIGINAL_BACKEND_ERROR", "\n".join(raw.splitlines()[-40:]))


if __name__ == "__main__":
    unittest.main()
