"""Contract tests for the experimental operator discovery rules."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
    "fla_check_build_ops", REPO_ROOT / "scripts" / "check_build_ops.py"
)
parse_changed_ops = _load_module(
    "fla_parse_changed_ops", REPO_ROOT / "scripts" / "ci" / "parse_changed_ops.py"
)


class ExperimentalBuildContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        operator = self.root / "experimental" / "rwkv_recurrent"
        operator.mkdir(parents=True)
        (operator / "CMakeLists.txt").write_text(
            "add_subdirectory(op_host)\n", encoding="utf-8"
        )
        (operator / "op_host").mkdir()
        (operator / "op_host" / "CMakeLists.txt").write_text(
            "# test operator\n", encoding="utf-8"
        )
        reserved = self.root / "experimental" / "_common"
        reserved.mkdir()
        (reserved / "CMakeLists.txt").write_text(
            "# reserved shared input\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_experimental_discovery_uses_flat_operator_directories(self):
        self.assertEqual(
            check_build_ops.discover_supported_ops(self.root, experimental=True),
            ["rwkv_recurrent"],
        )

    def test_default_discovery_stays_on_the_stable_tree(self):
        stable_host = self.root / "fla" / "ops" / "ascendc" / "gdn" / "stable_probe" / "op_host"
        stable_host.mkdir(parents=True)
        (stable_host / "CMakeLists.txt").write_text("# stable\n", encoding="utf-8")
        self.assertEqual(check_build_ops.discover_supported_ops(self.root), ["stable_probe"])

    def test_experimental_path_parser_uses_first_directory(self):
        previous_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            changed_file = "experimental/rwkv_recurrent/op_kernel/recurrent.cpp"
            self.assertEqual(
                parse_changed_ops.extract_operator_name(changed_file, "TRUE"),
                "rwkv_recurrent",
            )
        finally:
            os.chdir(previous_cwd)

    def test_reserved_directory_is_not_an_operator(self):
        self.assertEqual(
            parse_changed_ops.extract_operator_name(
                "experimental/_common/kernel.hpp", "TRUE"
            ),
            "",
        )

    def test_non_experimental_change_parser_keeps_legacy_results(self):
        previous_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            (self.root / "experimental" / "attention" / "legacy_probe").mkdir(parents=True)
            self.assertEqual(
                parse_changed_ops.extract_operator_name(
                    "attention/legacy_probe/op_host/def.cpp", "FALSE"
                ),
                "legacy_probe",
            )
            self.assertEqual(
                parse_changed_ops.extract_operator_name(
                    "attention/missing/op_host/def.cpp", "FALSE"
                ),
                "fused_infer_attention_score",
            )
            self.assertEqual(
                parse_changed_ops.extract_operator_name(
                    "fla/ops/ascendc/gdn/stable_probe/op_host/def.cpp", "FALSE"
                ),
                "",
            )
        finally:
            os.chdir(previous_cwd)

    @unittest.skipUnless(shutil.which("bash"), "bash is required for the CI entry point")
    def test_experimental_ci_cannot_enter_stable_runtime_stages(self):
        env = os.environ.copy()
        env.update(CI_EXPERIMENTAL="true", CI_STAGE="all")
        result = subprocess.run(
            ["bash", "ci/run_checks.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("supports only CI_STAGE=opp-package", result.stderr)


if __name__ == "__main__":
    unittest.main()
