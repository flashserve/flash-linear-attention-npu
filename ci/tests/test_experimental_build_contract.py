"""Contract tests for the experimental operator discovery rules."""

from __future__ import annotations

import importlib.util
import os
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


if __name__ == "__main__":
    unittest.main()
