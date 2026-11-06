#!/usr/bin/env python3
"""Contract tests for the changed-file CI classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_classify_changes", ROOT / "ci" / "classify_changes.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ChangeClassificationTests(unittest.TestCase):
    def classify(self, *paths: str):
        return MODULE.classify_paths(paths, ROOT)

    def test_documentation_only_is_a_noop(self):
        self.assertEqual(
            self.classify(
                "README.md", "LICENSES/LICENSE", "docs/开发者指南.md",
                "ci/README.md", ".github/README.md", "tests/README.md",
            ),
            ("docs-only", []),
        )

    def test_test_and_ci_files_skip_compilation(self):
        self.assertEqual(
            self.classify("tests/test_example.py", "examples/foo.py", "ci/run_checks.sh", ".github/workflows/ci.yml"),
            ("tests-only", []),
        )

    def test_documentation_and_tests_are_tests_only(self):
        self.assertEqual(
            self.classify("docs/开发者指南.md", "tests/test_example.py"),
            ("tests-only", []),
        )

    def test_nested_operator_names_are_extracted(self):
        self.assertEqual(
            self.classify(
                "fla/ops/ascendc/kda/chunk_kda_bwd/op_host/foo.cpp",
                "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/foo.cpp",
            ),
            ("partial", ["chunk_gated_delta_rule_fwd", "chunk_kda_bwd"]),
        )

    def test_unknown_ascendc_file_falls_back_to_full(self):
        self.assertEqual(
            self.classify("fla/ops/ascendc/CMakeLists.txt"),
            ("full", []),
        )

    def test_documentation_and_operators_are_partial(self):
        self.assertEqual(
            self.classify(
                "docs/开发者指南.md",
                "fla/ops/ascendc/kda/chunk_kda_bwd/op_host/foo.cpp",
            ),
            ("partial", ["chunk_kda_bwd"]),
        )

    def test_tests_and_operators_are_partial(self):
        self.assertEqual(
            self.classify(
                "tests/test_example.py",
                "fla/ops/ascendc/kda/chunk_kda_bwd/op_host/foo.cpp",
            ),
            ("partial", ["chunk_kda_bwd"]),
        )

    def test_documentation_and_unrelated_source_are_full(self):
        self.assertEqual(
            self.classify("docs/开发者指南.md", "torch_custom/fla_npu/fla_npu.py"),
            ("full", []),
        )

    def test_mixed_operator_and_source_change_is_full(self):
        self.assertEqual(
            self.classify(
                "fla/ops/ascendc/kda/chunk_kda_bwd/op_host/foo.cpp",
                "torch_custom/fla_npu/fla_npu.py",
            ),
            ("full", []),
        )

    def test_rename_considers_both_paths(self):
        self.assertEqual(
            MODULE.paths_from_name_status(
                ["R100\tdocs/old.md\ttorch_custom/new.py"]
            ),
            ["docs/old.md", "torch_custom/new.py"],
        )


if __name__ == "__main__":
    unittest.main()
