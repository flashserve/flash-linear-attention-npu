#!/usr/bin/env python3
"""Keep PR Example/ST cases without following untrusted filesystem paths."""

import ast
import os
import re
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_case_copy():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    step = next(
        step for step in workflow["jobs"]["ascend-npu"]["steps"]
        if step["name"] == "使用可信 CI 脚本并保留 PR 用例"
    )
    block = re.search(r"python3 - <<'PY'\n(.*?)\nPY", step["run"], re.S)[1]
    tree = ast.parse(block)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WORKFLOW_PATH), "exec"), namespace)
    return namespace["copy_case_file"]


@unittest.skipUnless(
    getattr(os, "O_NOFOLLOW", 0) and getattr(os, "O_DIRECTORY", 0),
    "case preservation requires Linux nofollow directory flags",
)
class WorkflowCaseSecurityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.copy_case_file = staticmethod(_load_case_copy())

    def test_valid_case_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "ci").mkdir(parents=True)
            payload = b'{"cases": [{"name": "new-pr-case", "enabled": true}]}\n'
            (repo / "ci" / "example_st_cases.json").write_bytes(payload)
            self.assertTrue(self.copy_case_file(repo, root / "saved.json"))
            self.assertEqual((root / "saved.json").read_bytes(), payload)

    def test_missing_ci_directory_does_not_create_a_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            self.assertFalse(self.copy_case_file(repo, root / "saved.json"))
            self.assertFalse((root / "saved.json").exists())

    def test_symlink_repo_and_output_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            (repo / "ci").mkdir(parents=True)
            (repo / "ci" / "example_st_cases.json").write_text("{}", encoding="utf-8")
            repo_link = root / "repo-link"
            repo_link.symlink_to(repo, target_is_directory=True)
            with self.assertRaises(OSError):
                self.copy_case_file(repo_link, root / "repo-link.json")
            output = root / "output-link"
            output.symlink_to(repo, target_is_directory=True)
            with self.assertRaises(OSError):
                self.copy_case_file(repo, output / "saved.json")
            self.assertFalse((repo / "saved.json").exists())

    def test_case_copy_missing_invalid_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            repo = temp_path / "repo"
            output_dir = temp_path / "output"
            repo.mkdir()
            (repo / "ci").mkdir()
            output_dir.mkdir()

            self.assertFalse(
                self.copy_case_file(repo, output_dir / "missing.json")
            )
            self.assertFalse((output_dir / "missing.json").exists())

            case_path = repo / "ci" / "example_st_cases.json"
            case_path.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.copy_case_file(repo, output_dir / "invalid.json")
            self.assertFalse((output_dir / "invalid.json").exists())
            case_path.unlink()

            outside = temp_path / "outside.json"
            outside.write_text('{"secret": true}', encoding="utf-8")
            case_path.symlink_to(outside)
            with self.assertRaises(OSError):
                self.copy_case_file(repo, output_dir / "symlink.json")

    def test_case_copy_rejects_ci_symlink_fifo_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            repo = temp_path / "repo"
            outside_ci = temp_path / "outside-ci"
            output_dir = temp_path / "output"
            repo.mkdir()
            outside_ci.mkdir()
            output_dir.mkdir()
            (outside_ci / "example_st_cases.json").write_text("{}", encoding="utf-8")
            (repo / "ci").symlink_to(outside_ci, target_is_directory=True)
            with self.assertRaises(OSError):
                self.copy_case_file(repo, output_dir / "ci-symlink.json")

            (repo / "ci").unlink()
            (repo / "ci").mkdir()
            case_path = repo / "ci" / "example_st_cases.json"
            os.mkfifo(case_path)
            with self.assertRaises(RuntimeError):
                self.copy_case_file(repo, output_dir / "fifo.json")
            case_path.unlink()

            case_path.write_text("{}", encoding="utf-8")
            existing_output = output_dir / "existing.json"
            existing_output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.copy_case_file(repo, existing_output)
            self.assertEqual(existing_output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
