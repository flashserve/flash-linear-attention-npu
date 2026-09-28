#!/usr/bin/env python3
"""Cover the base-delta replay used when GitHub's test merge lags the base branch.

The A5 job can be handed a test merge that GitHub computed before the base
branch advanced.  Instead of failing the run, the job replays the base delta
onto that merge; these cases pin down when that is safe and when the run has to
ask for a rebase instead.
"""

import ast
import re
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _python_blocks():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["ascend-npu"]["steps"]
    blocks = []
    for step in steps:
        run = step.get("run", "")
        blocks.extend(
            re.findall(r"python3? - <<'PY'\n(.*?)\nPY(?:\n|$)", run, flags=re.S)
        )
    return blocks


def _load_replay_functions():
    block = next(block for block in _python_blocks() if "def replay_base_delta" in block)
    tree = ast.parse(block)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef)
        and node.name in {"tree_digests", "replay_base_delta"}
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(WORKFLOW_PATH), "exec"),
        namespace,
    )
    return namespace["replay_base_delta"]


def _write_tree(root, files):
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _read_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class BaseDeltaReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay_base_delta = staticmethod(_load_replay_functions())

    def _replay(self, source, target, merged):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_tree = root / "source"
            target_tree = root / "target"
            repo_dir = root / "repo"
            for tree in (source_tree, target_tree, repo_dir):
                tree.mkdir()
            _write_tree(source_tree, source)
            _write_tree(target_tree, target)
            _write_tree(repo_dir, merged)
            conflicts = self.replay_base_delta(repo_dir, source_tree, target_tree)
            return sorted(conflicts), _read_tree(repo_dir)

    def test_base_only_edit_is_replayed(self):
        conflicts, tree = self._replay(
            {"a.py": "v1\n"}, {"a.py": "v2\n"}, {"a.py": "v1\n"}
        )
        self.assertEqual(conflicts, [])
        self.assertEqual(tree["a.py"], "v2\n")

    def test_base_only_addition_is_replayed(self):
        conflicts, tree = self._replay(
            {}, {"pkg/new.py": "new\n"}, {"pkg/keep.py": "keep\n"}
        )
        self.assertEqual(conflicts, [])
        self.assertEqual(tree["pkg/new.py"], "new\n")
        self.assertEqual(tree["pkg/keep.py"], "keep\n")

    def test_base_only_deletion_is_replayed(self):
        conflicts, tree = self._replay(
            {"gone.py": "old\n", "kept.py": "same\n"},
            {"kept.py": "same\n"},
            {"gone.py": "old\n", "kept.py": "same\n"},
        )
        self.assertEqual(conflicts, [])
        self.assertNotIn("gone.py", tree)

    def test_pr_side_edit_is_kept_when_base_did_not_touch_the_path(self):
        conflicts, tree = self._replay(
            {"base.py": "v1\n", "pr.py": "v1\n"},
            {"base.py": "v2\n", "pr.py": "v1\n"},
            {"base.py": "v1\n", "pr.py": "pr\n"},
        )
        self.assertEqual(conflicts, [])
        self.assertEqual(tree["base.py"], "v2\n")
        self.assertEqual(tree["pr.py"], "pr\n")

    def test_tree_that_already_has_the_base_state_is_untouched(self):
        conflicts, tree = self._replay(
            {"a.py": "v1\n"}, {"a.py": "v2\n"}, {"a.py": "v2\n"}
        )
        self.assertEqual(conflicts, [])
        self.assertEqual(tree["a.py"], "v2\n")

    def test_same_path_changed_on_both_sides_is_reported(self):
        conflicts, tree = self._replay(
            {"a.py": "v1\n", "b.py": "v1\n"},
            {"a.py": "v2\n", "b.py": "v2\n"},
            {"a.py": "pr\n", "b.py": "v1\n"},
        )
        self.assertEqual(conflicts, ["a.py"])
        self.assertEqual(tree["a.py"], "pr\n")
        self.assertEqual(tree["b.py"], "v2\n")

    def test_path_deleted_by_base_and_kept_by_pr_is_reported(self):
        conflicts, tree = self._replay(
            {"a.py": "v1\n"}, {}, {"a.py": "v1\n"}
        )
        self.assertEqual(conflicts, [])
        self.assertNotIn("a.py", tree)

        conflicts, tree = self._replay(
            {"a.py": "v1\n"}, {}, {"a.py": "pr\n"}
        )
        self.assertEqual(conflicts, ["a.py"])
        self.assertEqual(tree["a.py"], "pr\n")


if __name__ == "__main__":
    unittest.main()
