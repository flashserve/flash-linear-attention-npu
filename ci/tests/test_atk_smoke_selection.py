"""PR path selection against a real Git diff and the published smoke inventory."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests/atk/smoke"
ROOTS = json.loads((SMOKE / "source_roots.json").read_bytes())


class AtkSmokeSelectionTests(unittest.TestCase):
    def test_registered_roots_are_independent_operator_directories(self):
        suites = json.loads((SMOKE / "suites.json").read_bytes())
        self.assertEqual(set(ROOTS), {s["op"] for s in suites})
        for op, root in ROOTS.items():
            with self.subTest(op=op):
                self.assertTrue(root.startswith("fla/ops/ascendc/"))
                self.assertEqual(Path(root).name, op)
                self.assertNotIn("op_kernel", Path(root).parts)
                self.assertTrue((REPO / root / "op_host").is_dir())

    def test_changed_paths_drive_matrix_and_platform_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args):
                subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)

            def add_file(path):
                f = repo / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("fixture\n")
                return f

            def select(*args):
                return subprocess.check_output(
                    [sys.executable, str(SMOKE / "run.py"), *args, "--git-range", "HEAD^", "HEAD"],
                    env={**os.environ, "GIT_DIR": str(repo / ".git"), "GIT_WORK_TREE": str(repo)},
                    text=True,
                )

            git("init", "--quiet")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.org")
            old = add_file(ROOTS["chunk_fwd_h"] + "/op_kernel/old file.cpp")
            git("add", ".")
            git("commit", "-qm", "base")
            old.unlink()
            add_file(ROOTS["chunk_gated_delta_rule_fwd_h"] + "/op_kernel/new file.cpp")
            add_file(ROOTS["chunk_gated_delta_rule_fwd_h"] + "/op_host/api.cpp")
            add_file("fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
                     "op_kernel/internal/operators/chunk_fwd_o/op_kernel/internal.h")
            add_file("fla/ops/ascendc/kda/chunk_kda_bwd_other/kernel.cpp")
            add_file("ci/Dockerfile")
            git("add", "-A")
            git("commit", "-qm", "move and edit operators")
            self.assertEqual(select("affected", "--platform", "A2").splitlines(), ["chunk_fwd_h"])
            self.assertEqual(select("affected", "--platform", "A5").splitlines(),
                             ["chunk_fwd_h", "chunk_gated_delta_rule_fwd_h"])
            self.assertEqual(len(json.loads(select("matrix"))["include"]), 3)
            add_file("docs/common.md")
            git("add", ".")
            git("commit", "-qm", "unrelated change")
            self.assertEqual(json.loads(select("matrix")), {"include": []})
            add_file("tests/atk/smoke/cases/chunk_fwd_h/A5/cases.json")
            git("add", ".")
            git("commit", "-qm", "A5 test input only")
            self.assertEqual(json.loads(select("matrix")), {"include": [{"op": "chunk_fwd_h", "platform": "A5"}]})
            self.assertEqual(select("affected", "--platform", "A2"), "")
            add_file("tests/atk/smoke/helpers/bridges.py")
            git("add", ".")
            git("commit", "-qm", "shared ATK test harness")
            registered = json.loads((SMOKE / "suites.json").read_bytes())
            self.assertCountEqual(json.loads(select("matrix"))["include"], registered)
