"""Exercise platform selection with a temporary operator catalog and real Git diffs."""

import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests/atk/smoke"


class AtkSmokeSelectionTests(unittest.TestCase):
    def test_changed_paths_drive_matrix_and_platform_execution(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as fixture:
            repo = Path(directory)
            smoke = Path(fixture) / "tests/atk/smoke"
            smoke.mkdir(parents=True)
            ci = Path(fixture) / "ci"
            ci.mkdir()
            shutil.copy(SMOKE / "run.py", smoke)
            shutil.copy(REPO / "ci/smoke_cache.py", ci)
            platforms = {
                "chunk_fwd_h": {"A2": True, "A5": True},
                "chunk_gated_delta_rule_bwd_dhu": {"A5": True},
                "recurrent_kda": {"A2": True, "A5": True},
                "causal_conv1d": {"A2": True, "A5": True},
                "chunk_kda_bwd_recompute": {"A2": True, "A5": False},
            }
            roots = {op: f"fla/ops/ascendc/fixture/{op}" for op in platforms}
            for op, arches in platforms.items():
                folder = smoke / "cases" / op
                folder.mkdir(parents=True)
                (folder / "suite.json").write_text(json.dumps({"source_root": roots[op], "platforms": arches}))
                for arch in arches:
                    (folder / arch).mkdir()
                    (folder / arch / "cases.json").write_text("[]")

            def git(*args):
                subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)

            def add_file(path):
                f = repo / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("fixture\n")
                return f

            def select(*args):
                return subprocess.check_output(
                    [sys.executable, str(smoke / "run.py"), *args, "--git-range", "HEAD^", "HEAD"],
                    env={**os.environ, "GIT_DIR": str(repo / ".git"), "GIT_WORK_TREE": str(repo)},
                    text=True,
                )

            git("init", "--quiet")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.org")
            old = add_file(roots["chunk_fwd_h"] + "/op_kernel/old file.cpp")
            git("add", ".")
            git("commit", "-qm", "base")
            old.unlink()
            add_file(roots["chunk_gated_delta_rule_bwd_dhu"] + "/op_kernel/new file.cpp")
            add_file(roots["chunk_gated_delta_rule_bwd_dhu"] + "/op_host/api.cpp")
            add_file("fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/"
                     "op_kernel/internal/operators/chunk_fwd_o/op_kernel/internal.h")
            add_file("fla/ops/ascendc/kda/chunk_kda_bwd_other/kernel.cpp")
            add_file("ci/Dockerfile")
            git("add", "-A")
            git("commit", "-qm", "move and edit operators")
            self.assertEqual(select("affected", "--platform", "A2").splitlines(), ["chunk_fwd_h"])
            self.assertEqual(select("affected", "--platform", "A5").splitlines(),
                             ["chunk_fwd_h", "chunk_gated_delta_rule_bwd_dhu"])
            self.assertEqual(len(json.loads(select("matrix"))["include"]), 3)
            add_file("docs/common.md")
            add_file(roots["chunk_fwd_h"] + "/README.md")
            add_file(roots["recurrent_kda"] + "/docs/API.MD")
            add_file("tests/atk/smoke/cases/chunk_fwd_o/A5/README.md")
            git("add", ".")
            git("commit", "-qm", "documentation only, including operator directories")
            self.assertEqual(json.loads(select("matrix")), {"include": []})
            add_file(roots["recurrent_kda"] + "/docs/tiling.md")
            add_file("tests/atk/smoke/cases/chunk_fwd_h/A5/cases.json")
            git("add", ".")
            git("commit", "-qm", "A5 test input and unrelated operator documentation")
            self.assertEqual(json.loads(select("matrix")), {"include": [{"op": "chunk_fwd_h", "platform": "A5"}]})
            self.assertEqual(select("affected", "--platform", "A2"), "")
            add_file("tests/atk/chunk_fwd_h/fwd_h_cpu.py")
            git("add", ".")
            git("commit", "-qm", "edit legacy standalone ATK reference")
            self.assertEqual(json.loads(select("matrix")), {"include": []})
            add_file("tests/atk/smoke/cases/causal_conv1d/suite.json")
            git("add", ".")
            git("commit", "-qm", "register one operator")
            self.assertEqual(select("affected", "--platform", "A2").splitlines(), ["causal_conv1d"])
            self.assertEqual(select("affected", "--platform", "A5").splitlines(), ["causal_conv1d"])

            add_file("tests/atk/smoke/executor.py")
            git("add", ".")
            git("commit", "-qm", "shared ATK test harness")
            enabled = [{"op": op, "platform": arch} for op, arches in platforms.items() for arch, on in arches.items() if on]
            self.assertCountEqual(json.loads(select("matrix"))["include"], enabled)

            add_file(roots["chunk_kda_bwd_recompute"] + "/op_kernel/recompute.cpp")
            git("add", ".")
            git("commit", "-qm", "change only recompute")
            self.assertEqual(select("affected", "--platform", "A2").splitlines(), ["chunk_kda_bwd_recompute"])
            self.assertEqual(select("affected", "--platform", "A5"), "")
            self.assertEqual(json.loads(select("matrix"))["include"], [{"op": "chunk_kda_bwd_recompute", "platform": "A2"}])
