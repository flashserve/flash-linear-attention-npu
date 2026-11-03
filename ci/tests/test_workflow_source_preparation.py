#!/usr/bin/env python3
"""Exercise the shared source path without network or NPU resources."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def source_steps():
    steps = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]["ascend-npu"]["steps"]
    stop = next(i for i, step in enumerate(steps) if step.get("id") == "npu_ci")
    return steps[:stop]


class SharedSourcePreparationTest(unittest.TestCase):
    def test_platforms_share_one_source_path_and_use_pr_ci(self):
        steps = source_steps()
        self.assertEqual(sum(step.get("working-directory") == "repo" for step in steps), 1)
        for step in steps:
            self.assertNotIn("matrix.platform", step.get("if", ""))
            if step.get("uses") == "actions/checkout@v5" and step["with"]["path"] == "repo":
                expected = "base_sha"
                self.assertEqual(step["with"]["ref"], "${{ needs.prepare.outputs." + expected + " }}")
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        for removed in ("download_archive", "extract_archive", "replay_base_delta", "fla-npu-source-", "trusted_ci_sha", "use_pr_ci", "trusted-ci"):
            self.assertNotIn(removed, text)


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell test")
class SourceMergeBehaviorTest(unittest.TestCase):
    def _run_merge(self, **settings):
        step = next(step for step in source_steps() if step.get("working-directory") == "repo")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            git = bin_dir / "git"
            git.write_text(
                "#!" + sys.executable + "\n" + r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["GIT_TEST_LOG"])
with log.open("a") as stream:
    stream.write(json.dumps({"args": args, "proxy": os.getenv("http_proxy"),
                             "upper_proxy": os.getenv("HTTP_PROXY"),
                             "no_proxy": os.getenv("no_proxy")}) + "\n")
while args and args[0] == "-c":
    args = args[2:]
command = args[0]
if command == "rev-parse":
    if args[1] == "HEAD":
        print("a" * 40)
    elif args[1] == "--is-shallow-repository":
        print("true")
    else:
        print(("c" if os.getenv("MISMATCH") else "b") * 40)
elif command == "fetch":
    deepening = any(arg.startswith("--deepen=") or arg == "--unshallow" for arg in args)
    if os.getenv("FAIL_ALL_FETCHES") or (os.getenv("FAIL_DEEPEN") and deepening) or (os.getenv("FAIL_DIRECT") and not os.getenv("http_proxy")):
        sys.exit(1)
    if any(arg.startswith("--deepen=") or arg == "--unshallow" for arg in args):
        Path(os.environ["GIT_TEST_DEEPENED"]).touch()
elif command == "merge-base":
    if os.getenv("SHALLOW") and not Path(os.environ["GIT_TEST_DEEPENED"]).exists():
        sys.exit(1)
''',
                encoding="utf-8",
            )
            git.chmod(0o755)
            proxy = root / "proxy.sh"
            proxy.write_text("proxyon() { export http_proxy=http://runner-proxy:7899; }\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                PATH=str(bin_dir) + os.pathsep + env["PATH"],
                CI_PROXY_SCRIPT=str(proxy),
                NPU_CI_PR_NUMBER="527",
                NPU_CI_BASE_SHA="a" * 40,
                NPU_CI_TARGET_SHA="b" * 40,
                GIT_TEST_LOG=str(root / "git.jsonl"),
                GIT_TEST_DEEPENED=str(root / "deepened"),
                http_proxy="http://inherited-proxy:8080",
                HTTP_PROXY="http://inherited-proxy:8080",
                **{key: str(value) for key, value in settings.items()},
            )
            completed = subprocess.run(["bash", "-c", step["run"]], cwd=root, env=env, text=True, capture_output=True)
            records = [json.loads(line) for line in (root / "git.jsonl").read_text(encoding="utf-8").splitlines()]
            return completed, records

    def test_fetch_retries_through_proxy_and_merges_the_fixed_head(self):
        completed, records = self._run_merge(FAIL_DIRECT=1)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fetches = [record for record in records if "fetch" in record["args"]]
        self.assertEqual(len(fetches), 2)
        self.assertIsNone(fetches[0]["proxy"])
        self.assertIsNone(fetches[0]["upper_proxy"])
        self.assertEqual(fetches[0]["no_proxy"], "*")
        self.assertEqual(fetches[1]["proxy"], "http://runner-proxy:7899")
        self.assertIn(["merge", "--no-ff", "--no-edit", "b" * 40], [record["args"] for record in records])

    def test_history_deepening_also_uses_direct_then_proxy(self):
        completed, records = self._run_merge(FAIL_DIRECT=1, SHALLOW=1)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        fetches = [record for record in records if "fetch" in record["args"]]
        self.assertEqual(len(fetches), 4)
        self.assertEqual([bool(record["proxy"]) for record in fetches], [False, True, False, True])
        self.assertTrue(any("--deepen=50" in record["args"] for record in fetches))

    def test_failed_fetch_and_changed_head_never_merge(self):
        for settings in ({"FAIL_ALL_FETCHES": 1}, {"MISMATCH": 1}, {"MISMATCH": 1, "SHALLOW": 1}, {"FAIL_DEEPEN": 1, "SHALLOW": 1}):
            with self.subTest(settings=settings):
                completed, records = self._run_merge(**settings)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(any(record["args"][0] == "merge" for record in records))


if __name__ == "__main__":
    unittest.main()
