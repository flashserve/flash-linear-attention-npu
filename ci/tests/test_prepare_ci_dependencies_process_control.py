"""Real Git remote-helper regressions; no network or production cache access."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from ci import prepare_ci_dependencies as dependencies


# A real Git transport helper, with a grandchild that inherits the Git pipes.
# Each new attempt records whether a previous attempt still has live workers.
REMOTE_HELPER = r'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(os.environ["CI_TEST_HELPERS"])
mode = os.environ.get("CI_TEST_HELPER_MODE", "normal")
if mode != "normal":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if len(sys.argv) > 1 and sys.argv[1] == "--child":
    (root / (str(os.getpid()) + ".child")).write_text(str(os.getpgrp()))
    time.sleep(60)
else:
    overlap = []
    for record in root.glob("*.json"):
        for pid in json.loads(record.read_text())["pids"]:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state not in {"Z", "X"}:
                    overlap.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                pass
    child = subprocess.Popen([sys.executable, __file__, "--child"])
    record = {"pids": [os.getpid(), child.pid], "pgid": os.getpgrp(), "overlap": overlap}
    (root / (str(os.getpid()) + ".json")).write_text(json.dumps(record))
    while not (root / (str(child.pid) + ".child")).exists():
        time.sleep(.01)
    if mode == "leader-exits":
        os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(60)
'''


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("git"),
                     "requires Linux process groups and Git")
class GitProcessControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fla-git-processes-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.records = self.root / "helpers"
        self.records.mkdir()
        self.addCleanup(self.cleanup_helpers)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        helper = bin_dir / "git-remote-cihang"
        helper.write_text(f"#!{sys.executable}\n" + REMOTE_HELPER)
        helper.chmod(0o755)
        self.enterContext(mock.patch.dict(os.environ, {
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "GIT_ALLOW_PROTOCOL": "cihang",
            "CI_TEST_HELPERS": str(self.records),
            "CI_TEST_HELPER_MODE": "normal",
        }))
        self.enterContext(mock.patch.object(dependencies, "GIT_TIMEOUT_SECONDS", 1))
        self.enterContext(mock.patch.object(dependencies, "GIT_CLEANUP_GRACE_SECONDS", .15))

    def helper_records(self):
        return [json.loads(path.read_text()) for path in self.records.glob("*.json")]

    def cleanup_helpers(self):
        # Also stop the fixtures when testing the buggy implementation.
        for record in self.helper_records():
            if record["pgid"] != os.getpgrp():
                try:
                    os.killpg(record["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                for pid in record["pids"]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def assert_helpers_stopped(self, count=1):
        records = self.helper_records()
        self.assertEqual(len(records), count, "the real Git transport must have started")
        for record in records:
            self.assertEqual(record["overlap"], [], "retry overlapped a previous download")
            for pid in record["pids"]:
                try:
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                self.assertIn(fields[0], {"Z", "X"}, f"download process {pid} survived")

    def hanging_fetch(self):
        dependencies.run_git(self.repo, "init", "-q")
        return dependencies.run_git(self.repo, "fetch", "cihang::isolated")

    def test_timeout_stops_real_git_helper_and_grandchild(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.hanging_fetch()
        self.assert_helpers_stopped()

    def test_timeout_kills_stubborn_helpers_after_git_leader_exits(self):
        os.environ["CI_TEST_HELPER_MODE"] = "leader-exits"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.hanging_fetch()
        self.assert_helpers_stopped()

    def test_fetch_retries_only_after_previous_helpers_stop(self):
        os.environ["CI_TEST_HELPER_MODE"] = "ignore-term"
        dependency = dependencies.Dependency("example", "cihang::isolated", "0" * 40)
        with self.assertRaises(subprocess.TimeoutExpired):
            dependencies.fetch(self.repo, dependency)
        self.assert_helpers_stopped(count=3)

    def test_cleanup_failure_prevents_fetch_retry(self):
        dependencies.run_git(self.repo, "init", "-q")
        real_stop = dependencies._stop_git_processes

        def stop_then_fail(process):
            real_stop(process)
            if "fetch" in process.args:
                raise dependencies.GitCleanupError("cleanup could not be confirmed")

        dependency = dependencies.Dependency("example", "cihang::isolated", "0" * 40)
        with mock.patch.object(dependencies, "_stop_git_processes", side_effect=stop_then_fail):
            with self.assertRaises(dependencies.GitCleanupError):
                dependencies.fetch(self.repo, dependency)
        self.assert_helpers_stopped()

    def test_interrupt_stops_helpers_before_propagating(self):
        dependencies.run_git(self.repo, "init", "-q")

        def interrupt_when_ready(process, **kwargs):
            deadline = time.monotonic() + 5
            while not list(self.records.glob("*.child")) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(list(self.records.glob("*.child")))
            raise KeyboardInterrupt

        with mock.patch.object(subprocess.Popen, "communicate", interrupt_when_ready):
            with self.assertRaises(KeyboardInterrupt):
                dependencies.run_git(self.repo, "fetch", "cihang::isolated")
        self.assert_helpers_stopped()

    def test_unconfirmed_live_group_raises_nonretryable_error(self):
        process = mock.Mock(pid=123456)
        with mock.patch.object(os, "killpg") as kill, \
             mock.patch.object(dependencies, "_git_group_alive", return_value=True):
            with self.assertRaises(dependencies.GitCleanupError):
                dependencies._stop_git_processes(process)
        self.assertEqual(kill.call_args_list, [mock.call(process.pid, signal.SIGTERM),
                                               mock.call(process.pid, signal.SIGKILL)])

    def test_run_git_preserves_output_and_check_behavior(self):
        result = dependencies.run_git(self.repo, "--version")
        self.assertEqual(result.returncode, 0)
        self.assertIn("git version", result.stdout)
        result = dependencies.run_git(self.repo, "rev-parse", "HEAD", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr)
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            dependencies.run_git(self.repo, "rev-parse", "HEAD")
        self.assertEqual(raised.exception.returncode, result.returncode)
        self.assertEqual(raised.exception.stderr, result.stderr)


if __name__ == "__main__":
    unittest.main()
