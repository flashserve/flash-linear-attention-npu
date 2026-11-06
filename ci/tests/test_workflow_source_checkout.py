#!/usr/bin/env python3
"""Run the workflow's source preparation against real, local Git repositories."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("git"),
                     "The device workflow uses Linux Bash and Git")
class WorkflowSourceCheckoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        steps = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))["jobs"]["ascend-npu"]["steps"]
        cls.merge_script = next(step["run"] for step in steps if step.get("name") == "现场合并 PR 代码")
        cls.validate_script = next(step["run"] for step in steps if step.get("id") == "validate_pr")
        container_script = next(step["run"] for step in steps if step.get("id") == "npu_ci")
        cls.monitor_script = container_script[
            container_script.index("get_pr_head_sha() {"):container_script.index("mark_stale() {")
        ] + "\nget_pr_head_sha\n"
        cls.container_prefix = container_script[:container_script.index("summary_status=0")]
        cls.real_git = shutil.which("git")
        cls.seed_directory = tempfile.TemporaryDirectory(prefix="npu-ci-source-seed-")
        cls.addClassCleanup(cls.seed_directory.cleanup)
        cls.seed_root = Path(cls.seed_directory.name)
        author = cls.seed_root / "author"
        author.mkdir()
        cls.git(author, "init", "-b", "main")
        cls.git(author, "config", "user.name", "Source test")
        cls.git(author, "config", "user.email", "source-test@example.invalid")
        (author / "shared.txt").write_text("shared ancestor\n", encoding="utf-8")
        cls.git(author, "add", ".")
        cls.git(author, "commit", "-m", "shared ancestor")
        cls.root_sha = cls.git(author, "rev-parse", "HEAD").stdout.strip()
        cls.git(author, "branch", "pull-source")

        # Each side exceeds the workflow's fetch depth, so a real merge needs
        # history beyond the initial shallow fetch on both sides of the fork.
        for number in range(65):
            (author / "base.txt").write_text(f"base revision {number}\n", encoding="utf-8")
            cls.git(author, "add", "base.txt")
            cls.git(author, "commit", "-m", f"base {number}")
        cls.base_sha = cls.git(author, "rev-parse", "HEAD").stdout.strip()
        cls.git(author, "commit", "--allow-empty", "-m", "next base")
        cls.next_base_sha = cls.git(author, "rev-parse", "HEAD").stdout.strip()
        cls.git(author, "branch", "future-base")
        cls.git(author, "reset", "--hard", cls.base_sha)
        cls.git(author, "checkout", "pull-source")
        for number in range(65):
            (author / "pr.txt").write_text(f"PR revision {number}\n", encoding="utf-8")
            cls.git(author, "add", "pr.txt")
            cls.git(author, "commit", "-m", f"PR {number}")
        cls.head_sha = cls.git(author, "rev-parse", "HEAD").stdout.strip()
        cls.git(author, "checkout", "main")
        cls.seed_remote = cls.seed_root / "source.git"
        cls.git(cls.seed_root, "clone", "--bare", str(author), str(cls.seed_remote))
        cls.git(cls.seed_root, "--git-dir", str(cls.seed_remote), "update-ref", "refs/pull/17/head", cls.head_sha)

    @classmethod
    def git(cls, cwd, *arguments):
        return subprocess.run(
            [cls.real_git, *arguments], cwd=cwd, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                 "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "file"},
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="npu-ci-source-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote.git"
        self.git(self.root, "clone", "--bare", str(self.seed_remote), str(self.remote))
        self.git(self.root, "--git-dir", str(self.remote), "update-ref", "refs/pull/17/head", self.head_sha)
        self.git(self.root, "--git-dir", str(self.remote), "update-ref",
                 "refs/original/refs/pull/17/head", self.root_sha)
        self.repo = self.root / "repo"
        # file:// is deliberate: plain local paths ignore --depth.
        self.git(self.root, "clone", "--depth=1", "--single-branch", "--branch", "main",
                 self.remote.as_uri(), str(self.repo))
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.command_log = self.root / "git.jsonl"
        self.github_env = self.root / "github-env"
        self.proxy_calls = self.root / "proxy-calls"
        proxy_script = self.root / "proxy.sh"
        proxy_script.write_text(textwrap.dedent("""\
            if [ "${TEST_PROXY_SOURCE_OUTPUT:-0}" = 1 ]; then
              printf 'proxy configuration loaded\\n'
            fi
            proxyon() {
              printf 'enabled\\n' >> "$TEST_PROXY_CALLS"
              export http_proxy='http://127.0.0.1:17890' https_proxy='http://127.0.0.1:17890'
              export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
              export all_proxy='socks5h://127.0.0.1:17891' ALL_PROXY='socks5h://127.0.0.1:17891'
              export GIT_HTTP_PROXY="$http_proxy" GIT_HTTPS_PROXY="$https_proxy"
              export no_proxy='localhost,127.0.0.1' NO_PROXY='localhost,127.0.0.1'
              export PROXY_HTTP_PORT=17890 PROXY_SOCKS_PORT=17891 PROXY_MIXED_PORT=17892
            }
            """), encoding="utf-8")
        self.write_executable("git", f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            proxy_names = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                           'all_proxy', 'ALL_PROXY', 'GIT_HTTP_PROXY', 'GIT_HTTPS_PROXY',
                           'no_proxy', 'NO_PROXY')
            with open(os.environ['TEST_GIT_LOG'], 'a', encoding='utf-8') as log:
                log.write(json.dumps({'args': args, 'proxy': {
                    name: os.environ.get(name, '') for name in proxy_names}}) + '\\n')
            if args and args[0] in ('fetch', 'ls-remote', 'pull', 'push', 'clone'):
                if os.environ.get('TEST_DENY_NETWORK') == '1':
                    sys.stderr.write('TEST: offline validation attempted network access\\n')
                    sys.exit(92)
                if args[0] == 'fetch':
                    counter = Path(os.environ['TEST_FETCH_COUNT'])
                    count = int(counter.read_text() if counter.exists() else '0') + 1
                    counter.write_text(str(count))
                    if count <= int(os.environ.get('TEST_FETCH_FAILURES', '0')):
                        sys.stderr.write('TEST: simulated GitHub connection failure\\n')
                        sys.exit(1)
                if args[0] == 'ls-remote' and os.environ.get('TEST_FAIL_FIRST_LS_REMOTE') == '1':
                    marker = Path(os.environ['TEST_LS_REMOTE_MARKER'])
                    if not marker.exists():
                        marker.touch()
                        sys.stderr.write('TEST: simulated GitHub ref lookup failure\\n')
                        sys.exit(1)
            os.execv(os.environ['TEST_REAL_GIT'], [os.environ['TEST_REAL_GIT'], *args])
            """))
        self.write_executable("sleep", "#!/bin/sh\nprintf '%s\\n' \"$1\" >> \"$TEST_SLEEP_LOG\"\n")
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "CI_PROXY_SCRIPT": str(proxy_script), "GITHUB_ENV": str(self.github_env),
            "NPU_CI_PR_NUMBER": "17", "NPU_CI_BASE_REF": "main",
            "NPU_CI_BASE_SHA": self.base_sha, "NPU_CI_TARGET_SHA": self.head_sha,
            "NPU_CI_HEAD_SHA": self.head_sha, "NPU_CI_PROXY_ENABLED": "false",
            "TEST_REAL_GIT": self.real_git, "TEST_GIT_LOG": str(self.command_log),
            "TEST_FETCH_COUNT": str(self.root / "fetch-count"), "TEST_FETCH_FAILURES": "0",
            "TEST_SLEEP_LOG": str(self.root / "sleep-log"), "TEST_PROXY_CALLS": str(self.proxy_calls),
            "TEST_LS_REMOTE_MARKER": str(self.root / "ls-remote-attempted"),
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "file",
        }
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
                     "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"):
            self.env[name] = "http://inherited.invalid:9999"
        self.env.update(no_proxy="inherited.invalid", NO_PROXY="inherited.invalid")

    def write_executable(self, name, content):
        script = self.bin / name
        script.write_text(content, encoding="utf-8")
        script.chmod(0o755)

    def run_script(self, script, **overrides):
        return subprocess.run(
            ["bash", "-c", script], cwd=self.repo, env={**self.env, **overrides},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40,
        )

    def assert_succeeded(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def commands(self):
        if not self.command_log.exists():
            return []
        return [json.loads(line) for line in self.command_log.read_text(encoding="utf-8").splitlines()]

    def fetches(self):
        return [command for command in self.commands() if command["args"][0] == "fetch"]

    def validate_offline(self, **overrides):
        self.command_log.write_text("", encoding="utf-8")
        result = self.run_script(self.validate_script, TEST_DENY_NETWORK="1", **overrides)
        self.assertFalse(any(command["args"][0] in ("fetch", "ls-remote") for command in self.commands()))
        return result

    def test_real_fork_merge_deepens_history_and_clears_inherited_proxy(self):
        self.assertEqual(self.git(self.repo, "rev-parse", "--is-shallow-repository").stdout.strip(), "true")
        self.assert_succeeded(self.run_script(self.merge_script))
        fetches = self.fetches()
        self.assertIn("--depth=50", fetches[0]["args"])
        self.assertTrue(any("--deepen=50" in command["args"] for command in fetches))
        for name, value in fetches[0]["proxy"].items():
            self.assertEqual(value, "*" if name.lower() == "no_proxy" else "", name)
        self.assertFalse(self.proxy_calls.exists())
        self.assertEqual((self.repo / "base.txt").read_text(), "base revision 64\n")
        self.assertEqual((self.repo / "pr.txt").read_text(), "PR revision 64\n")
        self.assert_succeeded(self.validate_offline())

    def test_failed_direct_fetch_enables_proxy_and_persists_it(self):
        self.assert_succeeded(self.run_script(self.merge_script, TEST_FETCH_FAILURES="1"))
        fetches = self.fetches()
        self.assertEqual(fetches[0]["proxy"]["http_proxy"], "")
        self.assertEqual(fetches[1]["proxy"]["http_proxy"], "http://127.0.0.1:17890")
        self.assertTrue(all(command["proxy"]["no_proxy"] == "localhost,127.0.0.1" for command in fetches[1:]))
        self.assertEqual(self.proxy_calls.read_text(), "enabled\n")
        self.assertEqual((self.root / "sleep-log").read_text(), "15\n")
        persisted = dict(line.split("=", 1) for line in self.github_env.read_text().splitlines())
        self.assertEqual(persisted["NPU_CI_PROXY_ENABLED"], "true")
        self.assertEqual(persisted["https_proxy"], "http://127.0.0.1:17890")
        self.assertEqual(persisted["no_proxy"], "localhost,127.0.0.1")
        self.assertEqual(len(persisted), len({name.casefold() for name in persisted}))
        self.assertNotIn("HTTP_PROXY", persisted)
        self.assert_succeeded(self.validate_offline())

    def test_exhausted_proxy_retries_fail_with_connectivity_advice(self):
        result = self.run_script(self.merge_script, TEST_FETCH_FAILURES="99")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Git fetch failed after proxy retries", result.stderr)
        self.assertIn("check the runner proxy and GitHub connectivity", result.stderr)
        self.assertEqual(len(self.fetches()), 3)
        self.assertEqual((self.root / "sleep-log").read_text(), "15\n30\n")
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD").stdout.strip(), self.base_sha)

    def test_checkout_does_not_reject_a_changed_base_snapshot(self):
        result = self.run_script(self.merge_script, NPU_CI_BASE_SHA=self.root_sha)
        self.assert_succeeded(result)
        self.assertTrue(self.fetches())

    def test_changed_pr_head_is_rejected_before_merge(self):
        result = self.run_script(self.merge_script, NPU_CI_TARGET_SHA=self.root_sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PR head changed before merge", result.stdout + result.stderr)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD").stdout.strip(), self.base_sha)

    def test_offline_guard_allows_advanced_target_branch(self):
        self.git(self.root, "--git-dir", str(self.remote), "update-ref", "refs/heads/main", self.next_base_sha)
        self.assert_succeeded(self.run_script(self.merge_script))
        result = self.validate_offline()
        self.assert_succeeded(result)

    def test_offline_guard_rejects_changed_head_snapshot(self):
        self.assert_succeeded(self.run_script(self.merge_script))
        result = self.validate_offline(NPU_CI_HEAD_SHA=self.root_sha)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PR head changed before validation", result.stderr)

    def test_offline_guard_rejects_fetched_but_unmerged_pr(self):
        self.git(self.repo, "fetch", "--unshallow", "origin",
                 "+refs/heads/main:refs/remotes/origin/main",
                 "+refs/pull/17/head:refs/remotes/pull/17/head")
        result = self.validate_offline()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Merged source does not contain the PR head", result.stderr)
        self.assertIn("fetch and merge the PR again", result.stderr)

    def test_running_head_monitor_reads_the_pr_git_ref(self):
        result = self.run_script(self.monitor_script)
        self.assert_succeeded(result)
        self.assertEqual(result.stdout.strip(), self.head_sha)
        self.assertFalse(self.proxy_calls.exists())

    def test_running_head_monitor_retries_with_proxy_without_polluting_sha(self):
        result = self.run_script(self.monitor_script, TEST_FAIL_FIRST_LS_REMOTE="1",
                                 TEST_PROXY_SOURCE_OUTPUT="1")
        self.assert_succeeded(result)
        self.assertEqual(result.stdout.strip(), self.head_sha)
        self.assertIn("proxy configuration loaded", result.stderr)
        self.assertEqual(self.proxy_calls.read_text(), "enabled\n")
        lookups = [command for command in self.commands() if command["args"][0] == "ls-remote"]
        self.assertEqual(len(lookups), 2)
        self.assertEqual(lookups[1]["proxy"]["https_proxy"], "http://127.0.0.1:17890")

    def test_all_monitor_connections_fail_without_failing_the_container(self):
        ci_directory = self.repo / "ci"
        ci_directory.mkdir()
        (ci_directory / "run_ci_container.sh").write_text(
            "#!/bin/sh\nattempt=0\n"
            'while [ ! -f "$TEST_PROXY_CALLS" ]; do\n'
            '  attempt=$((attempt + 1))\n  [ "$attempt" -le 500 ] || exit 1\n'
            "  /bin/sleep 0.01\ndone\necho 'container completed'\n", encoding="utf-8"
        )
        script = self.container_prefix.replace("${{ matrix.name }}", "test-device")
        result = self.run_script(
            script + '\nexit "$ci_status"\n',
            TEST_DENY_NETWORK="1", CI_ACCURACY_REPORT_FILE=".ci-tmp/accuracy.json",
            CI_RAW_LOG_FILE=".ci-tmp/raw.log", CI_DIAGNOSTIC_FILE=".ci-tmp/diagnostic.json",
            CI_SUMMARY_FILE=".ci-tmp/summary.md", CI_CONTAINER_NAME="unused-container",
            NPU_CI_HEAD_POLL_SECONDS="0",
        )
        self.assert_succeeded(result)
        self.assertIn("Unable to check PR head while NPU CI is running; continuing", result.stdout)
        self.assertIn("container completed", (self.repo / ".ci-tmp/raw.log").read_text())


if __name__ == "__main__":
    unittest.main()
