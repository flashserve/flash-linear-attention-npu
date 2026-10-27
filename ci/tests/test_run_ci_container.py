"""Exercise host storage and container cleanup without a Docker daemon."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux shell is required")
class RunCiContainerTests(unittest.TestCase):
    def run_ci(self, root, *, run_status=0, cleanup_status=0, write_id=True, interrupt=False, broad_tmp=False):
        repo = root / "repo"
        (repo / "ci").mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "ci/run_ci_container.sh", repo / "ci/run_ci_container.sh")
        (repo / "ci/detect_npu.sh").write_text(
            'if [[ "$1" == --candidates ]]; then echo 0; else\n'
            'echo "NPU_SELECTED_DEVICE=0 NPU_SELECTED_NAME=test '
            'NPU_SELECTED_HEALTH=OK NPU_SELECTED_FREE=100 NPU_SOC=ascend950"\nfi\n'
        )
        commands = root / "bin"
        commands.mkdir()
        (commands / "npu-smi").write_text("#!/bin/sh\nexit 0\n")
        (commands / "npu-smi").chmod(0o755)
        (commands / "docker").write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, signal, sys, time\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['DOCKER_RECORD'], 'a') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "if args[:2] == ['image', 'inspect'] or args[:2] == ['rm', '-f']:\n"
            "    pid = pathlib.Path(os.environ['WAIT_PID'])\n"
            "    if args[:2] == ['rm', '-f'] and pid.exists():\n"
            "        os.kill(int(pid.read_text()), signal.SIGKILL)\n"
            "    sys.exit(0)\n"
            "if '--cidfile' in args:\n"
            "    if os.environ['WRITE_ID'] == 'true':\n"
            "        pathlib.Path(args[args.index('--cidfile') + 1]).write_text('owned-container-id')\n"
            "    if os.environ['RUN_WAIT'] == 'true':\n"
            "        signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "        pathlib.Path(os.environ['WAIT_PID']).write_text(str(os.getpid()))\n"
            "        while True: time.sleep(1)\n"
            "    sys.exit(int(os.environ['RUN_STATUS']))\n"
            "sys.exit(int(os.environ['CLEANUP_STATUS']))\n"
        )
        (commands / "docker").chmod(0o755)
        env = dict(os.environ)
        env.update(
            PATH=str(commands) + os.pathsep + env["PATH"],
            TMPDIR=str(root / "storage/host"),
            CI_TMPDIR="/tmp/fla-npu-ci",
            CI_HOST_TMPDIR=str(root / "storage/container"),
            CI_CACHE_ROOT=str(root / "storage/cache"),
            CI_NPU_LOCK_DIR=str(root / "storage/locks"),
            CI_IMAGE="test-image",
            CI_CONTAINER_NAME="ci-test",
            CI_REQUIRE_PRELOADED_IMAGE="true",
            CI_REBUILD_IMAGE="false",
            CI_REQUIRE_HEALTHY_NPU="false",
            DOCKER_RECORD=str(root / "docker.jsonl"),
            RUN_STATUS=str(run_status),
            CLEANUP_STATUS=str(cleanup_status),
            WRITE_ID=str(write_id).lower(),
            RUN_WAIT=str(interrupt).lower(),
            WAIT_PID=str(root / "waiting-pid"),
        )
        if broad_tmp:
            env["CI_HOST_TMPDIR"] = env["TMPDIR"]
        command = ["bash", "ci/run_ci_container.sh"]
        if interrupt:
            process = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not Path(env["WAIT_PID"]).exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise AssertionError("Mock container did not start")
                    time.sleep(0.02)
                process.terminate()
                stdout, stderr = process.communicate(timeout=10)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        else:
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, check=False)
        calls = [json.loads(line) for line in (root / "docker.jsonl").read_text().splitlines()]
        return result, calls, env

    def test_missing_storage_is_created_and_only_owned_container_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, calls, env = self.run_ci(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("TMPDIR", "CI_HOST_TMPDIR", "CI_CACHE_ROOT", "CI_NPU_LOCK_DIR"):
                self.assertTrue(Path(env[name]).is_dir(), name)
            self.assertTrue((Path(env["CI_CACHE_ROOT"]) / "third_party").is_dir())
            self.assertTrue((Path(env["CI_CACHE_ROOT"]) / "gdr_accuracy_golden").is_dir())
            self.assertIn(["rm", "-f", "owned-container-id"], calls)
            helper = calls[-1]
            self.assertEqual(helper[helper.index("--network") + 1], "none")
            mounts = [helper[index + 1] for index, arg in enumerate(helper) if arg == "-v"]
            self.assertEqual(mounts, [
                str(root / "repo") + ":/ci-repo", env["CI_HOST_TMPDIR"] + ":/ci-tmp",
            ])
            self.assertEqual(helper[-3:], [f"{os.getuid()}:{os.getgid()}", "/ci-repo", "/ci-tmp"])
            self.assertFalse(list(Path(env["TMPDIR"]).iterdir()))
            lock = Path(env["CI_NPU_LOCK_DIR"]) / "fla-npu-ci-npu-0.lock"
            self.assertTrue(lock.exists())
            self.assertEqual(subprocess.run(["flock", "-n", str(lock), "true"]).returncode, 0)

    def test_cleanup_preserves_ci_failure_and_reports_cleanup_failure(self):
        for run_status, cleanup_status, expected in (
            (17, 0, 17), (137, 0, 137), (0, 39, 1), (17, 39, 17),
        ):
            with self.subTest(run_status=run_status, cleanup_status=cleanup_status):
                with tempfile.TemporaryDirectory() as temporary:
                    result, calls, _ = self.run_ci(
                        Path(temporary), run_status=run_status, cleanup_status=cleanup_status,
                    )
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertIn(["rm", "-f", "owned-container-id"], calls)
                    self.assertIn("--entrypoint", calls[-1])

    def test_failed_docker_start_does_not_remove_another_named_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, env = self.run_ci(Path(temporary), run_status=125, write_id=False)
            self.assertEqual(result.returncode, 125, result.stdout + result.stderr)
            self.assertFalse(any(call[:2] == ["rm", "-f"] for call in calls))
            self.assertFalse(any("--entrypoint" in call for call in calls))
            self.assertFalse(list(Path(env["TMPDIR"]).iterdir()))

    def test_cancellation_stops_unresponsive_container_and_returns_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, env = self.run_ci(Path(temporary), interrupt=True)
            self.assertEqual(result.returncode, 143, result.stdout + result.stderr)
            self.assertIn(["rm", "-f", "owned-container-id"], calls)
            self.assertIn("--entrypoint", calls[-1])
            self.assertFalse(list(Path(env["TMPDIR"]).iterdir()))
            lock = Path(env["CI_NPU_LOCK_DIR"]) / "fla-npu-ci-npu-0.lock"
            self.assertEqual(subprocess.run(["flock", "-n", str(lock), "true"]).returncode, 0)

    def test_shared_host_temp_root_is_never_given_to_ownership_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, _ = self.run_ci(Path(temporary), broad_tmp=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("dedicated temporary subdirectory", result.stderr)
            self.assertFalse(any(call[0] == "run" for call in calls))


if __name__ == "__main__":
    unittest.main()
