import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ci.pkg.make_packages import (
    _terminate_process_group, _process_group_alive, _remote_build_startup, Remote, run_logged, build_arm, BuildCleanupError,
)


@unittest.skipUnless(os.name == "posix", "process-group semantics are POSIX-specific")
class ProcessControlTest(unittest.TestCase):
    def test_unconfirmed_cleanup_has_nonretryable_error_type(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("ci.pkg.make_packages._terminate_process_group", side_effect=RuntimeError("still alive")):
            with self.assertRaises(BuildCleanupError):
                run_logged([sys.executable, "-c", "pass"], Path(tmp)/"build.log")

    def test_docker_cleanup_after_cli_interruption(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("ci.pkg.make_packages.clean_build_state"), \
             mock.patch("ci.pkg.make_packages.subprocess.run") as run, \
             mock.patch("ci.pkg.make_packages.run_logged", side_effect=KeyboardInterrupt) as logged:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            path = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                build_arm(path, "ascend950", path/"wheels", path/"build.log")
            argv = logged.call_args.args[0]
            container = argv[argv.index("--name")+1]
            self.assertEqual(run.call_args.args[0], ["docker", "rm", "-f", container])

    def test_remote_build_records_process_group_and_cancellation(self):
        source = (REPO_ROOT / "ci" / "pkg" / "make_packages.py").read_text()
        self.assertIn("nohup setsid bash -c", source)
        self.assertIn("> pgid", source)
        self.assertIn("cancel_remote_process_group(out_dir)", source)

    def test_terminate_process_group_stops_child_tree(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys, time; "
                    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                    "time.sleep(60)"
                ),
            ],
            start_new_session=True,
        )
        _terminate_process_group(process, grace_seconds=1)
        self.assertIsNotNone(process.poll())

    def test_run_logged_stop_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = threading.Event()
            timer = threading.Timer(0.2, stop.set)
            timer.start()
            try:
                rc = run_logged(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('started', flush=True); time.sleep(60)",
                    ],
                    Path(tmp) / "build.log",
                    stop=stop,
                )
            finally:
                timer.cancel()
            self.assertNotEqual(rc, 0)
            self.assertIn("started", (Path(tmp) / "build.log").read_text())

    def test_child_ignoring_int_is_stopped_after_leader_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp)/"ready"
            child = ("import signal,time; from pathlib import Path; "
                     "signal.signal(signal.SIGINT,signal.SIG_IGN); "
                     "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                     f"Path({str(ready)!r}).touch(); time.sleep(60)")
            # Background shells may pass SIGINT=SIG_IGN to this fixture. The
            # leader must handle SIGINT so this case actually tests an exited
            # leader with a surviving child, regardless of how CI was started.
            parent = subprocess.Popen([sys.executable, "-c",
                "import signal,subprocess,sys,time; "
                "signal.signal(signal.SIGINT,signal.default_int_handler); "
                f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic()+5
                while not ready.exists() and time.monotonic()<deadline: time.sleep(.02)
                self.assertTrue(ready.exists())
                # Exercise the already-exited-leader case as well.
                parent.send_signal(signal.SIGINT)
                parent.wait(timeout=5)
                self.assertTrue(_process_group_alive(parent.pid))
                _terminate_process_group(parent, grace_seconds=.15)
                self.assertFalse(_process_group_alive(parent.pid))
            finally:
                try: os.killpg(parent.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                parent.wait()

    def test_run_logged_cleans_descendants_after_successful_leader_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            ready = path/"pgid"
            child = ("import os,signal,time; from pathlib import Path; "
                     "signal.signal(signal.SIGINT,signal.SIG_IGN); "
                     f"Path({str(ready)!r}).write_text(str(os.getpgrp())); time.sleep(60)")
            parent = (
                "import subprocess,sys,time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                f"ready=Path({str(ready)!r})\n"
                "for _ in range(250):\n"
                " if ready.exists(): break\n"
                " time.sleep(.02)\n"
                "assert ready.exists()\n")
            try:
                with mock.patch("ci.pkg.make_packages._terminate_process_group",
                                side_effect=lambda p: _terminate_process_group(p, grace_seconds=.15)):
                    self.assertEqual(run_logged([sys.executable, "-c", parent], path/"build.log"), 0)
                self.assertFalse(_process_group_alive(int(ready.read_text())))
            finally:
                if ready.exists():
                    try: os.killpg(int(ready.read_text()), signal.SIGKILL)
                    except ProcessLookupError: pass

    def test_real_detached_remote_topology_cancels_whole_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/"build.sh").write_text("trap '' INT TERM\necho $$ > worker.pid\necho ready > ready\nwhile true; do sleep 60; done\n")
            remote = Remote.__new__(Remote)
            remote.exec = lambda script, timeout: subprocess.run(
                ["bash", "-s"], input=script, text=True, capture_output=True, timeout=timeout)
            pgid = None
            try:
                remote.check(_remote_build_startup(str(path), 60), timeout=5)
                deadline = time.monotonic()+5
                while not (path/"ready").exists() and time.monotonic()<deadline: time.sleep(.02)
                self.assertTrue((path/"ready").exists())
                pgid = int((path/"pgid").read_text())
                self.assertEqual(os.getpgid(int((path/"worker.pid").read_text())), pgid)
                start_time = (path/"start_time").read_text()
                (path/"start_time").write_text("0\n")
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    remote.cancel_remote_process_group(str(path))
                self.assertTrue(_process_group_alive(pgid))
                (path/"start_time").write_text(start_time)
                remote.cancel_remote_process_group(str(path))
                self.assertFalse(_process_group_alive(pgid))
            finally:
                if pgid is not None:
                    try: os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError: pass


if __name__ == "__main__":
    unittest.main()
