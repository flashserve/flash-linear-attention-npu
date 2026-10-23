import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ci.pkg.make_packages import _terminate_process_group, run_logged


@unittest.skipUnless(os.name == "posix", "process-group semantics are POSIX-specific")
class ProcessControlTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
