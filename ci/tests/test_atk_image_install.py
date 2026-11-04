#!/usr/bin/env python3
"""Verify controlled ATK image installation using a tiny local wheel, without network."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux image installer test")
class AtkImageInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log = self.root / "calls.jsonl"
        self.wheel = self.make_wheel()
        self.target = self.root / "installed"
        curl = self.bin_dir / "curl"
        curl.write_text("#!" + sys.executable + "\n" + r'''
import json, os, shutil, sys
with open(os.environ["ATK_INSTALL_TEST_LOG"], "a") as stream:
    stream.write(json.dumps({"tool": "curl"}) + "\n")
shutil.copyfile(os.environ["ATK_INSTALL_TEST_WHEEL"], sys.argv[sys.argv.index("-o") + 1])
''')
        curl.chmod(0o755)
        python = self.bin_dir / "python3"
        python.write_text("#!" + sys.executable + "\n" + r'''
import json, os, subprocess, sys
if sys.argv[1:3] == ["-m", "pip"]:
    with open(os.environ["ATK_INSTALL_TEST_LOG"], "a") as stream:
        stream.write(json.dumps({"tool": "pip", "argv": sys.argv[1:]}) + "\n")
    # Exercise pip only in a temporary target, never in the host or container environment.
    raise SystemExit(subprocess.call([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                                     "--target", os.environ["ATK_INSTALL_TEST_TARGET"], sys.argv[-1]]))
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        python.chmod(0o755)

    def make_wheel(self, name="atk", version="26.9.24", tag="py3-none-any", summary="The API Toolkit for Ascend OP in Python"):
        path = self.root / f"{name}-{version}-{tag}.whl"
        prefix = f"{name}-{version}.dist-info"
        files = {
            f"{name}/__init__.py": "",
            f"{prefix}/METADATA": f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\nSummary: {summary}\n",
            f"{prefix}/WHEEL": f"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: {tag}\n",
            f"{prefix}/entry_points.txt": "[console_scripts]\natk = atk.__main__:main\n",
        }
        files[f"{prefix}/RECORD"] = "".join(f"{entry},,\n" for entry in files) + f"{prefix}/RECORD,,\n"
        with zipfile.ZipFile(path, "w") as archive:
            for entry, content in files.items():
                archive.writestr(entry, content)
        return path

    def run_installer(self, **overrides):
        environment = {**os.environ, "PATH": str(self.bin_dir) + os.pathsep + os.environ["PATH"],
                       "PYTHONPATH": str(self.target), "ATK_INSTALL_TEST_LOG": str(self.log),
                       "ATK_INSTALL_TEST_WHEEL": str(self.wheel), "ATK_INSTALL_TEST_TARGET": str(self.target),
                       "ATK_WHL_URL": "https://controlled.example/" + self.wheel.name,
                       "ATK_WHL_SHA256": hashlib.sha256(self.wheel.read_bytes()).hexdigest()}
        environment.update(overrides)
        return subprocess.run(["bash", str(REPO_ROOT / "ci/install_atk.sh")], env=environment,
                              capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_verified_wheel_installs_only_into_test_target(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Installed ATK 26.9.24", result.stdout)
        self.assertEqual([call["tool"] for call in self.calls()], ["curl", "pip"])
        self.assertTrue((self.target / "atk-26.9.24.dist-info/METADATA").exists())

    def test_unconfigured_image_build_is_explicitly_without_atk(self):
        result = self.run_installer(ATK_WHL_URL="", ATK_WHL_SHA256="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("this image has no ATK", result.stdout)
        self.assertEqual(self.calls(), [])

    def test_unsafe_url_and_missing_checksum_fail_before_download(self):
        for changes in ({"ATK_WHL_URL": "http://controlled.example/atk.whl"}, {"ATK_WHL_SHA256": ""}):
            with self.subTest(changes=changes):
                result = self.run_installer(**changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])

    def test_checksum_mismatch_never_installs(self):
        result = self.run_installer(ATK_WHL_SHA256="0" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call["tool"] for call in self.calls()], ["curl"])
        self.assertFalse(self.target.exists())

    def test_wrong_architecture_or_version_never_installs(self):
        for settings in ({"tag": "cp38-cp38-linux_aarch64"}, {"version": "1.0.0"}, {"summary": "Unrelated same-name package"}):
            with self.subTest(settings=settings):
                if self.log.exists():
                    self.log.unlink()
                self.wheel = self.make_wheel(**settings)
                result = self.run_installer()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual([call["tool"] for call in self.calls()], ["curl"])
                self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
