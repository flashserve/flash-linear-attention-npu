"""Both NPU images must install and run the same ABI-matched ATK release."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch


CI = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("atk_release", CI / "atk_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class AtkImageReleaseTests(unittest.TestCase):
    def test_ci_python_wheels_match_the_platform(self):
        with patch.object(release.sys, "platform", "linux"), patch.object(
            release.sys, "version_info", types.SimpleNamespace(major=3, minor=12)
        ):
            for arch in ("aarch64", "x86_64"):
                with self.subTest(arch=arch), patch.object(release.platform, "machine", return_value=arch):
                    name, digest = release.wheel_info()
                    self.assertEqual(name, f"atk-26.11.11-cp312-cp312-linux_{arch}.whl")
                    self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_bad_download_never_reaches_pip(self):
        with TemporaryDirectory() as tmp, patch.object(release, "wheel_info", return_value=("test.whl", "0" * 64)):
            (Path(tmp) / "test.whl").write_bytes(b"invalid wheel")
            with patch.object(release.subprocess, "run") as execute:
                with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                    release.install(Path(tmp))
                execute.assert_not_called()

    def test_a_shadowed_import_fails_the_ci_version_check(self):
        old = types.SimpleNamespace(PACKAGE_VERSION="26.9.8", __file__="/old/atk/__init__.py")
        with patch.dict(release.sys.modules, {"atk": old}), patch.object(
            release.metadata, "version", return_value=release.VERSION
        ), self.assertRaisesRegex(SystemExit, "imported=26.9.8"):
            release.check_version()

    def test_dockerfiles_and_ci_share_the_release_check(self):
        for file in (CI / "Dockerfile", CI / "Dockerfile.ascend950"):
            self.assertIn("RUN python3 /tmp/fla-ci/atk_release.py --install", file.read_text())
        self.assertIn("python3 ci/atk_release.py", (CI / "run_checks.sh").read_text())
