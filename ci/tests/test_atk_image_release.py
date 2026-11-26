"""Both NPU images must install and run the same ABI-matched ATK release."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch


CI = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("atk_release", CI / "atk_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
spec = importlib.util.spec_from_file_location("atk_smoke", CI.parent / "tests/atk/smoke/run.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


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

    def test_release_report_rows_are_checked_for_each_output(self):
        case = {"id": 1, "name": "trace", "standard": {"acc": smoke.STANDARD}}
        output = {"filename": "output_0.pt", "result": True, "new_benchmark_indicate": {"Benchmark_error": 0}}
        row = {"用例json信息": case, "cpu_cpu_golden_精度通过": True, "精度详情": {"cpu_cpu_golden": [output]}}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            report = folder / "atk_output/run/report/accuracy.json"
            report.parent.mkdir(parents=True)
            for role in ("dut", "low", "high"):
                (folder / f"events.{role}").write_text(json.dumps([1, "trace", [[[2], "float32"]]]))
            for data in ([row], {"statistic": [row], "remote_inf_nan": []}):
                report.write_text(json.dumps(data))
                smoke.check_report(folder, [case])
            output["result"] = False
            report.write_text(json.dumps({"statistic": [row]}))
            with self.assertRaisesRegex(ValueError, "1 failed"):
                smoke.check_report(folder, [case])
            report.write_text(json.dumps({"statistic": []}))
            with self.assertRaisesRegex(ValueError, "1 missing"):
                smoke.check_report(folder, [case])
