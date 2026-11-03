#!/usr/bin/env python3
"""Trusted CI replacement must retain PR case data without following PR links."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from ci.apply_trusted_ci import apply_trusted_ci


@unittest.skipUnless(os.name == "posix", "Linux directory-descriptor protection")
class ApplyTrustedCiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.trusted = self.root / "trusted-ci"
        for root in (self.repo, self.trusted):
            (root / "ci").mkdir(parents=True)
        (self.repo / "ci/run_checks.sh").write_text("PR script must not execute")
        (self.trusted / "ci/run_checks.sh").write_text("trusted script")
        (self.trusted / "ci/example_st_cases.json").write_text('{"cases": []}')

    def test_uses_trusted_scripts_and_current_pr_cases(self):
        cases = {"cases": [{"name": "new PR case", "enabled": True}]}
        (self.repo / "ci/example_st_cases.json").write_text(json.dumps(cases))
        (self.repo / "ci/untrusted_extra.py").write_text("bad")
        (self.repo / "kernel.cpp").write_text("PR operator")
        apply_trusted_ci(self.repo, self.trusted)
        self.assertEqual((self.repo / "ci/run_checks.sh").read_text(), "trusted script")
        self.assertEqual(json.loads((self.repo / "ci/example_st_cases.json").read_text()), cases)
        self.assertFalse((self.repo / "ci/untrusted_extra.py").exists())
        self.assertEqual((self.repo / "kernel.cpp").read_text(), "PR operator")

    def test_missing_pr_cases_keep_trusted_defaults(self):
        apply_trusted_ci(self.repo, self.trusted)
        self.assertEqual(json.loads((self.repo / "ci/example_st_cases.json").read_text()), {"cases": []})

    def test_rejects_invalid_json_before_modifying_pr_ci(self):
        (self.repo / "ci/example_st_cases.json").write_text("invalid JSON")
        with self.assertRaises(ValueError):
            apply_trusted_ci(self.repo, self.trusted)
        self.assertEqual((self.repo / "ci/run_checks.sh").read_text(), "PR script must not execute")

    def test_rejects_case_symlink_directory_and_fifo(self):
        cases = self.repo / "ci/example_st_cases.json"
        external = self.root / "secret.json"
        external.write_text('{"private": true}')
        for kind in ("symlink", "directory", "fifo"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    cases.symlink_to(external)
                elif kind == "directory":
                    cases.mkdir()
                else:
                    os.mkfifo(cases)
                try:
                    with self.assertRaises((OSError, ValueError)):
                        apply_trusted_ci(self.repo, self.trusted)
                    self.assertEqual((self.repo / "ci/run_checks.sh").read_text(), "PR script must not execute")
                finally:
                    cases.rmdir() if cases.is_dir() else cases.unlink()

    def test_rejects_ci_symlink_without_touching_external_directory(self):
        (self.repo / "ci/run_checks.sh").unlink()
        (self.repo / "ci").rmdir()
        external = self.root / "external"
        external.mkdir()
        (external / "keep").write_text("private")
        (self.repo / "ci").symlink_to(external, target_is_directory=True)
        with self.assertRaises(OSError):
            apply_trusted_ci(self.repo, self.trusted)
        self.assertEqual((external / "keep").read_text(), "private")

    def test_rejects_overlapping_checkouts_and_missing_trusted_ci(self):
        for path in (self.repo, self.repo / "nested", self.root, self.root / "missing"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                apply_trusted_ci(self.repo, path)
        self.assertEqual((self.repo / "ci/run_checks.sh").read_text(), "PR script must not execute")


if __name__ == "__main__":
    unittest.main()
