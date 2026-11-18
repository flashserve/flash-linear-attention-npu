"""Offline tests for daily SHA coordination and public matrix validation."""

from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import patch

import daily_plan
import verify_daily_public as public_verify


class DailyPlanTests(unittest.TestCase):
    def test_plan_uses_fixed_shanghai_cutoff(self):
        sha_by_branch = {"main": "a" * 40, "v26.10.0": "b" * 40}
        with patch.object(daily_plan, "has_formal_release",
                          return_value=(False, "")), \
                patch.object(daily_plan, "resolve_commit",
                             side_effect=lambda branch, *_: sha_by_branch[branch]):
            plan = daily_plan.build_plan(
                dt.date(2026, 10, 31), 1,
                ["main", "v26.10.0"], "owner/repo", "https://api.test", "token")
        self.assertEqual(plan["cutoff"], "2026-10-31T01:00:00+08:00")
        self.assertEqual(plan["cutoff_utc"], "2026-10-30T17:00:00Z")
        self.assertEqual(
            [(row["branch"], row["sha"]) for row in plan["branches"]],
            list(sha_by_branch.items()),
        )

    def test_formal_release_branch_is_excluded(self):
        def released(branch, *_):
            return (branch == "v26.10.0", "https://release.test")

        with patch.object(daily_plan, "has_formal_release", side_effect=released), \
                patch.object(daily_plan, "resolve_commit", return_value="a" * 40):
            plan = daily_plan.build_plan(
                dt.date(2026, 10, 31), 1,
                ["main", "v26.10.0"], "owner/repo", "https://api.test", "token")
        self.assertEqual([row["branch"] for row in plan["branches"]], ["main"])
        self.assertEqual(plan["skipped"][0]["branch"], "v26.10.0")

    def test_invalid_sha_is_rejected(self):
        plan = {
            "schema": daily_plan.SCHEMA,
            "date": "20261031",
            "branches": [{"branch": "main", "sha": "short"}],
        }
        with self.assertRaisesRegex(RuntimeError, "非法"):
            daily_plan.validate_plan(plan)


class PublicVerifyTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "schema": daily_plan.SCHEMA,
            "date": "20261031",
            "branches": [
                {"branch": "main", "sha": "a" * 40},
                {"branch": "v26.10.0", "sha": "b" * 40},
            ],
        }

    def test_two_branches_two_arches_all_platforms_is_twelve(self):
        rows = public_verify.expected_rows(self.plan)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["arch"] for row in rows}, {"arm", "x86"})

    def test_filename_match_checks_commit_and_arch(self):
        row = public_verify.expected_rows(self.plan)[0]
        expected = (
            f"{row['package']}-{row['version']}"
            f"-py3-none-manylinux_2_34_{row['wheel_arch']}.whl"
        )
        wrong_arch = expected.replace(row["wheel_arch"], "x86_64")
        self.assertEqual(public_verify.match_filename(row, {wrong_arch, expected}), expected)


if __name__ == "__main__":
    unittest.main()
