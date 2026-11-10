"""Validate the strict designated Code Owner approval policy."""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github/scripts/codeowner_approval_gate.js"
WORKFLOW = ROOT / ".github/workflows/codeowner-approval.yml"
EXPECTED = ["weinachuan", "weiwei-612", "chen-linxin", "Ensley0304", "zhangshuolei-hfut"]
NODE = os.environ.get("CODEOWNER_NODE") or shutil.which("node") or shutil.which("nodejs")


class CodeOwnerApprovalGateTest(unittest.TestCase):
    def run_helper(self, operation, **options):
        wrapper = r"""
const gate = require(process.env.CODEOWNER_APPROVAL_SCRIPT);
const input = JSON.parse(process.env.TEST_INPUT);
try {
  const result = gate[input.operation](...(input.args || []));
  process.stdout.write(JSON.stringify({result}));
} catch (error) {
  process.stdout.write(JSON.stringify({error:error.message}));
}
"""
        env = os.environ.copy()
        env.update(
            CODEOWNER_APPROVAL_SCRIPT=str(HELPER),
            TEST_INPUT=json.dumps({"operation": operation, "args": options.get("args", [])}),
        )
        if NODE.lower().endswith(".exe"):
            existing = env.get("WSLENV", "")
            env["WSLENV"] = ":".join(filter(
                None, (existing, "CODEOWNER_APPROVAL_SCRIPT/p", "TEST_INPUT"),
            ))
        completed = subprocess.run(
            [NODE], input=wrapper, text=True, encoding="utf-8",
            capture_output=True, cwd=ROOT, env=env, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_codeowners_must_be_exactly_the_designated_users(self):
        text = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
        result = self.run_helper("parseCodeowners", args=[text])
        self.assertNotIn("error", result)
        self.assertEqual(result["result"], EXPECTED)

    def test_path_specific_owner_outside_designated_users_fails_closed(self):
        text = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
        result = self.run_helper(
            "parseCodeowners",
            args=[text + "\nfla/special/** @outsider\n"],
        )
        self.assertIn("所有路径级 Code Owner", result["error"])

    def test_outsider_approvals_do_not_count(self):
        reviews = [
            {"user": {"login": "outsider"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:00:00Z"},
            {"user": {"login": "weinachuan"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:01:00Z"},
        ]
        result = self.run_helper("computeActiveApprovals", args=[reviews, EXPECTED])
        self.assertEqual(result["result"], ["weinachuan"])

    def test_two_designated_approvals_pass_with_outsider_present(self):
        reviews = [
            {"user": {"login": "outsider"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:00:00Z"},
            {"user": {"login": "chen-linxin"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:01:00Z"},
            {"user": {"login": "Ensley0304"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:02:00Z"},
        ]
        result = self.run_helper("computeActiveApprovals", args=[reviews, EXPECTED])
        self.assertEqual(result["result"], ["chen-linxin", "Ensley0304"])

    def test_dismissed_or_new_changes_requested_review_removes_approval(self):
        for final_state in ("DISMISSED", "CHANGES_REQUESTED"):
            with self.subTest(final_state=final_state):
                reviews = [
                    {"user": {"login": "weiwei-612"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:00:00Z"},
                    {"user": {"login": "weiwei-612"}, "state": final_state, "submitted_at": "2026-11-06T10:01:00Z"},
                ]
                result = self.run_helper("computeActiveApprovals", args=[reviews, EXPECTED])
                self.assertEqual(result["result"], [])

    def test_comment_after_approval_does_not_clear_effective_approval(self):
        reviews = [
            {"user": {"login": "weiwei-612"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:00:00Z"},
            {"user": {"login": "weiwei-612"}, "state": "COMMENTED", "submitted_at": "2026-11-06T10:01:00Z"},
        ]
        result = self.run_helper("computeActiveApprovals", args=[reviews, EXPECTED])
        self.assertEqual(result["result"], ["weiwei-612"])

    def test_designated_owner_without_write_permission_fails_closed(self):
        result = self.run_helper(
            "requireReviewerWritePermissions",
            args=[{"weinachuan": "write", "weiwei-612": "read"}],
        )
        self.assertIn("至少需要 write", result["error"])

    def test_codeowners_ref_uses_current_target_branch_not_stale_base_sha(self):
        result = self.run_helper(
            "getTargetBranchRef",
            args=[{"base": {"ref": "main", "sha": "98e0ddf6cba844e46e412372f84685b1dae42b44"}}],
        )
        self.assertEqual(result["result"], "main")

    def test_missing_target_branch_ref_fails_closed(self):
        result = self.run_helper("getTargetBranchRef", args=[{"base": {"sha": "98e0ddf"}}])
        self.assertIn("无法确定 PR 当前目标分支", result["error"])

    def test_workflow_runs_on_review_and_pr_state_changes(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        events = workflow.get("on", workflow.get(True))
        self.assertEqual(
            events["pull_request_review"]["types"],
            ["submitted", "edited", "dismissed"],
        )
        self.assertIn("synchronize", events["pull_request_target"]["types"])
        self.assertEqual(workflow["permissions"]["statuses"], "write")
        checkout = workflow["jobs"]["evaluate"]["steps"][0]["with"]
        self.assertEqual(checkout["ref"], "${{ github.workflow_sha }}")
        script = workflow["jobs"]["evaluate"]["steps"][1]["with"]["script"]
        self.assertIn("gate.run", script)


if __name__ == "__main__":
    unittest.main()
