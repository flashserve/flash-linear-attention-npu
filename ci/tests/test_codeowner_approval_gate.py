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
REVIEW_TRIGGER = ROOT / ".github/workflows/codeowner-review-trigger.yml"
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

    def test_added_designated_owner_approval_counts(self):
        reviews = [
            {"user": {"login": "outsider"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:00:00Z"},
            {"user": {"login": "zhangshuolei-hfut"}, "state": "APPROVED", "submitted_at": "2026-11-06T10:01:00Z"},
        ]
        result = self.run_helper("computeActiveApprovals", args=[reviews, EXPECTED])
        self.assertEqual(result["result"], ["zhangshuolei-hfut"])

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

    def test_codeowners_ref_uses_current_target_branch_not_stale_base_sha(self):
        result = self.run_helper(
            "getTargetBranchRef",
            args=[{"base": {"ref": "main", "sha": "98e0ddf6cba844e46e412372f84685b1dae42b44"}}],
        )
        self.assertEqual(result["result"], "main")

    def test_missing_target_branch_ref_fails_closed(self):
        result = self.run_helper("getTargetBranchRef", args=[{"base": {"sha": "98e0ddf"}}])
        self.assertIn("无法确定 PR 当前目标分支", result["error"])

    def test_push_target_branch_comes_from_event_ref(self):
        result = self.run_helper("getPushTargetBranch", args=[{"ref": "refs/heads/v26.10.0"}])
        self.assertEqual(result["result"], "v26.10.0")

    def test_push_target_branch_rejects_non_branch_ref(self):
        result = self.run_helper("getPushTargetBranch", args=[{"ref": "refs/tags/v26.10.0"}])
        self.assertIn("无法确定 push 事件的目标分支", result["error"])

    def test_pull_number_parser_accepts_positive_integers(self):
        for value, expected in ((1144, 1144), ("1144", 1144), (" 1144 ", 1144)):
            with self.subTest(value=value):
                result = self.run_helper("parsePullNumber", args=[value])
                self.assertEqual(result["result"], expected)

    def test_pull_number_parser_rejects_invalid_values(self):
        for value in (None, "", 0, -1, "1.5", "1144x", 9007199254740992):
            with self.subTest(value=value):
                result = self.run_helper("parsePullNumber", args=[value])
                self.assertIn("PR 编号必须是合法的正整数", result["error"])

    def test_workflow_runs_on_review_and_pr_state_changes(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        events = workflow.get("on", workflow.get(True))
        self.assertNotIn("pull_request_review", events)
        self.assertEqual(events["workflow_run"]["workflows"], ["Code Owner Approval Review Trigger"])
        self.assertEqual(events["workflow_run"]["types"], ["completed"])
        self.assertIn("synchronize", events["pull_request_target"]["types"])
        self.assertEqual(events["push"]["branches"], ["main"])
        self.assertIn(".github/scripts/codeowner_approval_gate.js", events["push"]["paths"])
        self.assertIn(".github/CODEOWNERS", events["push"]["paths"])
        self.assertIn(".github/workflows/codeowner-review-trigger.yml", events["push"]["paths"])
        self.assertEqual(workflow["permissions"]["actions"], "read")
        self.assertEqual(workflow["permissions"]["statuses"], "write")
        job = workflow["jobs"]["evaluate"]
        self.assertIn("workflow_run.conclusion == 'success'", job["if"])
        checkout = job["steps"][0]["with"]
        self.assertEqual(checkout["ref"], "${{ github.event.pull_request.base.ref || github.ref_name }}")
        download = job["steps"][1]
        self.assertEqual(download["uses"], "actions/download-artifact@v4")
        self.assertEqual(download["with"]["run-id"], "${{ github.event.workflow_run.id }}")
        self.assertIn("github.run_id", workflow["concurrency"]["group"])
        script = job["steps"][2]["with"]["script"]
        self.assertIn("gate.parsePullNumber", script)
        self.assertIn("gate.run", script)

        trigger = yaml.safe_load(REVIEW_TRIGGER.read_text(encoding="utf-8"))
        trigger_events = trigger.get("on", trigger.get(True))
        self.assertEqual(
            trigger_events["pull_request_review"]["types"],
            ["submitted", "edited", "dismissed"],
        )
        upload = trigger["jobs"]["capture"]["steps"][1]
        self.assertEqual(upload["uses"], "actions/upload-artifact@v4")
        self.assertEqual(upload["with"]["name"], "codeowner-review-pr")


if __name__ == "__main__":
    unittest.main()
