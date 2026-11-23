"""Keep the web forms, API policy and trusted workflow entry point aligned."""

import json
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class IssuePolicyContractTest(unittest.TestCase):
    def test_forms_and_server_require_the_same_fields(self):
        result = subprocess.run(
            ["node", "-e", "const p=require('./.github/scripts/issue_policy');"
             "process.stdout.write(JSON.stringify({fields:p.FIELDS,required:p.REQUIRED}));"],
            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
        )
        policy = json.loads(result.stdout)
        for kind, filename in [("bug", "bug_report.yml"), ("enhancement", "feature_request.yml"),
                               ("question", "question.yml")]:
            form = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE" / filename).read_text(encoding="utf-8"))
            fields = {item["attributes"]["label"]: item for item in form["body"]
                      if item["type"] != "markdown"}
            for name in policy["fields"] + policy["required"][kind]:
                self.assertTrue(fields[name]["validations"]["required"], (kind, name))
            self.assertEqual(fields["Issue 类型"]["attributes"]["options"], [kind])
            self.assertNotIn("value", fields["涉及分支"]["attributes"])

    def test_workflow_uses_default_branch_with_only_issue_write_permission(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/issue-policy.yml").read_text(encoding="utf-8"))
        events = workflow.get("on", workflow.get(True))
        self.assertTrue({"opened", "edited", "reopened", "labeled", "unlabeled"}.issubset(
            events["issues"]["types"]))
        self.assertIn("workflow_dispatch", events)
        self.assertEqual(workflow["permissions"], {"contents": "read", "issues": "write"})
        self.assertFalse(workflow["concurrency"]["cancel-in-progress"])
        checkout = workflow["jobs"]["validate"]["steps"][0]
        self.assertEqual(checkout["with"]["ref"], "${{ github.event.repository.default_branch }}")
        self.assertFalse(checkout["with"]["persist-credentials"])
        script = workflow["jobs"]["validate"]["steps"][1]["with"]["script"]
        self.assertIn("policy.run({ github, context, core })", script)
        self.assertNotIn("${{", script)

    def test_web_blank_entry_is_disabled_and_runtime_tests_run_in_ci(self):
        config = yaml.safe_load((ROOT / ".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8"))
        self.assertFalse(config["blank_issues_enabled"])
        ci = yaml.safe_load((ROOT / ".github/workflows/ci-contract-tests.yml").read_text(encoding="utf-8"))
        self.assertTrue(any(step.get("run") == "node --test .github/scripts/tests/issue_policy.test.js"
                            for step in ci["jobs"]["contract-tests"]["steps"]))


if __name__ == "__main__":
    unittest.main()
