"""Exercise issue label policy with GitHub issue/event payloads, without API writes."""

from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "issue_label_policy", ROOT / ".github/scripts/issue_label_policy.py"
)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def form(issue_type="bug", branches="main"):
    return f"### 问题类型\n\n{issue_type}\n\n### 目标分支\n\n{branches}\n"


class IssueLabelPolicyTests(unittest.TestCase):
    branches = ["main", "v26.9.1", "v26.10.0", "v26.11.0", "A6-dev"]

    def plan(self, labels=(), body="", event=None, **extra):
        issue = {
            "number": 123,
            "title": "Issue policy test",
            "state": "open",
            "labels": [{"name": name} for name in labels],
            "body": body,
            **extra,
        }
        return POLICY.plan_issue(issue, self.branches, event or {"action": "opened"})

    def assert_valid(self, plan):
        self.assertFalse(plan["skip"])
        self.assertEqual(plan["errors"], [])
        self.assertNotIn("needs-issue-info", plan["add"])

    def assert_invalid(self, plan):
        self.assertFalse(plan["skip"])
        self.assertTrue(plan["errors"])
        self.assertIn("needs-issue-info", plan["add"])

    def test_new_bug_form_adds_type_and_main(self):
        plan = self.plan(body=form())
        self.assert_valid(plan)
        self.assertEqual(set(plan["add"]), {"bug", "main"})
        self.assertEqual(plan["remove"], [])

    def test_enhancement_can_target_multiple_branches(self):
        plan = self.plan(labels=["enhancement"], body=form("enhancement", "main, v26.11.0"))
        self.assert_valid(plan)
        self.assertEqual(set(plan["add"]), {"main", "v26.11.0"})

    def test_api_issue_with_valid_labels_needs_no_form(self):
        for issue_type in ("bug", "enhancement"):
            with self.subTest(issue_type=issue_type):
                plan = self.plan(labels=[issue_type, "v26.11.0"])
                self.assert_valid(plan)
                self.assertEqual(plan["add"], [])
                self.assertEqual(plan["remove"], [])

    def test_labels_can_be_strings_in_api_payload(self):
        plan = POLICY.plan_issue(
            {"number": 123, "labels": ["bug", "main"], "body": None},
            self.branches,
            {"action": "opened"},
        )
        self.assert_valid(plan)

    def test_missing_type_or_branch_is_marked(self):
        for labels in ([], ["bug"], ["main"]):
            with self.subTest(labels=labels):
                self.assert_invalid(self.plan(labels=labels))

    def test_two_type_labels_without_a_form_are_ambiguous(self):
        self.assert_invalid(self.plan(labels=["bug", "enhancement", "main"]))

    def test_nonexistent_and_disallowed_branches_are_rejected(self):
        for branch in ("v26.12.0", "A6-dev", "main; echo unsafe", "_No response_"):
            with self.subTest(branch=branch):
                plan = self.plan(labels=["bug"], body=form(branches=branch))
                self.assert_invalid(plan)
                self.assertNotIn(branch, plan["add"])

    def test_nonexistent_version_label_does_not_satisfy_requirement(self):
        self.assert_invalid(self.plan(labels=["bug", "v26.12.0"]))

    def test_invalid_type_cannot_create_an_arbitrary_label(self):
        plan = self.plan(body=form("administrator"))
        self.assert_invalid(plan)
        self.assertNotIn("administrator", plan["add"])

    def test_duplicate_type_fields_are_not_treated_as_valid_input(self):
        body = form() + "\n### 问题类型\n\nenhancement\n"
        plan = self.plan(body=body)
        self.assert_invalid(plan)
        self.assertFalse({"bug", "enhancement"} & set(plan["add"]))

    def test_duplicate_branch_fields_are_not_treated_as_valid_input(self):
        body = form() + "\n### 目标分支\n\nv26.11.0\n"
        plan = self.plan(labels=["bug"], body=body)
        self.assert_invalid(plan)
        self.assertFalse({"main", "v26.11.0"} & set(plan["add"]))

    def test_fenced_example_does_not_supply_required_fields(self):
        for fence in ("```", "~~~~"):
            with self.subTest(fence=fence):
                plan = self.plan(body=f"{fence}\n{form()}{fence}\n")
                self.assert_invalid(plan)
                self.assertFalse({"bug", "main"} & set(plan["add"]))

    def test_fenced_example_does_not_override_real_form_values(self):
        body = "```markdown\n" + form("bug", "v26.9.1") + "```\n" + form("enhancement", "main")
        plan = self.plan(body=body)
        self.assert_valid(plan)
        self.assertEqual(set(plan["add"]), {"enhancement", "main"})

    def test_unrelated_edit_preserves_maintainer_branch_labels(self):
        old_body = form(branches="v26.9.1")
        plan = self.plan(
            labels=["bug", "main", "v26.11.0"],
            body=old_body + "\n### 其他补充\n\nUpdated reproduction.\n",
            event={"action": "edited", "changes": {"body": {"from": old_body}}},
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])

    def test_label_event_preserves_maintainer_type_and_branch(self):
        plan = self.plan(
            labels=["enhancement", "v26.11.0"],
            body=form("bug", "main"),
            event={"action": "labeled"},
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])

    def test_missing_last_branch_can_be_restored_from_form(self):
        plan = self.plan(labels=["bug"], body=form(), event={"action": "unlabeled"})
        self.assert_valid(plan)
        self.assertEqual(plan["add"], ["main"])

    def test_missing_type_can_be_restored_from_form(self):
        plan = self.plan(labels=["main"], body=form(), event={"action": "unlabeled"})
        self.assert_valid(plan)
        self.assertEqual(plan["add"], ["bug"])

    def test_editing_branch_field_replaces_managed_branch_labels(self):
        plan = self.plan(
            labels=["bug", "v26.9.1", "op:kda", "performance"],
            body=form(branches="main, v26.11.0"),
            event={"action": "edited", "changes": {"body": {"from": form(branches="v26.9.1")}}},
        )
        self.assert_valid(plan)
        self.assertEqual(set(plan["add"]), {"main", "v26.11.0"})
        self.assertEqual(set(plan["remove"]), {"v26.9.1"})

    def test_editing_type_field_replaces_type_without_reverting_branches(self):
        plan = self.plan(
            labels=["bug", "v26.11.0", "op:kda"],
            body=form("enhancement", "main"),
            event={"action": "edited", "changes": {"body": {"from": form("bug", "main")}}},
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], ["enhancement"])
        self.assertEqual(plan["remove"], ["bug"])

    def test_invalid_branch_edit_does_not_remove_existing_valid_labels(self):
        plan = self.plan(
            labels=["bug", "main", "op:kda"],
            body=form(branches="does-not-exist"),
            event={"action": "edited", "changes": {"body": {"from": form()}}},
        )
        self.assert_invalid(plan)
        self.assertEqual(plan["remove"], [])

    def test_unrelated_events_do_not_clear_unfixed_form_error(self):
        for event in (
            {"action": "labeled"},
            {"action": "edited", "changes": {"body": {"from": form(branches="invalid")}}},
        ):
            with self.subTest(event=event["action"]):
                plan = self.plan(
                    labels=["bug", "main", "needs-issue-info", "op:kda"],
                    body=form(branches="invalid") + "\n### 其他补充\n\nMore information.\n",
                    event=event,
                )
                self.assertTrue(plan["errors"])
                self.assertNotIn("needs-issue-info", plan["remove"])
                self.assertNotIn("invalid", plan["add"])

    def test_old_opened_event_preserves_later_manual_branch_choice(self):
        body = form(branches="main")
        plan = self.plan(
            labels=["bug", "v26.11.0"], body=body,
            event={"action": "opened", "issue": {"body": body, "labels": ["bug", "main"]}},
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])

    def test_old_edited_event_preserves_later_manual_type_and_branch_choice(self):
        body = form("enhancement", "v26.11.0")
        plan = self.plan(
            labels=["bug", "v26.10.0"], body=body,
            event={
                "action": "edited",
                "issue": {"body": body, "labels": ["enhancement", "main"]},
                "changes": {"body": {"from": form("bug", "main")}},
            },
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])

    def test_auxiliary_label_added_after_edit_does_not_discard_form_choice(self):
        body = form("bug", "v26.11.0")
        plan = self.plan(
            labels=["bug", "main", "performance"], body=body,
            event={
                "action": "edited",
                "issue": {"body": body, "labels": ["bug", "main"]},
                "changes": {"body": {"from": form("bug", "main")}},
            },
        )
        self.assert_valid(plan)
        self.assertEqual(plan["add"], ["v26.11.0"])
        self.assertEqual(plan["remove"], ["main"])

    def test_fixing_an_api_issue_clears_pending_marker(self):
        plan = self.plan(
            labels=["enhancement", "main", "needs-issue-info", "op:kda"],
            event={"action": "labeled"},
        )
        self.assert_valid(plan)
        self.assertEqual(plan["remove"], ["needs-issue-info"])

    def test_special_issue_categories_need_no_branch(self):
        for label in ("question", "committer-promotion"):
            with self.subTest(label=label):
                plan = self.plan(labels=[label])
                self.assert_valid(plan)
                self.assertEqual(plan["add"], [])
                self.assertEqual(plan["remove"], [])

    def test_system_index_is_exempt(self):
        plan = self.plan(labels=["requirements-index"])
        self.assertTrue(plan["skip"])
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])

    def test_special_label_cannot_bypass_bug_requirements(self):
        self.assert_invalid(self.plan(labels=["bug", "question"]))

    def test_multiple_primary_types_are_invalid_even_with_a_branch(self):
        for labels in (
            ["bug", "question", "main"],
            ["enhancement", "committer-promotion", "main"],
            ["question", "committer-promotion"],
        ):
            with self.subTest(labels=labels):
                self.assert_invalid(self.plan(labels=labels))

    def test_explicit_form_type_change_removes_old_primary_type(self):
        body = form("bug", "main")
        for previous in ("question", "committer-promotion"):
            with self.subTest(previous=previous):
                plan = self.plan(
                    labels=[previous, "performance"], body=body,
                    event={
                        "action": "edited",
                        "issue": {"body": body, "labels": [previous, "performance"]},
                        "changes": {"body": {"from": "### 背景\n\nOriginal inquiry.\n"}},
                    },
                )
                self.assert_valid(plan)
                self.assertEqual(set(plan["add"]), {"bug", "main"})
                self.assertEqual(plan["remove"], [previous])

    def test_manual_change_to_special_type_is_not_reverted_by_old_form(self):
        for label in ("question", "committer-promotion"):
            with self.subTest(label=label):
                plan = self.plan(
                    labels=[label, "needs-issue-info"], body=form("bug", "v26.99.0"),
                    event={"action": "labeled"},
                )
                self.assert_valid(plan)
                self.assertEqual(plan["add"], [])
                self.assertEqual(plan["remove"], ["needs-issue-info"])

    def test_pull_requests_are_exempt(self):
        plan = self.plan(pull_request={"url": "https://api.github.com/repos/example/repo/pulls/123"})
        self.assertTrue(plan["skip"])
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["remove"], [])


class IssueLabelRuntimeTests(unittest.TestCase):
    def make_api(self, labels, body, existing=None, failures=None, ignored_writes=()):
        issue = {
            "number": 123,
            "state": "open",
            "body": body,
            "labels": [{"name": name} for name in labels],
        }
        calls = []
        failures = failures or {}

        def request(method, path, data=None):
            calls.append((method, path, deepcopy(data)))
            failure = failures.get((method, path))
            if failure:
                raise HTTPError("https://api.github.com/test", failure, "mock failure", {}, None)
            if method == "GET" and path == "issues/123":
                return deepcopy(issue)
            if method == "GET" and path == "":
                return {"default_branch": "main"}
            if method == "GET" and path.startswith("labels/"):
                return {"name": path[len("labels/"):]}
            if method == "POST" and path == "issues/123/labels" and "add" not in ignored_writes:
                names = {entry["name"] for entry in issue["labels"]} | set(data["labels"])
                issue["labels"] = [{"name": name} for name in sorted(names)]
            if method == "DELETE" and path.startswith("issues/123/labels/") and "remove" not in ignored_writes:
                target = path[len("issues/123/labels/"):]
                issue["labels"] = [entry for entry in issue["labels"] if entry["name"] != target]
            return None

        def list_all(path):
            if path == "branches":
                return [{"name": name} for name in ["main", "v26.11.0"]]
            if path == "labels":
                return [{"name": name} for name in (existing if existing is not None else labels)]
            raise AssertionError(f"Unexpected list request: {path}")

        api = Mock()
        api.request.side_effect = request
        api.list_all.side_effect = list_all
        api.calls = calls
        api.issue_snapshot = deepcopy(issue)
        return api

    def run_main(self, api, event):
        self.output = io.StringIO()
        event = deepcopy(event)
        if set(event.get("issue", {})) == {"number"}:
            event["issue"] = deepcopy(api.issue_snapshot)
        with tempfile.TemporaryDirectory() as directory:
            event_file = Path(directory) / "event.json"
            event_file.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
            with patch.dict(os.environ, {"GITHUB_EVENT_PATH": str(event_file)}):
                with patch.object(POLICY, "GitHub", return_value=api), redirect_stdout(self.output):
                    return POLICY.main()

    def test_creates_missing_branch_label_before_assigning_it(self):
        api = self.make_api(["enhancement"], form("enhancement", "v26.11.0"))
        result = self.run_main(api, {"action": "opened", "issue": {"number": 123}})
        self.assertEqual(result, 0)
        creation = next(call for call in api.calls if call[:2] == ("POST", "labels"))
        assignment = ("POST", "issues/123/labels", {"labels": ["v26.11.0"]})
        self.assertEqual(creation[2]["name"], "v26.11.0")
        self.assertLess(api.calls.index(creation), api.calls.index(assignment))
        self.assertIn(
            ("POST", "actions/workflows/requirements-index.yml/dispatches", {"ref": "main"}),
            api.calls,
        )

    def test_replaces_branch_by_adding_first_and_preserves_auxiliary_labels(self):
        api = self.make_api(
            ["bug", "main", "op:kda", "performance"],
            form("bug", "v26.11.0"),
            existing=["bug", "main", "v26.11.0", "op:kda", "performance"],
        )
        event = {
            "action": "edited",
            "issue": {"number": 123},
            "changes": {"body": {"from": form("bug", "main")}},
        }
        self.assertEqual(self.run_main(api, event), 0)
        assignment = ("POST", "issues/123/labels", {"labels": ["v26.11.0"]})
        removal = ("DELETE", "issues/123/labels/main", None)
        self.assertLess(api.calls.index(assignment), api.calls.index(removal))
        self.assertEqual([call for call in api.calls if call[0] == "DELETE"], [removal])
        self.assertFalse(any(call[0] in {"PUT", "PATCH"} for call in api.calls))
        readback = next(index for index, call in enumerate(api.calls)
                        if index > api.calls.index(removal) and call[:2] == ("GET", "issues/123"))
        dispatch = next(index for index, call in enumerate(api.calls)
                        if call[:2] == ("POST", "actions/workflows/requirements-index.yml/dispatches"))
        self.assertLess(readback, dispatch)

    def test_concurrent_label_creation_reads_back_after_422(self):
        api = self.make_api(
            ["bug"], form(), failures={("POST", "labels"): 422},
        )
        self.assertEqual(self.run_main(api, {"action": "opened", "issue": {"number": 123}}), 0)
        readback = ("GET", "labels/main", None)
        assignment = ("POST", "issues/123/labels", {"labels": ["main"]})
        self.assertLess(api.calls.index(readback), api.calls.index(assignment))

    def test_failed_422_readback_does_not_claim_success(self):
        api = self.make_api(
            ["bug"], form(),
            failures={("POST", "labels"): 422, ("GET", "labels/main"): 404},
        )
        with self.assertRaises(HTTPError):
            self.run_main(api, {"action": "opened", "issue": {"number": 123}})
        self.assertNotIn("检查通过", self.output.getvalue())
        self.assertFalse(any(call[:2] == ("POST", "issues/123/labels") for call in api.calls))

    def test_assignment_failure_stops_before_removing_old_branch(self):
        api = self.make_api(
            ["bug", "main"], form("bug", "v26.11.0"),
            existing=["bug", "main", "v26.11.0"],
            failures={("POST", "issues/123/labels"): 403},
        )
        with self.assertRaises(HTTPError):
            self.run_main(api, {
                "action": "edited",
                "issue": {"number": 123},
                "changes": {"body": {"from": form()}},
            })
        self.assertFalse(any(call[0] == "DELETE" for call in api.calls))
        self.assertNotIn("检查通过", self.output.getvalue())

    def test_repository_request_omits_trailing_slash(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"default_branch": "main"}'
        env = {
            "GITHUB_API_URL": "https://api.github.com",
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_TOKEN": "test-token",
        }
        with patch.dict(os.environ, env), patch.object(POLICY, "urlopen", return_value=response) as opener:
            api = POLICY.GitHub()
            self.assertEqual(api.request("GET", ""), {"default_branch": "main"})
            self.assertEqual(opener.call_args[0][0].full_url, "https://api.github.com/repos/example/repo")
            api.request("GET", "issues/123")
            self.assertEqual(opener.call_args[0][0].full_url, "https://api.github.com/repos/example/repo/issues/123")

    def test_success_response_without_added_label_fails_readback(self):
        api = self.make_api(["bug"], form(), ignored_writes=["add"])
        with self.assertRaisesRegex(RuntimeError, "读回不一致"):
            self.run_main(api, {"action": "opened", "issue": {"number": 123}})
        self.assertNotIn("检查通过", self.output.getvalue())
        self.assertFalse(any("dispatches" in call[1] for call in api.calls))

    def test_success_response_without_removed_label_fails_readback(self):
        api = self.make_api(
            ["bug", "main"], form("bug", "v26.11.0"), ignored_writes=["remove"],
        )
        with self.assertRaisesRegex(RuntimeError, "读回不一致"):
            self.run_main(api, {
                "action": "edited", "issue": {"number": 123},
                "changes": {"body": {"from": form("bug", "main")}},
            })
        self.assertNotIn("检查通过", self.output.getvalue())
        self.assertFalse(any("dispatches" in call[1] for call in api.calls))


class IssueLabelConfigurationTests(unittest.TestCase):
    def load(self, path):
        return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))

    def test_bug_and_enhancement_forms_require_type_and_same_branch_choices(self):
        branches = []
        for filename, issue_type in (("bug_report.yml", "bug"), ("feature_request.yml", "enhancement")):
            with self.subTest(filename=filename):
                template = self.load(f".github/ISSUE_TEMPLATE/{filename}")
                fields = {entry.get("id"): entry for entry in template["body"]}
                type_field = fields["issue_type"]
                branch_field = fields["target_branches"]
                self.assertEqual(type_field["type"], "dropdown")
                self.assertTrue(type_field["validations"]["required"])
                self.assertEqual(type_field["attributes"]["options"], [issue_type])
                self.assertIn(issue_type, template["labels"])
                self.assertEqual(branch_field["type"], "dropdown")
                self.assertTrue(branch_field["validations"]["required"])
                self.assertTrue(branch_field["attributes"]["multiple"])
                choices = branch_field["attributes"]["options"]
                self.assertIn("main", choices)
                self.assertEqual(len(choices), len(set(choices)))
                self.assertTrue(all(POLICY.BRANCH.fullmatch(name) for name in choices))
                branches.append(choices)
        self.assertEqual(branches[0], branches[1])

    def test_special_templates_keep_their_own_classification(self):
        for filename, label in (("question.yml", "question"), ("committer_promotion.yml", "committer-promotion")):
            with self.subTest(filename=filename):
                template = self.load(f".github/ISSUE_TEMPLATE/{filename}")
                self.assertEqual(template["labels"], [label])
                self.assertFalse(any(
                    entry.get("id") in {"issue_type", "target_branches"}
                    for entry in template["body"]
                ))

    def test_blank_issues_disabled_and_workflow_runs_trusted_default_branch(self):
        config = self.load(".github/ISSUE_TEMPLATE/config.yml")
        self.assertFalse(config["blank_issues_enabled"])
        workflow = self.load(".github/workflows/issue-labels.yml")
        self.assertEqual(workflow["concurrency"]["queue"], "max")
        version_workflow = self.load(".github/workflows/version-tags.yml")
        self.assertEqual(version_workflow["concurrency"]["queue"], "max")
        self.assertEqual(workflow["permissions"], {
            "contents": "read", "issues": "write", "actions": "write",
        })
        steps = workflow["jobs"]["labels"]["steps"]
        checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
        self.assertEqual(checkout["with"]["ref"], "${{ github.event.repository.default_branch }}")
        self.assertFalse(checkout["with"]["persist-credentials"])
        script = next(step for step in steps if step.get("run"))
        self.assertEqual(script["run"], "python .github/scripts/issue_label_policy.py")
        self.assertEqual(script["env"], {"GITHUB_TOKEN": "${{ github.token }}"})


if __name__ == "__main__":
    unittest.main()
