"""Run default-state failure handling without checkout or live GitHub writes."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/npu-ci-default-status.yml"
DEFINITION = ROOT / ".github/scripts/npu_ci_checks.json"


class DefaultStatusFailureTests(unittest.TestCase):
    def run_workflow(self, **options):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["mark-not-run"]["steps"]
        scripts = [step["with"]["script"] for step in steps if step.get("uses") == "actions/github-script@v8"]
        wrapper = r'''
const fs = require('fs');
const settings = JSON.parse(process.env.TEST_OPTIONS);
const definition = JSON.parse(fs.readFileSync(process.env.TEST_DEFINITION, 'utf8'));
const contexts = definition.required.map(item => item.context);
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const eventPr = {number:527,state:'open',draft:false,changed_files:1,user:{login:settings.author || 'contributor'},
  html_url:'https://github.com/example/repo/pull/527',head:{sha:'a'.repeat(40)},
  base:{ref:'main',sha:'c'.repeat(40)}};
const pr = {...eventPr, ...settings.current_pr};
const context = {repo:{owner:'example',repo:'repo'},payload:{pull_request:eventPr,
  repository:{default_branch:'main'},action:settings.action || 'edited',
  changes:settings.title_edit ? {title:{from:'old'}} : {base:{ref:{from:'old'}}}}};
const listFiles = () => {};
const listStatuses = () => {};
const listComments = () => {};
const history = settings.history || contexts.map(context => ({context,state:'success',
  target_url:'https://github.com/example/repo/actions/runs/7'}));
let normalFailed = false;
const github = {paginate:async (endpoint) => {
  if (endpoint === listFiles) {
    if (settings.files_error) throw new Error('file list unavailable');
    return [{filename:settings.file || 'fla/changed.cpp',status:'modified'}];
  }
  if (endpoint === listStatuses) {
    if (settings.history_error) throw new Error('status history unavailable');
    return history;
  }
  return [];
}, rest:{pulls:{listFiles,get:async () => {
  if (settings.pr_error) throw new Error('PR read unavailable');
  return {data:pr};
}}, repos:{listCommitStatusesForRef:listStatuses,getCollaboratorPermissionLevel:async request => {
  record('permission',request.username);
  if (settings.permission_error) throw new Error('permission unavailable');
  return {data:{permission:(settings.permissions || {})[request.username] || 'read'}};
},getContent:async request => {
  record('content',request);
  if (request.ref === pr.base.sha && settings.target_profile) {
    if (settings.legacy_profile && request.path.endsWith('.json')) {
      throw Object.assign(new Error('legacy profile'), {status:404});
    }
    const text = settings.legacy_profile
      ? settings.target_profile.map(name => `'${name}'`).join('\n')
      : JSON.stringify({required:settings.target_profile.map(context => ({context}))});
    return {data:{type:'file',encoding:'base64',content:Buffer.from(text).toString('base64')}};
  }
  if (request.ref !== process.env.NPU_CI_WORKFLOW_SHA || settings.definition_error) {
    throw Object.assign(new Error('profile unavailable'), {status:403});
  }
  return {data:{type:'file',encoding:'base64',content:Buffer.from(JSON.stringify(definition)).toString('base64')}};
}, createCommitStatus:async payload => {
  record('status',payload);
  if (settings.status_error && !normalFailed) throw new Error('initial status write unavailable');
}}, issues:{listComments,createComment:async payload => {record('comment',payload);return {data:{id:1}};},updateComment:async () => {}}}};
const core = {notice:message => record('notice',message),warning:message => record('warning',message),
  setFailed:message => record('failed',message)};
(async () => {
  if (!settings.checkout_error) {
    try { await (async () => { NORMAL_SCRIPT })(); }
    catch (error) { normalFailed = true; record('normal_error',error.message); }
  } else { normalFailed = true; record('normal_error','checkout failed'); }
  if (normalFailed) {
    try { await (async () => { FALLBACK_SCRIPT })(); }
    catch (error) { record('fallback_error',error.message); }
  }
  process.stdout.write(JSON.stringify(records));
})();
'''.replace("NORMAL_SCRIPT", scripts[0]).replace("FALLBACK_SCRIPT", scripts[1])
        env = dict(os.environ, TEST_OPTIONS=json.dumps(options), TEST_DEFINITION=str(DEFINITION),
                   NPU_CI_AUTH_SCRIPT=str(ROOT / ".github/scripts/npu_ci_authorization.js"),
                   NPU_CI_CHECKS_SCRIPT=str(ROOT / ".github/scripts/npu_ci_checks.js"),
                   NPU_CI_WORKFLOW_SHA="b" * 40)
        if options.get("module_error"):
            env["NPU_CI_AUTH_SCRIPT"] = str(ROOT / "missing-control-module.js")
        result = subprocess.run(["node"], input=wrapper, env=env, cwd=ROOT,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def statuses(self, records):
        return [item["payload"] for item in records if item["kind"] == "status"]

    def assert_invalidated(self, records, sha="a" * 40):
        written = self.statuses(records)
        required = json.loads(DEFINITION.read_text(encoding="utf-8"))["required"]
        last_by_context = {item["context"]: item for item in written}
        for item in required:
            self.assertEqual(last_by_context[item["context"]]["state"], "failure")
            self.assertEqual(last_by_context[item["context"]]["sha"], sha)
        self.assertTrue(any(item["kind"] in ("failed", "fallback_error") for item in records))

    def test_fallback_is_independent_and_runs_after_failure_or_cancellation(self):
        steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["mark-not-run"]["steps"]
        fallback = steps[-1]
        self.assertEqual(fallback["if"], "${{ always() && (failure() || cancelled()) }}")
        self.assertNotIn("require(", fallback["with"]["script"])
        self.assertEqual(fallback["env"]["NPU_CI_WORKFLOW_SHA"], "${{ github.workflow_sha }}")

    def test_checkout_module_file_list_and_pr_read_errors_invalidate_previous_success(self):
        for error in ("checkout_error", "module_error", "files_error", "pr_error"):
            with self.subTest(error=error):
                records = self.run_workflow(**{error: True})
                self.assert_invalidated(records)
                reads = [item["payload"] for item in records if item["kind"] == "content"]
                self.assertTrue(any(item["ref"] == "b" * 40 for item in reads))

    def test_unknown_target_profile_invalidates_default_and_legacy_successes_on_same_head(self):
        legacy = "NPU CI / release / old gate"
        records = self.run_workflow(current_pr={"base": {"ref": "release", "sha": "d" * 40}},
                                    history=[{"context": legacy, "state": "success"}])
        self.assert_invalidated(records)
        self.assertIn(legacy, [item["context"] for item in self.statuses(records)])

    def test_partial_normal_status_write_failure_is_retried_as_failure(self):
        records = self.run_workflow(status_error=True)
        self.assert_invalidated(records)
        self.assertTrue(any(item["kind"] == "normal_error" for item in records))

    def test_fallback_uses_current_head_instead_of_stale_event(self):
        records = self.run_workflow(checkout_error=True, current_pr={"head": {"sha": "e" * 40}})
        self.assert_invalidated(records, sha="e" * 40)

    def test_api_failures_still_invalidate_every_context_that_can_be_determined(self):
        for options in ({"history_error": True}, {"definition_error": True}):
            with self.subTest(options=options):
                records = self.run_workflow(checkout_error=True, **options)
                self.assert_invalidated(records)
                self.assertTrue(any(item["kind"] == "fallback_error" for item in records))

    def test_target_gate_is_invalidated_when_history_api_fails(self):
        target = ["NPU CI / release / build", "NPU CI / release / test"]
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                records = self.run_workflow(checkout_error=True, history_error=True,
                                            current_pr={"base": {"ref": "release", "sha": "d" * 40}},
                                            target_profile=target, legacy_profile=legacy)
                self.assert_invalidated(records)
                written = {item["context"]: item["state"] for item in self.statuses(records)}
                self.assertTrue(all(written[name] == "failure" for name in target))

    def test_unrelated_title_edit_does_not_invalidate_success(self):
        records = self.run_workflow(checkout_error=True, title_edit=True)
        self.assertEqual(self.statuses(records), [])

    def test_ci_changes_from_admin_author_need_no_confirmation(self):
        records = self.run_workflow(file="ci/run_checks.sh", author="pr-admin",
                                    permissions={"pr-admin": "admin"}, history=[])
        self.assertEqual([item["state"] for item in self.statuses(records)], ["pending"] * 6)
        self.assertEqual([item["payload"] for item in records if item["kind"] == "permission"], ["pr-admin"])
        self.assertFalse(any(item["kind"] in ("normal_error", "fallback_error") for item in records))

    def test_non_admin_ci_author_cannot_keep_old_success(self):
        records = self.run_workflow(file=".github/workflows/ci.yml", author="contributor",
                                    permissions={"ci-admin": "admin"})
        self.assertEqual([item["state"] for item in self.statuses(records)], ["failure"] * 6)
        body = next(item["payload"]["body"] for item in records if item["kind"] == "comment")
        self.assertIn("作者", body)
        self.assertIn("Admin", body)

    def test_admin_ci_author_keeps_existing_execution_without_approval_marker(self):
        records = self.run_workflow(file="ci/run_checks.sh", author="pr-admin",
                                    permissions={"pr-admin": "admin"}, action="synchronize")
        self.assertEqual(self.statuses(records), [])
        self.assertFalse(any(item["kind"] in ("normal_error", "fallback_error") for item in records))

    def test_author_permission_read_failure_blocks_ci_changes(self):
        records = self.run_workflow(file="ci/run_checks.sh", permission_error=True)
        self.assertEqual([item["state"] for item in self.statuses(records)], ["failure"] * 6)
        body = next(item["payload"]["body"] for item in records if item["kind"] == "comment")
        self.assertIn("permission unavailable", body)

    def test_non_ci_change_does_not_read_author_permission(self):
        records = self.run_workflow(permission_error=True, history=[])
        self.assertEqual([item["state"] for item in self.statuses(records)], ["pending"] * 6)
        self.assertFalse(any(item["kind"] == "permission" for item in records))


if __name__ == "__main__":
    unittest.main()
