"""Exercise CI requester authorization and PR-author policy at each workflow entry point."""

import json
import os
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github/scripts/npu_ci_authorization.js"


def workflow_script(filename, job, step_id=None):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text(encoding="utf-8"))
    return next(step["with"]["script"] for step in workflow["jobs"][job]["steps"]
                if step.get("uses") == "actions/github-script@v8"
                and (step_id is None or step.get("id") == step_id))


class CiAuthorizationTest(unittest.TestCase):
    def run_entry(self, entry="helper", **options):
        scripts = {
            "dispatch": workflow_script("npu-ci-comment-dispatch.yml", "dispatch"),
            "package-dispatch": workflow_script("npu-ci-comment-dispatch.yml", "dispatch-package"),
            "package-guard": workflow_script("package.yml", "guard"),
            "prepare": workflow_script("ci.yml", "prepare", "prepare"),
            "default": workflow_script("npu-ci-default-status.yml", "mark-not-run"),
            "helper": "result = await auth[settings.method]({github, context, pr, ...settings.args});",
        }
        wrapper = r"""
const settings = JSON.parse(process.env.TEST_OPTIONS);
const auth = require(process.env.NPU_CI_AUTH_SCRIPT);
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const files = (settings.files || ['fla/example.cpp']).map((file) =>
  typeof file === 'string' ? {filename:file,status:'modified'} : file);
const pr = {number:527, state:'open', draft:false, changed_files:settings.file_count ?? files.length,
  merge_commit_sha:'c'.repeat(40),
  html_url:'https://github.com/example/repo/pull/527',user:{login:settings.author || 'ci-author'},
  base:{ref:settings.base_ref || 'main',sha:'c'.repeat(40)},head:{sha:'a'.repeat(40),repo:{full_name:'example/fork'}},...settings.initial_pr};
const context = {actor:settings.actor || 'ci-user',serverUrl:'https://github.com',runId:99,runNumber:12,
  ref:'refs/heads/main',
  repo:{owner:'example',repo:'repo'},payload:{repository:{default_branch:'main'},
    action:settings.action || 'synchronize',pull_request:pr,issue:{number:pr.number},
    changes:settings.changes,
    comment:{body:settings.command || (settings.entry.startsWith('package-') ? '/package' : '/run-npu-ci quick')},
    inputs:{pr_number:'527',ci_mode:'quick',build_repo:'example/repo',build_sha:'c'.repeat(40),...settings.inputs}}};
const listFiles = () => {};
const listStatuses = () => {};
const listComments = () => {};
let prReads = 0;
const github = {
  paginate: async (endpoint) => endpoint === listFiles ? files : endpoint === listStatuses ? settings.statuses || [] : [],
  rest:{
    pulls:{listFiles,get:async () => ({data:settings.current_pr || (settings.changed_pr && prReads++ > 0 ? {...pr, head:{sha:'d'.repeat(40)}} : pr)})},
    repos:{
      getContent:async (request) => {
        record('content',request);
        if (request.path === '.github/npu-ci-users.json') {
          if (settings.list_error) throw new Error('allowlist unavailable');
          return {data:{type:'file',encoding:'base64',content:Buffer.from(JSON.stringify(
            settings.config || {users:settings.users || ['ci-user','ci-admin']}
          )).toString('base64')}};
        }
        if (request.path === '.github/scripts/npu_ci_authorization.js') {
          return {data:{type:'file',encoding:'base64',content:require('fs').readFileSync(process.env.NPU_CI_AUTH_SCRIPT).toString('base64')}};
        }
        if (request.path === '.github/scripts/npu_ci_checks.json') {
          if (settings.profile_error) throw Object.assign(new Error('profile unavailable'), {status:settings.profile_error});
          return {data:{type:'file',encoding:'base64',content:Buffer.from(JSON.stringify(
            settings.branch_checks || require(process.env.NPU_CI_CHECKS_SCRIPT.replace(/\.js$/, '.json'))
          )).toString('base64')}};
        }
        return {data:{type:'file',content:Buffer.from(settings.workflow_text ?? require('fs').readFileSync('.github/workflows/ci.yml','utf8')).toString('base64')}};
      },
      getCollaboratorPermissionLevel:async ({username}) => {
        record('permission',username);
        if (settings.permission_error) throw new Error('permission unavailable');
        return {data:{permission:(settings.permissions || {})[username] || 'read'}};
      },
      getBranch:async () => ({data:{protected:settings.protected !== false,commit:{sha:settings.branch_sha || pr.base.sha}}}),
      getCommit:async () => ({data:{}}),
      createCommitStatus:async (payload) => record('status',payload),
      listCommitStatusesForRef:listStatuses,
    },
    issues:{listComments,createComment:async (payload) => {record('comment',payload);return {data:{id:123}};},
      updateComment:async (payload) => record('comment',payload)},
    actions:{createWorkflowDispatch:async (payload) => {
        if (settings.accepted_inputs && Object.keys(payload.inputs).some((name) => !settings.accepted_inputs.includes(name))) {
          throw new Error('422 Unexpected inputs: ' + Object.keys(payload.inputs).filter((name) => !settings.accepted_inputs.includes(name)).join(','));
        }
        record('dispatch',payload);
      },
      getWorkflowRun:async () => ({data:{status:'in_progress'}})},
  },
};
const core = {notice:(payload) => record('notice',payload),warning:(payload) => record('warning',payload),
  info:(payload) => record('info',payload),
  setFailed:(payload) => record('failed',payload),setOutput:(name,value) => record('output',{name,value})};
(async () => {
  let result;
  try {
    await (async () => { SCRIPT_BODY })();
    process.stdout.write(JSON.stringify({records,result}));
  } catch(error) {
    process.stdout.write(JSON.stringify({records,error:error.message}));
  }
})();
""".replace("SCRIPT_BODY", scripts[entry])
        env = os.environ.copy()
        env.update(TEST_OPTIONS=json.dumps({"entry":entry, **options}), NPU_CI_AUTH_SCRIPT=str(HELPER),
                   NPU_CI_CHECKS_SCRIPT=str(ROOT / '.github/scripts/npu_ci_checks.js'),
                   NPU_CI_WORKFLOW_SHA=options.get("workflow_sha", "c" * 40),
                   NPU_CI_WORKFLOW_REF=options.get("workflow_ref", options.get("base_ref", "main")), NPU_CI_RUN_ATTEMPT=str(options.get("attempt", 1)))
        completed = subprocess.run(["node"], input=wrapper, text=True, encoding="utf-8",
                                   capture_output=True, cwd=ROOT, env=env, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def records(self, result, kind):
        return [item["payload"] for item in result["records"] if item["kind"] == kind]

    def test_listed_reader_can_dispatch_without_repository_permission_lookup(self):
        for entry in ("dispatch", "package-dispatch", "package-guard"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, base_ref="v26.10.0")
                self.assertNotIn("error", result)
                self.assertFalse(self.records(result, "failed"))
                if entry != "package-guard":
                    self.assertEqual(len(self.records(result, "dispatch")), 1)
                self.assertEqual(self.records(result, "permission"), [])
                allowlist = self.records(result, "content")[0]
                self.assertEqual((allowlist["path"], allowlist["ref"]), (".github/npu-ci-users.json", "main"))

    def test_repository_admin_outside_list_cannot_dispatch(self):
        for entry in ("dispatch", "package-dispatch", "package-guard"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, actor="outsider", permissions={"outsider":"admin"})
                self.assertIn("npu-ci-users.json", result["error"])
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_allowlist_is_case_insensitive(self):
        for entry in ("dispatch", "package-dispatch", "package-guard"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, actor="CI-USER")
                self.assertNotIn("error", result)
                self.assertFalse(self.records(result, "failed"))

    def test_missing_or_invalid_allowlist_fails_closed(self):
        for entry in ("dispatch", "package-dispatch", "package-guard"):
            for options in ({"list_error":True}, {"config":{"users":[]}}, {"config":{"users":["invalid/name"]}}):
                with self.subTest(entry=entry, options=options):
                    result = self.run_entry(entry, **options)
                    self.assertIn("error", result)
                    self.assertEqual(self.records(result, "dispatch"), [])

    def test_package_ci_changes_check_author_admin_separately_from_requester(self):
        for entry in ("package-dispatch", "package-guard"):
            with self.subTest(entry=entry):
                allowed = self.run_entry(entry, files=["ci/run.sh"], permissions={"ci-author":"admin"})
                self.assertNotIn("error", allowed)
                self.assertFalse(self.records(allowed, "failed"))
                self.assertEqual(self.records(allowed, "permission"), ["ci-author"])
                denied = self.run_entry(entry, files=[".github/workflows/package.yml"],
                                        permissions={"ci-user":"admin"})
                self.assertIn("PR 作者", denied["error"])
                self.assertEqual(self.records(denied, "dispatch"), [])

    def test_package_guard_checks_comment_requester_and_ignores_manual_spoofing(self):
        allowed = self.run_entry("package-guard", actor="github-actions[bot]", inputs={"requested_by":"ci-user"})
        self.assertNotIn("error", allowed)
        self.assertFalse(self.records(allowed, "failed"))
        for actor, requester in (("github-actions[bot]", "outsider"), ("outsider", "ci-user")):
            with self.subTest(actor=actor):
                denied = self.run_entry("package-guard", actor=actor, inputs={"requested_by":requester})
                self.assertIn("outsider", denied["error"])
                self.assertEqual(self.records(denied, "info"), [])

    def test_prepare_ignores_spoofed_requester_input_for_human_actor(self):
        result = self.run_entry("prepare", actor="outsider", inputs={"requested_by":"ci-admin"})
        self.assertIn("outsider", result["error"])
        self.assertEqual(self.records(result, "status"), [])

    def test_bot_dispatch_validates_original_requester(self):
        result = self.run_entry("prepare", actor="github-actions[bot]", inputs={"requested_by":"ci-user", "expected_head_sha":"a"*40,
                                        "expected_base_ref":"main", "expected_base_sha":"c"*40,
                                        "expected_workflow_sha":"c"*40})
        self.assertNotIn("error", result)
        self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))

    def test_bot_dispatch_without_requester_fails(self):
        result = self.run_entry("prepare", actor="github-actions[bot]")
        self.assertIn("error", result)

    def test_dispatch_captures_complete_source_and_workflow_identity(self):
        result = self.run_entry("dispatch")
        self.assertNotIn("error", result)
        inputs = self.records(result, "dispatch")[0]["inputs"]
        self.assertEqual({name: inputs[name] for name in (
            "expected_head_sha", "expected_base_ref", "expected_base_sha", "expected_workflow_sha")}, {
                "expected_head_sha": "a"*40, "expected_base_ref": "main",
                "expected_base_sha": "c"*40, "expected_workflow_sha": "c"*40})
        workflow_read = next(item for item in self.records(result, "content")
                             if item["path"] == ".github/workflows/ci.yml")
        self.assertEqual(workflow_read["ref"], "c"*40)

    def test_custom_ci_ref_cannot_select_an_unapproved_workflow(self):
        for parameter in ("ci_ref=malicious", "branch=malicious", "ci_ref=main", "ci_modified=true", "head=" + "a"*40):
            with self.subTest(parameter=parameter):
                result = self.run_entry("dispatch", command="/run-npu-ci " + parameter)
                self.assertIn("不支持", self.records(result, "failed")[0])
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_unprotected_pr_base_cannot_supply_executable_workflow(self):
        for entry in ("dispatch", "prepare"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, base_ref="malicious", protected=False)
                self.assertIn("受保护", result["error"])
                self.assertEqual(self.records(result, "dispatch"), [])
                self.assertEqual(self.records(result, "status"), [])

    def test_legacy_workflow_dispatches_only_its_four_declared_inputs(self):
        # These interfaces match the inspected v26.9.0, v26.9.1 and v26.10.0
        # workflows; the first two also have issue_comment before dispatch.
        declared = {"pr_number":{"required":True}, "ci_mode":{"required":False,"default":"quick"},
                    "ops":{"required":False,"default":""}, "requested_by":{"required":False,"default":""}}
        for branch, old_comments in (("v26.9.0",True), ("v26.9.1",True), ("v26.10.0",False)):
            for contract in (None, "2"):
                with self.subTest(branch=branch, contract=contract):
                    events = {"issue_comment":{"types":["created"]}} if old_comments else {}
                    events["workflow_dispatch"] = {"inputs":declared}
                    workflow = {"on":events, "jobs":{}}
                    if contract:
                        workflow["env"] = {"NPU_CI_DISPATCH_CONTRACT":contract}
                    result = self.run_entry("dispatch", base_ref=branch,
                                            workflow_text=yaml.safe_dump(workflow,sort_keys=False),
                                            accepted_inputs=list(declared))
                    self.assertNotIn("error", result)
                    dispatch = self.records(result, "dispatch")[0]
                    self.assertEqual(dispatch["ref"], branch)
                    self.assertEqual(dispatch["inputs"], {"pr_number":"527", "ci_mode":"quick", "ops":"", "requested_by":"ci-user"})

    def test_partially_updated_workflow_receives_supported_snapshot_fields_only(self):
        current = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        partial = current.replace("      expected_head_sha:", "      ignored_head_sha:")
        allowed = ["pr_number", "ci_mode", "ops", "requested_by", "expected_base_ref", "expected_base_sha", "expected_workflow_sha"]
        result = self.run_entry("dispatch", workflow_text=partial, accepted_inputs=allowed)
        self.assertNotIn("error", result)
        inputs = self.records(result, "dispatch")[0]["inputs"]
        self.assertEqual(set(inputs), set(allowed))
        self.assertNotIn("expected_head_sha", inputs)

    def test_workflow_with_no_inputs_can_dispatch_without_new_contract(self):
        for text in ("on:\n  workflow_dispatch:\n", "on:\n  workflow_dispatch: {}\n"):
            with self.subTest(workflow=text):
                result = self.run_entry("dispatch", workflow_text=text, accepted_inputs=[])
                self.assertNotIn("error", result)
                self.assertEqual(self.records(result, "dispatch")[0]["inputs"], {})

    def test_missing_dispatch_entry_is_still_a_configuration_error(self):
        result = self.run_entry("dispatch", workflow_text="on:\n  push:\n")
        self.assertIn("workflow_dispatch", result["error"])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_legacy_target_keeps_current_entry_allowlist_and_author_policy(self):
        legacy = "on:\n  workflow_dispatch:\n    inputs:\n      pr_number:\n        required: true\n"
        unlisted = self.run_entry("dispatch", actor="outsider", permissions={"outsider":"admin"}, workflow_text=legacy)
        self.assertIn("npu-ci-users.json", unlisted["error"])
        self.assertEqual(self.records(unlisted, "dispatch"), [])
        ci_author = self.run_entry("dispatch", files=["ci/run.sh"], actor="ci-admin",
                                   permissions={"ci-admin":"admin"}, workflow_text=legacy)
        self.assertIn("PR 作者", self.records(ci_author, "failed")[0])
        self.assertEqual(self.records(ci_author, "dispatch"), [])

    def test_initial_dispatch_and_prepare_reject_closed_merged_or_draft_pr(self):
        for entry in ("dispatch", "prepare"):
            for change in ({"state":"closed"}, {"merged":True}, {"merged_at":"2026-11-04T00:00:00Z"}, {"draft":True}):
                with self.subTest(entry=entry, change=change):
                    result = self.run_entry(entry, initial_pr=change)
                    self.assertIn("error", result)
                    self.assertEqual(self.records(result, "dispatch"), [])
                    self.assertEqual(self.records(result, "status"), [])

    def test_prepare_rejects_head_or_target_changes_but_allows_base_advance(self):
        expected = {"requested_by":"ci-user", "expected_head_sha":"a"*40,
                    "expected_base_ref":"main", "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for field, old_value in (("expected_head_sha", "d"*40), ("expected_base_ref", "release")):
            with self.subTest(field=field):
                result = self.run_entry("prepare", actor="github-actions[bot]", files=["ci/run_checks.sh"],
                                        inputs={**expected, field:old_value}, permissions={"ci-admin":"admin"})
                self.assertIn("触发快照", result["error"])
                self.assertEqual(self.records(result, "status"), [])
                self.assertFalse(any(item["name"] == "ci_approval_key" for item in self.records(result, "output")))
        result = self.run_entry("prepare", actor="github-actions[bot]", files=["fla/example.cpp"],
                                workflow_sha="d"*40, branch_sha="d"*40,
                                inputs=expected)
        self.assertNotIn("error", result)
        self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))

    def test_bot_dispatch_requires_head_and_target_identity(self):
        expected = {"requested_by":"ci-user", "expected_head_sha":"a"*40, "expected_base_ref":"main",
                    "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for field in ("expected_head_sha", "expected_base_ref"):
            with self.subTest(field=field):
                result = self.run_entry("prepare", actor="github-actions[bot]", inputs={**expected, field:""})
                self.assertIn("触发快照", result["error"])

    def test_manual_workflow_cannot_run_another_ref_but_allows_outdated_base(self):
        result = self.run_entry("prepare", workflow_ref="other")
        self.assertIn("error", result)
        self.assertEqual(self.records(result, "status"), [])
        for options in ({"workflow_sha":"d"*40}, {"branch_sha":"d"*40}):
            with self.subTest(options=options):
                result = self.run_entry("prepare", **options)
                self.assertNotIn("error", result)

    def test_ci_documentation_does_not_need_admin_author(self):
        result = self.run_entry("dispatch", files=["ci/README.MD", ".github/notes.md"])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "permission"), [])

    def test_rename_out_of_ci_still_needs_admin_author(self):
        result = self.run_entry("dispatch", files=[{"filename":"tools/run.sh", "previous_filename":"ci/run.sh", "status":"renamed"}])
        self.assertTrue(self.records(result, "failed"))

    def test_incomplete_duplicate_or_invalid_file_list_rejects(self):
        for options in ({"file_count":2}, {"files":["a.py","a.py"]}, {"files":[{"filename":"a.py","status":"renamed"}]}):
            with self.subTest(options=options):
                result = self.run_entry("dispatch", **options)
                self.assertIn("error", result)
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_pr_change_during_inspection_rejects(self):
        result = self.run_entry("dispatch", changed_pr=True)
        self.assertIn("error", result)

    def test_default_status_fails_ci_changes_from_non_admin_author(self):
        result = self.run_entry("default", files=[".github/scripts/ci.js"])
        statuses = self.records(result, "status")
        self.assertEqual(len(statuses), 6)
        self.assertTrue(all(status["state"] == "failure" for status in statuses))

    def test_title_edit_keeps_existing_status(self):
        result = self.run_entry("default", files=["ci/run_checks.sh"], action="edited", changes={"title":{"from":"old"}})
        self.assertEqual(self.records(result, "status"), [])

    def test_default_status_uses_target_branch_profile_at_fixed_sha(self):
        contexts = ["NPU CI / release / build", "NPU CI / release / test"]
        result = self.run_entry("default", base_ref="release",
                                branch_checks={"required":[{"context":item} for item in contexts]})
        self.assertNotIn("error", result)
        self.assertEqual([status["context"] for status in self.records(result, "status")], contexts)
        profile_read = self.records(result, "content")[0]
        self.assertEqual((profile_read["path"], profile_read["ref"]),
                         (".github/scripts/npu_ci_checks.json", "c" * 40))

    def test_default_status_rejects_unknown_branch_profile(self):
        result = self.run_entry("default", base_ref="release", profile_error=403)
        self.assertIn("无法确定", result["error"])
        self.assertEqual(self.records(result, "status"), [])
        self.assertIn("profile unavailable", self.records(result, "warning")[0])

    def test_rerun_preserves_initial_authorization_without_reloading_allowlist(self):
        for entry in ("dispatch", "prepare", "package-dispatch", "package-guard"):
            with self.subTest(entry=entry):
                snapshot = {"expected_head_sha":"a"*40, "expected_base_ref":"main",
                            "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
                result = self.run_entry(entry, attempt=2, list_error=True, inputs=snapshot)
                self.assertNotIn("error", result)
                self.assertFalse(any(item["path"] == ".github/npu-ci-users.json"
                                     for item in self.records(result, "content")))
                self.assertEqual(self.records(result, "permission"), [])
                if entry in ("dispatch", "package-dispatch"):
                    self.assertEqual(len(self.records(result, "dispatch")), 1)
                elif entry == "prepare":
                    self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))
                else:
                    self.assertTrue(self.records(result, "info"))

    def test_manual_rerun_without_head_or_target_snapshot_refuses_new_prepare(self):
        snapshot = {"expected_head_sha":"a"*40, "expected_base_ref":"main",
                    "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for inputs in ({}, {**snapshot, "expected_head_sha":""}, {**snapshot, "expected_base_ref":""}):
            with self.subTest(inputs=inputs):
                result = self.run_entry("prepare", attempt=2, list_error=True, inputs=inputs)
                self.assertIn("新建 Run workflow", result["error"])
                self.assertEqual(self.records(result, "status"), [])
                self.assertEqual(self.records(result, "permission"), [])

    def test_complete_snapshot_rerun_rejects_changed_head_without_reauthorization(self):
        result = self.run_entry("prepare", attempt=2, list_error=True, initial_pr={"head":{"sha":"d"*40}},
                                inputs={"expected_head_sha":"a"*40, "expected_base_ref":"main",
                                        "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40})
        self.assertIn("触发快照", result["error"])
        self.assertEqual(self.records(result, "status"), [])
        self.assertEqual(self.records(result, "permission"), [])

    def test_admin_requester_cannot_allow_non_admin_author_ci_changes(self):
        for entry in ("dispatch", "prepare"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, files=["ci/run_checks.sh"], actor="ci-admin", permissions={"ci-admin":"admin"})
                self.assertTrue(self.records(result, "failed"))
                self.assertIn("PR 作者 @ci-author", self.records(result, "comment")[0]["body"])
                self.assertEqual(self.records(result, "permission"), ["ci-author"])
                self.assertEqual(self.records(result, "dispatch"), [])
                self.assertTrue(all(item["state"] == "failure" for item in self.records(result, "status")))

    def test_listed_reader_can_run_admin_author_ci_changes_without_confirmation(self):
        for entry in ("dispatch", "prepare"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, files=[".github/workflows/ci.yml"], permissions={"ci-author":"admin"})
                self.assertNotIn("error", result)
                self.assertEqual(self.records(result, "failed"), [])
                self.assertEqual(self.records(result, "permission"), ["ci-author"])
                if entry == "dispatch":
                    inputs = self.records(result, "dispatch")[0]["inputs"]
                    self.assertNotIn("ci_modified", inputs)
                    self.assertEqual(len(inputs), 8)
                else:
                    self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))
                    self.assertTrue(all(item["state"] == "pending" for item in self.records(result, "status")))
                    self.assertFalse(any("ci-ok=" in item["description"] for item in self.records(result, "status")))
                    self.assertFalse(any(item["name"] == "ci_approval_key" for item in self.records(result, "output")))

    def test_ci_author_does_not_need_to_be_in_ci_allowlist(self):
        result = self.run_entry("dispatch", files=["ci/run.sh"], author="admin-author", users=["ci-user"],
                                permissions={"admin-author":"admin"})
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "permission"), ["admin-author"])
        self.assertEqual(len(self.records(result, "dispatch")), 1)

    def test_ci_author_permission_failure_or_missing_author_fails_closed(self):
        for options in ({"permission_error":True}, {"initial_pr":{"user":None}}):
            with self.subTest(options=options):
                result = self.run_entry("prepare", files=["ci/run.sh"], **options)
                self.assertTrue(self.records(result, "failed"))
                self.assertEqual([item["state"] for item in self.records(result, "status")], ["failure"] * 6)

    def test_default_status_preserves_admin_author_execution_but_rejects_non_admin_author(self):
        prepared = self.run_entry("prepare", files=["ci/run_checks.sh"], permissions={"ci-author":"admin"})
        statuses = self.records(prepared, "status")
        allowed = self.run_entry("default", files=["ci/run_checks.sh"], statuses=statuses, permissions={"ci-author":"admin"})
        self.assertEqual(self.records(allowed, "status"), [])
        rejected = self.run_entry("default", files=["ci/run_checks.sh"], statuses=statuses)
        self.assertEqual([status["state"] for status in self.records(rejected, "status")], ["failure"] * 6)

if __name__ == "__main__":
    unittest.main()
