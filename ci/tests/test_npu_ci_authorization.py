"""Exercise CI authorization and approval at each workflow entry point."""

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
            "prepare": workflow_script("ci.yml", "prepare", "prepare"),
            "default": workflow_script("npu-ci-default-status.yml", "mark-not-run"),
            "guard": workflow_script("ci.yml", "ascend-npu", "authorize_rerun"),
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
  html_url:'https://github.com/example/repo/pull/527',
  base:{ref:settings.base_ref || 'main',sha:'c'.repeat(40)},head:{sha:'a'.repeat(40),repo:{full_name:'example/fork'}}};
const context = {actor:settings.actor || 'ci-user',serverUrl:'https://github.com',runId:99,runNumber:12,
  repo:{owner:'example',repo:'repo'},payload:{repository:{default_branch:'main'},
    action:settings.action || 'synchronize',pull_request:pr,issue:{number:pr.number},
    changes:settings.changes,
    comment:{body:settings.command || '/run-npu-ci quick'},
    inputs:{pr_number:'527',ci_mode:'quick',...settings.inputs}}};
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
        return {data:{type:'file',content:Buffer.from('on:\n  workflow_dispatch:\n').toString('base64')}};
      },
      getCollaboratorPermissionLevel:async ({username}) => {
        record('permission',username);
        if (settings.permission_error) throw new Error('permission unavailable');
        return {data:{permission:(settings.permissions || {})[username] || 'read'}};
      },
      getBranch:async () => ({data:{commit:{sha:pr.base.sha}}}),
      createCommitStatus:async (payload) => record('status',payload),
      listCommitStatusesForRef:listStatuses,
    },
    issues:{listComments,createComment:async (payload) => {record('comment',payload);return {data:{id:123}};},
      updateComment:async (payload) => record('comment',payload)},
    actions:{createWorkflowDispatch:async (payload) => record('dispatch',payload),
      getWorkflowRun:async () => ({data:{status:'in_progress'}})},
  },
};
const core = {notice:(payload) => record('notice',payload),warning:(payload) => record('warning',payload),
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
        env.update(TEST_OPTIONS=json.dumps(options), NPU_CI_AUTH_SCRIPT=str(HELPER),
                   NPU_CI_CHECKS_SCRIPT=str(ROOT / '.github/scripts/npu_ci_checks.js'),
                   NPU_CI_WORKFLOW_SHA="b" * 40, NPU_CI_RUN_ATTEMPT=str(options.get("attempt", 1)),
                   NPU_CI_TRIGGERING_ACTOR=options.get("rerun_by", ""),
                   NPU_CI_PR_NUMBER="527", NPU_CI_HEAD_SHA="a"*40, NPU_CI_BASE_SHA="c"*40,
                   NPU_CI_BASE_REF="main", NPU_CI_APPROVAL_KEY=options.get("approval_key", ""))
        completed = subprocess.run(["node"], input=wrapper, text=True, encoding="utf-8",
                                   capture_output=True, cwd=ROOT, env=env, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def records(self, result, kind):
        return [item["payload"] for item in result["records"] if item["kind"] == kind]

    def test_listed_reader_can_dispatch_without_repository_permission_lookup(self):
        result = self.run_entry("dispatch")
        self.assertNotIn("error", result)
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "permission"), [])
        allowlist = self.records(result, "content")[0]
        self.assertEqual((allowlist["path"], allowlist["ref"]), (".github/npu-ci-users.json", "main"))

    def test_repository_admin_outside_list_cannot_dispatch(self):
        result = self.run_entry("dispatch", actor="outsider", permissions={"outsider":"admin"})
        self.assertIn("npu-ci-users.json", result["error"])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_allowlist_is_case_insensitive(self):
        result = self.run_entry("dispatch", actor="CI-USER")
        self.assertNotIn("error", result)

    def test_missing_or_invalid_allowlist_fails_closed(self):
        for options in ({"list_error":True}, {"config":{"users":[]}}, {"config":{"users":["invalid/name"]}}):
            with self.subTest(options=options):
                result = self.run_entry("dispatch", **options)
                self.assertIn("error", result)
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_comment_rerun_checks_actual_rerun_user(self):
        result = self.run_entry("dispatch", attempt=2, rerun_by="outsider")
        self.assertIn("outsider", result["error"])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_prepare_ignores_spoofed_requester_input_for_human_actor(self):
        result = self.run_entry("prepare", actor="outsider", inputs={"requested_by":"ci-admin"})
        self.assertIn("outsider", result["error"])
        self.assertEqual(self.records(result, "status"), [])

    def test_bot_dispatch_validates_original_requester(self):
        result = self.run_entry("prepare", actor="github-actions[bot]", inputs={"requested_by":"ci-user"})
        self.assertNotIn("error", result)
        self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))

    def test_bot_dispatch_without_requester_fails(self):
        result = self.run_entry("prepare", actor="github-actions[bot]")
        self.assertIn("error", result)

    def test_ci_changes_require_explicit_admin_confirmation(self):
        for entry in ("dispatch", "prepare"):
            with self.subTest(entry=entry):
                result = self.run_entry(entry, files=["ci/run_checks.sh"], actor="ci-admin", permissions={"ci-admin":"admin"})
                self.assertTrue(self.records(result, "failed"))
                self.assertIn("ci_modified=true", self.records(result, "comment")[0]["body"])
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_listed_non_admin_cannot_confirm_ci_changes(self):
        result = self.run_entry("dispatch", files=[".github/workflows/ci.yml"], command="/run-npu-ci ci_modified=true")
        self.assertIn("Admin", self.records(result, "failed")[0])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_admin_confirmation_is_forwarded(self):
        result = self.run_entry("dispatch", files=["ci/run_checks.sh"], actor="ci-admin",
                                command="/run-npu-ci ci_modified=true", permissions={"ci-admin":"admin"})
        self.assertEqual(self.records(result, "dispatch")[0]["inputs"]["ci_modified"], "true")

    def test_prepare_admin_confirmation_marks_all_pending_statuses(self):
        result = self.run_entry("prepare", files=["ci/run_checks.sh"], actor="ci-admin",
                                inputs={"ci_modified":"true"}, permissions={"ci-admin":"admin"})
        self.assertNotIn("error", result)
        statuses = self.records(result, "status")
        self.assertEqual(len(statuses), 6)
        self.assertTrue(all(status["state"] == "pending" and " ci-ok=" in status["description"] for status in statuses))
        self.assertTrue(any(item["name"] == "ci_approval_key" for item in self.records(result, "output")))

    def test_ci_rerun_also_requires_admin_for_actual_rerunner(self):
        result = self.run_entry("prepare", files=["ci/run_checks.sh"], actor="ci-admin", attempt=2,
                                rerun_by="ci-user", inputs={"ci_modified":"true"}, permissions={"ci-admin":"admin"})
        self.assertIn("ci-user", self.records(result, "failed")[0])

    def test_permission_api_failure_rejects_confirmation(self):
        result = self.run_entry("prepare", files=["ci/run_checks.sh"], actor="ci-admin", inputs={"ci_modified":"true"}, permission_error=True)
        self.assertIn("permission unavailable", self.records(result, "failed")[0])

    def test_ci_modified_rejects_values_other_than_true_false(self):
        for value in ("", "yes", "1", "TRUE"):
            with self.subTest(value=value):
                result = self.run_entry("dispatch", command=f"/run-npu-ci ci_modified={value}")
                self.assertIn("true", result["error"])

    def test_ci_documentation_does_not_need_admin_confirmation(self):
        result = self.run_entry("dispatch", files=["ci/README.MD", ".github/notes.md"])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "permission"), [])

    def test_rename_out_of_ci_still_needs_confirmation(self):
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

    def test_default_status_fails_ci_changes_before_confirmation(self):
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

    def test_default_status_skips_unknown_branch_profile(self):
        result = self.run_entry("default", base_ref="release", profile_error=403)
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "status"), [])
        self.assertIn("profile unavailable", self.records(result, "warning")[0])

    def test_default_status_preserves_only_this_pr_base_head_approval(self):
        prepared = self.run_entry("prepare", files=["ci/run_checks.sh"], actor="ci-admin",
                                  inputs={"ci_modified":"true"}, permissions={"ci-admin":"admin"})
        statuses = self.records(prepared, "status")
        approved = self.run_entry("default", files=["ci/run_checks.sh"], statuses=statuses)
        self.assertEqual(self.records(approved, "status"), [])
        invalid = [{**status, "description":status["description"].replace("ci-ok=", "other-pr=")} for status in statuses]
        rejected = self.run_entry("default", files=["ci/run_checks.sh"], statuses=invalid)
        self.assertEqual([status["state"] for status in self.records(rejected, "status")], ["failure"] * 6)

    def test_approval_key_changes_with_each_bound_identity(self):
        args = {"prNumber":527,"baseRef":"main","baseSha":"c"*40,"headSha":"a"*40}
        original = self.run_entry(method="buildCiApprovalKey", args=args)["result"]
        for key, value in (("prNumber",528),("baseRef","release"),("baseSha","d"*40),("headSha","b"*40)):
            with self.subTest(key=key):
                other = self.run_entry(method="buildCiApprovalKey", args={**args,key:value})["result"]
                self.assertNotEqual(original, other)

    def test_device_rerun_guard_precedes_all_source_steps_on_both_platforms(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        guard = workflow["jobs"]["ascend-npu"]["steps"][0]
        self.assertEqual(guard["id"], "authorize_rerun")
        self.assertEqual(guard["if"], "github.run_attempt > 1")
        self.assertNotIn("matrix.platform", guard["if"])
        self.assertEqual(guard["env"]["NPU_CI_TRIGGERING_ACTOR"], "${{ github.triggering_actor }}")

    def test_device_rerun_guard_reads_fixed_module_and_current_default_list(self):
        result = self.run_entry("guard", rerun_by="ci-user")
        self.assertNotIn("error", result)
        reads = self.records(result, "content")
        self.assertEqual([(item["path"],item["ref"]) for item in reads], [
            (".github/scripts/npu_ci_authorization.js", "b"*40), (".github/npu-ci-users.json", "main")])
        self.assertEqual(self.records(result, "permission"), [])

    def test_device_rerun_guard_denies_unlisted_actual_rerunner(self):
        result = self.run_entry("guard", rerun_by="outsider", permissions={"outsider":"admin"})
        self.assertIn("outsider", result["error"])

    def test_device_rerun_guard_requires_admin_for_ci_confirmation(self):
        key = self.run_entry(method="buildCiApprovalKey", args={"prNumber":527,"baseRef":"main","baseSha":"c"*40,"headSha":"a"*40})["result"]
        rejected = self.run_entry("guard", rerun_by="ci-user", approval_key=key)
        self.assertIn("Admin", rejected["error"])
        accepted = self.run_entry("guard", rerun_by="ci-admin", approval_key=key, permissions={"ci-admin":"admin"})
        self.assertNotIn("error", accepted)

    def test_device_rerun_guard_rejects_expired_approval(self):
        result = self.run_entry("guard", rerun_by="ci-admin", approval_key="deadbeef1234", permissions={"ci-admin":"admin"})
        self.assertIn("ci_modified=true", result["error"])

    def test_device_rerun_guard_rejects_stale_head_base_or_closed_pr(self):
        current = {"number":527,"state":"open","draft":False,
                   "head":{"sha":"a"*40},"base":{"ref":"main","sha":"c"*40}}
        for change in ({"head":{"sha":"d"*40}}, {"base":{"ref":"release","sha":"c"*40}},
                       {"base":{"ref":"main","sha":"d"*40}}, {"state":"closed"}, {"draft":True}):
            with self.subTest(change=change):
                result = self.run_entry("guard", rerun_by="ci-user", current_pr={**current,**change})
                self.assertIn("prepare", result["error"])


if __name__ == "__main__":
    unittest.main()
