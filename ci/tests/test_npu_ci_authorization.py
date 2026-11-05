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
            "prepare": workflow_script("ci.yml", "prepare", "prepare"),
            "default": workflow_script("npu-ci-default-status.yml", "mark-not-run"),
            "helper": "result = await auth[settings.method]({github, context, pr, ...settings.args});",
            "helper_tag": "result = auth.buildPrWorkflowTag(settings.pr_number, settings.workflow_sha);",
            "helper_files": "result = auth[settings.method](files);",
        }
        wrapper = r"""
const settings = JSON.parse(process.env.TEST_OPTIONS);
const auth = require(process.env.NPU_CI_AUTH_SCRIPT);
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const files = (settings.files || ['fla/example.cpp']).map((file) =>
  typeof file === 'string' ? {filename:file,status:'modified'} : file);
const pr = {number:527, state:'open', draft:false, merge_commit_sha:'d'.repeat(40), changed_files:settings.file_count ?? files.length,
  html_url:'https://github.com/example/repo/pull/527',user:{login:settings.author || 'ci-author'},
  base:{ref:settings.base_ref || 'main',sha:'c'.repeat(40)},head:{sha:'a'.repeat(40),repo:{full_name:'example/fork'}},...settings.initial_pr};
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
        return {data:{type:'file',content:Buffer.from(settings.workflow_text ?? require('fs').readFileSync('.github/workflows/ci.yml','utf8')).toString('base64')}};
      },
      getCollaboratorPermissionLevel:async ({username}) => {
        record('permission',username);
        if (settings.permission_error) throw new Error('permission unavailable');
        return {data:{permission:(settings.permissions || {})[username] || 'read'}};
      },
      getBranch:async () => ({data:{protected:settings.protected !== false,commit:{sha:settings.branch_sha || pr.base.sha}}}),
      getCommit:async (request) => {
        record('commit',request);
        return {data:settings.workflow_commit || {sha:request.ref,parents:[{sha:pr.base.sha},{sha:pr.head.sha}]}};
      },
      createCommitStatus:async (payload) => record('status',payload),
      listCommitStatusesForRef:listStatuses,
    },
    git:{
      getRef:async (request) => {
        record('ref',request);
        if (settings.existing_tag) return {data:{object:{type:'commit',sha:settings.existing_tag}}};
        throw Object.assign(new Error('tag not found'), {status:404});
      },
      createRef:async (request) => {
        record('create_ref',request);
        if (settings.ref_error) throw Object.assign(new Error('workflow ref unavailable'), {status:settings.ref_error});
        return {data:{ref:request.ref,object:{type:'commit',sha:request.sha}}};
      },
      deleteRef:async (request) => record('delete_ref',request),
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
const core = {notice:(payload) => record('notice',payload),info:(payload) => record('info',payload),warning:(payload) => record('warning',payload),
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
        workflow_sha = options.get("workflow_sha", "c" * 40)
        env.update(TEST_OPTIONS=json.dumps(options), NPU_CI_AUTH_SCRIPT=str(HELPER),
                   NPU_CI_CHECKS_SCRIPT=str(ROOT / '.github/scripts/npu_ci_checks.js'),
                   NPU_CI_WORKFLOW_SCRIPT=str(ROOT / '.github/scripts/npu_ci_workflow.js'),
                   NPU_CI_WORKFLOW_SHA=workflow_sha if isinstance(workflow_sha, str) else "c" * 40,
                   NPU_CI_WORKFLOW_REF=options.get("workflow_ref", options.get("base_ref", "main")), NPU_CI_RUN_ATTEMPT=str(options.get("attempt", 1)),
                   NPU_CI_PR_NUMBER="527", NPU_CI_HEAD_SHA="a"*40, NPU_CI_BASE_SHA="c"*40,
                   NPU_CI_BASE_REF="main")
        completed = subprocess.run(["node"], input=wrapper, text=True, encoding="utf-8",
                                   capture_output=True, cwd=ROOT, env=env, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def records(self, result, kind):
        return [item["payload"] for item in result["records"] if item["kind"] == kind]

    def test_pr_workflow_tag_encodes_only_fixed_pr_and_commit(self):
        result = self.run_entry("helper_tag", pr_number=527, workflow_sha="d" * 40)
        self.assertEqual(result["result"], "npu-ci/pr-527/" + "d" * 40)
        for pr_number in (0, -1, 527.5, "527", 2 ** 53):
            with self.subTest(pr_number=pr_number):
                rejected = self.run_entry("helper_tag", pr_number=pr_number, workflow_sha="d" * 40)
                self.assertIn("无效", rejected["error"])
        for workflow_sha in ("", "d" * 39, "D" * 40, "refs/heads/main", ["d" * 40], None):
            with self.subTest(workflow_sha=workflow_sha):
                rejected = self.run_entry("helper_tag", pr_number=527, workflow_sha=workflow_sha)
                self.assertIn("无效", rejected["error"])

    def protected_base_args(self, **changes):
        return {"branch":{"protected":True,"commit":{"sha":"c" * 40}},
                "workflowRef":"npu-ci/pr-527/" + "d" * 40, "workflowSha":"d" * 40,
                "workflowSource":"pr-merge",
                "workflowCommit":{"sha":"d" * 40,"parents":[{"sha":"c" * 40},{"sha":"a" * 40}]},
                **changes}

    def test_pr_workflow_identity_accepts_exact_merge_of_protected_base_and_head(self):
        result = self.run_entry(method="requireProtectedBase", args=self.protected_base_args())
        self.assertNotIn("error", result)

    def test_pr_workflow_identity_rejects_unprotected_or_moved_base(self):
        for branch in ({"protected":False,"commit":{"sha":"c" * 40}},
                       {"protected":True,"commit":{"sha":"e" * 40}},
                       {"protected":True,"commit":{"sha":"invalid"}}):
            with self.subTest(branch=branch):
                result = self.run_entry(method="requireProtectedBase", args=self.protected_base_args(branch=branch))
                self.assertIn("error", result)

    def test_pr_workflow_identity_rejects_wrong_tag_or_workflow_commit(self):
        for changes in ({"workflowRef":"main"},
                        {"workflowRef":"npu-ci/pr-528/" + "d" * 40},
                        {"workflowRef":"npu-ci/pr-527/" + "e" * 40},
                        {"workflowRef":"refs/tags/npu-ci/pr-527/" + "d" * 40},
                        {"workflowSha":"a" * 40},
                        {"workflowSha":"not-a-commit"},
                        {"workflowCommit":{"sha":"e" * 40,"parents":[{"sha":"c" * 40},{"sha":"a" * 40}]}},
                        {"workflowCommit":None}):
            with self.subTest(changes=changes):
                result = self.run_entry(method="requireProtectedBase", args=self.protected_base_args(**changes))
                self.assertIn("error", result)

    def test_pr_workflow_identity_rejects_other_parent_order_or_history(self):
        for parents in ([], [{"sha":"c" * 40}],
                        [{"sha":"a" * 40},{"sha":"c" * 40}],
                        [{"sha":"e" * 40},{"sha":"a" * 40}],
                        [{"sha":"c" * 40},{"sha":"e" * 40}],
                        [{"sha":"c" * 40},{"sha":"a" * 40},{"sha":"e" * 40}],
                        [{"sha":"c" * 40},{}]):
            with self.subTest(parents=parents):
                args = self.protected_base_args(workflowCommit={"sha":"d" * 40,"parents":parents})
                result = self.run_entry(method="requireProtectedBase", args=args)
                self.assertIn("error", result)

    def test_manual_source_selection_cannot_bypass_fixed_workflow_identity(self):
        for workflow_source in ("base", "head", "", "PR-MERGE", None):
            with self.subTest(workflow_source=workflow_source):
                result = self.run_entry(method="requireProtectedBase",
                                        args=self.protected_base_args(workflowSource=workflow_source))
                self.assertIn("error", result)

    def test_base_workflow_identity_retains_strict_branch_and_commit_contract(self):
        base_args = {"branch":{"protected":True,"commit":{"sha":"c" * 40}}}
        result = self.run_entry(method="requireProtectedBase", args=base_args)
        self.assertNotIn("error", result)
        for changes in ({"workflowRef":"other"}, {"workflowSha":"d" * 40},
                        {"workflowRef":"npu-ci/pr-527/" + "d" * 40,"workflowSha":"d" * 40}):
            with self.subTest(changes=changes):
                result = self.run_entry(method="requireProtectedBase", args={**base_args, **changes})
                self.assertIn("error", result)

    def test_workflow_change_detection_handles_both_rename_paths_and_only_ci_yaml(self):
        for file in (".github/workflows/ci.yml",
                     {"filename":"other.yml","previous_filename":".github/workflows/ci.yml","status":"renamed"},
                     {"filename":".github/workflows/ci.yml","previous_filename":"other.yml","status":"renamed"}):
            with self.subTest(file=file):
                result = self.run_entry("helper_files", method="containsWorkflowChanges", files=[file])
                self.assertTrue(result["result"])
        for file in ("ci/run.sh", ".github/workflows/ci-contract-tests.yml", ".github/workflows/ci.yml.md"):
            with self.subTest(file=file):
                result = self.run_entry("helper_files", method="containsWorkflowChanges", files=[file])
                self.assertFalse(result["result"])

    def test_shared_changed_files_are_returned_after_snapshot_validation(self):
        result = self.run_entry(method="listPrFiles", files=["ci/run.sh","README.md"])
        self.assertEqual([file["filename"] for file in result["result"]], ["ci/run.sh","README.md"])
        self.assertEqual(self.records(result, "permission"), [])

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
        allowed = ["pr_number", "ci_mode", "ops", "requested_by", "expected_base_ref", "expected_base_sha", "expected_workflow_sha", "workflow_source"]
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

    def test_prepare_rejects_head_base_or_workflow_changes_after_dispatch(self):
        expected = {"requested_by":"ci-user", "expected_head_sha":"a"*40,
                    "expected_base_ref":"main", "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for field, old_value in (("expected_head_sha", "d"*40), ("expected_base_ref", "release"),
                                 ("expected_base_sha", "d"*40), ("expected_workflow_sha", "d"*40)):
            with self.subTest(field=field):
                result = self.run_entry("prepare", actor="github-actions[bot]", files=["ci/run_checks.sh"],
                                        inputs={**expected, field:old_value}, permissions={"ci-admin":"admin"})
                self.assertIn("触发快照", result["error"])
                self.assertEqual(self.records(result, "status"), [])
                self.assertFalse(any(item["name"] == "ci_approval_key" for item in self.records(result, "output")))

    def test_bot_dispatch_cannot_omit_any_identity_field(self):
        expected = {"requested_by":"ci-user", "expected_head_sha":"a"*40, "expected_base_ref":"main",
                    "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for field in ("expected_head_sha", "expected_base_ref", "expected_base_sha", "expected_workflow_sha"):
            with self.subTest(field=field):
                result = self.run_entry("prepare", actor="github-actions[bot]", inputs={**expected, field:""})
                self.assertIn("触发快照", result["error"])

    def test_manual_workflow_cannot_run_another_ref_or_outdated_base(self):
        for options in ({"workflow_ref":"other"}, {"workflow_sha":"d"*40}, {"branch_sha":"d"*40}):
            with self.subTest(options=options):
                result = self.run_entry("prepare", **options)
                self.assertIn("error", result)
                self.assertEqual(self.records(result, "status"), [])

    def pr_workflow_options(self, **changes):
        return {"files":[".github/workflows/ci.yml"], "permissions":{"ci-author":"admin"},
                "workflow_sha":"d" * 40, "workflow_ref":"npu-ci/pr-527/" + "d" * 40,
                "inputs":{"workflow_source":"pr-merge", "expected_head_sha":"a" * 40,
                          "expected_base_ref":"main", "expected_base_sha":"c" * 40,
                          "expected_workflow_sha":"d" * 40}, **changes}

    def test_prepare_accepts_pr_merge_workflow_with_fixed_snapshot_and_admin_author(self):
        result = self.run_entry("prepare", **self.pr_workflow_options())
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "failed"), [])
        self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))
        self.assertEqual([item["ref"] for item in self.records(result, "commit")], ["d" * 40])
        self.assertEqual(self.records(result, "permission"), ["ci-author"])

    def test_prepare_base_workflow_does_not_read_merge_commit(self):
        result = self.run_entry("prepare")
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "commit"), [])

    def test_prepare_rejects_base_source_for_workflow_changes(self):
        result = self.run_entry("prepare", files=[".github/workflows/ci.yml"], permissions={"ci-author":"admin"})
        self.assertIn("不能用旧 workflow", result["error"])
        self.assertEqual(self.records(result, "status"), [])

    def test_prepare_pr_merge_always_requires_admin_author_even_without_ci_files(self):
        result = self.run_entry("prepare", **self.pr_workflow_options(
            files=["fla/example.cpp"], actor="ci-admin", permissions={"ci-admin":"admin"}))
        self.assertEqual(self.records(result, "permission"), ["ci-author"])
        self.assertIn("PR 作者 @ci-author", self.records(result, "failed")[0])
        self.assertFalse(any(item == {"name":"should_run","value":"true"} for item in self.records(result, "output")))

    def test_manual_pr_merge_source_requires_every_fixed_snapshot_field(self):
        options = self.pr_workflow_options()
        for field in ("expected_head_sha", "expected_base_ref", "expected_base_sha", "expected_workflow_sha"):
            with self.subTest(field=field):
                result = self.run_entry("prepare", **{**options, "inputs":{**options["inputs"], field:""}})
                self.assertIn("触发快照", result["error"])
                self.assertEqual(self.records(result, "status"), [])

    def test_prepare_pr_merge_rejects_workflow_history_before_running_pr_code(self):
        for parents in ([{"sha":"c" * 40},{"sha":"e" * 40}],
                        [{"sha":"a" * 40},{"sha":"c" * 40}]):
            with self.subTest(parents=parents):
                result = self.run_entry("prepare", **self.pr_workflow_options(
                    workflow_commit={"sha":"d" * 40,"parents":parents}))
                self.assertIn("合并版本不匹配", result["error"])
                self.assertEqual(self.records(result, "status"), [])
                self.assertEqual(self.records(result, "permission"), [])

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
        for entry in ("dispatch", "prepare"):
            with self.subTest(entry=entry):
                snapshot = {"expected_head_sha":"a"*40, "expected_base_ref":"main",
                            "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
                result = self.run_entry(entry, attempt=2, list_error=True, inputs=snapshot)
                self.assertNotIn("error", result)
                self.assertFalse(any(item["path"] == ".github/npu-ci-users.json"
                                     for item in self.records(result, "content")))
                self.assertEqual(self.records(result, "permission"), [])
                if entry == "dispatch":
                    self.assertEqual(len(self.records(result, "dispatch")), 1)
                else:
                    self.assertIn({"name":"should_run","value":"true"}, self.records(result, "output"))

    def test_manual_rerun_without_complete_snapshot_refuses_new_prepare(self):
        snapshot = {"expected_head_sha":"a"*40, "expected_base_ref":"main",
                    "expected_base_sha":"c"*40, "expected_workflow_sha":"c"*40}
        for inputs in ({}, *({**snapshot, name:""} for name in snapshot)):
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
                result = self.run_entry(entry, files=["ci/run_checks.sh"], permissions={"ci-author":"admin"})
                self.assertNotIn("error", result)
                self.assertEqual(self.records(result, "failed"), [])
                self.assertEqual(self.records(result, "permission"), ["ci-author"])
                if entry == "dispatch":
                    inputs = self.records(result, "dispatch")[0]["inputs"]
                    self.assertNotIn("ci_modified", inputs)
                    self.assertEqual(len(inputs), 9)
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

    def test_device_version_guard_runs_offline_after_source_merge(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        steps = workflow["jobs"]["ascend-npu"]["steps"]
        guard_index = next(index for index, step in enumerate(steps) if step.get("id") == "validate_pr")
        merge_index = next(index for index, step in enumerate(steps) if step.get("name") == "现场合并 PR 代码")
        container_index = next(index for index, step in enumerate(steps)
                               if "bash ci/run_ci_container.sh" in step.get("run", ""))
        self.assertLess(merge_index, guard_index)
        self.assertLess(guard_index, container_index)
        guard = steps[guard_index]
        self.assertEqual(guard["working-directory"], "repo")
        self.assertNotIn("if", guard)
        self.assertNotIn("uses", guard)
        self.assertNotIn("NPU_CI_TRIGGERING_ACTOR", guard["env"])
        self.assertFalse(any(step.get("uses", "").startswith("actions/github-script@") for step in steps))


if __name__ == "__main__":
    unittest.main()
