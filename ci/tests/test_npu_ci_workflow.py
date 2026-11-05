"""Exercise PR workflow selection, immutable references and failure handling."""

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PrWorkflowDispatchTest(unittest.TestCase):
    def run_dispatch(self, **settings):
        script = r"""
const settings = JSON.parse(process.env.TEST_OPTIONS);
const workflow = require('./.github/scripts/npu_ci_workflow.js');
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const base = 'a'.repeat(40), head = 'b'.repeat(40), merge = 'c'.repeat(40);
const pr = {number:1079, state:'open', draft:false, user:{login:'admin'},
  base:{ref:settings.base_ref || 'main', sha:base}, head:{sha:head},
  merge_commit_sha:merge, mergeable:true};
const context = {repo:{owner:'example',repo:'repo'}, serverUrl:'https://github.com',runId:500,
  payload:{comment:{id:100}}};
let reads = 0, commits = 0, dispatches = 0, tagReads = 0, commentReads = 0;
const runReads = {};
const cancelled = new Set();
let newRun, visibilityReads = 0, newRunReads = 0;
const registrations = settings.register_runs === false ? [] : (settings.runs || []).map(run => ({
  user:{login:'github-actions[bot]',type:'Bot'},
  body:`<!-- npu-ci-request:pr=1079:head=${(settings.run_heads || {})[run.id] || 'd'.repeat(40)}:run=${run.id} -->`,
}));
const github = {paginate:async (endpoint,payload) => {
  if (endpoint === github.rest.issues.listComments) {
    record('read-comments',payload);
    return [...registrations, ...((settings.comment_batches || [])[commentReads++] || settings.comments || [])];
  }
  if (endpoint === github.rest.repos.listCommitStatusesForRef) {
    record('read-statuses',payload);
    return settings.statuses || [];
  }
  if (endpoint === github.rest.actions.listWorkflowRuns) {
    record('list-runs',payload);
    if (payload.head_sha) {
      visibilityReads++;
      return visibilityReads <= (settings.visible_after || 0) ? [] :
        [settings.visible_run || newRun];
    }
    return (settings.runs || []).filter(run => run.status === payload.status);
  }
  throw new Error('unexpected pagination endpoint');
},rest:{
  issues:{listComments:() => {},createComment:async payload => {
    record('register-run',payload);
    if (settings.registration_error) throw new Error('comment registration denied');
    return {data:{id:10000}};
  }},
  pulls:{get:async () => {
    record('read-pr', ++reads);
    return {data:{...pr,...(settings.snapshots || [])[reads - 1]}};
  }},
  repos:{
    listCommitStatusesForRef:() => {},
    createCommitStatus:async payload => {
      record('status',payload);
      if (settings.status_error) throw new Error('status reset failed');
    },
    getContent:async (payload) => {
      record('read-workflow',payload);
      if (settings.workflow_missing) throw Object.assign(new Error('not found'),{status:404});
      const names = settings.legacy ? ['pr_number','ci_mode','ops','requested_by'] :
        ['pr_number','ci_mode','ops','requested_by','expected_head_sha','expected_base_ref',
         'expected_base_sha','expected_workflow_sha','workflow_source'];
      const text = 'on:\n  workflow_dispatch:\n    inputs:\n' + names.map(name =>
        '      '+name+':\n        type: string\n').join('');
      return {data:{type:'file',encoding:'base64',content:Buffer.from(text).toString('base64')}};
    },
    getCommit:async (payload) => {
      record('read-commit',payload);
      const status = (settings.commit_errors || [])[commits++];
      if (status) throw Object.assign(new Error('merge not ready'),{status});
      return {data:{sha:payload.ref,parents:(settings.parents || [base,head]).map(sha => ({sha}))}};
    },
  },
  actions:{listWorkflowRuns:() => {},
  cancelWorkflowRun:async payload => {
    record('cancel-run',payload);
    if (settings.cancel_error) throw Object.assign(new Error('cancel denied'),{status:settings.cancel_error});
    cancelled.add(payload.run_id);
  },
  getWorkflowRun:async payload => {
    record('read-run',payload);
    if (newRun && payload.run_id === newRun.id) {
      newRunReads++;
      const status = (settings.visibility_get_errors || [])[newRunReads - 1];
      if (status) throw Object.assign(new Error('new run unavailable'),{status});
      return {data:{...newRun,status:cancelled.has(newRun.id)
        ? (settings.new_run_cancel_stuck ? 'in_progress' : 'completed') : newRun.status}};
    }
    const run = (settings.runs || []).find(run => run.id === payload.run_id);
    const index = runReads[run.id] || 0;
    runReads[run.id] = index + 1;
    const states = (settings.run_states || {})[run.id] || ['completed'];
    return {data:{...run,status:states[Math.min(index,states.length-1)]}};
  },
  createWorkflowDispatch:async (payload) => {
    record('dispatch',payload);
    const status = (settings.dispatch_errors || [])[dispatches++];
    if (status) throw Object.assign(new Error('dispatch reference unavailable'),{status});
    if (settings.dispatch_error) throw new Error('dispatch unavailable');
    newRun = {id:9000,event:'workflow_dispatch',display_title:'NPU CI · PR #1079',
      repository:{full_name:'example/repo'},head_branch:payload.ref,
      head_sha:payload.inputs.expected_workflow_sha || base,path:'.github/workflows/ci.yml',
      status:'queued',...(settings.new_run || {})};
    return settings.dispatch_response === undefined
      ? {status:200,data:{workflow_run_id:newRun.id}} : settings.dispatch_response;
  }},
}};
const refGithub = {rest:{git:{
  createRef:async (payload) => {
    record('create-tag',payload);
    if (settings.tag_error) throw Object.assign(new Error('tag denied'),{status:settings.tag_error});
    return {data:{object:{type:'commit',sha:payload.sha}}};
  },
  getRef:async (payload) => {
    record('read-tag',payload);
    const status = (settings.read_tag_errors || [])[tagReads++] || settings.read_tag_error;
    if (status) throw Object.assign(new Error('tag not found'),{status});
    return {data:{object:{type:settings.tag_type || 'commit',sha:settings.tag_sha || merge}}};
  },
  deleteRef:async (payload) => record('delete-tag',payload),
}}};
(async () => {
  let result, error;
  try {
    const isCurrentRequest = settings.check_latest
      ? () => workflow.isLatestCiComment({github,context,prNumber:pr.number,users:['ci-user','admin']})
      : async () => true;
    if (settings.parse_body !== undefined) {
      result = workflow.parseCiCommand(settings.parse_body);
    } else result = await workflow.dispatchNpuWorkflow({github,refGithub,context,pr,
      baseBranch:{commit:{sha:base}}, files:settings.files || [{filename:'.github/workflows/ci.yml'}],
      isCurrentRequest,
      visibilityPolls:3,cancellationPolls:3,
      beforeDispatch:settings.replace ? async execution => workflow.replaceActiveNpuRuns({github,context,pr,execution,
        statusContexts:['NPU CI / first','NPU CI / final'],isCurrentRequest,
        cancellationPolls:3,wait:async ms => record('wait',ms)}) : undefined,
      inputs:{pr_number:String(pr.number),ci_mode:settings.ci_mode || 'quick',ops:settings.ops || '',requested_by:'ci-user',
        expected_head_sha:head,expected_base_ref:pr.base.ref,expected_base_sha:base,
        expected_workflow_sha:base}, wait:async ms => record('wait',ms)});
  } catch (caught) { error = caught.message; }
  process.stdout.write(JSON.stringify({result,error,records}));
})();
"""
        env = os.environ.copy()
        env["TEST_OPTIONS"] = json.dumps(settings)
        completed = subprocess.run(["node"], input=script, text=True, encoding="utf-8",
                                   cwd=ROOT, env=env, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    @staticmethod
    def records(result, kind):
        return [record["payload"] for record in result["records"] if record["kind"] == kind]

    @staticmethod
    def active_run(run_id=10, **overrides):
        return {"id": run_id, "event": "workflow_dispatch", "status": "in_progress",
                "display_title": "NPU CI · PR #1079", "repository": {"full_name": "example/repo"},
                "head_branch": "main", "head_sha": "a"*40,
                "path": ".github/workflows/ci.yml", **overrides}

    def test_command_parser_is_shared_with_latest_comment_filter(self):
        for body, expected in (("/run-npu-ci", {"ciMode": "quick", "ops": ""}),
                               (" /run-npu-ci full ops=foo,bar ", {"ciMode": "full", "ops": "foo,bar"}),
                               ("please /run-npu-ci", None)):
            with self.subTest(body=body):
                result = self.run_dispatch(parse_body=body)
                self.assertNotIn("error", result)
                self.assertEqual(result["result"], expected)
        for body in ("/run-npu-ci full quick", "/run-npu-ci ops=", "/run-npu-ci ops=a,,b",
                     "/run-npu-ci ops=a ops=b", "/run-npu-ci unsupported"):
            with self.subTest(body=body):
                self.assertIn("error", self.run_dispatch(parse_body=body))

    def test_only_newer_authorized_valid_comments_supersede_request(self):
        ignored = [
            {"id": 99, "body": "/run-npu-ci full", "user": {"login": "admin"}},
            {"id": 101, "body": "/run-npu-ci", "user": {"login": "outsider"}},
            {"id": 102, "body": "/run-npu-ci ops=", "user": {"login": "admin"}},
            {"id": 103, "body": "text /run-npu-ci", "user": {"login": "admin"}},
        ]
        result = self.run_dispatch(check_latest=True, comments=ignored)
        self.assertNotIn("error", result)
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        result = self.run_dispatch(check_latest=True, comments=ignored + [
            {"id": 104, "body": "/run-npu-ci full", "user": {"login": "ADMIN"}}])
        self.assertTrue(result["result"]["skipped"])
        self.assertEqual(self.records(result, "create-tag"), [])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_replacement_cancels_same_pr_across_commits_refs_and_modes_only(self):
        runs = [self.active_run(), self.active_run(11, status="queued", head_branch="release-v1"),
                self.active_run(12, status="waiting", head_branch="npu-ci/pr-1079/"+"d"*40,
                                path=".github/workflows/ci.yml@npu-ci/pr-1079/"+"d"*40),
                self.active_run(13, display_title="NPU CI · PR #1080"),
                self.active_run(14, path=".github/workflows/package.yml"),
                self.active_run(15, event="pull_request"),
                self.active_run(16, repository={"full_name": "other/repo"}),
                self.active_run(17, path=".github/workflows/ci.yml@wrong-ref"),
                self.active_run(18, status="completed")]
        result = self.run_dispatch(replace=True, runs=runs, run_states={"10": ["in_progress", "completed"]})
        self.assertNotIn("error", result)
        self.assertEqual({run["run_id"] for run in self.records(result, "cancel-run")}, {10, 11, 12})
        self.assertEqual(len(self.records(result, "wait")), 1)
        kinds = [item["kind"] for item in result["records"]]
        old_run_reads = [i for i, item in enumerate(result["records"])
                         if item["kind"] == "read-run" and item["payload"]["run_id"] != 9000]
        self.assertGreater(kinds.index("status"), max(old_run_reads))
        self.assertGreater(kinds.index("dispatch"), kinds.index("status"))
        statuses = self.records(result, "status")
        self.assertEqual([item["context"] for item in statuses], ["NPU CI / first", "NPU CI / final"])
        self.assertTrue(all(item["state"] == "pending" and item["sha"] == "b"*40 and
                            item["target_url"].endswith("/actions/runs/500") for item in statuses))

    def test_completed_results_are_reset_and_never_reused_for_new_comment(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}], legacy=True)
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "cancel-run"), [])
        self.assertEqual(len(self.records(result, "status")), 2)
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "create-tag"), [])

    def test_same_snapshot_active_run_is_kept_without_dispatch_or_status_reset(self):
        for status in ("queued", "in_progress"):
            for source in ("base", "pr-merge"):
                with self.subTest(status=status, source=source):
                    run = self.active_run(status=status)
                    files = [{"filename": "README.md"}]
                    if source == "pr-merge":
                        run.update(head_sha="c"*40, head_branch="npu-ci/pr-1079/"+"c"*40)
                        files = [{"filename": "ci/run_checks.sh"}]
                    result = self.run_dispatch(replace=True, runs=[run], files=files,
                                               ci_mode="full", ops="changed_scope",
                                               run_heads={"10": "b"*40})
                    self.assertNotIn("error", result)
                    self.assertTrue(result["result"]["skipped"])
                    self.assertEqual(result["result"]["reason"], "duplicate")
                    self.assertEqual(result["result"]["runId"], 10)
                    for kind in ("cancel-run", "status", "create-tag", "dispatch", "register-run"):
                        self.assertEqual(self.records(result, kind), [], kind)

    def test_new_pr_head_replaces_run_using_same_base_workflow(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                   runs=[self.active_run()])
        self.assertNotIn("error", result)
        self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [10])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "create-tag"), [])

    def test_same_head_with_new_base_or_target_replaces_outdated_merge_snapshot(self):
        for change in ({"head_sha": "d"*40}, {"head_branch": "release-v1"}):
            with self.subTest(change=change):
                result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                           runs=[self.active_run(**change)], run_heads={"10": "b"*40})
                self.assertNotIn("error", result)
                self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [10])
                self.assertEqual(len(self.records(result, "dispatch")), 1)

    def test_existing_duplicates_keep_newest_matching_run_and_cancel_only_others(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                   runs=[self.active_run(), self.active_run(11, status="queued"),
                                         self.active_run(12, head_sha="d"*40)],
                                   run_states={"11": ["in_progress"]},
                                   run_heads={"10": "b"*40, "11": "b"*40, "12": "d"*40})
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["runId"], 11)
        self.assertEqual({item["run_id"] for item in self.records(result, "cancel-run")}, {10, 12})
        for kind in ("status", "create-tag", "dispatch"):
            self.assertEqual(self.records(result, kind), [])

    def test_keeper_finishing_during_duplicate_cleanup_gets_fresh_verification(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                   runs=[self.active_run(), self.active_run(11)],
                                   run_heads={"10": "b"*40, "11": "b"*40})
        self.assertNotIn("error", result)
        self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [10])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(len(self.records(result, "status")), 2)

    def test_unknown_legacy_run_is_preserved_instead_of_guessing_workflow_head(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                   runs=[self.active_run(head_sha="b"*40)], register_runs=False)
        self.assertIn("无法确认已有 NPU CI", result["error"])
        self.assertIn("保留任务", result["error"])
        for kind in ("cancel-run", "status", "create-tag", "dispatch"):
            self.assertEqual(self.records(result, kind), [])

    def test_existing_prepare_comment_associates_legacy_run_with_pr_head(self):
        for suffix in (")", "/job/123)"):
            with self.subTest(suffix=suffix):
                result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                           runs=[self.active_run()], register_runs=False, comments=[{
                                               "user": {"login": "github-actions[bot]"},
                                               "body": "<!-- npu-ci:pr=1079:sha="+"b"*40+" -->\n"
                                                       "[run](https://github.com/example/repo/actions/runs/10"+suffix,
                                           }])
                self.assertNotIn("error", result)
                self.assertEqual(result["result"]["runId"], 10)
                self.assertEqual(self.records(result, "dispatch"), [])
                self.assertEqual(self.records(result, "read-statuses"), [])

    def test_historical_current_head_statuses_associate_unchanged_old_branch_run(self):
        result = self.run_dispatch(replace=True, legacy=True, files=[{"filename": "README.md"}],
                                   runs=[self.active_run()], register_runs=False, statuses=[
                                       {"context": "NPU CI / first", "target_url":
                                        "https://github.com/example/repo/actions/runs/99"},
                                       {"context": "NPU CI / first", "target_url":
                                        "https://github.com/example/repo/actions/runs/10"},
                                   ])
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["runId"], 10)
        self.assertEqual(self.records(result, "read-statuses")[0]["ref"], "b"*40)
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_untrusted_comments_and_unrelated_status_urls_cannot_register_active_run(self):
        for comments, statuses in (([{
                "user": {"login": "ci-user"},
                "body": "<!-- npu-ci-request:pr=1079:head="+"b"*40+":run=10 -->",
            }], []), ([{
                "user": {"login": "github-actions[bot]"},
                "body": "<!-- npu-ci-request:pr=1080:head="+"b"*40+":run=10 -->",
            }], []), ([], [{"context": "other CI", "target_url":
                            "https://github.com/example/repo/actions/runs/10"}]),
                ([], [{"context": "NPU CI / first", "target_url":
                       "https://github.com/other/repo/actions/runs/10"}])):
            with self.subTest(comments=comments, statuses=statuses):
                result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                           runs=[self.active_run()], register_runs=False,
                                           comments=comments, statuses=statuses)
                self.assertIn("无法确认已有 NPU CI", result["error"])
                self.assertEqual(self.records(result, "cancel-run"), [])
                self.assertEqual(self.records(result, "dispatch"), [])

    def test_conflicting_bot_registrations_preserve_active_run_and_fail(self):
        result = self.run_dispatch(replace=True, runs=[self.active_run()], comments=[{
            "user": {"login": "github-actions[bot]"},
            "body": "<!-- npu-ci-request:pr=1079:head="+"b"*40+":run=10 -->",
        }])
        self.assertIn("登记冲突", result["error"])
        self.assertEqual(self.records(result, "cancel-run"), [])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_new_run_is_registered_with_pr_head_and_confirmed_run_link(self):
        result = self.run_dispatch(files=[{"filename": "README.md"}], legacy=True)
        self.assertNotIn("error", result)
        comment = self.records(result, "register-run")[0]
        self.assertEqual(comment["issue_number"], 1079)
        self.assertIn("<!-- npu-ci-request:pr=1079:head="+"b"*40+":run=9000 -->", comment["body"])
        self.assertIn("https://github.com/example/repo/actions/runs/9000", comment["body"])
        kinds = [item["kind"] for item in result["records"]]
        self.assertGreater(kinds.index("register-run"), kinds.index("read-run"))

    def test_registration_failure_cancels_confirmed_run_before_dispatcher_releases(self):
        result = self.run_dispatch(registration_error=True)
        self.assertIn("无法登记 NPU CI #9000", result["error"])
        self.assertIn("任务已取消", result["error"])
        self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [9000])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "read-run")[-1]["run_id"], 9000)

    def test_registration_failure_with_stuck_cancellation_cannot_redispatch(self):
        result = self.run_dispatch(registration_error=True, new_run_cancel_stuck=True)
        self.assertIn("尚未停止", result["error"])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(self.records(result, "wait"), [5000, 5000])

    def test_failed_or_timed_out_cancellation_never_resets_or_dispatches(self):
        for options in ({"cancel_error": 403}, {"run_states": {"10": ["in_progress"]}}):
            with self.subTest(options=options):
                result = self.run_dispatch(replace=True, runs=[self.active_run()], **options)
                self.assertIn("error", result)
                self.assertEqual(self.records(result, "status"), [])
                self.assertEqual(self.records(result, "dispatch"), [])
                self.assertLessEqual(len(self.records(result, "wait")), 2)

    def test_run_finishing_during_cancel_is_verified_before_replacement(self):
        result = self.run_dispatch(replace=True, runs=[self.active_run()], cancel_error=409)
        self.assertNotIn("error", result)
        self.assertEqual([item["run_id"] for item in self.records(result, "read-run")], [10, 9000])
        self.assertEqual(len(self.records(result, "dispatch")), 1)

    def test_newer_comment_during_cancellation_prevents_old_request_dispatch(self):
        newer = [{"id": 101, "body": "/run-npu-ci", "user": {"login": "ci-user"}}]
        result = self.run_dispatch(replace=True, check_latest=True, runs=[self.active_run()],
                                   comment_batches=[[], [], [], newer], comments=newer)
        self.assertNotIn("error", result)
        self.assertTrue(result["result"]["skipped"])
        self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [10])
        self.assertEqual(self.records(result, "status"), [])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_pr_changes_while_canceling_do_not_start_old_snapshot(self):
        result = self.run_dispatch(replace=True, files=[{"filename": "README.md"}],
                                   snapshots=[{"head": {"sha": "d"*40}}])
        self.assertIn("已变化", result["error"])
        self.assertEqual(self.records(result, "status"), [])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_failed_status_reset_does_not_dispatch_legacy_workflow(self):
        result = self.run_dispatch(replace=True, status_error=True)
        self.assertIn("status reset failed", result["error"])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_dispatch_waits_for_run_visibility_with_pinned_api_response(self):
        result = self.run_dispatch(visible_after=2)
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["runId"], 9000)
        dispatch = self.records(result, "dispatch")[0]
        self.assertEqual(dispatch["headers"], {"X-GitHub-Api-Version": "2026-03-10"})
        self.assertEqual(self.records(result, "wait"), [2000, 2000])
        visibility_queries = self.records(result, "list-runs")
        self.assertEqual(len(visibility_queries), 3)
        self.assertTrue(all(query["workflow_id"] == "ci.yml" and
                            query["event"] == "workflow_dispatch" and
                            query["head_sha"] == "c"*40 for query in visibility_queries))
        self.assertEqual(self.records(result, "cancel-run"), [])

    def test_transient_get_run_404_is_confirmation_retry_not_redispatch(self):
        result = self.run_dispatch(visibility_get_errors=[404, 404])
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "wait"), [2000, 2000])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(len(self.records(result, "create-tag")), 1)
        self.assertEqual(self.records(result, "read-tag"), [])

    def test_completed_run_needs_no_visibility_poll(self):
        result = self.run_dispatch(new_run={"status": "completed"})
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["runId"], 9000)
        self.assertEqual(self.records(result, "list-runs"), [])
        self.assertEqual(self.records(result, "wait"), [])

    def test_invalid_dispatch_response_never_redispatches_or_cancels_unknown_run(self):
        for response in (None, {"status": 204}, {"status": 200, "data": {}},
                         {"status": 200, "data": {"workflow_run_id": 0}},
                         {"status": 200, "data": {"workflow_run_id": "9000"}}):
            with self.subTest(response=response):
                result = self.run_dispatch(dispatch_response=response)
                self.assertIn("有效的 NPU CI 运行编号", result["error"])
                self.assertEqual(len(self.records(result, "dispatch")), 1)
                self.assertEqual(self.records(result, "read-run"), [])
                self.assertEqual(self.records(result, "cancel-run"), [])
                self.assertEqual(self.records(result, "read-tag"), [])

    def test_new_run_must_match_repository_pr_workflow_ref_and_commit(self):
        for change in ({"repository": {"full_name": "other/repo"}},
                       {"display_title": "NPU CI · PR #1080"}, {"event": "push"},
                       {"head_branch": "main"}, {"head_sha": "d"*40},
                       {"path": ".github/workflows/package.yml"}):
            with self.subTest(change=change):
                result = self.run_dispatch(new_run=change)
                self.assertIn("不匹配", result["error"])
                self.assertIn("#9000", result["error"])
                self.assertEqual(len(self.records(result, "dispatch")), 1)
                self.assertEqual(self.records(result, "cancel-run"), [])

    def test_visibility_timeout_cancels_known_run_before_releasing_queue(self):
        result = self.run_dispatch(visible_after=99)
        self.assertIn("运行列表超时", result["error"])
        self.assertIn("该任务已取消", result["error"])
        self.assertIn("https://github.com/example/repo/actions/runs/9000", result["error"])
        self.assertEqual([item["run_id"] for item in self.records(result, "cancel-run")], [9000])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(len(self.records(result, "create-tag")), 1)
        self.assertEqual(self.records(result, "read-tag"), [])
        self.assertEqual(self.records(result, "read-run")[-1]["run_id"], 9000)

    def test_invisible_run_404_timeout_does_not_recreate_tag_or_dispatch(self):
        result = self.run_dispatch(visibility_get_errors=[404, 404, 404])
        self.assertIn("#9000", result["error"])
        self.assertIn("运行列表超时", result["error"])
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        self.assertEqual(len(self.records(result, "create-tag")), 1)
        self.assertEqual(self.records(result, "read-tag"), [])
        self.assertEqual(self.records(result, "cancel-run"), [])

    def test_failed_visibility_cancellation_is_bounded_and_actionable(self):
        result = self.run_dispatch(visible_after=99, new_run_cancel_stuck=True)
        self.assertIn("取消确认失败", result["error"])
        self.assertIn("请在 Actions 中确认已停止", result["error"])
        self.assertEqual(self.records(result, "wait"), [2000, 2000, 5000, 5000])
        self.assertEqual(len(self.records(result, "dispatch")), 1)

    def test_invalid_response_after_actual_missing_tag_retry_is_not_retried_again(self):
        result = self.run_dispatch(dispatch_errors=[404], read_tag_errors=[404], dispatch_response=None)
        self.assertIn("有效的 NPU CI 运行编号", result["error"])
        self.assertEqual(len(self.records(result, "dispatch")), 2)
        self.assertEqual(len(self.records(result, "create-tag")), 2)

    def test_unchanged_legacy_workflow_uses_target_without_new_contract(self):
        for files in ([{"filename":"fla/ops/example.cpp"}],
                      [{"filename":"README.md"}],
                      [{"filename":"ci/README.MD"}, {"filename":".github/notes.md"}]):
            with self.subTest(files=files):
                result = self.run_dispatch(files=files, legacy=True, base_ref="release-v1")
                self.assertNotIn("error", result)
                dispatch = self.records(result,"dispatch")[0]
                self.assertEqual(dispatch["ref"],"release-v1")
                self.assertEqual(set(dispatch["inputs"]),{"pr_number","ci_mode","ops","requested_by"})
                self.assertEqual(self.records(result,"read-pr"),[])
                self.assertEqual(self.records(result,"create-tag"),[])

    def test_ci_code_changes_use_pr_merge_even_without_yaml_changes(self):
        for filename in ("ci/run_checks.sh", ".github/scripts/publish_npu_ci_status.js",
                         ".github/npu-ci-users.json", ".github/workflows/ci-contract-tests.yml"):
            with self.subTest(filename=filename):
                result = self.run_dispatch(files=[{"filename":filename}])
                self.assertNotIn("error", result)
                self.assertEqual(result["result"]["source"], "pr-merge")
                self.assertEqual(self.records(result,"read-workflow")[0]["ref"],"c"*40)
                self.assertEqual(len(self.records(result,"create-tag")), 1)

    def test_modified_workflow_dispatches_fixed_pr_merge_tag(self):
        result = self.run_dispatch()
        self.assertNotIn("error",result)
        dispatch = self.records(result,"dispatch")[0]
        self.assertEqual(dispatch["ref"],"npu-ci/pr-1079/"+"c"*40)
        self.assertEqual(dispatch["inputs"]["expected_workflow_sha"],"c"*40)
        self.assertEqual(dispatch["inputs"]["workflow_source"],"pr-merge")
        self.assertEqual(self.records(result,"read-workflow")[0]["ref"],"c"*40)
        self.assertEqual(self.records(result,"create-tag")[0]["sha"],"c"*40)
        self.assertNotIn("token",json.dumps(dispatch))

    def test_workflow_renames_are_checked_on_both_paths(self):
        for old,new in ((".github/workflows/ci.yml","other.yml"),
                        ("other.yml",".github/workflows/ci.yml")):
            with self.subTest(old=old):
                result = self.run_dispatch(files=[{"filename":new,"previous_filename":old,"status":"renamed"}])
                self.assertEqual(result["result"]["source"],"pr-merge")

    def test_deleted_or_unreadable_workflow_never_falls_back(self):
        result = self.run_dispatch(workflow_missing=True)
        self.assertIn("保留可执行",result["error"])
        self.assertEqual(self.records(result,"dispatch"),[])
        self.assertEqual(self.records(result,"create-tag"),[])

    def test_modified_legacy_yaml_requires_contract_in_pr_only(self):
        for filename in (".github/workflows/ci.yml", "ci/run_checks.sh"):
            with self.subTest(filename=filename):
                result = self.run_dispatch(legacy=True, files=[{"filename":filename}])
                self.assertIn("workflow_source",result["error"])
                self.assertIn("非 Markdown",result["error"])
                self.assertEqual(self.records(result,"dispatch"),[])

    def test_merge_must_have_exact_base_and_head_parents(self):
        for parents in (("a"*40,"d"*40),("b"*40,"a"*40),("a"*40,),
                        ("a"*40,"b"*40,"d"*40)):
            with self.subTest(parents=parents):
                result = self.run_dispatch(parents=parents)
                self.assertIn("合并预览",result["error"])
                self.assertEqual(len(self.records(result,"wait")),3)
                self.assertEqual(self.records(result,"create-tag"),[])
                self.assertEqual(self.records(result,"dispatch"),[])

    def test_transient_merge_metadata_is_retried(self):
        result = self.run_dispatch(snapshots=[{"merge_commit_sha":None}],commit_errors=[404])
        self.assertNotIn("error",result)
        self.assertEqual(len(self.records(result,"wait")),2)

    def test_conflicting_pr_fails_immediately(self):
        result = self.run_dispatch(snapshots=[{"mergeable":False}])
        self.assertIn("合并冲突",result["error"])
        self.assertEqual(self.records(result,"wait"),[])
        self.assertEqual(self.records(result,"dispatch"),[])

    def test_head_or_base_change_before_selection_fails(self):
        for change in ({"head":{"sha":"d"*40}},
                       {"base":{"sha":"d"*40,"ref":"main"}},
                       {"base":{"sha":"a"*40,"ref":"release"}}):
            with self.subTest(change=change):
                result = self.run_dispatch(snapshots=[change])
                self.assertIn("已变化",result["error"])
                self.assertEqual(self.records(result,"create-tag"),[])

    def test_head_change_after_tag_creation_keeps_tag_for_other_requests(self):
        result = self.run_dispatch(snapshots=[{}, {"head":{"sha":"d"*40}}])
        self.assertIn("已变化",result["error"])
        self.assertIn("已保留",result["error"])
        self.assertEqual(self.records(result,"delete-tag"),[])
        self.assertEqual(self.records(result,"dispatch"),[])

    def test_existing_matching_tag_is_reused(self):
        result = self.run_dispatch(tag_error=422)
        self.assertNotIn("error",result)
        self.assertEqual(len(self.records(result,"read-tag")),1)
        self.assertEqual(len(self.records(result,"dispatch")),1)

    def test_tampered_tag_is_never_overwritten(self):
        for settings in ({"tag_sha":"d"*40},{"tag_type":"tag"}):
            with self.subTest(settings=settings):
                result = self.run_dispatch(tag_error=422,**settings)
                self.assertIn("拒绝覆盖",result["error"])
                self.assertEqual(self.records(result,"dispatch"),[])
                self.assertEqual(self.records(result,"delete-tag"),[])

    def test_missing_tag_permissions_has_actionable_error_and_no_base_fallback(self):
        result = self.run_dispatch(tag_error=403)
        self.assertIn("NPU_CI_DISPATCH_TOKEN",result["error"])
        self.assertIn("Contents 和 Workflows",result["error"])
        self.assertEqual(self.records(result,"dispatch"),[])

    def test_reference_update_failed_without_existing_tag_explains_required_token(self):
        result = self.run_dispatch(tag_error=422,read_tag_error=404)
        self.assertIn("NPU_CI_DISPATCH_TOKEN",result["error"])
        self.assertIn("Contents 和 Workflows",result["error"])
        self.assertEqual(self.records(result,"dispatch"),[])

    def test_failed_dispatch_keeps_new_and_reused_tags_for_concurrent_requests(self):
        for settings in ({},{"tag_error":422}):
            with self.subTest(settings=settings):
                result = self.run_dispatch(dispatch_error=True,**settings)
                self.assertIn("dispatch unavailable",result["error"])
                self.assertIn("npu-ci/pr-1079/"+"c"*40,result["error"])
                self.assertIn("已保留",result["error"])
                self.assertEqual(self.records(result,"delete-tag"),[])

    def test_deleted_tag_is_recreated_and_dispatch_retried_once(self):
        for status in (404,422):
            with self.subTest(status=status):
                result = self.run_dispatch(dispatch_errors=[status],read_tag_errors=[404])
                self.assertNotIn("error",result)
                self.assertEqual(len(self.records(result,"create-tag")),2)
                self.assertEqual(len(self.records(result,"dispatch")),2)
                self.assertEqual(len(self.records(result,"read-pr")),3)
                self.assertEqual(self.records(result,"create-tag")[0],self.records(result,"create-tag")[1])

    def test_persistent_reference_failure_is_not_retried_more_than_once(self):
        result = self.run_dispatch(dispatch_errors=[404,404],read_tag_errors=[404])
        self.assertIn("dispatch reference unavailable",result["error"])
        self.assertEqual(len(self.records(result,"dispatch")),2)
        self.assertEqual(len(self.records(result,"create-tag")),2)

    def test_dispatch_failure_with_existing_tag_does_not_retry(self):
        for status in (404,422):
            with self.subTest(status=status):
                result = self.run_dispatch(dispatch_errors=[status])
                self.assertIn("dispatch reference unavailable",result["error"])
                self.assertEqual(len(self.records(result,"dispatch")),1)
                self.assertEqual(len(self.records(result,"create-tag")),1)

    def test_retry_rechecks_pr_snapshot_before_dispatching(self):
        result = self.run_dispatch(dispatch_errors=[404],read_tag_errors=[404],
                                   snapshots=[{}, {}, {"head":{"sha":"d"*40}}])
        self.assertIn("已变化",result["error"])
        self.assertEqual(len(self.records(result,"dispatch")),1)
        self.assertEqual(len(self.records(result,"create-tag")),2)


if __name__ == "__main__":
    unittest.main()
