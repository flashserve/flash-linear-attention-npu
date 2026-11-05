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
const context = {repo:{owner:'example',repo:'repo'}};
let reads = 0, commits = 0, dispatches = 0, tagReads = 0;
const github = {rest:{
  pulls:{get:async () => {
    record('read-pr', ++reads);
    return {data:{...pr,...(settings.snapshots || [])[reads - 1]}};
  }},
  repos:{
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
  actions:{createWorkflowDispatch:async (payload) => {
    record('dispatch',payload);
    const status = (settings.dispatch_errors || [])[dispatches++];
    if (status) throw Object.assign(new Error('dispatch reference unavailable'),{status});
    if (settings.dispatch_error) throw new Error('dispatch unavailable');
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
    result = await workflow.dispatchNpuWorkflow({github,refGithub,context,pr,
      baseBranch:{commit:{sha:base}}, files:settings.files || [{filename:'.github/workflows/ci.yml'}],
      inputs:{pr_number:String(pr.number),ci_mode:'quick',ops:'',requested_by:'ci-user',
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

    def test_unchanged_legacy_workflow_uses_target_without_new_contract(self):
        result = self.run_dispatch(files=[{"filename":"src/feature.cpp"}], legacy=True,
                                   base_ref="release-v1")
        self.assertNotIn("error", result)
        dispatch = self.records(result,"dispatch")[0]
        self.assertEqual(dispatch["ref"],"release-v1")
        self.assertEqual(set(dispatch["inputs"]),{"pr_number","ci_mode","ops","requested_by"})
        self.assertEqual(self.records(result,"read-pr"),[])
        self.assertEqual(self.records(result,"create-tag"),[])

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
        result = self.run_dispatch(legacy=True)
        self.assertIn("workflow_source",result["error"])
        self.assertEqual(self.records(result,"dispatch"),[])

    def test_modified_ci_control_script_also_requires_pr_workflow_contract(self):
        result = self.run_dispatch(legacy=True, files=[{"filename":".github/scripts/changed.js"}])
        self.assertIn("workflow_source", result["error"])
        self.assertEqual(self.records(result, "dispatch"), [])

    def test_markdown_under_ci_control_path_keeps_target_workflow(self):
        result = self.run_dispatch(legacy=True, files=[{"filename":".github/scripts/README.md"}])
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "dispatch")[0]["ref"], "main")

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
