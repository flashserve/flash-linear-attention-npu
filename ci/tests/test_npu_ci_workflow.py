"""Behavioral checks for selecting and dispatching a PR's CI workflow."""

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PrWorkflowDispatchTest(unittest.TestCase):
    def run_dispatch(self, **settings):
        script = r"""
const options = JSON.parse(process.env.TEST_OPTIONS);
const workflow = require('./.github/scripts/npu_ci_workflow.js');
const records = [], record = (kind, payload) => records.push({kind, payload});
const base = 'a'.repeat(40), head = 'b'.repeat(40), merge = 'c'.repeat(40);
const pr = {number:1079,state:'open',draft:false,user:{login:'author'},
  base:{ref:'main',sha:base},head:{sha:head},merge_commit_sha:merge,mergeable:true};
const context = {repo:{owner:'example',repo:'repo'}};
let reads = 0, current = pr;
const github = {rest:{
  pulls:{get:async payload => {
    record('read-pr',payload);
    const snapshot = (options.snapshots || [])[reads++] || {};
    current = {...pr,...snapshot,base:{...pr.base,sha:options.advanced_base || base,
      ...(snapshot.base || {})},head:{...pr.head,...(snapshot.head || {})}};
    return {data:current};
  }},
  repos:{
    getCollaboratorPermissionLevel:async payload => {
      record('admin',payload); return {data:{permission:options.permission || 'admin'}};
    },
    getContent:async payload => {
      record('read-workflow',payload);
      if (options.read_error) throw Object.assign(new Error('workflow unavailable'),{status:404});
      const names = options.legacy ? ['pr_number','ci_mode','ops','requested_by'] :
        ['pr_number','ci_mode','ops','requested_by','expected_head_sha','expected_base_ref',
         'expected_base_sha','expected_workflow_sha','workflow_source'];
      const text = 'on:\n  workflow_dispatch:\n    inputs:\n' + names.map(name =>
        '      '+name+':\n        type: string\n').join('');
      return {data:{type:'file',encoding:'base64',content:Buffer.from(text).toString('base64')}};
    },
    getCommit:async payload => {
      record('read-commit',payload);
      return {data:{sha:payload.ref,parents:(options.parents || [current.base.sha,head])
        .map(sha => ({sha}))}};
    },
  },
  actions:{createWorkflowDispatch:async payload => record('dispatch',payload)},
  git:{createRef:async () => { throw new Error('must use isolated ref client'); }},
}};
const refGithub = {rest:{git:{
  createRef:async payload => {
    record('tag-create',payload);
    if (options.tag_error) throw Object.assign(new Error('tag denied'),{status:403});
    if (options.tag_exists) throw Object.assign(new Error('already exists'),{status:422});
  },
  getRef:async payload => {
    record('tag-read',payload);
    return {data:{object:{type:options.tag_type || 'commit',sha:options.tag_target || merge}}};
  },
}}};
(async () => {
  let result, error;
  try {
    result = await workflow[options.select_only ? 'selectWorkflowExecution' : 'dispatchNpuWorkflow']({
      github,refGithub,context,pr,baseBranch:{commit:{sha:base}},
      files:(options.files || ['fla/op.cpp']).map(file => typeof file === 'string' ? {filename:file} : file),
      inputs:{pr_number:'1079',ci_mode:'quick',ops:'foo',requested_by:'runner',
        expected_head_sha:head,expected_base_ref:'main',expected_base_sha:base,
        expected_workflow_sha:base},wait:async ms => record('wait',ms),
    });
  } catch (caught) { error = caught.message; }
  process.stdout.write(JSON.stringify({result,error,records}));
})();
"""
        env = {**os.environ, "TEST_OPTIONS": json.dumps(settings)}
        completed = subprocess.run(["node"], input=script, text=True, encoding="utf-8",
                                   cwd=ROOT, env=env, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    @staticmethod
    def records(result, kind):
        return [record["payload"] for record in result["records"] if record["kind"] == kind]

    def test_source_only_uses_base_workflow_without_tag_or_admin_requirement(self):
        result = self.run_dispatch(permission="write")
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["source"], "base")
        self.assertEqual(self.records(result, "read-workflow")[0]["ref"], "a" * 40)
        self.assertEqual(self.records(result, "dispatch")[0]["ref"], "main")
        self.assertFalse(self.records(result, "tag-create"))
        self.assertFalse(self.records(result, "admin"))

    def test_legacy_workflow_filters_unknown_inputs_for_both_sources(self):
        for files in (["fla/op.cpp"], [".github/workflows/ci.yml"]):
            with self.subTest(files=files):
                result = self.run_dispatch(files=files, legacy=True)
                self.assertNotIn("error", result)
                self.assertEqual(self.records(result, "dispatch")[0]["inputs"], {
                    "pr_number": "1079", "ci_mode": "quick", "ops": "foo", "requested_by": "runner"})

    def test_admin_ci_changes_use_merge_workflow_and_isolated_tag_client(self):
        for filename in ("ci/run.py", ".github/scripts/helper.js", ".github/workflows/ci.yml"):
            with self.subTest(filename=filename):
                result = self.run_dispatch(files=[filename])
                self.assertNotIn("error", result)
                dispatch = self.records(result, "dispatch")[0]
                ref = "npu-ci/pr-1079/" + "c" * 40
                self.assertEqual(dispatch["ref"], ref)
                self.assertEqual(dispatch["inputs"]["workflow_source"], "pr-merge")
                self.assertEqual(dispatch["inputs"]["expected_workflow_sha"], "c" * 40)
                self.assertEqual(self.records(result, "read-workflow")[0]["ref"], "c" * 40)
                self.assertEqual(self.records(result, "tag-create"), [{
                    "owner": "example", "repo": "repo", "ref": "refs/tags/" + ref, "sha": "c" * 40}])

    def test_nonadmin_ci_changes_fail_before_reading_or_dispatching_workflow(self):
        for files in (["ci/run.py"], [".github/workflows/ci.yml"],
                      [{"filename": "src/moved.py", "previous_filename": "ci/run.py"}]):
            with self.subTest(files=files):
                result = self.run_dispatch(files=files, permission="write")
                self.assertIn("不是仓库 Admin", result["error"])
                for kind in ("read-workflow", "read-commit", "tag-create", "dispatch"):
                    self.assertFalse(self.records(result, kind))

    def test_markdown_files_follow_the_same_directory_policy(self):
        files = ["ci/README.md", ".github/README.MD"]
        result = self.run_dispatch(files=files, permission="read")
        self.assertIn("不是仓库 Admin", result["error"])
        self.assertFalse(self.records(result, "dispatch"))
        result = self.run_dispatch(files=files)
        self.assertNotIn("error", result)
        self.assertEqual(result["result"]["source"], "pr-merge")

    def test_workflow_read_or_tag_creation_failure_prevents_dispatch(self):
        for settings in ({"read_error": True}, {"tag_error": True}):
            with self.subTest(settings=settings):
                result = self.run_dispatch(files=["ci/run.py"], **settings)
                self.assertIn("error", result)
                self.assertFalse(self.records(result, "dispatch"))

    def test_merge_preview_must_match_both_parents(self):
        for parents in (["d" * 40, "b" * 40], ["a" * 40, "d" * 40], ["b" * 40]):
            with self.subTest(parents=parents):
                result = self.run_dispatch(files=["ci/run.py"], parents=parents)
                self.assertIn("合并预览", result["error"])
                self.assertEqual(len(self.records(result, "read-commit")), 4)
                self.assertFalse(self.records(result, "read-workflow"))
                self.assertFalse(self.records(result, "dispatch"))

    def test_conflicting_pr_cannot_dispatch(self):
        result = self.run_dispatch(files=["ci/run.py"], snapshots=[{"mergeable": False}])
        self.assertIn("冲突", result["error"])
        self.assertFalse(self.records(result, "dispatch"))

    def test_base_advance_uses_current_valid_merge_snapshot(self):
        result = self.run_dispatch(files=["ci/run.py"], advanced_base="d" * 40)
        self.assertNotIn("error", result)
        self.assertEqual(self.records(result, "dispatch")[0]["inputs"]["expected_base_sha"], "d" * 40)

    def test_head_or_target_change_before_dispatch_is_rejected(self):
        for change in ({"head": {"sha": "d" * 40}}, {"base": {"ref": "release"}}):
            with self.subTest(change=change):
                result = self.run_dispatch(files=["ci/run.py"], snapshots=[{}, change])
                self.assertIn("变化", result["error"])
                self.assertFalse(self.records(result, "dispatch"))

    def test_advancing_base_after_selection_does_not_invalidate_snapshot(self):
        result = self.run_dispatch(files=["ci/run.py"], snapshots=[{}, {"base": {"sha": "d" * 40}}])
        self.assertNotIn("error", result)
        self.assertEqual(len(self.records(result, "dispatch")), 1)

    def test_immutable_tag_can_be_reused_only_at_same_commit(self):
        result = self.run_dispatch(files=["ci/run.py"], tag_exists=True)
        self.assertNotIn("error", result)
        self.assertEqual(len(self.records(result, "tag-read")), 1)
        self.assertEqual(len(self.records(result, "dispatch")), 1)
        for settings in ({"tag_target": "d" * 40}, {"tag_type": "tag"}):
            with self.subTest(settings=settings):
                result = self.run_dispatch(files=["ci/run.py"], tag_exists=True, **settings)
                self.assertIn("拒绝覆盖", result["error"])
                self.assertFalse(self.records(result, "dispatch"))


if __name__ == "__main__":
    unittest.main()
