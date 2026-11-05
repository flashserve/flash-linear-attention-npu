"""Run the default-branch tag cleanup against mocked GitHub API responses."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/npu-ci-workflow-cleanup.yml'
SHA = 'a' * 40
TAG = f'npu-ci/pr-17/{SHA}'


class NpuCiWorkflowCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
        cls.script = cls.workflow['jobs']['cleanup']['steps'][0]['with']['script']

    def run_cleanup(self, *, run=None, tag=None, active=None, missing=False,
                    delete_missing=False, repository='owner/repo'):
        spec = {
            'run': {'id': 7, 'event': 'workflow_dispatch', 'status': 'completed',
                    'path': '.github/workflows/ci.yml', 'head_branch': TAG,
                    'head_sha': SHA, 'repository': {'full_name': 'owner/repo'},
                    'head_repository': {'full_name': 'owner/repo'}, **(run or {})},
            'tag': {'ref': f'refs/tags/{TAG}',
                    'object': {'type': 'commit', 'sha': SHA}, **(tag or {})},
            'active': active or [], 'missing': missing,
            'delete_missing': delete_missing, 'repository': repository,
        }
        harness = r'''
const spec = JSON.parse(process.env.CLEANUP_SPEC);
const calls = [];
const context = { repo: {owner: 'owner', repo: 'repo'}, payload: {
  repository: {full_name: spec.repository}, workflow_run: spec.run,
}};
const github = {rest: {git: {
  getRef: async args => {
    calls.push(['get', args]);
    if (spec.missing) throw Object.assign(new Error('missing'), {status:404});
    return {data: spec.tag};
  },
  deleteRef: async args => {
    calls.push(['delete', args]);
    if (spec.delete_missing) throw Object.assign(new Error('missing'), {status:404});
  },
}, actions: {listWorkflowRuns: () => {}}},
paginate: async (method, args) => {calls.push(['list', args]); return spec.active;}};
const core = {notice: () => {}};
(async () => {
  await (async () => {
''' + self.script + r'''
  })();
  process.stdout.write(JSON.stringify(calls));
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
        result = subprocess.run(['node'], input=harness, text=True, encoding='utf-8',
                                capture_output=True, cwd=ROOT,
                                env={**os.environ, 'CLEANUP_SPEC': json.dumps(spec)})
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_branch_cleanup_never_loads_pr_code(self):
        triggers = self.workflow.get('on', self.workflow.get(True))
        self.assertEqual(triggers, {'workflow_run': {
            'workflows': ['NPU CI'], 'types': ['completed']}})
        self.assertEqual(self.workflow['permissions'], {'actions': 'read', 'contents': 'write'})
        self.assertEqual(self.workflow['jobs']['cleanup']['runs-on'], 'ubuntu-latest')
        self.assertEqual(len(self.workflow['jobs']['cleanup']['steps']), 1)
        self.assertNotIn('checkout', self.workflow['jobs']['cleanup']['steps'][0].get('uses', ''))

    def test_completed_run_deletes_only_exact_tag(self):
        calls = self.run_cleanup()
        self.assertEqual([item[0] for item in calls], ['get', 'list', 'delete'])
        self.assertEqual(calls[-1][1]['ref'], f'tags/{TAG}')
        self.assertEqual(calls[1][1]['head_sha'], SHA)
        self.assertEqual(calls[1][1]['per_page'], 100)

    def test_workflow_path_accepts_exact_tag_suffix(self):
        calls = self.run_cleanup(run={'path': f'.github/workflows/ci.yml@{TAG}'})
        self.assertEqual(calls[-1][0], 'delete')

    def test_unrelated_events_repositories_paths_and_refs_are_untouched(self):
        cases = [
            {'run': {'event': 'pull_request'}}, {'run': {'status': 'in_progress'}},
            {'run': {'path': '.github/workflows/other.yml'}},
            {'run': {'path': '.github/workflows/ci.yml@main'}},
            {'run': {'path': f'.github/workflows/ci.yml@{TAG}-other'}},
            {'run': {'path': f'.github/workflows/other.yml@{TAG}'}},
            {'run': {'repository': {'full_name': 'fork/repo'}}},
            {'run': {'head_repository': {'full_name': 'fork/repo'}}},
            {'repository': 'fork/repo'}, {'run': {'head_branch': 'main'}},
            {'run': {'head_branch': f'npu-ci/pr-0/{SHA}'}},
            {'run': {'head_branch': TAG + '-extra'}}, {'run': {'head_sha': 'b' * 40}},
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.run_cleanup(**case), [])

    def test_changed_annotated_or_wrong_ref_target_is_preserved(self):
        for tag in ({'object': {'type': 'commit', 'sha': 'b' * 40}},
                    {'object': {'type': 'tag', 'sha': SHA}}, {'ref': 'refs/tags/unrelated'}):
            with self.subTest(tag=tag):
                self.assertEqual([call[0] for call in self.run_cleanup(tag=tag)], ['get'])

    def test_duplicate_active_runs_keep_tag(self):
        for status in ('queued', 'in_progress', 'waiting', 'pending', 'requested'):
            active = [{'id': 8, 'event': 'workflow_dispatch', 'head_branch': TAG,
                       'head_sha': SHA, 'status': status}]
            with self.subTest(status=status):
                self.assertEqual([call[0] for call in self.run_cleanup(active=active)], ['get', 'list'])

    def test_completed_and_unrelated_runs_do_not_block_old_tag_cleanup(self):
        active = [
            {'id': 7, 'event': 'workflow_dispatch', 'head_branch': TAG,
             'head_sha': SHA, 'status': 'completed'},
            {'id': 8, 'event': 'workflow_dispatch', 'head_branch': TAG,
             'head_sha': SHA, 'status': 'completed'},
            {'id': 9, 'event': 'workflow_dispatch', 'head_branch': 'main',
             'head_sha': SHA, 'status': 'in_progress'},
        ]
        self.assertEqual(self.run_cleanup(active=active)[-1][0], 'delete')

    def test_missing_tag_and_concurrent_delete_are_idempotent(self):
        self.assertEqual([call[0] for call in self.run_cleanup(missing=True)], ['get'])
        self.assertEqual(self.run_cleanup(delete_missing=True)[-1][0], 'delete')


if __name__ == '__main__':
    unittest.main()
