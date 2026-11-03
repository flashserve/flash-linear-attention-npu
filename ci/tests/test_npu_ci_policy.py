#!/usr/bin/env python3
"""Exercise NPU CI authorization, change inspection, preparation, and comments."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = r"""
const config = JSON.parse(process.env.NPU_POLICY_TEST_CONFIG);
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const head = 'a'.repeat(40);
const files = config.files || [{filename: 'operator.cpp', status: 'modified'}];
const pr = {
  number: 527, state: config.state || 'open', draft: config.draft || false,
  html_url: 'https://github.com/example/repo/pull/527', user: {login: 'pr-author'},
  changed_files: config.changed_files === undefined ? files.length : config.changed_files,
  base: {ref: config.baseRef || 'main', sha: config.baseSha || 'b'.repeat(40)},
  head: {sha: head, repo: {full_name: 'author/repo'}},
};
const context = {
  actor: config.actor || 'ci-user', repo: {owner: 'example', repo: 'repo'},
  runId: 901, runNumber: 23, serverUrl: 'https://github.com',
  payload: {
    repository: {default_branch: 'main'}, pull_request: pr, action: config.action || 'opened',
    inputs: {pr_number: '527', ci_mode: 'quick', ...(config.inputs || {})},
    issue: {number: 527}, comment: {body: config.comment || '/run-npu-ci quick'},
  },
};
if (config.noRepositoryPayload) delete context.payload.repository;
let getCalls = 0;
const github = {
  rest: {
    repos: {
      get: async (params) => {record('repository', params); return {data: {default_branch: 'main'}};},
      getContent: async (params) => {
        record('content', params);
        let content;
        if (params.path.endsWith('npu-ci-users.json')) {
          if (config.allowlistError) throw new Error('allowlist unavailable');
          content = JSON.stringify(config.allowlist || {users: ['ci-user']});
        } else if (params.path.endsWith('npu-ci-default-status.yml')) {
          content = config.branchWorkflow || "const contexts = ['NPU CI / legacy / 01'];";
        } else {
          content = 'name: NPU CI\non:\n  workflow_dispatch:\n';
        }
        return {data: {type: 'file', encoding: 'base64', content: Buffer.from(content).toString('base64')}};
      },
      getBranch: async (params) => {record('branch', params); return {data: {commit: {sha: config.baseSha || 'b'.repeat(40)}}};},
      getCollaboratorPermissionLevel: async (params) => {
        record('author-permission', params);
        if (config.permissionError) throw new Error('permission API unavailable');
        return {data: {permission: config.authorPermission || 'read'}};
      },
      listCommitStatusesForRef: async (params) => {record('statuses', params); return {data: config.statuses || []};},
      createCommitStatus: async (params) => {record('status', params); return {data: params};},
    },
    pulls: {
      get: async (params) => {
        record('pull', params);
        getCalls += 1;
        const result = JSON.parse(JSON.stringify(pr));
        const afterInitial = getCalls > (config.mode === 'prepare' ? 1 : 0);
        if (config.headRace && afterInitial) result.head.sha = 'd'.repeat(40);
        return {data: result};
      },
      listFiles: async (params) => {
        record('files', params);
        if (config.filesError) throw new Error('files unavailable');
        return {data: files};
      },
    },
    issues: {createComment: async (params) => {record('comment', params); return {data: {id: records.length}};}},
    actions: {
      createWorkflowDispatch: async (params) => record('dispatch', params),
      getWorkflowRun: async (params) => {record('run', params); return {data: {status: 'in_progress'}};},
    },
  },
  paginate: async (method, params) => {record('paginate', params); return (await method(params)).data;},
};
const core = {
  setOutput: (name, value) => record('output', {name, value}),
  notice: (message) => record('notice', message), warning: (message) => record('warning', message),
};
process.env.NPU_CI_WORKFLOW_SHA = config.workflowSha || 'c'.repeat(40);
process.env.NPU_CI_WORKFLOW_REF = 'ci-definition';
process.env.NPU_CI_RUN_ATTEMPT = String(config.attempt || 1);
process.env.NPU_CI_TRIGGERING_ACTOR = config.triggeringActor === undefined ? 'ci-user' : config.triggeringActor;
process.env.GITHUB_WORKSPACE = process.cwd();
(async () => {
  const policy = require('./.github/scripts/npu_ci_policy');
  try {
    let result;
    if (config.mode === 'markdown') result = await policy.isMarkdownOnly({github, context, pr});
    else if (config.mode === 'authorization') result = await policy.requireAuthorizedCiUser({github, context, username: context.actor});
    else if (config.mode === 'matrix-guard') {
      // Execute the workflow's exact script while mapping its trusted checkout
      // path to the same policy module, without creating a fixture checkout.
      const guardRequire = (path) => {
        const expectedPath = `${process.env.GITHUB_WORKSPACE}/${config.policyCheckoutPath}/.github/scripts/npu_ci_policy.js`;
        if (path !== expectedPath) throw new Error(`Unexpected policy require path: ${path}`);
        record('policy-require', path);
        return policy;
      };
      const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
      await new AsyncFunction('github', 'context', 'core', 'require', config.guardScript)(github, context, core, guardRequire);
    }
    else {
      const entry = {prepare: 'prepare_npu_ci', default: 'default_npu_ci_status', dispatch: 'dispatch_npu_ci'}[config.mode];
      await require('./.github/scripts/' + entry)({github, context, core});
    }
    record('result', result === undefined ? null : result);
  } catch (error) {record('error', error.message);}
  process.stdout.write(JSON.stringify(records));
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""


def run_policy(mode, **options):
    env = os.environ.copy()
    env['NPU_POLICY_TEST_CONFIG'] = json.dumps(dict(mode=mode, **options))
    completed = subprocess.run(['node'], input=HARNESS, text=True, encoding='utf-8',
                               capture_output=True, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


def values(records, kind):
    return [item['payload'] for item in records if item['kind'] == kind]


def outputs(records):
    return {item['name']: item['value'] for item in values(records, 'output')}


class NpuCiPolicyTest(unittest.TestCase):
    def test_authorization_uses_default_branch_file_and_ignores_repository_role(self):
        records = run_policy('authorization', actor='CI-USER', noRepositoryPayload=True)
        self.assertFalse(values(records, 'error'))
        self.assertEqual(values(records, 'content')[0]['ref'], 'main')
        self.assertEqual(values(records, 'content')[0]['path'], '.github/npu-ci-users.json')
        self.assertTrue(values(records, 'repository'))
        self.assertFalse(values(records, 'author-permission'))

    def test_repository_admin_outside_allowlist_cannot_trigger(self):
        records = run_policy('prepare', actor='outside-admin', authorPermission='admin')
        self.assertTrue(values(records, 'error'))
        self.assertFalse(values(records, 'pull'))
        self.assertFalse(values(records, 'status'))

    def test_missing_or_invalid_allowlist_fails_closed(self):
        for option in ({'allowlistError': True}, {'allowlist': {'users': []}},
                       {'allowlist': {'users': ['bad user']}}):
            with self.subTest(option=option):
                records = run_policy('prepare', **option)
                self.assertTrue(values(records, 'error'))
                self.assertFalse(values(records, 'status'))

    def test_manual_actor_cannot_spoof_requester(self):
        records = run_policy('prepare', actor='outside-user', inputs={'requested_by': 'ci-user'})
        self.assertTrue(values(records, 'error'))
        self.assertFalse(values(records, 'status'))

    def test_bot_dispatch_requires_listed_original_requester(self):
        for requested, permitted in [('ci-user', True), ('outside-user', False), ('', False)]:
            with self.subTest(requested=requested):
                records = run_policy('prepare', actor='github-actions[bot]', inputs={'requested_by': requested})
                self.assertEqual(not values(records, 'error'), permitted)

    def test_rerun_requires_actual_triggering_actor_in_allowlist(self):
        for user, allowed in [('ci-user', True), ('outside-user', False), ('', False)]:
            with self.subTest(user=user):
                records = run_policy('prepare', attempt=2, triggeringActor=user)
                self.assertEqual(not values(records, 'error'), allowed)
                if not allowed:
                    self.assertFalse(values(records, 'pull'))
                    self.assertFalse(values(records, 'status'))

    def test_failed_matrix_rerun_guard_checks_actual_user_before_npu_execution(self):
        workflow = yaml.safe_load((REPO_ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
        steps = workflow['jobs']['ascend-npu']['steps']
        guard_index, guard = next((index, step) for index, step in enumerate(steps)
                                  if step.get('uses') == 'actions/github-script@v8'
                                  and 'NPU_CI_TRIGGERING_ACTOR' in step.get('env', {}))
        checkout = steps[guard_index - 1]
        self.assertEqual(guard['if'], 'github.run_attempt > 1')
        self.assertEqual(checkout['if'], 'github.run_attempt > 1')
        self.assertTrue(checkout['uses'].startswith('actions/checkout@'))
        self.assertEqual(checkout['with']['ref'], '${{ github.workflow_sha }}')
        self.assertEqual(checkout['with']['sparse-checkout'], '.github/scripts')
        self.assertFalse(checkout['with']['persist-credentials'])
        self.assertEqual(guard['env']['NPU_CI_TRIGGERING_ACTOR'], '${{ github.triggering_actor }}')
        self.assertNotIn('continue-on-error', guard)
        self.assertNotIn('continue-on-error', checkout)
        self.assertLess(guard_index, next(index for index, step in enumerate(steps)
                                         if step.get('id') == 'checkout_pr_1'))
        self.assertLess(guard_index, next(index for index, step in enumerate(steps)
                                         if step.get('id') == 'npu_ci'))

        # context.actor is still the authorized original requester, as happens
        # when GitHub reuses successful prepare outputs on Re-run failed jobs.
        for triggering_actor, allowed in [('ci-user', True), ('outside-writer', False), ('', False)]:
            with self.subTest(triggering_actor=triggering_actor):
                records = run_policy('matrix-guard', actor='ci-user', attempt=2,
                                     triggeringActor=triggering_actor,
                                     guardScript=guard['with']['script'],
                                     policyCheckoutPath=checkout['with']['path'])
                self.assertEqual(not values(records, 'error'), allowed)
                self.assertEqual(values(records, 'result'), [None] if allowed else [])
                self.assertEqual(values(records, 'content')[0]['ref'], 'main')
                self.assertEqual(values(records, 'content')[0]['path'], '.github/npu-ci-users.json')
                self.assertEqual(len(values(records, 'policy-require')), 1)

    def test_markdown_check_accepts_modified_deleted_and_markdown_rename(self):
        files = [{'filename': 'README.md', 'status': 'modified'},
                 {'filename': 'docs/old.md', 'status': 'removed'},
                 {'filename': 'docs/new.md', 'status': 'renamed', 'previous_filename': 'old.md'}]
        records = run_policy('markdown', files=files)
        self.assertEqual(values(records, 'result'), [True])
        self.assertEqual(values(records, 'files')[0]['per_page'], 100)
        self.assertTrue(values(records, 'paginate'))
        self.assertTrue(values(records, 'pull'))

    def test_empty_nonmarkdown_or_renamed_source_never_skip_npu(self):
        for files in ([], [{'filename': 'README.md', 'status': 'renamed', 'previous_filename': 'kernel.cpp'}],
                      [{'filename': 'README.md', 'status': 'renamed'}],
                      [{'filename': 'README.md', 'status': 'renamed', 'previous_filename': ''}],
                      [{'filename': 'README.md.bak', 'status': 'modified'}]):
            with self.subTest(files=files):
                self.assertEqual(values(run_policy('markdown', files=files), 'result'), [False])

    def test_truncated_duplicate_or_failed_file_list_does_not_pass(self):
        for options in ({'changed_files': 3010}, {'filesError': True},
                        {'files': [{'filename': 'a.md'}, {'filename': 'a.md'}]}):
            with self.subTest(options=options):
                records = run_policy('default', **options)
                self.assertTrue(values(records, 'error'))
                self.assertFalse(values(records, 'status'))

    def test_head_race_does_not_publish_markdown_success(self):
        for mode in ('default', 'prepare'):
            with self.subTest(mode=mode):
                records = run_policy(mode, files=[{'filename': 'README.md'}], headRace=True)
                self.assertTrue(values(records, 'error'))
                self.assertFalse(values(records, 'status'))
                self.assertFalse(values(records, 'comment'))

    def test_markdown_default_and_manual_requests_pass_without_npu(self):
        for mode in ('default', 'prepare'):
            with self.subTest(mode=mode):
                records = run_policy(mode, files=[{'filename': 'README.md'}])
                self.assertFalse(values(records, 'error'))
                self.assertEqual(len(values(records, 'status')), 6)
                self.assertEqual({item['state'] for item in values(records, 'status')}, {'success'})
                self.assertEqual(len(values(records, 'comment')), 1)
                self.assertFalse(values(records, 'author-permission'))
                if mode == 'prepare':
                    self.assertEqual(outputs(records)['should_run'], 'false')

    def test_markdown_scoped_request_only_writes_scoped_context(self):
        records = run_policy('prepare', files=[{'filename': 'README.md'}], inputs={'ops': 'chunk_fwd_o'})
        statuses = values(records, 'status')
        self.assertEqual(len(statuses), 1)
        self.assertTrue(statuses[0]['context'].endswith('/ 定向诊断'))
        self.assertEqual(statuses[0]['state'], 'success')
        self.assertEqual(outputs(records)['should_run'], 'false')

    def test_prepare_trusts_only_admin_pr_author(self):
        for permission, expected in [('admin', 'true'), ('write', 'false'), ('read', 'false')]:
            with self.subTest(permission=permission):
                records = run_policy('prepare', authorPermission=permission)
                self.assertFalse(values(records, 'error'))
                self.assertEqual(outputs(records)['should_run'], 'true')
                self.assertEqual(outputs(records)['use_pr_ci'], expected)
                self.assertEqual(values(records, 'author-permission')[0]['username'], 'pr-author')
                self.assertEqual(len(values(records, 'comment')), 1)
                self.assertNotIn('comment_id', outputs(records))

    def test_admin_lookup_failure_uses_trusted_ci(self):
        records = run_policy('prepare', permissionError=True)
        self.assertEqual(outputs(records)['use_pr_ci'], 'false')
        self.assertEqual(outputs(records)['should_run'], 'true')
        self.assertTrue(values(records, 'warning'))

    def test_prepare_reuses_success_only_for_identical_source_and_ci_request(self):
        for ops, prefix in [('', 'all-'), ('chunk_fwd_o', 'scoped-')]:
            with self.subTest(ops=ops):
                initial = run_policy('prepare', inputs={'ops': ops})
                request_key = outputs(initial)['request_key']
                self.assertTrue(request_key.startswith(prefix))
                succeeded = [dict(item, state='success',
                                  target_url='https://github.com/example/repo/actions/runs/900')
                             for item in values(initial, 'status')]

                same_request = run_policy('prepare', inputs={'ops': ops}, statuses=succeeded)
                self.assertFalse(values(same_request, 'error'))
                self.assertEqual(outputs(same_request)['should_run'], 'false')
                self.assertEqual(outputs(same_request)['request_key'], request_key)
                self.assertFalse(values(same_request, 'status'))
                self.assertEqual(len(values(same_request, 'comment')), 1)

                for changed in ({'baseSha': 'd' * 40}, {'workflowSha': 'e' * 40}):
                    with self.subTest(changed=changed):
                        rerun = run_policy('prepare', inputs={'ops': ops}, statuses=succeeded, **changed)
                        self.assertFalse(values(rerun, 'error'))
                        self.assertEqual(outputs(rerun)['should_run'], 'true')
                        self.assertNotEqual(outputs(rerun)['request_key'], request_key)
                        self.assertEqual({item['state'] for item in values(rerun, 'status')}, {'pending'})

    def test_default_nonmarkdown_pr_is_pending_and_appends_comment(self):
        records = run_policy('default')
        self.assertEqual(len(values(records, 'status')), 6)
        self.assertEqual({item['state'] for item in values(records, 'status')}, {'pending'})
        self.assertIn('.github/npu-ci-users.json', values(records, 'comment')[0]['body'])

    def test_default_preserves_maintenance_branch_contexts(self):
        records = run_policy('default', baseRef='release')
        self.assertEqual([item['context'] for item in values(records, 'status')], ['NPU CI / legacy / 01'])

    def test_dispatch_uses_allowlist_and_preserves_target_branch(self):
        records = run_policy('dispatch', comment='/run-npu-ci full ci_ref=release ops=z,a')
        self.assertFalse(values(records, 'error'))
        self.assertFalse(values(records, 'author-permission'))
        dispatch = values(records, 'dispatch')[0]
        self.assertEqual(dispatch['ref'], 'release')
        self.assertEqual(dispatch['inputs']['ci_mode'], 'full')
        self.assertEqual(dispatch['inputs']['ops'], 'a,z')
        self.assertEqual(values(records, 'content')[0]['ref'], 'main')

    def test_dispatch_rerun_checks_both_original_and_triggering_users(self):
        for actor, rerun_by, allowed in [
            ('ci-user', 'CI-USER', True),
            ('ci-user', 'outside-user', False),
            ('ci-user', '', False),
            ('outside-user', 'ci-user', False),
        ]:
            with self.subTest(actor=actor, rerun_by=rerun_by):
                records = run_policy('dispatch', actor=actor, attempt=2, triggeringActor=rerun_by)
                self.assertEqual(bool(values(records, 'dispatch')), allowed)
                self.assertEqual(not values(records, 'error'), allowed)
                allowlist_reads = [item for item in values(records, 'content')
                                   if item['path'] == '.github/npu-ci-users.json']
                self.assertEqual(len(allowlist_reads), 1)
                if not allowed:
                    self.assertFalse(values(records, 'pull'))


    def test_dispatch_rejects_duplicate_or_empty_command_arguments(self):
        for comment in ('/run-npu-ci quick full', '/run-npu-ci ops=',
                        '/run-npu-ci ci_ref=', '/run-npu-ci ops=a,,b',
                        '/run-npu-ci ci_ref=main branch=release'):
            with self.subTest(comment=comment):
                records = run_policy('dispatch', comment=comment)
                self.assertTrue(values(records, 'error'))
                self.assertFalse(values(records, 'dispatch'))


if __name__ == '__main__':
    unittest.main()
