#!/usr/bin/env python3
"""Exercise NPU CI authorization, change inspection, preparation, and comments."""

import json
import os
from pathlib import Path
import subprocess
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
# Include the repository's real build roots: fixtures must not hide layout changes.
_TRACKED_ASCENDC_FILES = subprocess.check_output(
    ['git', 'ls-files', '-z', '--', 'fla/ops/ascendc'], cwd=REPO_ROOT,
).decode('utf-8').split('\0')
DEFAULT_HEAD_TREE = [
    {'path': path, 'type': 'blob'} for path in _TRACKED_ASCENDC_FILES
    if path.endswith('/op_host/CMakeLists.txt')
] + [
    {'path': 'tests/atk/chunk_fwd_o/chunk_fwd_o.yaml', 'type': 'blob'},
    {'path': 'tests/atk/recurrent_kda/recurrent_kda.yaml', 'type': 'blob'},
    {'path': 'tests/atk/common/helper.py', 'type': 'blob'},
]
HARNESS = r"""
const config = JSON.parse(process.env.NPU_POLICY_TEST_CONFIG);
const records = [];
const record = (kind, payload) => records.push({kind, payload});
const head = 'a'.repeat(40);
const files = config.files || [{filename: 'operator.cpp', status: 'modified'}];
const pr = {
  number: config.prNumber || 527, state: config.state || 'open', draft: config.draft || false,
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
    inputs: {pr_number: String(pr.number), ci_mode: 'quick', ...(config.inputs || {})},
    issue: {number: pr.number}, comment: {body: config.comment || '/run-npu-ci quick'},
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
        return {data: {permission: config.permissions?.[params.username] || config.authorPermission || 'read'}};
      },
      listCommitStatusesForRef: async (params) => {record('statuses', params); return {data: config.statuses || []};},
      createCommitStatus: async (params) => {record('status', params); return {data: params};},
    },
    git: {getTree: async (params) => {
      record('tree', params);
      return {data: {truncated: config.treeTruncated || false,
        tree: config.tree}};
    }},
    pulls: {
      get: async (params) => {
        record('pull', params);
        getCalls += 1;
        const result = JSON.parse(JSON.stringify(pr));
        const afterInitial = getCalls > (config.mode === 'prepare' ? 1 : 0);
        if (config.headRace && afterInitial) result.head.sha = 'd'.repeat(40);
        if (config.baseRefRace && afterInitial) result.base.ref = 'v26.9.1';
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
    if (config.mode === 'markdown') result = (await policy.inspectChanges({github, context, pr})).documentation_only;
    else if (config.mode === 'plan') result = await policy.inspectChanges({github, context, pr, ...(config.planOptions || {})});
    else if (config.mode === 'normalize-atk') result = policy.normalizeAtkOps(config.atkOps);
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
    options.setdefault('tree', DEFAULT_HEAD_TREE)
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
                      [{'filename': 'README.md.bak', 'status': 'modified'}]):
            with self.subTest(files=files):
                self.assertEqual(values(run_policy('markdown', files=files), 'result'), [False])

    def test_truncated_duplicate_or_failed_file_list_does_not_pass(self):
        for options in ({'changed_files': 3010}, {'filesError': True},
                        {'files': [{'filename': 'a.md'}, {'filename': 'a.md'}]},
                        {'files': [{'filename': 'README.md', 'status': 'renamed'}]},
                        {'files': [{'filename': 'README.md', 'status': 'renamed', 'previous_filename': ''}]}):
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

    def test_same_sha_base_branch_race_does_not_publish_success_or_schedule(self):
        for mode in ('default', 'prepare'):
            with self.subTest(mode=mode):
                records = run_policy(mode, files=[{'filename': 'README.md'}], baseRefRace=True)
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
        self.assertEqual(statuses[0]['state'], 'pending')
        self.assertEqual(outputs(records)['should_run'], 'true')
        self.assertEqual(outputs(records)['build_ops'], 'chunk_fwd_o')

    def test_prepare_always_uses_pr_ci_without_author_permission_lookup(self):
        for permission in ('admin', 'write', 'read'):
            with self.subTest(permission=permission):
                records = run_policy('prepare', authorPermission=permission)
                self.assertFalse(values(records, 'error'))
                self.assertEqual(outputs(records)['should_run'], 'true')
                self.assertNotIn('use_pr_ci', outputs(records))
                self.assertNotIn('trusted_ci_sha', outputs(records))
                self.assertFalse(values(records, 'author-permission'))
                self.assertEqual(len(values(records, 'comment')), 1)
                self.assertNotIn('comment_id', outputs(records))

    def test_ordinary_pr_does_not_depend_on_repository_permission_api(self):
        records = run_policy('prepare', permissionError=True)
        self.assertEqual(outputs(records)['should_run'], 'true')
        self.assertFalse(values(records, 'author-permission'))

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

    def test_same_sha_different_base_branch_does_not_reuse_daily_request(self):
        initial = run_policy('prepare', files=[{'filename': 'examples/example.py'}])
        prior = [dict(item, state='success', target_url='https://github.com/example/repo/actions/runs/900')
                 for item in values(initial, 'status')]
        retargeted = run_policy('prepare', files=[{'filename': 'examples/example.py'}],
                                baseRef='v26.9.1', statuses=prior)
        self.assertFalse(values(retargeted, 'error'))
        self.assertEqual(outputs(retargeted)['base_ref'], 'v26.9.1')
        self.assertEqual(outputs(retargeted)['should_run'], 'true')
        self.assertNotEqual(outputs(retargeted)['request_key'], outputs(initial)['request_key'])

    def test_ci_approval_marker_is_pr_and_base_bound_before_reusing_statuses(self):
        args = dict(files=[{'filename': 'ci/run_checks.sh'}], authorPermission='admin', inputs={'ci_modified': 'true'})
        initial = run_policy('prepare', **args)
        key = outputs(initial)['ci_approval_key']
        self.assertRegex(key, r'^[a-f0-9]{12}$')
        self.assertTrue(all('ci-ok=' + key in s['description'] for s in values(initial, 'status')))
        self.assertTrue(all(len(s['description']) <= 100 for s in values(initial, 'status')))
        prior = [dict(item, state='success', target_url='https://github.com/example/repo/actions/runs/900')
                 for item in values(initial, 'status')]
        same = run_policy('prepare', statuses=prior, **args)
        self.assertEqual(outputs(same)['should_run'], 'false')
        for changes in ({'prNumber': 528}, {'baseRef': 'v26.9.1'}):
            with self.subTest(changes=changes):
                other = run_policy('prepare', statuses=prior, **changes, **args)
                self.assertEqual(outputs(other)['should_run'], 'true')
                self.assertNotEqual(outputs(other)['ci_approval_key'], key)

    def test_default_nonmarkdown_pr_is_pending_and_appends_comment(self):
        records = run_policy('default')
        self.assertEqual(len(values(records, 'status')), 6)
        self.assertEqual({item['state'] for item in values(records, 'status')}, {'pending'})
        self.assertIn('.github/npu-ci-users.json', values(records, 'comment')[0]['body'])

    def test_default_preserves_maintenance_branch_contexts(self):
        records = run_policy('default', baseRef='release')
        self.assertEqual([item['context'] for item in values(records, 'status')], ['NPU CI / legacy / 01'])

    def test_dispatch_uses_allowlist_and_preserves_target_branch(self):
        records = run_policy('dispatch', comment='/run-npu-ci quick ci_ref=release ops=z,a')
        self.assertFalse(values(records, 'error'))
        self.assertFalse(values(records, 'author-permission'))
        dispatch = values(records, 'dispatch')[0]
        self.assertEqual(dispatch['ref'], 'release')
        self.assertEqual(dispatch['inputs']['ci_mode'], 'quick')
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
                        '/run-npu-ci ci_ref=main branch=release',
                        '/run-npu-ci full ops=chunk_fwd_o',
                        '/run-npu-ci ops=chunk_fwd_o atk-ops=recurrent_kda',
                        '/run-npu-ci atk-ops=', '/run-npu-ci atk-ops=a,,b',
                        '/run-npu-ci atk-ops=null,a', '/run-npu-ci atk-ops=a,null',
                        '/run-npu-ci atk-ops=a atk-ops=b',
                        '/run-npu-ci device-id=', '/run-npu-ci device-id=-1',
                        '/run-npu-ci device-id=1.5', '/run-npu-ci device-id=9007199254740992',
                        '/run-npu-ci ci_modified=yes', '/run-npu-ci ci_modified=',
                        '/run-npu-ci ci_modified=true ci_modified=false'):
            with self.subTest(comment=comment):
                records = run_policy('dispatch', comment=comment)
                self.assertTrue(values(records, 'error'))
                self.assertFalse(values(records, 'dispatch'))

    def test_documentation_and_licenses_skip_for_nonmarkdown_assets(self):
        files = [{'filename': path} for path in ('docs/diagram.svg', 'LICENSES/MIT.txt', 'ci/README.MD')]
        for mode in ('default', 'prepare'):
            with self.subTest(mode=mode):
                records = run_policy(mode, files=files)
                self.assertFalse(values(records, 'error'))
                self.assertEqual({s['state'] for s in values(records, 'status')}, {'success'})
                self.assertFalse(values(records, 'author-permission'))
        # Moving code into docs is still a code change.
        plan = values(run_policy('plan', files=[{
            'filename': 'docs/kernel.txt', 'status': 'renamed', 'previous_filename': 'common/kernel.cpp',
        }]), 'result')[0]
        self.assertFalse(plan['documentation_only'])
        self.assertTrue(plan['full_build'])

    def test_scoped_diagnostic_only_compiles_explicit_ops_and_does_not_request_atk(self):
        records = run_policy('prepare', files=[
            {'filename': 'scripts/build_wheel.py'},
            {'filename': 'fla/ops/ascendc/kda/recurrent_kda/op_kernel/kernel.cpp'},
            {'filename': 'tests/atk/common/helper.py'},
        ], inputs={'ops': 'chunk_fwd_o'})
        self.assertFalse(values(records, 'error'))
        self.assertEqual(outputs(records)['full_build'], 'false')
        self.assertEqual(outputs(records)['build_kind'], 'incremental')
        self.assertEqual(outputs(records)['build_ops'], 'chunk_fwd_o')
        self.assertEqual(outputs(records)['atk_ops'], '')
        self.assertFalse(values(records, 'tree'))
        self.assertEqual(len(values(records, 'status')), 1)

    def test_change_plan_classifies_build_and_atk_without_unnecessary_compilation(self):
        cases = [
            (['common/runtime.h'], 'full', '', ''),
            (['cmake/dependencies.cmake'], 'full', '', ''),
            (['fla/ops/ascendc/common/tensor.h'], 'full', '', ''),
            (['fla/ops/ascendc/gdn/CMakeLists.txt'], 'full', '', ''),
            (['fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/kernel.cpp'],
             'incremental', 'chunk_kda_fwd', 'chunk_kda_fwd'),
            (['fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/op_host/tiling.cpp',
              'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/kernel.cpp'],
             'incremental', 'causal_conv1d,chunk_fwd_o', 'causal_conv1d,chunk_fwd_o'),
            (['tests/atk/chunk_fwd_o/executor.py'], 'base', '', 'chunk_fwd_o'),
            (['torch_custom/fla_npu/setup.py'], 'torch', '', ''),
            (['torch_custom/fla_npu/setup.py', 'tests/atk/recurrent_kda/cases.json'], 'torch', '', 'recurrent_kda'),
            (['examples/gdr.py', 'tests/test_runtime.py'], 'base', '', ''),
            (['ci/run_checks.sh', '.github/workflows/ci.yml'], 'base', '', ''),
            (['scripts/build_wheel.py'], 'full', '', ''),
        ]
        for paths, kind, compiled, tested in cases:
            with self.subTest(paths=paths):
                records = run_policy('plan', files=[{'filename': path} for path in paths])
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], kind)
                self.assertEqual(plan['full_build'], kind == 'full')
                self.assertEqual(plan['build_ops'], compiled)
                self.assertEqual(plan['atk_ops'], tested)
                self.assertEqual(plan['device_id'], '')

    def test_full_build_retains_modified_operator_and_manual_atk_targets(self):
        records = run_policy('prepare', files=[
            {'filename': 'common/header.h'},
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/kernel.cpp'},
            {'filename': 'tests/atk/recurrent_kda/cases.json'},
        ], inputs={'atk_ops': 'solve_tri,chunk_fwd_o', 'device_id': '02'})
        self.assertFalse(values(records, 'error'))
        self.assertEqual(outputs(records)['full_build'], 'true')
        self.assertEqual(outputs(records)['build_ops'], '')
        self.assertEqual(outputs(records)['atk_ops'], 'chunk_fwd_o,recurrent_kda,solve_tri')
        self.assertEqual(outputs(records)['device_id'], '2')

    def test_manual_full_forces_full_build_even_for_example_or_documentation(self):
        for path in ('examples/example.py', 'docs/guide.md'):
            with self.subTest(path=path):
                records = run_policy('prepare', files=[{'filename': path}], inputs={'ci_mode': 'full'})
                self.assertEqual(outputs(records)['should_run'], 'true')
                self.assertEqual(outputs(records)['full_build'], 'true')
                self.assertEqual(outputs(records)['build_kind'], 'full')

    def test_renames_compile_and_test_both_existing_operators(self):
        plan = values(run_policy('plan', files=[{
            'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/kernel.cpp', 'status': 'renamed',
            'previous_filename': 'fla/ops/ascendc/kda/recurrent_kda/op_kernel/kernel.cpp',
        }]), 'result')[0]
        self.assertEqual(plan['build_ops'], 'chunk_fwd_o,recurrent_kda')
        self.assertEqual(plan['atk_ops'], 'chunk_fwd_o,recurrent_kda')

    def test_nested_gdn_paths_use_real_leaf_operator_names(self):
        for relative, expected in (
            ('chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp', 'chunk_fwd_h'),
            ('recurrent_gdn/recurrent_gated_delta_rule/op_host/tiling.cpp', 'recurrent_gated_delta_rule'),
            ('gdn_preprocess/causal_conv1d/op_kernel/kernel.cpp', 'causal_conv1d'),
        ):
            with self.subTest(relative=relative):
                records = run_policy('plan', files=[{'filename': 'fla/ops/ascendc/gdn/' + relative}])
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], 'incremental')
                self.assertEqual(plan['build_ops'], expected)
                self.assertEqual(plan['atk_ops'], expected)
                self.assertEqual(len(values(records, 'tree')), 1)
                self.assertEqual(values(records, 'tree')[0]['tree_sha'], 'a' * 40)

    def test_nearest_buildable_op_host_handles_nested_internal_operators(self):
        for path, expected in (
            ('kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post/op_kernel/kernel.cpp', 'kda_gate_bwd_post'),
            ('kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post_varlen/op_host/tiling.cpp', 'kda_gate_bwd_post_varlen'),
            ('gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/'
             'fwd_h/op_host/fwd_h_tiling.h', 'chunk_gated_delta_rule_fwd'),
            ('gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/op_kernel/internal/arch22/stages/'
             'recompute_wu/op_host/recompute_wu_tiling.h', 'chunk_gated_delta_rule_fwd'),
        ):
            with self.subTest(path=path):
                tree = DEFAULT_HEAD_TREE + [{
                    'path': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/'
                            'op_kernel/internal/arch22/stages/fwd_h/op_host', 'type': 'tree',
                }, {
                    'path': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/'
                            'op_kernel/internal/arch22/stages/recompute_wu/op_host/tiling.h', 'type': 'blob',
                }]
                records = run_policy('plan', files=[{'filename': 'fla/ops/ascendc/' + path}], tree=tree)
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_ops'], expected)
                self.assertEqual(plan['atk_ops'], expected)

    def test_operator_discovery_supports_new_groups_and_arbitrary_depth_in_head(self):
        for root in ('fla/ops/ascendc/direct_op',
                     'fla/ops/ascendc/new_group/stage/subgroup/new_op'):
            with self.subTest(root=root):
                records = run_policy('plan', files=[{'filename': root + '/op_kernel/kernel.cpp'}],
                                     tree=[{'path': root + '/op_host/CMakeLists.txt'}])
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], 'incremental')
                self.assertEqual(plan['build_ops'], root.rsplit('/', 1)[-1])

    def test_shared_unknown_and_removed_operator_paths_force_full(self):
        for path in ('fla/ops/ascendc/gdn/chunk_gdn_fwd/CMakeLists.txt',
                     'fla/ops/ascendc/gdn/recurrent_gdn/shared.h',
                     'fla/ops/ascendc/gdn/gdn_preprocess/common/header.h',
                     'fla/ops/ascendc/unknown/not_an_operator/kernel.cpp',
                     'fla/ops/ascendc/gdn/deleted_op/op_host/CMakeLists.txt'):
            with self.subTest(path=path):
                records = run_policy('plan', files=[{'filename': path, 'status': 'removed'}])
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], 'full')
                self.assertEqual(plan['build_ops'], '')
                self.assertEqual(plan['atk_ops'], '')

    def test_removed_rename_source_does_not_generate_missing_operator(self):
        records = run_policy('plan', files=[{
            'filename': 'fla/ops/ascendc/new_group/new_op/op_kernel/kernel.cpp',
            'status': 'renamed',
            'previous_filename': 'fla/ops/ascendc/old_group/old_op/op_kernel/kernel.cpp',
        }], tree=[{'path': 'fla/ops/ascendc/new_group/new_op/op_host/CMakeLists.txt'}])
        self.assertFalse(values(records, 'error'))
        plan = values(records, 'result')[0]
        self.assertEqual(plan['build_kind'], 'full')
        self.assertEqual(plan['build_ops'], '')
        self.assertEqual(plan['atk_ops'], 'new_op')

    def test_operator_and_torch_custom_changes_rebuild_full_wheel(self):
        for extra in ([], [{'filename': 'common/header.h'}]):
            with self.subTest(extra=extra):
                records = run_policy('plan', files=[
                    {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
                    {'filename': 'torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py'},
                    *extra,
                ], planOptions={'atkOps': 'recurrent_kda'})
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertTrue(plan['full_build'])
                self.assertEqual(plan['build_kind'], 'full')
                self.assertEqual(plan['build_ops'], '')
                self.assertEqual(plan['atk_ops'], 'chunk_fwd_h,recurrent_kda')

    def test_operator_and_shared_atk_reuse_one_complete_head_tree(self):
        records = run_policy('plan', files=[
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
            {'filename': 'tests/atk/common/helper.py'},
        ])
        self.assertFalse(values(records, 'error'))
        self.assertEqual(len(values(records, 'tree')), 1)
        self.assertEqual(values(records, 'result')[0]['atk_ops'], 'chunk_fwd_h,chunk_fwd_o,recurrent_kda')

    def test_operator_discovery_rejects_truncated_tree(self):
        records = run_policy('plan', treeTruncated=True, files=[
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
        ])
        self.assertTrue(values(records, 'error'))
        self.assertFalse(values(records, 'result'))

    def test_unrelated_docs_and_ci_changes_do_not_read_tree(self):
        for paths in (['fla/ops/ascendc/gdn/chunk_gdn_fwd/README.md'],
                      ['ci/run_checks.sh', '.github/workflows/ci.yml'],
                      ['torch_custom/setup.py'], ['examples/example.py']):
            with self.subTest(paths=paths):
                records = run_policy('plan', files=[{'filename': path} for path in paths], treeTruncated=True)
                self.assertFalse(values(records, 'error'))
                self.assertFalse(values(records, 'tree'))

    def test_atk_ops_normalizer_reserves_only_standalone_lowercase_null(self):
        for value, expected in (('null', 'null'), (' null ', 'null'),
                                ('', ''), ('z,a,z', 'a,z'), ('NULL', 'NULL')):
            with self.subTest(value=value):
                records = run_policy('normalize-atk', atkOps=value)
                self.assertFalse(values(records, 'error'))
                self.assertEqual(values(records, 'result'), [expected])
        for value in ('null,a', 'a,null', 'null,null', 'a,,b', None):
            with self.subTest(value=value):
                records = run_policy('normalize-atk', atkOps=value)
                self.assertTrue(values(records, 'error'))

    def test_atk_null_overrides_auto_targets_without_changing_build_plan(self):
        paths = [
            'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp',
            'tests/atk/recurrent_kda/cases.json',
            'tests/atk/common/helper.py',
        ]
        for ci_mode, extras, expected_kind, expected_ops in (
            ('quick', [], 'incremental', 'chunk_fwd_h'),
            ('full', [], 'full', ''),
            ('quick', ['torch_custom/setup.py'], 'full', ''),
        ):
            with self.subTest(ci_mode=ci_mode, extras=extras):
                records = run_policy('plan', files=[{'filename': path} for path in paths + extras],
                                     planOptions={'atkOps': ' null ', 'ciMode': ci_mode})
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], expected_kind)
                self.assertEqual(plan['build_ops'], expected_ops)
                self.assertEqual(plan['atk_ops'], '')
                self.assertFalse(plan['shared_atk_modified'])
                self.assertEqual(len(values(records, 'tree')), 1)

    def test_atk_null_skips_shared_expansion_even_when_no_atk_tree_exists(self):
        records = run_policy('plan', files=[{'filename': 'tests/atk/common/helper.py'}],
                             planOptions={'atkOps': 'null'}, tree=[], treeTruncated=True)
        self.assertFalse(values(records, 'error'))
        plan = values(records, 'result')[0]
        self.assertEqual(plan['build_kind'], 'base')
        self.assertEqual(plan['atk_ops'], '')
        self.assertFalse(plan['shared_atk_modified'])
        self.assertFalse(values(records, 'tree'))

    def test_manual_atk_is_deduplicated_with_modified_operator_and_test_targets(self):
        records = run_policy('plan', files=[
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
            {'filename': 'tests/atk/recurrent_kda/cases.json'},
        ], planOptions={'atkOps': 'recurrent_kda,chunk_fwd_h,recurrent_kda'})
        self.assertFalse(values(records, 'error'))
        self.assertEqual(values(records, 'result')[0]['atk_ops'], 'chunk_fwd_h,recurrent_kda')

    def test_dispatch_forwards_explicit_atk_disable_sentinel(self):
        records = run_policy('dispatch', comment='/run-npu-ci quick atk-ops=null', files=[
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
        ])
        self.assertFalse(values(records, 'error'))
        self.assertEqual(values(records, 'dispatch')[0]['inputs']['atk_ops'], 'null')

    def test_prepare_atk_null_disables_operator_tests(self):
        records = run_policy('prepare', files=[
            {'filename': 'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/op_kernel/kernel.cpp'},
            {'filename': 'tests/atk/common/helper.py'},
        ], inputs={'atk_ops': 'null'})
        self.assertFalse(values(records, 'error'))
        self.assertEqual(outputs(records)['should_run'], 'true')
        self.assertEqual(outputs(records)['build_ops'], 'chunk_fwd_h')
        self.assertEqual(outputs(records)['atk_ops'], '')

    def test_docs_only_with_atk_null_keeps_documentation_fast_path(self):
        records = run_policy('prepare', files=[{'filename': 'docs/guide.md'}], inputs={'atk_ops': 'null'})
        self.assertFalse(values(records, 'error'))
        self.assertEqual(outputs(records)['should_run'], 'false')
        self.assertEqual({s['state'] for s in values(records, 'status')}, {'success'})
        self.assertFalse(values(records, 'tree'))

    def test_shared_atk_changes_expand_all_current_atk_directories(self):
        for path in ('tests/atk/run_test_cpu.sh', 'tests/atk/common/result.py'):
            with self.subTest(path=path):
                records = run_policy('plan', files=[{'filename': path}])
                self.assertFalse(values(records, 'error'))
                plan = values(records, 'result')[0]
                self.assertEqual(plan['build_kind'], 'base')
                self.assertEqual(plan['atk_ops'], 'chunk_fwd_o,recurrent_kda')
                self.assertEqual(values(records, 'tree')[0]['tree_sha'], 'a' * 40)
        records = run_policy('plan', files=[{'filename': 'tests/atk/common/result.py'}], treeTruncated=True)
        self.assertTrue(values(records, 'error'))

    def test_ci_modification_requires_admin_acknowledgment_at_dispatch_and_prepare(self):
        files = [{'filename': 'ci/run_checks.sh'}]
        for mode in ('dispatch', 'prepare'):
            for confirmed, permission, allowed in [('true', 'admin', True), ('false', 'admin', False),
                                                  ('true', 'write', False), ('true', 'read', False)]:
                with self.subTest(mode=mode, confirmed=confirmed, permission=permission):
                    records = run_policy(mode, files=files, authorPermission=permission,
                                         inputs={'ci_modified': confirmed},
                                         comment=f'/run-npu-ci ci_modified={confirmed}')
                    self.assertEqual(not values(records, 'error'), allowed)
                    if allowed:
                        self.assertEqual(values(records, 'author-permission')[0]['username'], 'ci-user')
                        if mode == 'prepare':
                            self.assertEqual(outputs(records)['ci_modified'], 'true')
                            self.assertEqual(outputs(records)['build_kind'], 'base')
                    else:
                        self.assertEqual(len(values(records, 'comment')), 1)
                        self.assertIn('ci_modified=true', values(records, 'comment')[0]['body'])
                        self.assertEqual({s['state'] for s in values(records, 'status')}, {'failure'})
                        self.assertFalse(values(records, 'dispatch'))

    def test_ci_modification_rerun_requires_actual_rerunner_to_be_admin(self):
        for mode in ('prepare', 'dispatch'):
            for actual_permission, allowed in [('admin', True), ('write', False)]:
                with self.subTest(mode=mode, actual_permission=actual_permission):
                    records = run_policy(mode, files=[{'filename': '.github/workflows/ci.yml'}],
                                         allowlist={'users': ['ci-user', 'rerunner']}, attempt=2,
                                         triggeringActor='rerunner',
                                         permissions={'ci-user': 'admin', 'rerunner': actual_permission},
                                         inputs={'ci_modified': 'true'}, comment='/run-npu-ci ci_modified=true')
                    self.assertEqual(not values(records, 'error'), allowed)
                    self.assertEqual([r['username'] for r in values(records, 'author-permission')],
                                     ['ci-user', 'rerunner'])

    def test_ci_permission_read_error_fails_with_actionable_comment(self):
        records = run_policy('prepare', files=[{'filename': 'ci/run_checks.sh'}],
                             permissionError=True, inputs={'ci_modified': 'true'})
        self.assertTrue(values(records, 'error'))
        self.assertIn('恢复权限读取', values(records, 'comment')[0]['body'])
        self.assertEqual(outputs(records)['should_run'], 'false')

    def test_default_ci_modification_fails_until_admin_confirmation(self):
        records = run_policy('default', files=[{'filename': '.github/workflows/ci.yml'}])
        self.assertFalse(values(records, 'error'))
        self.assertEqual({s['state'] for s in values(records, 'status')}, {'failure'})
        self.assertIn('ci_modified=true', values(records, 'comment')[0]['body'])

    def test_command_forwards_normalized_atk_device_and_ci_acknowledgment(self):
        records = run_policy('dispatch', comment='/run-npu-ci full atk-ops=z,a,z device-id=03 ci_modified=true')
        self.assertFalse(values(records, 'error'))
        inputs = values(records, 'dispatch')[0]['inputs']
        self.assertEqual(inputs['ci_mode'], 'full')
        self.assertEqual(inputs['atk_ops'], 'a,z')
        self.assertEqual(inputs['device_id'], '3')
        self.assertEqual(inputs['ci_modified'], 'true')

    def test_changed_atk_device_or_compilation_plan_cannot_reuse_success(self):
        initial = run_policy('prepare', files=[{'filename': 'examples/example.py'}])
        succeeded = [dict(item, state='success', target_url='https://github.com/example/repo/actions/runs/900')
                     for item in values(initial, 'status')]
        for options in ({'inputs': {'device_id': '1'}}, {'inputs': {'atk_ops': 'chunk_fwd_o'}},
                        {'files': [{'filename': 'torch_custom/setup.py'}]}):
            with self.subTest(options=options):
                options.setdefault('files', [{'filename': 'examples/example.py'}])
                records = run_policy('prepare', statuses=succeeded, **options)
                self.assertEqual(outputs(records)['should_run'], 'true')
                self.assertNotEqual(outputs(records)['request_key'], outputs(initial)['request_key'])


if __name__ == '__main__':
    unittest.main()
