#!/usr/bin/env python3
"""Do not reuse Markdown shortcuts or execution results after a base change."""

import hashlib
import json
from pathlib import Path
import unittest

import yaml

from ci.tests.test_npu_ci_policy import run_policy, values


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = [
    'NPU CI / A2+A5 / ' + label
    for label in (
        '01 环境、wheel 与运行时契约', '02 全量 OPP 构建',
        '03 torch_custom wheel 与 OPP 布局', '04 OPP 安装与 PyTorch 适配',
        '05 GDR Example/ST', '06 报告与 commit 校验',
    )
]


def approval_marker(*, pr_number=527, base_ref='main', base_sha='b' * 40, head_sha='a' * 40):
    payload = json.dumps([pr_number, base_ref, base_sha, head_sha], separators=(',', ':'))
    return 'ci-ok=' + hashlib.sha256(payload.encode()).hexdigest()[:12]


def statuses(kind, *, contexts=CONTEXTS, newer=True, approval='', state='success'):
    markdown = kind == 'markdown'
    return [
        {
            'context': name,
            'state': state,
            'description': '仅修改 Markdown，跳过 NPU CI' if markdown else f'A2+A5 已通过 {approval}'.strip(),
            'target_url': ('https://github.com/example/repo/pull/526' if markdown
                           else 'https://github.com/example/repo/actions/runs/900'),
            'created_at': '2026-11-03T01:00:00Z' if newer else '2026-11-02T01:00:00Z',
            'id': index + (20 if newer else 1),
        }
        for index, name in enumerate(contexts)
    ]


class DefaultNpuCiStatusTest(unittest.TestCase):
    def assert_pending(self, records, contexts=CONTEXTS):
        self.assertFalse(values(records, 'error'))
        written = values(records, 'status')
        self.assertEqual([item['context'] for item in written], contexts)
        self.assertEqual({item['state'] for item in written}, {'pending'})
        self.assertEqual(len(values(records, 'comment')), 1)

    def test_nonmarkdown_diff_invalidates_another_prs_markdown_success(self):
        records = run_policy('default', statuses=statuses('markdown'))
        self.assert_pending(records)

    def test_historical_execution_does_not_preserve_latest_markdown_success(self):
        records = run_policy(
            'default', statuses=statuses('execution', newer=False) + statuses('markdown'),
        )
        self.assert_pending(records)

    def test_non_documentation_diff_invalidates_extended_documentation_shortcut(self):
        prior = statuses('markdown')
        for item in prior:
            item['description'] = '仅修改文档或 LICENSES，跳过 NPU CI'
        self.assert_pending(run_policy('default', statuses=prior))

    def test_mixed_latest_results_reset_markdown_shortcuts_only(self):
        records = run_policy('default', statuses=(
            statuses('markdown', contexts=CONTEXTS[:2]) +
            statuses('execution', contexts=CONTEXTS[2:])
        ))
        self.assert_pending(records, CONTEXTS[:2])

    def test_new_execution_results_are_preserved(self):
        records = run_policy(
            'default', statuses=statuses('markdown', newer=False) + statuses('execution'),
        )
        self.assertFalse(values(records, 'error'))
        self.assertFalse(values(records, 'status'))
        self.assertFalse(values(records, 'comment'))
        self.assertTrue(values(records, 'notice'))

    def test_ci_changes_preserve_only_same_pr_base_approved_run(self):
        for state in ('success', 'pending', 'failure'):
            for action in ('opened', 'synchronize', 'edited'):
                with self.subTest(state=state, action=action):
                    records = run_policy('default', action=action,
                                         files=[{'filename': 'ci/run_checks.sh'}],
                                         statuses=statuses('execution', approval=approval_marker(), state=state))
                    self.assertFalse(values(records, 'error'))
                    self.assertFalse(values(records, 'status'))
                    self.assertFalse(values(records, 'comment'))
                    self.assertTrue(values(records, 'notice'))

    def test_ci_changes_invalidate_other_pr_base_or_unapproved_shared_commit_results(self):
        markers = ['', approval_marker(pr_number=526), approval_marker(base_ref='v26.9.1'),
                   approval_marker(base_sha='c' * 40), approval_marker(head_sha='d' * 40)]
        for marker in markers:
            with self.subTest(marker=marker):
                records = run_policy('default', files=[{'filename': 'ci/run_checks.sh'}],
                                     statuses=statuses('execution', approval=marker))
                self.assertFalse(values(records, 'error'))
                self.assertEqual({s['state'] for s in values(records, 'status')}, {'failure'})
                self.assertEqual(len(values(records, 'status')), 6)
                self.assertIn('ci_modified=true', values(records, 'comment')[0]['body'])

    def test_ci_approval_marker_cannot_be_mixed_between_runs_or_missing_stages(self):
        for change in ('missing-marker', 'different-run', 'missing-stage', 'foreign-repo'):
            with self.subTest(change=change):
                prior = statuses('execution', approval=approval_marker())
                if change == 'missing-marker':
                    prior[0]['description'] = 'A2+A5 已通过'
                elif change == 'different-run':
                    prior[0]['target_url'] = 'https://github.com/example/repo/actions/runs/899'
                elif change == 'missing-stage':
                    prior.pop()
                else:
                    for status in prior:
                        status['target_url'] = 'https://github.com/other/repo/actions/runs/900'
                records = run_policy('default', files=[{'filename': '.github/workflows/ci.yml'}], statuses=prior)
                self.assertEqual({s['state'] for s in values(records, 'status')}, {'failure'})
                self.assertEqual(len(values(records, 'status')), 6)

    def test_ci_changes_retargeted_to_same_sha_branch_require_its_own_approval(self):
        records = run_policy('default', action='edited', baseRef='v26.9.1',
                             branchWorkflow="const contexts = " + repr(CONTEXTS) + ';',
                             files=[{'filename': 'ci/run_checks.sh'}],
                             statuses=statuses('execution', approval=approval_marker()))
        self.assertEqual({s['state'] for s in values(records, 'status')}, {'failure'})
        self.assertEqual(len(values(records, 'status')), 6)

    def test_base_change_invalidates_all_old_execution_results(self):
        records = run_policy('default', action='edited', statuses=statuses('execution'))
        self.assert_pending(records)
        self.assertTrue(all('目标分支已变化' in item['description'] for item in values(records, 'status')))
        self.assertIn('旧 base', values(records, 'comment')[0]['body'])

    def test_base_change_uses_the_new_branch_contexts(self):
        legacy = ['NPU CI / legacy / 01']
        records = run_policy(
            'default', action='edited', baseRef='release',
            statuses=statuses('execution', contexts=legacy),
        )
        self.assert_pending(records, legacy)

    def test_markdown_diff_still_passes_after_base_change(self):
        records = run_policy(
            'default', action='edited', files=[{'filename': 'README.md', 'status': 'modified'}],
            statuses=statuses('execution'),
        )
        self.assertFalse(values(records, 'error'))
        self.assertEqual(len(values(records, 'status')), 6)
        self.assertEqual({item['state'] for item in values(records, 'status')}, {'success'})

    def test_workflow_limits_edited_events_to_base_changes(self):
        workflow = yaml.safe_load((REPO_ROOT / '.github/workflows/npu-ci-default-status.yml').read_text(encoding='utf-8'))
        triggers = workflow.get('on', workflow.get(True))
        self.assertIn('edited', triggers['pull_request_target']['types'])
        self.assertEqual(
            workflow['jobs']['mark-not-run']['if'],
            "github.event.action != 'edited' || github.event.changes.base.ref",
        )


if __name__ == '__main__':
    unittest.main()
