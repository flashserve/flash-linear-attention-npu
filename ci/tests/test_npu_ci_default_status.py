#!/usr/bin/env python3
"""Do not reuse Markdown shortcuts or execution results after a base change."""

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


def statuses(kind, *, contexts=CONTEXTS, newer=True):
    markdown = kind == 'markdown'
    return [
        {
            'context': name,
            'state': 'success',
            'description': '仅修改 Markdown，跳过 NPU CI' if markdown else 'A2+A5 已通过',
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
