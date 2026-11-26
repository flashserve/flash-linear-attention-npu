"""Exercise transfer retry routing and the workflow's report-presence check."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
UPLOAD_STEPS = WORKFLOW['jobs']['ascend-npu']['steps']
UPLOAD_STEPS = UPLOAD_STEPS[next(i for i, step in enumerate(UPLOAD_STEPS)
                               if step.get('id') == 'upload_results_1'):]
FINALIZE_STEPS = WORKFLOW['jobs']['finalize']['steps']
DOWNLOAD_STEPS = FINALIZE_STEPS[
    next(i for i, step in enumerate(FINALIZE_STEPS) if step.get('id') == 'download_results'):
    next(i for i, step in enumerate(FINALIZE_STEPS) if step.get('id') == 'publish_statuses')]
CHECK_SCRIPT = next(step['run'] for step in DOWNLOAD_STEPS
                    if step.get('id') == 'check_results_1').split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]


class ArtifactRetryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = dict(os.environ, GITHUB_RUN_ID='42', GITHUB_RUN_ATTEMPT='1', NPU_OPS='')

    def reports(self, platform, attempt=1, run_id=42, accuracy=True):
        directory = self.root / 'accuracy-reports' / f'npu-ci-result-{platform}-{run_id}-{attempt}'
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('npu-ci-result', 'npu-ci-stages', 'npu-ci-diagnostics'):
            (directory / f'{name}-{platform}.json').write_text('{}', encoding='utf-8')
        (directory / f'npu-ci-summary-{platform}.md').write_text('summary', encoding='utf-8')
        if accuracy:
            (directory / f'gdr_accuracy_report-{platform}.json').write_text('{}', encoding='utf-8')
        return directory

    def check_reports(self, script=CHECK_SCRIPT):
        return subprocess.run([sys.executable, '-c', script], cwd=self.root, env=self.env,
                              capture_output=True, text=True, encoding='utf-8', timeout=10)

    def run_transfers(self, steps, transfer, already_failed=False):
        # Evaluate the actual workflow conditions with step outcomes, while
        # preserving the distinction between outcome and continue-on-error.
        outcomes, calls, waits = {}, [], []
        failed = already_failed
        for step in steps:
            condition = step.get('if', 'success()')
            condition = re.sub(r'steps\.([a-zA-Z0-9_]+)\.outcome',
                               lambda match: repr(outcomes.get(match[1], 'skipped')), condition)
            condition = condition.replace('&&', ' and ').replace('||', ' or ')
            enabled = eval(condition, {'__builtins__': {}},
                           {'always': lambda: True, 'success': lambda: not failed})
            outcome = 'skipped'
            if enabled:
                if 'uses' in step:
                    calls.append(step['id'])
                    outcome = transfer(step, len(calls))
                elif step['run'].startswith('sleep '):
                    waits.append(int(step['run'].split()[1]))
                    outcome = 'success'
                else:
                    script = step['run'].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
                    outcome = 'success' if self.check_reports(script).returncode == 0 else 'failure'
                if outcome == 'failure' and not step.get('continue-on-error', False):
                    failed = True
            if 'id' in step:
                outcomes[step['id']] = outcome
        return calls, waits, failed

    def test_upload_stops_on_success_and_preserves_prior_build_failure(self):
        for success_at in (1, 2, 3, 4, None):
            for already_failed in (False, True):
                with self.subTest(success_at=success_at, already_failed=already_failed):
                    calls, waits, failed = self.run_transfers(
                        UPLOAD_STEPS,
                        lambda step, attempt: 'success' if attempt == success_at else 'failure',
                        already_failed=already_failed)
                    count = success_at or 4
                    self.assertEqual(len(calls), count)
                    self.assertEqual(waits, [10, 30, 60][:count - 1])
                    self.assertEqual(failed, already_failed or success_at is None)

    def test_retries_reuse_the_same_artifact_and_only_transfer_reports(self):
        uploads = [step for step in UPLOAD_STEPS if 'uses' in step]
        self.assertEqual(len(uploads), 4)
        for step in uploads:
            self.assertEqual(step['uses'], 'actions/upload-artifact@v6')
            self.assertEqual(step['with'], uploads[0]['with'])
            self.assertTrue(step['with']['overwrite'])
            self.assertEqual(len(step['with']['path'].splitlines()), 5)
        self.assertFalse(any('run_ci_container' in step.get('run', '') for step in UPLOAD_STEPS))

    def test_download_retries_transport_failure_then_missing_platform(self):
        def download(step, attempt):
            if attempt == 1:
                return 'failure'
            self.reports('a2')
            if attempt == 3:
                self.reports('a5')
            return 'success'

        calls, waits, failed = self.run_transfers(DOWNLOAD_STEPS, download)
        self.assertEqual(len(calls), 3)
        self.assertEqual(waits, [10, 30])
        self.assertFalse(failed)

    def test_download_missing_platform_exhausts_retries(self):
        self.reports('a2')
        calls, waits, failed = self.run_transfers(DOWNLOAD_STEPS, lambda *_: 'success')
        self.assertEqual(len(calls), 4)
        self.assertEqual(waits, [10, 30, 60])
        self.assertTrue(failed)
        self.assertIn('a5: result artifact', self.check_reports().stderr)

    def test_download_transport_failure_exhausts_retries(self):
        calls, waits, failed = self.run_transfers(DOWNLOAD_STEPS, lambda *_: 'failure')
        self.assertEqual(len(calls), 4)
        self.assertEqual(waits, [10, 30, 60])
        self.assertTrue(failed)

    def test_partial_rerun_reuses_prior_platform_but_checks_latest_attempt(self):
        self.env['GITHUB_RUN_ATTEMPT'] = '2'
        self.reports('a2', attempt=1)
        self.reports('a5', attempt=2)
        calls, waits, failed = self.run_transfers(DOWNLOAD_STEPS, lambda *_: 'success')
        self.assertEqual(len(calls), 1)
        self.assertEqual(waits, [])
        self.assertFalse(failed)
        self.reports('a5', attempt=1)
        latest = self.reports('a5', attempt=2)
        (latest / 'npu-ci-stages-a5.json').unlink()
        result = self.check_reports()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('npu-ci-result-a5-42-2', result.stderr)

    def test_other_runs_and_future_attempts_do_not_satisfy_presence(self):
        self.reports('a2')
        self.reports('a5', run_id=41)
        self.reports('a5', attempt=2)
        self.assertNotEqual(self.check_reports().returncode, 0)

    def test_accuracy_is_required_only_for_full_scope(self):
        self.reports('a2', accuracy=False)
        self.reports('a5', accuracy=False)
        result = self.check_reports()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('gdr_accuracy_report-a5.json', result.stderr)
        self.env['NPU_OPS'] = 'chunk_fwd_o'
        self.assertEqual(self.check_reports().returncode, 0)


if __name__ == '__main__':
    unittest.main()
