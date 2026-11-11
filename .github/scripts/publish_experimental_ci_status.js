'use strict';

// Runs on GitHub-hosted hardware from the trusted workflow revision.
const fs = require('fs');
const path = require('path');
const checks = require('./npu_ci_checks');

function validateReport(report, expected) {
  for (const key of ['head_sha', 'run_id', 'run_attempt', 'platform', 'soc', 'mode']) {
    if (report[key] !== expected[key]) throw new Error(`Experimental report identity mismatch: ${key}`);
  }
  if (report.schema !== 'experimental-ci-v1' || report.complete !== true || report.status !== 'success') {
    throw new Error('Experimental report is incomplete or failed');
  }
  const runtime = expected.platform === 'a2';
  if (report.build_only !== !runtime) throw new Error('Wrong experimental validation scope');
  const phases = report.phases || {};
  const names = ['dependencies', 'build-host', 'load-isolation', 'accuracy-queue', 'accuracy-inline'];
  for (const [index, name] of names.entries()) {
    const status = runtime || index < 2 ? 'success' : 'skipped';
    if (phases[name]?.status !== status) throw new Error(`Missing/failed experimental phase: ${name}`);
  }
  if (!runtime) return;
  const isolation = phases['load-isolation'].result;
  const isolationIds = ['L01', 'L07', ...Array.from({length: 20}, (_, i) => `L${String(i + 9).padStart(2, '0')}`)];
  if (isolation?.complete !== true || isolation.protected_unchanged !== true ||
      isolation.installation_rejections !== 3 || isolation.planned !== isolationIds.length ||
      isolation.counts?.PASS !== isolationIds.length || isolation.counts?.FAIL !== 0 || isolation.counts?.BLOCKED !== 0 ||
      !Array.isArray(isolation.results) || isolation.results.length !== isolationIds.length ||
      isolation.results.some((item, i) => item.id !== isolationIds[i] || item.status !== 'PASS')) {
    throw new Error('Experimental installation/load isolation is incomplete');
  }
  for (const launch of ['queue', 'inline']) {
    const result = phases[`accuracy-${launch}`].result;
    const count = expected.mode === 'quick' ? 3 : 209;
    if (result?.schema !== 'experimental-accuracy-v1' || result.backend !== 'cpu-reference' ||
        result.standard !== 'mixed_tolerance_bm-v1-atk26.9.8' || result.complete !== true || result.all_pass !== true ||
        result.total !== count || result.check_pass !== count || result.exec_fail !== 0 || result.check_fail !== 0 ||
        result.expected !== count || !Array.isArray(result.case_indices) || result.case_indices.length !== count ||
        result.case_indices.some((value, i) => value !== i) ||
        !Array.isArray(result.results) || result.results.length !== count || result.results.some((item, i) =>
          item.index !== i || item.status !== 'PASS' || ['o', 's', 'sa'].some((name) => item.outputs?.[name]?.pass !== true)) ||
        !Array.isArray(result.execution_evidence) || !result.execution_evidence.length ||
        result.execution_evidence.some((item) => item.synchronized_after_real_call !== true ||
          (launch === 'inline' ? item.launch_mode !== 'inline' : !['queue', 'inline'].includes(item.launch_mode)))) {
      throw new Error(`Experimental ${launch} CPU-reference accuracy is incomplete`);
    }
  }
}

async function publish({ github, context, core }) {
  const env = process.env;
  const sha = env.NPU_EXPECTED_HEAD_SHA;
  if (!/^[0-9a-f]{40}$/.test(sha || '')) throw new Error('Invalid experimental target commit');
  let failure = '';
  try {
    const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: Number(env.NPU_PR_NUMBER) });
    if (pr.state !== 'open' || pr.draft || pr.merged || pr.merged_at || pr.head?.sha !== sha ||
        !env.NPU_EXPECTED_BASE_REF || pr.base?.ref !== env.NPU_EXPECTED_BASE_REF) {
      throw new Error('PR state/head/base changed; experimental results are stale');
    }
    if (env.NPU_MATRIX_RESULT !== 'success') throw new Error('Experimental execution/upload jobs did not succeed');
    if (!['quick', 'full'].includes(env.NPU_CI_MODE)) throw new Error('Invalid experimental CI mode');
    for (const [platform, soc] of [['a2', 'ascend910b'], ['a5', 'ascend950']]) {
      const directory = path.join('experimental-reports',
        `experimental-npu-ci-result-${platform}-${env.NPU_EXPECTED_RUN_ID}-${env.NPU_EXPECTED_RUN_ATTEMPT}`);
      const report = JSON.parse(fs.readFileSync(path.join(directory, 'experimental/summary.json'), 'utf8'));
      validateReport(report, { platform, soc, mode: env.NPU_CI_MODE, head_sha: sha,
        run_id: env.NPU_EXPECTED_RUN_ID, run_attempt: env.NPU_EXPECTED_RUN_ATTEMPT });
    }
  } catch (error) {
    failure = error.message;
    core.warning(failure);
  }
  await github.rest.repos.createCommitStatus({ ...context.repo, sha,
    state: failure ? 'failure' : 'success', context: checks.experimental.context,
    description: failure ? '实验独立门禁未通过；请检查实验任务及结果报告'
      : `实验 A2 精度与隔离、A5 构建均通过 (${env.NPU_CI_MODE},experimental,A2+A5)`,
    target_url: `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}` });
  core.setOutput('status_published', 'true');
  if (failure) core.setFailed(failure);
}

module.exports = publish;
module.exports.validateReport = validateReport;
