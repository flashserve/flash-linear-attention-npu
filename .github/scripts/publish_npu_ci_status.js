'use strict';

module.exports = async function publishNpuCiStatus({ github, context, core }) {
  const fs = require('fs');
  const path = require('path');
  const mode = process.env.NPU_CI_MODE || '';
  const ops = process.env.NPU_OPS || '';
  const prNumber = process.env.NPU_PR_NUMBER || '';
  const commentId = process.env.NPU_COMMENT_ID || '';
  const sha = process.env.NPU_EXPECTED_HEAD_SHA;
  const expectedRunId = process.env.NPU_EXPECTED_RUN_ID;
  const expectedRunAttempt = process.env.NPU_EXPECTED_RUN_ATTEMPT;
  const matrixResult = process.env.NPU_MATRIX_RESULT || 'failure';
  const targetUrl = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  const platformDefs = [
    { key: 'a2', label: 'A2', soc: 'ascend910b' },
    { key: 'a5', label: 'A5', soc: 'ascend950' },
  ];
  const truncateDescription = (value) => value.length > 140 ? `${value.slice(0, 137)}...` : value;
  const formatValue = (value) => {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value.toExponential(3) : String(value);
    }
    if (typeof value === 'boolean') {
      return value ? 'true' : 'false';
    }
    return value === undefined || value === null || value === '' ? '-' : String(value);
  };
  const redactSecretAssignment = (match, quote, key, separator) => {
    const isSpacedLowercaseTokenVariable = !quote && key === key.toLowerCase() &&
      /(?:^|[-_])token$/.test(key) && /^\s+[:=]\s+$/.test(separator);
    return isSpacedLowercaseTokenVariable
      ? match
      : `${quote}${key}${quote}${separator}<redacted>`;
  };
  const redactHostAssignment = (match, quote, key, separator) => {
    const isSpacedLowercaseBusinessVariable = !quote &&
      ['host', 'worker', 'node'].includes(key) && /^\s+[:=]\s+$/.test(separator);
    return isSpacedLowercaseBusinessVariable
      ? match
      : `${quote}${key}${quote}${separator}<internal-host>`;
  };
  const sanitizePublicText = (value, maxLength = 800) => String(value ?? '')
    .replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, '')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '')
    .replace(/\b(?:10(?:\.\d{1,3}){3}|127(?:\.\d{1,3}){3}|169\.254(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b/g, '<internal-host>')
    .replace(/(?<![0-9a-f:])(?:::1|(?:f[cd][0-9a-f]{2}|fe[89ab][0-9a-f]):(?:[0-9a-f]{0,4}:){1,6}[0-9a-f]{0,4})(?![0-9a-f:])/gi, '<internal-host>')
    .replace(/(https?:\/\/)[^\s\/@:]+:[^\s\/@]+@/gi, '$1<redacted>@')
    .replace(/(?<![a-z0-9_-])(--(?:[a-z][a-z0-9]*[-_])*(?:authorization|token|password|passwd|secret|api[-_]?key|access[-_]?key|secret[-_]?key|private[-_]?key))(\s*=\s*|\s+)(?:<redacted>|"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[^\s,;\r\n}\]]+)/gi, '$1$2<redacted>')
    .replace(/(?<![a-z0-9_-])([\"']?)((?:[a-z][a-z0-9]*[-_])*(?:authorization|token|password|passwd|secret|api[-_]?key|access[-_]?key|secret[-_]?key|private[-_]?key))\1(\s*[:=]\s*)(?:<redacted>|\"(?:\\.|[^\"\\\r\n])*\"|'(?:\\.|[^'\\\r\n])*'|(?:bearer\s+|basic\s+)?[^,;\r\n}\]]+?(?=\s+[\"']?[a-z][a-z0-9_-]*[\"']?\s*[:=]|[,;\r\n}\]]|$))/gi, redactSecretAssignment)
    .replace(/\b(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|glpat-[A-Za-z0-9_-]{12,})\b/gi, '<redacted>')
    .replace(/\b[a-z_][a-z0-9_.-]{0,63}@[a-z0-9_.-]+\b/gi, '<user-host>')
    .replace(/\b(?:[a-z0-9-]+\.)+(?:local|internal|corp)\b/gi, '<internal-host>')
    .replace(/(?<![a-z0-9_-])(["']?)(host(?:name)?|runner|machine|worker|node)\1(\s*[:=]\s*)(?:<internal-host>|"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'|[a-z0-9][a-z0-9_.-]*)/gi, redactHostAssignment)
    .replace(/\b((?:on|via|from)\s+host\s+)[a-z0-9][a-z0-9_.-]*/gi, '$1<internal-host>')
    .replace(/\b(?=[a-z0-9-]*(?:runner|worker|host|node))(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b/gi, '<internal-host>')
    .replace(/[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s]*/g, '<path>')
    .replace(/(?<![\w.])\/(?:data|workspace|root|home|tmp|opt|usr|var|mnt|etc|srv|run|__w|github|runner)(?:\/[^\s\"'<>:]*)*(?=$|[\s\"'<>:,;.\)\]])/g, '<path>')
    .replace(/(?<![:\/\w.])\/(?:[^\s\/\"'<>:]+\/)+[^\s\"'<>:]*/g, '<path>')
    .slice(0, maxLength);
  const sanitizeInline = (value, maxLength = 800) => sanitizePublicText(value, maxLength)
    .replace(/[\r\n\t]+/g, ' ')
    .trim();
  const sanitizeMarkdownInline = (value, maxLength = 800) => sanitizeInline(value, maxLength)
    .replace(/@/g, '_at_')
    .replace(/[`<>\[\]]/g, "'");
  const isRecord = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
  const escapeCell = (value) => sanitizeMarkdownInline(formatValue(value), 300)
    .replace(/\|/g, '\\|');
  const diagnosticLines = (value) => Array.isArray(value)
    ? value.slice(0, 4).map((line) => sanitizeInline(line, 280)).filter(Boolean)
    : [];
  const renderComment = (lines) => {
    const maxChars = 48000;
    const notice = '_其余诊断因评论长度限制已省略，完整结构化结果见平台 artifact。_';
    const kept = [];
    let used = 0;
    let truncated = false;
    for (const line of lines) {
      const value = String(line);
      if (used + value.length + 1 > maxChars - notice.length - 2) {
        truncated = true;
        break;
      }
      kept.push(value);
      used += value.length + 1;
    }
    if (truncated) {
      kept.push('', notice);
    }
    return kept.join('\n');
  };
  const readJson = (filePath, description, errors) => {
    if (!fs.existsSync(filePath)) {
      errors.push(`${description}未生成`);
      return null;
    }
    try {
      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      if (parsed === null) {
        errors.push(`${description}根节点无效`);
        return null;
      }
      return parsed;
    } catch (error) {
      errors.push(`${description}无法解析`);
      core.warning(`Unable to parse ${filePath}: ${error.message}`);
      return null;
    }
  };
  const validateIdentity = (metadata, def, description, errors) => {
    if (!isRecord(metadata)) {
      errors.push(`${description}缺少身份信息`);
      return;
    }
    const expected = {
      platform: def.key,
      soc: def.soc,
      head_sha: sha,
      run_id: expectedRunId,
      run_attempt: expectedRunAttempt,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (String(metadata[key] || '') !== String(value)) {
        errors.push(`${description}${key}不属于本次触发`);
      }
    }
  };
  const validatePlatform = (def) => {
    const errors = [];
    const executionErrors = [];
    const accuracyErrors = [];
    const diagnosticErrors = [];
    const resultPath = path.join('accuracy-reports', `npu-ci-result-${def.key}.json`);
    const reportPath = path.join('accuracy-reports', `gdr_accuracy_report-${def.key}.json`);
    const diagnosticPath = path.join('accuracy-reports', `npu-ci-diagnostics-${def.key}.json`);
    const execution = readJson(resultPath, '执行结果', executionErrors);
    if (execution !== null) {
      if (!isRecord(execution)) {
        executionErrors.push('执行结果根节点无效');
      } else if (execution.schema !== 'npu-ci-platform-result-v1') {
        executionErrors.push('执行结果 schema 不受支持');
      }
      if (isRecord(execution)) {
        validateIdentity(execution.metadata, def, '执行结果', executionErrors);
        if (!['success', 'failure'].includes(execution.status)) {
          executionErrors.push('执行结果 status 无效');
        } else if (execution.status !== 'success') {
          executionErrors.push(`执行状态为 ${execution.status || 'unknown'}`);
        }
      }
    }
    const report = readJson(reportPath, '精度报告', accuracyErrors);
    let summary = null;
    let reportAccuracyState = null;
    if (report !== null) {
      if (!isRecord(report)) {
        accuracyErrors.push('精度报告根节点无效');
      } else if (report.schema !== 'gdr-accuracy-report-v1') {
        accuracyErrors.push('精度报告 schema 不受支持');
      }
      if (isRecord(report)) {
        validateIdentity(report.metadata, def, '精度报告', accuracyErrors);
        if (report.complete !== true) {
          accuracyErrors.push('精度报告未完整结束');
        }
        const derivedExecution = { passed: 0, failed: 0, not_run: 0 };
        const derivedAccuracy = { passed: 0, failed: 0, not_run: 0 };
        let casesValid = true;
        const metricFailures = [];
        if (!Array.isArray(report.cases)) {
          accuracyErrors.push('精度报告 cases 无效');
          casesValid = false;
        } else {
          for (const [caseIndex, caseItem] of report.cases.entries()) {
            if (!isRecord(caseItem)) {
              accuracyErrors.push(`精度报告 case ${caseIndex} 不是对象`);
              casesValid = false;
              continue;
            }
            const caseName = sanitizeMarkdownInline(caseItem.name || `case-${caseIndex}`, 120);
            const caseStatus = String(caseItem.status || 'unknown').toLowerCase();
            const accuracyStatus = String(caseItem.accuracy_status || 'not_requested').toLowerCase();
            const executionStatusValid = Object.hasOwn(derivedExecution, caseStatus);
            const accuracyStatusValid = ['passed', 'failed', 'not_run', 'not_requested'].includes(accuracyStatus);
            if (!executionStatusValid) {
              accuracyErrors.push(`${caseName}: 执行状态无效`);
              casesValid = false;
            } else {
              derivedExecution[caseStatus] += 1;
            }
            const returnCode = caseItem.return_code;
            const returnCodeValid =
              (caseStatus === 'passed' && Number.isInteger(returnCode) && returnCode === 0) ||
              (caseStatus === 'failed' && Number.isInteger(returnCode) && returnCode !== 0) ||
              (caseStatus === 'not_run' && (returnCode === null || returnCode === undefined));
            if (executionStatusValid && !returnCodeValid) {
              executionErrors.push(`${caseName}: 执行状态与 return_code 不一致`);
              casesValid = false;
            }
            if (typeof caseItem.accuracy_check !== 'boolean' || !accuracyStatusValid) {
              accuracyErrors.push(`${caseName}: 精度状态无效`);
              casesValid = false;
            }
            const isAccuracyCase = caseItem.accuracy_check === true || accuracyStatus === 'not_run';
            if ((caseItem.accuracy_check === true && accuracyStatus === 'not_requested') ||
                (!isAccuracyCase && accuracyStatus !== 'not_requested')) {
              accuracyErrors.push(`${caseName}: 精度状态与检查开关不一致`);
              casesValid = false;
            } else if (isAccuracyCase && Object.hasOwn(derivedAccuracy, accuracyStatus)) {
              derivedAccuracy[accuracyStatus] += 1;
            }
            if (!Array.isArray(caseItem.metrics)) {
              accuracyErrors.push(`${caseName}: metrics 无效`);
              casesValid = false;
              continue;
            }
            if (accuracyStatus === 'passed' && caseItem.metrics.length === 0) {
              accuracyErrors.push(`${caseName}: 精度通过但缺少 metric`);
              casesValid = false;
            }
            for (const [metricIndex, metric] of caseItem.metrics.entries()) {
              if (!isRecord(metric)) {
                accuracyErrors.push(`${caseName}: metric ${metricIndex} 不是对象`);
                casesValid = false;
                continue;
              }
              if (isAccuracyCase && ['finite', 'allclose', 'cosine_ok'].some((field) => typeof metric[field] !== 'boolean')) {
                accuracyErrors.push(`${caseName}: metric ${metricIndex} 检查字段无效`);
                casesValid = false;
              }
              if ((metric.finite === false || metric.allclose === false || metric.cosine_ok === false) && metricFailures.length < 6) {
                metricFailures.push(`${caseName}/${sanitizeMarkdownInline(metric.tensor || 'unknown', 80)}`);
              }
            }
          }
        }
        if (metricFailures.length > 0) {
          accuracyErrors.push(`精度异常：${metricFailures.slice(0, 6).join('、')}`);
        }
        summary = report.summary;
        if (!isRecord(summary)) {
          accuracyErrors.push('精度报告 summary 无效');
        } else {
          const fields = ['total', 'passed', 'failed', 'not_run', 'accuracy_total', 'accuracy_passed', 'accuracy_failed', 'accuracy_not_run'];
          if (fields.some((field) => !Number.isInteger(summary[field]) || summary[field] < 0)) {
            accuracyErrors.push('精度报告计数无效');
          } else {
            const expected = {
              total: Array.isArray(report.cases) ? report.cases.length : 0,
              ...derivedExecution,
              accuracy_total: Object.values(derivedAccuracy).reduce((sum, value) => sum + value, 0),
              accuracy_passed: derivedAccuracy.passed,
              accuracy_failed: derivedAccuracy.failed,
              accuracy_not_run: derivedAccuracy.not_run,
            };
            if (!casesValid || fields.some((field) => summary[field] !== expected[field])) {
              accuracyErrors.push('精度报告 summary 与 cases 不一致');
            } else if (summary.accuracy_total === 0) {
              reportAccuracyState = 'failure';
              accuracyErrors.push('未发现精度检查用例');
            } else if (summary.accuracy_failed > 0 || summary.accuracy_not_run > 0 || summary.accuracy_passed !== summary.accuracy_total) {
              reportAccuracyState = 'failure';
              accuracyErrors.push(
                `精度未通过 (${summary.accuracy_passed}/${summary.accuracy_total}，失败 ${summary.accuracy_failed}，未执行 ${summary.accuracy_not_run})`
              );
            } else {
              reportAccuracyState = 'success';
            }
          }
        }
      }
    }
    const diagnostic = readJson(diagnosticPath, '关键诊断', diagnosticErrors);
    let diagnosticTopValid = false;
    let diagnosticExecutionValid = false;
    let diagnosticAccuracyValid = false;
    let diagnosticHasFailurePayload = false;
    if (diagnostic !== null) {
      if (!isRecord(diagnostic)) {
        diagnosticErrors.push('关键诊断根节点无效');
      } else if (diagnostic.schema !== 'npu-ci-diagnostics-v1') {
        diagnosticErrors.push('关键诊断 schema 不受支持');
      }
      if (isRecord(diagnostic)) {
        validateIdentity(diagnostic.metadata, def, '关键诊断', diagnosticErrors);
        if (!['success', 'failure'].includes(diagnostic.status)) {
          diagnosticErrors.push('关键诊断 status 无效');
        } else {
          diagnosticTopValid = true;
          if (diagnostic.status !== 'success') {
            diagnosticErrors.push(`关键诊断状态为 ${diagnostic.status}`);
          }
        }
        if (!isRecord(diagnostic.execution) ||
            !['success', 'failure'].includes(diagnostic.execution.status) ||
            !Number.isInteger(diagnostic.execution.exit_code)) {
          diagnosticErrors.push('关键诊断 execution 无效');
        } else {
          diagnosticExecutionValid = true;
          if ((diagnostic.execution.exit_code === 0) !== (diagnostic.execution.status === 'success')) {
            diagnosticErrors.push('关键诊断 execution 状态与退出码不一致');
          }
        }
        const diagnosticAccuracyFields = ['total', 'passed', 'failed', 'not_run'];
        if (!isRecord(diagnostic.accuracy) ||
            !['success', 'failure', 'not_available'].includes(diagnostic.accuracy.status) ||
            !Array.isArray(diagnostic.accuracy.failures) ||
            diagnosticAccuracyFields.some((field) =>
              !Number.isInteger(diagnostic.accuracy[field]) || diagnostic.accuracy[field] < 0)) {
          diagnosticErrors.push('关键诊断 accuracy 无效');
        } else {
          diagnosticAccuracyValid = true;
          if (diagnostic.accuracy.failures.length > 0) {
            diagnosticHasFailurePayload = true;
            if (diagnostic.accuracy.status === 'success') {
              accuracyErrors.push('关键诊断精度成功但包含失败用例');
            }
          }
          for (const [failureIndex, failure] of diagnostic.accuracy.failures.entries()) {
            if (!isRecord(failure) || !Array.isArray(failure.metrics)) {
              diagnosticErrors.push(`关键诊断 accuracy failure ${failureIndex} 无效`);
              continue;
            }
            for (const [metricIndex, metric] of failure.metrics.entries()) {
              if (!isRecord(metric) || !isRecord(metric.details)) {
                diagnosticErrors.push(`关键诊断 accuracy metric ${failureIndex}/${metricIndex} 无效`);
              }
            }
          }
        }
        if (!isRecord(diagnostic.diagnostics)) {
          diagnosticErrors.push('关键诊断 diagnostics 无效');
        } else {
          for (const category of ['compile', 'runtime', 'infrastructure']) {
            const blocks = diagnostic.diagnostics[category];
            if (!Array.isArray(blocks)) {
              diagnosticErrors.push(`关键诊断 ${category} 无效`);
              continue;
            }
            if (blocks.length > 0) {
              diagnosticHasFailurePayload = true;
            }
            for (const [blockIndex, block] of blocks.entries()) {
              if (!isRecord(block) || !Array.isArray(block.lines) ||
                  block.lines.some((line) => typeof line !== 'string')) {
                diagnosticErrors.push(`关键诊断 ${category} block ${blockIndex} 无效`);
              }
            }
          }
        }
        if (!Array.isArray(diagnostic.reproduction) ||
            diagnostic.reproduction.some((line) => typeof line !== 'string')) {
          diagnosticErrors.push('关键诊断 reproduction 无效');
        } else if (diagnostic.reproduction.length > 0) {
          diagnosticHasFailurePayload = true;
        }
        if (diagnosticTopValid && diagnosticExecutionValid && diagnosticAccuracyValid) {
          const expectedDiagnosticStatus =
            diagnostic.execution.status === 'success' &&
            diagnostic.accuracy.status === 'success' &&
            !diagnosticHasFailurePayload
              ? 'success'
              : 'failure';
          if (diagnostic.status !== expectedDiagnosticStatus) {
            diagnosticErrors.push('关键诊断顶层状态与 execution/accuracy 不一致');
          }
        }
        if (diagnosticTopValid && isRecord(execution) &&
            ['success', 'failure'].includes(execution.status) &&
            execution.status !== diagnostic.status) {
          diagnosticErrors.push('执行结果与关键诊断顶层状态不一致');
        }
        if (diagnosticAccuracyValid && reportAccuracyState !== null) {
          const countsMatch =
            diagnostic.accuracy.total === summary.accuracy_total &&
            diagnostic.accuracy.passed === summary.accuracy_passed &&
            diagnostic.accuracy.failed === summary.accuracy_failed &&
            diagnostic.accuracy.not_run === summary.accuracy_not_run;
          if (diagnostic.accuracy.status !== reportAccuracyState || !countsMatch) {
            accuracyErrors.push('关键诊断 accuracy 与精度报告不一致');
          }
        }
      }
    }
    errors.push(...executionErrors, ...accuracyErrors, ...diagnosticErrors);
    return {
      ...def,
      execution,
      report,
      diagnostic,
      summary,
      executionOk: executionErrors.length === 0 && diagnosticErrors.length === 0,
      accuracyOk: accuracyErrors.length === 0,
      executionErrors,
      accuracyErrors,
      diagnosticErrors,
      errors,
    };
  };

  let currentHeadSha = '';
  let headCheckError = '';
  try {
    const { data: currentPr } = await github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: Number(prNumber),
    });
    currentHeadSha = currentPr.head.sha;
  } catch (error) {
    headCheckError = error.message;
    core.warning(`Unable to inspect current PR head before publishing NPU CI status: ${error.message}`);
  }
  const staleRun = Boolean(currentHeadSha) && currentHeadSha !== sha;
  const platforms = platformDefs.map(validatePlatform);
  const matrixArtifactMismatch = matrixResult !== 'success' &&
    platforms.every((item) => item.executionOk && item.accuracyOk);
  const matrixFailureReason = matrixArtifactMismatch
    ? `执行矩阵状态为 ${sanitizeMarkdownInline(matrixResult, 40)}，但双平台 artifact 均记录为通过；请查看失败 job 或 post-action。`
    : '';
  const executionOk = !headCheckError && !staleRun && matrixResult === 'success' && platforms.every((item) => item.executionOk);
  const accuracyOk = !headCheckError && !staleRun && platforms.every((item) => item.accuracyOk);
  const invalidReason = headCheckError
    ? '无法确认 PR head，本次 A2+A5 结果失效'
    : (staleRun ? 'PR head 已变化，本次 A2+A5 结果失效' : '');
  const executionDescription = invalidReason || matrixFailureReason || (
    executionOk
      ? `NPU CI 通过 (${mode}+example,A2+A5)`
      : `NPU CI 失败 (${mode}+example,A2+A5)`
  );
  const accuracyTotals = platforms.reduce(
    (totals, item) => {
      if (item.summary) {
        totals.total += Number(item.summary.accuracy_total || 0);
        totals.passed += Number(item.summary.accuracy_passed || 0);
        totals.failed += Number(item.summary.accuracy_failed || 0);
        totals.notRun += Number(item.summary.accuracy_not_run || 0);
      }
      return totals;
    },
    { total: 0, passed: 0, failed: 0, notRun: 0 }
  );
  const failedPlatformSummary = platforms
    .filter((item) => !item.accuracyOk)
    .map((item) => `${item.label}: ${item.accuracyErrors.join('；') || '精度失败'}`)
    .join('；');
  const accuracyDescription = invalidReason || (accuracyOk
    ? `A2+A5 精度通过：${accuracyTotals.passed}/${accuracyTotals.total} 个用例`
    : truncateDescription(`A2+A5 精度失败：${failedPlatformSummary || '报告不完整'}`));

  await Promise.all([
    github.rest.repos.createCommitStatus({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha,
      state: executionOk ? 'success' : 'failure',
      context: 'NPU CI / A2+A5 手动验证',
      description: truncateDescription(executionDescription),
      target_url: targetUrl,
    }),
    github.rest.repos.createCommitStatus({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha,
      state: accuracyOk ? 'success' : 'failure',
      context: 'NPU CI / A2+A5 精度检查',
      description: truncateDescription(accuracyDescription),
      target_url: targetUrl,
    }),
  ]);

  const commentBody = [
    `<!-- npu-ci:pr=${prNumber}:sha=${sha} -->`,
    `NPU CI ${executionOk && accuracyOk ? '通过' : '失败'}：\`${sha}\`。`,
    '',
    `- 模式：\`${sanitizeMarkdownInline(mode, 40)}\``,
    `- 触发人：@${context.actor}`,
    '- 验证平台：A2 (`ascend910b`) + A5 (`ascend950`)',
    invalidReason ? `- 失效原因：${invalidReason}` : null,
    `- 执行矩阵：\`${sanitizeMarkdownInline(matrixResult, 40)}\``,
    matrixFailureReason ? `- 汇总异常：${matrixFailureReason}` : null,
    '- Example ST：两平台均必跑',
    `- 精度检查：${accuracyDescription}`,
    ops ? `- 算子：\`${sanitizeMarkdownInline(ops, 500)}\`` : null,
    `- 运行链接：[NPU CI #${context.runNumber}](${targetUrl})`,
    '',
    '### 双平台汇总',
    '',
    '| 平台 | SOC | 执行 | 精度 | 精度用例 | 说明 |',
    '| --- | --- | --- | --- | --- | --- |',
    ...platforms.map((item) => {
      const summary = item.summary || {};
      const accuracyCases = Number.isInteger(summary.accuracy_passed) && Number.isInteger(summary.accuracy_total)
        ? `${summary.accuracy_passed}/${summary.accuracy_total}`
        : '-/-';
      return `| ${item.label} | \`${item.soc}\` | ${item.executionOk ? '通过' : '失败'} | ${item.accuracyOk ? '通过' : '失败'} | ${escapeCell(accuracyCases)} | ${escapeCell(item.errors.join('；'))} |`;
    }),
  ].filter((line) => line !== null);

  const categoryLabels = {
    compile: '编译错误',
    runtime: '运行异常',
    infrastructure: 'CI 环境异常',
  };
  for (const item of platforms.filter((platform) => !platform.executionOk || !platform.accuracyOk)) {
    commentBody.push('', `### ${item.label} 关键诊断`, '');
    let detailAdded = false;
    const groups = item.diagnostic && item.diagnostic.diagnostics;
    if (groups && typeof groups === 'object') {
      for (const category of ['compile', 'runtime', 'infrastructure']) {
        const blocks = Array.isArray(groups[category]) ? groups[category].slice(0, 2) : [];
        for (const block of blocks) {
          const lines = diagnosticLines(block && block.lines);
          if (lines.length === 0) {
            continue;
          }
          const rawOccurrences = Number(block.occurrences || 1);
          const occurrences = Number.isSafeInteger(rawOccurrences) && rawOccurrences > 1
            ? Math.min(rawOccurrences, 9999)
            : 1;
          commentBody.push(`#### ${categoryLabels[category]}${occurrences > 1 ? `（重复 ${occurrences} 次）` : ''}`, '');
          commentBody.push(...lines.map((line) => `    ${line}`), '');
          detailAdded = true;
        }
      }
    }

    const accuracy = item.diagnostic && item.diagnostic.accuracy;
    const failures = accuracy && Array.isArray(accuracy.failures)
      ? accuracy.failures.filter(isRecord).slice(0, 4)
      : [];
    if (failures.length > 0) {
      commentBody.push('#### 精度异常', '');
      commentBody.push('| 用例 | Tensor | 失败判据 | tol | cosine / cos_min | max_abs | bad_frac | 退出码 |');
      commentBody.push('| --- | --- | --- | --- | --- | --- | --- | --- |');
      for (const failure of failures) {
        const validMetrics = Array.isArray(failure.metrics)
          ? failure.metrics.filter(isRecord).slice(0, 3)
          : [];
        const metrics = validMetrics.length > 0
          ? validMetrics
          : [{ output: '-', status: failure.status || 'failed', details: {} }];
        for (const metric of metrics) {
          const details = isRecord(metric.details) ? metric.details : {};
          const failedChecks = ['finite', 'allclose', 'cosine_ok']
            .filter((field) => details[field] === false)
            .map((field) => `${field}=false`);
          const criterion = failedChecks.join(', ') || metric.status || failure.status || 'failed';
          commentBody.push(
            `| ${escapeCell(failure.name)} | ${escapeCell(metric.output)} | ${escapeCell(criterion)} | ` +
            `${escapeCell(details.tol)} | ${escapeCell(details.cosine)} / ${escapeCell(details.cos_min)} | ` +
            `${escapeCell(details.max_abs)} | ${escapeCell(details.bad_frac)} | ${escapeCell(failure.return_code)} |`
          );
        }
      }
      detailAdded = true;
    }

    const reproduction = item.diagnostic && Array.isArray(item.diagnostic.reproduction)
      ? item.diagnostic.reproduction.filter((line) => typeof line === 'string').slice(0, 2).map((line) => sanitizeInline(line, 2400)).filter(Boolean)
      : [];
    if (reproduction.length > 0) {
      commentBody.push('', '#### 复现命令', '', ...reproduction.map((line) => `    ${line}`));
      detailAdded = true;
    }

    if (!detailAdded) {
      const fallback = item.errors.length > 0 ? item.errors : ['未生成可用的关键诊断，请查看对应平台 job。'];
      commentBody.push(...fallback.slice(0, 5).map((line) => `- ${sanitizeMarkdownInline(line, 300)}`));
    }
  }

  if (executionOk && accuracyOk) {
    commentBody.push('', '全部检查通过；成功用例的逐 Tensor 指标已收纳到平台 artifact，不在评论中重复展开。');
  }

  try {
    const body = renderComment(commentBody);
    if (commentId) {
      await github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: Number(commentId),
        body,
      });
    } else if (prNumber) {
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: Number(prNumber),
        body,
      });
    }
  } catch (error) {
    core.warning(`Unable to create or update PR comment: ${error.message}`);
  }

  if (!executionOk || !accuracyOk) {
    core.setFailed('A2+A5 NPU CI 汇总未通过。');
  }
};
