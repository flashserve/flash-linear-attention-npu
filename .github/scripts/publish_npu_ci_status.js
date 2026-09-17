'use strict';

module.exports = async function publishNpuCiStatus({ github, context, core }) {
  const fs = require('fs');
  const path = require('path');
  const mode = process.env.NPU_CI_MODE || '';
  const ops = process.env.NPU_OPS || '';
  const opsItems = ops.split(',').map((item) => item.trim());
  if (ops.trim() && opsItems.some((item) =>
    !item || !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(item))) {
    throw new Error('ops 参数必须是以逗号分隔的合法非空算子名');
  }
  const normalizedOps = [...new Set(opsItems.filter(Boolean))].sort().join(',');
  const requestScope = normalizedOps ? 'scoped' : 'all';
  const requestKey = requestScope === 'all'
    ? 'all'
    : require('crypto').createHash('sha256').update(normalizedOps).digest('hex').slice(0, 10);
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
  const stageDefs = [
    {
      key: 'environment-contracts',
      context: 'NPU CI / A2+A5 / 01 环境、wheel 与运行时契约',
      label: '环境、wheel 配置及运行时契约',
    },
    {
      key: 'opp-package',
      context: 'NPU CI / A2+A5 / 02 全量 OPP 构建',
      label: '目标 SOC 全量 OPP run 包构建',
    },
    {
      key: 'standalone-layout',
      context: 'NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局',
      label: '独立 torch_custom wheel 与 OPP 安装布局',
    },
    {
      key: 'torch-adapter',
      context: 'NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配',
      label: 'OPP 安装与 PyTorch 适配构建',
    },
    {
      key: 'gdr-example-st',
      context: 'NPU CI / A2+A5 / 05 GDR Example/ST',
      label: 'GDR Example/ST（4 条精度用例）',
    },
    {
      key: 'scoped-overlay',
      context: 'NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装',
      label: 'chunk_fwd_o run 包覆盖 wheel OPP',
    },
  ];
  const finalStatusDef = {
    context: 'NPU CI / A2+A5 / 07 报告与 commit 校验',
    label: '报告完整性与 PR commit 一致性',
  };
  const scopedStatusDef = {
    context: 'NPU CI / A2+A5 / 定向诊断',
    label: '定向诊断',
  };
  const accuracyThresholdContract = {
    output_tol: 5e-3,
    grad_tol: 8e-3,
    beta_grad_tol: 2e-2,
    gate_grad_tol: 2e-2,
    output_cos_min: 0.999,
    grad_cos_min: 0.999,
    beta_grad_cos_min: 0.99,
    gate_grad_cos_min: 0.99,
  };
  const metricThresholdFields = {
    o: ['output_tol', 'output_cos_min'],
    dq: ['grad_tol', 'grad_cos_min'],
    dk: ['grad_tol', 'grad_cos_min'],
    dv: ['grad_tol', 'grad_cos_min'],
    dbeta: ['beta_grad_tol', 'beta_grad_cos_min'],
    dg: ['gate_grad_tol', 'gate_grad_cos_min'],
  };
  const allAccuracyTensors = ['o', 'dq', 'dk', 'dv', 'dbeta', 'dg'];
  const baseGdrContract = {
    script: 'examples/flash_gated_delta_rule.py',
    chunk_size: 64,
    key_dim: 128,
    value_dim: 128,
    gate_source: 'g',
    gate_function: 'logsigmoid',
    initial_state: 'none',
    output_final_state: false,
    demo_model: false,
    conv_kernel: 4,
    accuracy_thresholds: accuracyThresholdContract,
  };
  const requiredAccuracyCases = [
    {
      name: 'case1_current_default',
      contract: {
        ...baseGdrContract,
        batch: 1, tokens: 4087, query_heads: 32, value_heads: 32,
        dtype: 'bf16', varlen: true, mean_len: 1024, qk_l2norm: true,
        seed: 20260630, scale: null, accuracy_tensors: ['o'],
      },
      sequenceCount: null,
    },
    {
      name: 'gdr_accuracy_dense_b2_t128_h2_d128_fp16',
      contract: {
        ...baseGdrContract,
        batch: 2, tokens: 128, query_heads: 2, value_heads: 2,
        dtype: 'fp16', varlen: false, mean_len: 128, qk_l2norm: false,
        seed: 42, scale: 0.1, accuracy_tensors: allAccuracyTensors,
      },
      sequenceCount: null,
    },
    {
      name: 'gdr_accuracy_varlen_64_64_h2_d128_fp16',
      contract: {
        ...baseGdrContract,
        batch: 1, tokens: 128, query_heads: 2, value_heads: 2,
        dtype: 'fp16', varlen: true, mean_len: 1024, qk_l2norm: false,
        seed: 43, scale: 0.1, accuracy_tensors: allAccuracyTensors,
        cu_seqlens: [0, 64, 128],
      },
      sequenceCount: 2,
    },
    {
      name: 'gdr_accuracy_tnd_3seq_t3991_h2_d128_fp16',
      contract: {
        ...baseGdrContract,
        batch: 1, tokens: 3991, query_heads: 2, value_heads: 2,
        dtype: 'fp16', varlen: true, mean_len: 1024, qk_l2norm: false,
        seed: 44, scale: 0.1, accuracy_tensors: ['o'],
      },
      sequenceCount: 3,
    },
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
  const contractValueMatches = (actual, expected) => {
    if (Array.isArray(expected)) {
      return Array.isArray(actual) && expected.length === actual.length &&
        expected.every((value, index) => contractValueMatches(actual[index], value));
    }
    if (isRecord(expected)) {
      return isRecord(actual) && Object.keys(expected).length === Object.keys(actual).length &&
        Object.entries(expected).every(([key, value]) =>
          Object.hasOwn(actual, key) && contractValueMatches(actual[key], value)
        );
    }
    return actual === expected;
  };
  const shellQuote = (value) => `'${String(value).replace(/'/g, `'"'"'`)}'`;
  const buildFallbackReproduction = (platform, failedStage) => {
    const assignments = [
      ['CI_MODE', mode],
      ['CI_SOC', platform.soc],
      ['FLA_NPU_SOC', platform.soc],
      ['CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK', 'true'],
      ['CI_RUN_SCOPED_WHEEL_INSTALL_CHECK', 'true'],
      ['CI_STAGE', failedStage || 'all'],
    ];
    if (platform.key === 'a5') {
      assignments.push(
        ['CI_IMAGE', 'fla-npu-ci:9.1.0-950'],
        ['CI_DOCKERFILE', 'ci/Dockerfile.ascend950'],
        ['CI_REQUIRE_PRELOADED_IMAGE', 'true']
      );
    } else {
      assignments.push(
        ['CI_IMAGE', 'fla-npu-ci:9.1.0-910b'],
        ['CI_DOCKERFILE', 'ci/Dockerfile']
      );
    }
    if (normalizedOps) {
      assignments.push(['CI_OPS', normalizedOps]);
    }
    return `${assignments.map(([key, value]) => `${key}=${shellQuote(value)}`).join(' ')} bash ci/run_ci_container.sh`;
  };
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
  const validateIdentity = (metadata, def, description, errors, artifactRunAttempt) => {
    if (!isRecord(metadata)) {
      errors.push(`${description}缺少身份信息`);
      return;
    }
    const expected = {
      platform: def.key,
      soc: def.soc,
      head_sha: sha,
      run_id: expectedRunId,
      run_attempt: artifactRunAttempt,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (String(metadata[key] || '') !== String(value)) {
        errors.push(`${description}${key}不属于本次触发`);
      }
    }
  };
  const reportsRoot = 'accuracy-reports';
  const resolvePlatformArtifact = (def) => {
    const flatResult = path.join(reportsRoot, `npu-ci-result-${def.key}.json`);
    if (fs.existsSync(flatResult)) {
      return { directory: reportsRoot, runAttempt: expectedRunAttempt };
    }

    const expectedAttemptNumber = Number.parseInt(expectedRunAttempt, 10);
    let entries = [];
    try {
      entries = fs.readdirSync(reportsRoot, { withFileTypes: true });
    } catch (error) {
      core.warning(`Unable to inspect downloaded NPU CI artifacts: ${error.message}`);
    }
    const candidates = entries
      .filter((entry) => entry.isDirectory())
      .map((entry) => {
        const match = entry.name.match(/^npu-ci-result-(a2|a5)-(\d+)-(\d+)$/);
        if (!match || match[1] !== def.key || match[2] !== expectedRunId) {
          return null;
        }
        const attempt = Number.parseInt(match[3], 10);
        if (!Number.isSafeInteger(attempt) || attempt <= 0 ||
            !Number.isSafeInteger(expectedAttemptNumber) || attempt > expectedAttemptNumber) {
          return null;
        }
        return {
          directory: path.join(reportsRoot, entry.name),
          runAttempt: String(attempt),
          attempt,
        };
      })
      .filter(Boolean)
      .sort((left, right) => right.attempt - left.attempt);
    if (candidates.length > 0) {
      return candidates[0];
    }
    return {
      directory: path.join(
        reportsRoot,
        `npu-ci-result-${def.key}-${expectedRunId}-${expectedRunAttempt}`
      ),
      runAttempt: expectedRunAttempt,
    };
  };
  const validatePlatform = (def) => {
    const errors = [];
    const executionErrors = [];
    const accuracyErrors = [];
    const diagnosticErrors = [];
    const stageReportErrors = [];
    const artifact = resolvePlatformArtifact(def);
    const resultPath = path.join(artifact.directory, `npu-ci-result-${def.key}.json`);
    const reportPath = path.join(artifact.directory, `gdr_accuracy_report-${def.key}.json`);
    const diagnosticPath = path.join(artifact.directory, `npu-ci-diagnostics-${def.key}.json`);
    const stageReportPath = path.join(artifact.directory, `npu-ci-stages-${def.key}.json`);
    const stageReport = readJson(stageReportPath, '分项结果', stageReportErrors);
    const stageResults = {};
    if (stageReport !== null) {
      if (!isRecord(stageReport)) {
        stageReportErrors.push('分项结果根节点无效');
      } else if (stageReport.schema !== 'npu-ci-stage-report-v1') {
        stageReportErrors.push('分项结果 schema 不受支持');
      }
      if (isRecord(stageReport)) {
        validateIdentity(
          stageReport.metadata,
          def,
          '分项结果',
          stageReportErrors,
          artifact.runAttempt
        );
        if (stageReport.complete !== true) {
          stageReportErrors.push('分项结果未完整结束');
        }
        if (!['success', 'failure'].includes(stageReport.status)) {
          stageReportErrors.push('分项结果顶层 status 无效');
        }
        if (!isRecord(stageReport.stages)) {
          stageReportErrors.push('分项结果 stages 无效');
        } else {
          for (const stageDef of stageDefs) {
            const stage = stageReport.stages[stageDef.key];
            const stageErrors = [];
            if (!isRecord(stage)) {
              stageErrors.push('结果缺失');
            } else {
              const status = String(stage.status || '');
              const exitCode = stage.exit_code;
              if (!['not_run', 'running', 'success', 'failure', 'skipped'].includes(status)) {
                stageErrors.push('status 无效');
              }
              if (typeof stage.reason !== 'string') {
                stageErrors.push('reason 无效');
              }
              if ((status === 'success' && exitCode !== 0) ||
                  (status === 'failure' && (!Number.isInteger(exitCode) || exitCode === 0)) ||
                  (['not_run', 'running', 'skipped'].includes(status) && exitCode !== null)) {
                stageErrors.push('status 与 exit_code 不一致');
              }
              stageResults[stageDef.key] = {
                status,
                exitCode,
                reason: sanitizeMarkdownInline(stage.reason || '', 240),
                valid: stageErrors.length === 0,
              };
            }
            if (stageErrors.length > 0) {
              stageReportErrors.push(`${stageDef.label}：${stageErrors.join('；')}`);
              stageResults[stageDef.key] = {
                status: 'invalid',
                exitCode: null,
                reason: stageErrors.join('；'),
                valid: false,
              };
            }
          }
          const stageValues = stageDefs.map((stageDef) => stageResults[stageDef.key]);
          if (stageValues.some((stage) => stage && ['not_run', 'running'].includes(stage.status))) {
            stageReportErrors.push('分项结果包含未完成阶段');
          }
          if (stageValues.every((stage) => stage && stage.valid)) {
            let failedStageIndex = null;
            for (const [index, stage] of stageValues.entries()) {
              if (failedStageIndex !== null && stage.status !== 'skipped') {
                const failedStageLabel = stageDefs[failedStageIndex].label;
                stageReportErrors.push(
                  `${failedStageLabel}失败后，${stageDefs[index].label}必须跳过`
                );
                break;
              }
              if (failedStageIndex === null && stage.status === 'failure') {
                failedStageIndex = index;
              }
            }
            const expectedStatus = stageValues.some((stage) => stage.status === 'failure')
              ? 'failure'
              : 'success';
            if (stageReport.status !== expectedStatus) {
              stageReportErrors.push('分项结果顶层 status 与 stages 不一致');
            }
          }
        }
      }
    }
    const execution = readJson(resultPath, '执行结果', executionErrors);
    if (execution !== null) {
      if (!isRecord(execution)) {
        executionErrors.push('执行结果根节点无效');
      } else if (execution.schema !== 'npu-ci-platform-result-v1') {
        executionErrors.push('执行结果 schema 不受支持');
      }
      if (isRecord(execution)) {
        validateIdentity(
          execution.metadata,
          def,
          '执行结果',
          executionErrors,
          artifact.runAttempt
        );
        if (!['success', 'failure'].includes(execution.status)) {
          executionErrors.push('执行结果 status 无效');
        } else if (execution.status !== 'success') {
          executionErrors.push(`执行状态为 ${execution.status || 'unknown'}`);
        }
      }
    }
    const report = requestScope === 'all'
      ? readJson(reportPath, '精度报告', accuracyErrors)
      : null;
    let summary = null;
    let reportAccuracyState = null;
    if (report !== null) {
      if (!isRecord(report)) {
        accuracyErrors.push('精度报告根节点无效');
      } else if (report.schema !== 'gdr-accuracy-report-v1') {
        accuracyErrors.push('精度报告 schema 不受支持');
      }
      if (isRecord(report)) {
        validateIdentity(
          report.metadata,
          def,
          '精度报告',
          accuracyErrors,
          artifact.runAttempt
        );
        if (report.complete !== true) {
          accuracyErrors.push('精度报告未完整结束');
        }
        const derivedExecution = { passed: 0, failed: 0, not_run: 0 };
        const derivedAccuracy = { passed: 0, failed: 0, not_run: 0 };
        const casesByName = new Map();
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
            const rawCaseName = typeof caseItem.name === 'string' ? caseItem.name : '';
            if (!rawCaseName) {
              accuracyErrors.push(`精度报告 case ${caseIndex} 名称无效`);
              casesValid = false;
            } else if (casesByName.has(rawCaseName)) {
              accuracyErrors.push(`精度报告包含重复用例：${caseName}`);
              casesValid = false;
            } else {
              casesByName.set(rawCaseName, caseItem);
            }
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
        if (requestScope === 'all') {
          for (const requiredCase of requiredAccuracyCases) {
            const caseItem = casesByName.get(requiredCase.name);
            if (!caseItem) {
              accuracyErrors.push(`缺少必跑精度用例：${requiredCase.name}`);
              casesValid = false;
              continue;
            }
            if (caseItem.status !== 'passed' || caseItem.accuracy_check !== true ||
                caseItem.accuracy_status !== 'passed') {
              accuracyErrors.push(`必跑精度用例未通过：${requiredCase.name}`);
              casesValid = false;
            }
            const contract = caseItem.contract;
            if (!isRecord(contract)) {
              accuracyErrors.push(`必跑精度用例契约缺失：${requiredCase.name}`);
              casesValid = false;
              continue;
            }
            const mismatches = Object.entries(requiredCase.contract)
              .filter(([field, expected]) => !contractValueMatches(contract[field], expected))
              .map(([field]) => field);
            const cuSeqlens = contract.cu_seqlens;
            const validCuSeqlens = Array.isArray(cuSeqlens) &&
              cuSeqlens.every((value) => Number.isInteger(value)) &&
              (requiredCase.contract.varlen
                ? cuSeqlens.length >= 2 && cuSeqlens[0] === 0 &&
                  cuSeqlens.at(-1) === requiredCase.contract.tokens &&
                  cuSeqlens.slice(1).every((value, index) => value > cuSeqlens[index])
                : cuSeqlens.length === 0) &&
              (requiredCase.sequenceCount === null ||
                cuSeqlens.length === requiredCase.sequenceCount + 1);
            if (!validCuSeqlens && !mismatches.includes('cu_seqlens')) {
              mismatches.push('cu_seqlens');
            }
            if (mismatches.length > 0) {
              accuracyErrors.push(
                `必跑精度用例契约不匹配：${requiredCase.name} (${mismatches.join(', ')})`
              );
              casesValid = false;
            }
            if (!Array.isArray(caseItem.metrics)) {
              continue;
            }
            const metricNames = caseItem.metrics.map((metric) =>
              isRecord(metric) && typeof metric.tensor === 'string' ? metric.tensor : ''
            );
            const duplicateMetricNames = [...new Set(metricNames.filter((name, index) =>
              name && metricNames.indexOf(name) !== index
            ))];
            if (duplicateMetricNames.length > 0) {
              accuracyErrors.push(
                `必跑精度用例包含重复 metric：${requiredCase.name} (${duplicateMetricNames.join(', ')})`
              );
              casesValid = false;
            }
            const expectedMetricNames = requiredCase.contract.accuracy_tensors;
            const metricSetMatches = metricNames.length === expectedMetricNames.length &&
              expectedMetricNames.every((name) => metricNames.includes(name));
            if (!metricSetMatches) {
              accuracyErrors.push(
                `必跑精度用例 metric 集合不匹配：${requiredCase.name} ` +
                `(期望 ${expectedMetricNames.join(',')}，实际 ${metricNames.filter(Boolean).join(',') || '空'})`
              );
              casesValid = false;
            }
            for (const metric of caseItem.metrics) {
              if (!isRecord(metric) || !Object.hasOwn(metricThresholdFields, metric.tensor)) {
                continue;
              }
              const [tolField, cosMinField] = metricThresholdFields[metric.tensor];
              const expectedTol = accuracyThresholdContract[tolField];
              const expectedCosMin = accuracyThresholdContract[cosMinField];
              if (metric.tol !== expectedTol || metric.cos_min !== expectedCosMin) {
                accuracyErrors.push(
                  `必跑精度用例 metric 阈值不匹配：${requiredCase.name}/${metric.tensor}`
                );
                casesValid = false;
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
        validateIdentity(
          diagnostic.metadata,
          def,
          '关键诊断',
          diagnosticErrors,
          artifact.runAttempt
        );
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
          const scopedAccuracyOk = requestScope === 'scoped' &&
            diagnostic.accuracy.status === 'not_available';
          const expectedDiagnosticStatus =
            diagnostic.execution.status === 'success' &&
            (diagnostic.accuracy.status === 'success' || scopedAccuracyOk) &&
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
    errors.push(...executionErrors, ...accuracyErrors, ...diagnosticErrors, ...stageReportErrors);
    return {
      ...def,
      execution,
      report,
      diagnostic,
      stageReport,
      stageResults,
      summary,
      executionOk: executionErrors.length === 0 && diagnosticErrors.length === 0,
      accuracyOk: accuracyErrors.length === 0,
      stageReportOk: stageReportErrors.length === 0,
      executionErrors,
      accuracyErrors,
      diagnosticErrors,
      stageReportErrors,
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
    platforms.every((item) => item.executionOk && item.accuracyOk && item.stageReportOk);
  const matrixFailureReason = matrixArtifactMismatch
    ? `执行矩阵状态为 ${sanitizeMarkdownInline(matrixResult, 40)}，但双平台 artifact 均记录为通过；请查看失败 job 或 post-action。`
    : '';
  const invalidReason = headCheckError
    ? '无法确认 PR head，本次 A2+A5 结果失效'
    : (staleRun ? 'PR head 已变化，本次 A2+A5 结果失效' : '');
  const stageOutcomes = stageDefs.map((stageDef) => {
    const failures = [];
    for (const platform of platforms) {
      const result = platform.stageResults[stageDef.key];
      if (!platform.stageReportOk || !result || !result.valid) {
        failures.push(`${platform.label} 分项报告无效`);
        continue;
      }
      if (result.status !== 'success') {
        const reason = result.reason ? `（${result.reason}）` : '';
        failures.push(`${platform.label} ${result.status}${reason}`);
        continue;
      }
      if (stageDef.key === 'gdr-example-st' && !platform.accuracyOk) {
        failures.push(`${platform.label} 精度未通过`);
      }
    }
    const ok = !invalidReason && failures.length === 0;
    return {
      ...stageDef,
      ok,
      description: invalidReason || (ok
        ? `${stageDef.label}：A2、A5 均通过`
        : `${stageDef.label}：${failures.join('；') || '未通过'}`),
    };
  });
  const accuracyOk = !headCheckError && !staleRun && platforms.every((item) => item.accuracyOk);
  const requiredStageOutcomes = requestScope === 'all'
    ? stageOutcomes
    : stageOutcomes.filter((item) =>
      ['environment-contracts', 'opp-package'].includes(item.key)
    );
  const executionOk = !headCheckError && !staleRun && matrixResult === 'success' &&
    platforms.every((item) => item.executionOk && item.stageReportOk) &&
    requiredStageOutcomes.every((item) => item.ok);
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

  const finalOk = executionOk && accuracyOk;
  const finalDescription = invalidReason || matrixFailureReason || (finalOk
    ? `报告与 commit 校验通过 (${mode},${requestKey},A2+A5)`
    : `报告或执行结果未通过 (${mode},${requestKey},A2+A5)`);
  const statusPayloads = requestScope === 'all' ? stageOutcomes.map((item) => ({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha,
      state: item.ok ? 'success' : 'failure',
      context: item.context,
      description: truncateDescription(item.description),
      target_url: targetUrl,
    })) : [];
  if (requestScope === 'all') {
    statusPayloads.push({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha,
      state: finalOk ? 'success' : 'failure',
      context: finalStatusDef.context,
      description: truncateDescription(finalDescription),
      target_url: targetUrl,
    });
  } else {
    statusPayloads.push({
      owner: context.repo.owner,
      repo: context.repo.repo,
      sha,
      state: finalOk ? 'success' : 'failure',
      context: scopedStatusDef.context,
      description: truncateDescription(
        invalidReason || (finalOk
          ? `${scopedStatusDef.label}通过 (${mode},${requestKey},A2+A5)`
          : `${scopedStatusDef.label}未通过 (${mode},${requestKey},A2+A5)`)
      ),
      target_url: targetUrl,
    });
  }
  const publishCommitStatus = async (payload) => {
    let lastError;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        await github.rest.repos.createCommitStatus(payload);
        return;
      } catch (error) {
        lastError = error;
        if (attempt < 3) {
          await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
        }
      }
    }
    throw lastError;
  };
  const stageStatusPayloads = requestScope === 'all' ? statusPayloads.slice(0, -1) : [];
  const finalStatusPayload = statusPayloads.at(-1);
  const stagePublishResults = await Promise.allSettled(
    stageStatusPayloads.map((payload) => publishCommitStatus(payload))
  );
  const stagePublishFailures = stagePublishResults.filter((result) => result.status === 'rejected');
  if (stagePublishFailures.length > 0) {
    if (typeof core.setOutput === 'function') {
      core.setOutput('statuses_published', 'false');
    }
    throw new Error(`无法发布 ${stagePublishFailures.length} 个 NPU CI 分项状态`);
  }
  try {
    await publishCommitStatus(finalStatusPayload);
  } catch (error) {
    if (typeof core.setOutput === 'function') {
      core.setOutput('statuses_published', 'false');
    }
    throw error;
  }
  if (typeof core.setOutput === 'function') {
    core.setOutput('statuses_published', 'true');
  }

  const stageCell = (platform, stageDef) => {
    const result = platform.stageResults[stageDef.key];
    if (!platform.stageReportOk || !result || !result.valid) {
      return '报告无效';
    }
    if (stageDef.key === 'gdr-example-st' && result.status === 'success' && !platform.accuracyOk) {
      return '精度失败';
    }
    const labels = {
      success: '通过',
      failure: '失败',
      skipped: '未执行',
      not_run: '未执行',
      running: '未完成',
    };
    return labels[result.status] || '无效';
  };
  const displayedStageOutcomes = requestScope === 'all'
    ? stageOutcomes
    : requiredStageOutcomes;
  const finalDisplayLabel = requestScope === 'all'
    ? finalStatusDef.label
    : scopedStatusDef.label;
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
    requestScope === 'all'
      ? '- Example ST：两平台均必跑'
      : '- 范围：环境契约与指定算子 OPP 构建；不执行后续无关分项',
    requestScope === 'all' ? `- 精度检查：${accuracyDescription}` : null,
    ops ? `- 算子：\`${sanitizeMarkdownInline(ops, 500)}\`` : null,
    `- 运行链接：[NPU CI #${context.runNumber}](${targetUrl})`,
    '',
    '### 分项结果',
    '',
    '| 检查项 | A2 | A5 | 汇总 |',
    '| --- | --- | --- | --- |',
    ...displayedStageOutcomes.map((outcome) =>
      `| ${outcome.label} | ${stageCell(platforms[0], outcome)} | ${stageCell(platforms[1], outcome)} | ${outcome.ok ? '通过' : '失败'} |`
    ),
    `| ${finalDisplayLabel} | - | - | ${finalOk ? '通过' : '失败'} |`,
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
      const accuracyCell = requestScope === 'all'
        ? (item.accuracyOk ? '通过' : '失败')
        : '未执行';
      const accuracyCasesCell = requestScope === 'all' ? accuracyCases : '-/-';
      return `| ${item.label} | \`${item.soc}\` | ${item.executionOk ? '通过' : '失败'} | ${accuracyCell} | ${escapeCell(accuracyCasesCell)} | ${escapeCell(item.errors.join('；'))} |`;
    }),
  ].filter((line) => line !== null);

  const categoryLabels = {
    compile: '编译错误',
    runtime: '运行异常',
    infrastructure: 'CI 环境异常',
  };
  for (const item of platforms.filter((platform) => {
    const requiredDefs = requestScope === 'all'
      ? stageDefs
      : stageDefs.filter((stageDef) =>
        ['environment-contracts', 'opp-package'].includes(stageDef.key)
      );
    const hasIncompleteRequiredStage = requiredDefs.some((stageDef) => {
      const stage = platform.stageResults[stageDef.key];
      return !stage || !stage.valid || stage.status !== 'success';
    });
    return !platform.executionOk || !platform.accuracyOk ||
      !platform.stageReportOk || hasIncompleteRequiredStage;
  })) {
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

    let reproduction = item.diagnostic && Array.isArray(item.diagnostic.reproduction)
      ? item.diagnostic.reproduction.filter((line) => typeof line === 'string').slice(0, 2).map((line) => sanitizeInline(line, 2400)).filter(Boolean)
      : [];
    if (reproduction.length === 0) {
      const failedStage = stageDefs.find((stageDef) => {
        const stage = item.stageResults[stageDef.key];
        return stage && stage.valid && stage.status === 'failure';
      });
      reproduction = [buildFallbackReproduction(
        item,
        failedStage
          ? failedStage.key
          : (!item.accuracyOk ? 'gdr-example-st' : (requestScope === 'scoped' ? 'opp-package' : 'all'))
      )];
    }
    if (reproduction.length > 0) {
      commentBody.push('', '#### 复现命令', '', ...reproduction.map((line) => `    ${line}`));
      detailAdded = true;
    }

    if (!detailAdded) {
      const fallback = item.errors.length > 0 ? item.errors : ['未生成可用的关键诊断，请查看对应平台 job。'];
      commentBody.push(...fallback.slice(0, 5).map((line) => `- ${sanitizeMarkdownInline(line, 300)}`));
    }
  }

  if (matrixFailureReason && platforms.every((item) =>
    item.executionOk && item.accuracyOk && item.stageReportOk
  )) {
    commentBody.push('', '### 汇总异常复现', '');
    for (const item of platforms) {
      commentBody.push(`#### ${item.label}`, '', `    ${buildFallbackReproduction(item, 'all')}`, '');
    }
  }

  if (executionOk && accuracyOk) {
    commentBody.push(
      '',
      requestScope === 'all'
        ? '全部检查通过；成功用例的逐 Tensor 指标已收纳到平台 artifact，不在评论中重复展开。'
        : '定向编译诊断通过；未改写 01-07 正式门禁状态。'
    );
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
