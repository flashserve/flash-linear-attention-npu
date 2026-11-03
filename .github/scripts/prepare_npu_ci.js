'use strict';

const {
  REQUIRED_CHECKS, SCOPED_CHECK, requireAuthorizedCiUser, isMarkdownOnly,
  assertUnchangedPullRequest, normalizeOps, buildRequestKey, createComment, writeStatuses,
} = require('./npu_ci_policy');

function latestByContext(statuses) {
  const result = new Map();
  const sorted = [...statuses].sort((a, b) =>
    new Date(b.created_at) - new Date(a.created_at) || Number(b.id || 0) - Number(a.id || 0));
  for (const status of sorted) if (!result.has(status.context)) result.set(status.context, status);
  return result;
}

async function isActiveRun({ github, context, core, status }) {
  if (status?.state !== 'pending') return false;
  const runId = Number((status.target_url || '').match(/\/actions\/runs\/(\d+)/)?.[1]);
  if (!runId || runId === Number(context.runId)) return false;
  try {
    const { data: run } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: runId });
    return ['queued', 'in_progress', 'waiting', 'pending', 'requested'].includes(run.status);
  } catch (error) {
    core.warning(`无法确认历史 NPU CI 是否仍在运行，允许重新执行：${error.message}`);
    return false;
  }
}

async function prAuthorIsAdmin({ github, context, core, pr }) {
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      ...context.repo, username: pr.user.login,
    });
    return data.permission === 'admin';
  } catch (error) {
    // An unavailable permission check must never grant the PR code CI privileges.
    core.warning(`无法确认 PR 作者的 Admin 身份，使用可信 CI：${error.message}`);
    return false;
  }
}

module.exports = async function prepareNpuCi({ github, context, core }) {
  core.setOutput('should_run', 'false');
  const inputs = context.payload.inputs || {};
  const requestedBy = context.actor === 'github-actions[bot]' ? inputs.requested_by : context.actor;
  const users = await requireAuthorizedCiUser({ github, context, username: requestedBy });
  if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) {
    // GitHub preserves context.actor on reruns; the person requesting the rerun
    // is github.triggering_actor and must independently have CI authorization.
    const rerunBy = process.env.NPU_CI_TRIGGERING_ACTOR || '';
    if (!users.some((user) => user.toLowerCase() === rerunBy.toLowerCase())) {
      throw new Error(`${rerunBy || '(空用户名)'} 无权重新运行 NPU CI：须列入默认分支的 .github/npu-ci-users.json。`);
    }
  }

  const prNumber = Number(inputs.pr_number);
  const ciMode = inputs.ci_mode || 'quick';
  const ops = normalizeOps(inputs.ops || '');
  if (!Number.isInteger(prNumber) || prNumber <= 0) throw new Error(`无效的 PR 编号：${inputs.pr_number}`);
  if (!['quick', 'full'].includes(ciMode)) throw new Error(`不支持的 CI 模式：${ciMode}`);

  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: prNumber });
  if (pr.state !== 'open') throw new Error(`PR #${prNumber} 已关闭。`);
  if (pr.draft) throw new Error(`PR #${prNumber} 仍是草稿，请先标记为 ready for review。`);
  const { data: baseBranch } = await github.rest.repos.getBranch({ ...context.repo, branch: pr.base.ref });
  const baseSha = baseBranch.commit.sha;
  const workflowSha = process.env.NPU_CI_WORKFLOW_SHA || '';
  if (![baseSha, workflowSha, pr.head.sha].every((sha) => /^[0-9a-f]{40}$/.test(sha))) {
    throw new Error('无法确定本次 NPU CI 的 base、head 或可信 CI commit。');
  }
  if (!pr.head.repo?.full_name) throw new Error('无法确定 PR 源码仓库。');

  // A previous success is reusable only for the same source merge and CI version.
  const requestKey = buildRequestKey({ ciMode, ops, baseSha, workflowSha });
  const outputs = {
    pr_number: String(prNumber), base_ref: pr.base.ref, base_sha: baseSha, head_sha: pr.head.sha,
    trusted_ci_sha: workflowSha, ci_mode: ciMode, ops, request_key: requestKey, use_pr_ci: 'false',
  };
  const checks = ops ? [SCOPED_CHECK] : REQUIRED_CHECKS;
  const runUrl = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  const details = [
    `- 模式：\`${ciMode}\``, `- 触发人：@${requestedBy}`,
    `- CI 定义分支：\`${process.env.NPU_CI_WORKFLOW_REF || ''}\``,
    '- 验证平台：A2 (`ascend910b`) + A5 (`ascend950`)',
    ops ? `- 定向编译诊断：\`${ops}\`；仅执行 01-02，不改写正式门禁状态` : '',
  ].filter(Boolean);
  const comment = async (title, extra = []) => createComment({
    github, context, pr, body: [title, '', ...details, ...extra].join('\n'),
  });
  const finish = (shouldRun) => {
    for (const [name, value] of Object.entries(outputs)) core.setOutput(name, value);
    core.setOutput('should_run', String(shouldRun));
  };

  if (await isMarkdownOnly({ github, context, pr })) {
    await writeStatuses({ github, context, pr, checks, state: 'success',
      description: '仅修改 Markdown，跳过 NPU CI', targetUrl: runUrl });
    await comment(`PR \`${pr.head.sha}\` 仅修改 Markdown，NPU CI 直接通过，不占用 NPU。`);
    finish(false);
    return;
  }

  const signature = `(${ciMode},${requestKey},A2+A5)`;
  const statuses = await github.paginate(github.rest.repos.listCommitStatusesForRef, {
    ...context.repo, ref: pr.head.sha, per_page: 100,
  });
  const latest = latestByContext(statuses);
  const finalStatus = latest.get(checks.at(-1)[0]);
  const matchesRequest = finalStatus?.description?.includes(signature);
  const completeSuccess = finalStatus?.target_url && checks.every(([name]) =>
    latest.get(name)?.state === 'success' && latest.get(name)?.target_url === finalStatus.target_url);
  if (process.env.NPU_CI_RUN_ATTEMPT === '1' && matchesRequest && completeSuccess) {
    await comment(`本次 NPU CI 请求已经在 \`${pr.head.sha}\` 上通过，不会重复运行。`,
      [`- 已有结果：[查看 NPU CI](${finalStatus.target_url})`]);
    finish(false);
    return;
  }
  if (matchesRequest && await isActiveRun({ github, context, core, status: finalStatus })) {
    await comment(`NPU CI 已在验证 \`${pr.head.sha}\`，本次请求不会重复启动新任务。`,
      [`- 运行链接：[查看已有 NPU CI](${finalStatus.target_url})`]);
    finish(false);
    return;
  }

  outputs.use_pr_ci = String(await prAuthorIsAdmin({ github, context, core, pr }));
  await assertUnchangedPullRequest({ github, context, pr });
  await writeStatuses({ github, context, pr, checks, state: 'pending',
    description: (label) => `${label}等待执行 ${signature}`, targetUrl: runUrl });
  await comment(`NPU CI 已开始验证 \`${pr.head.sha}\`。`, [
    outputs.use_pr_ci === 'true' ? '- CI 脚本：PR 作者为仓库 Admin，使用当前 PR 中的 CI' : '- CI 脚本：使用固定 workflow commit 的可信 CI',
    '- 执行方式：按顺序执行；前置失败时后续项跳过',
    `- 运行链接：[NPU CI #${context.runNumber}](${runUrl})`,
  ]);
  finish(true);
};
