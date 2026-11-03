'use strict';

const {
  REQUIRED_CHECKS, decodeFile, defaultBranch, changedFiles, changedPaths, isDocumentationPath,
  createComment, writeStatuses,
} = require('./npu_ci_policy');

async function branchContexts({ github, context, core, pr }) {
  const main = await defaultBranch({ github, context });
  if (pr.base.ref === main) return REQUIRED_CHECKS.map(([name]) => name);
  // Maintenance branches may still use older status profiles. Read their names
  // as data, without executing that branch's JavaScript.
  for (const path of ['.github/workflows/npu-ci-default-status.yml', '.github/scripts/npu_ci_policy.js']) {
    try {
      const { data } = await github.rest.repos.getContent({ ...context.repo, path, ref: pr.base.ref });
      const text = decodeFile(data, path);
      const names = [...text.matchAll(/'NPU CI \/ [^']+'/g)].map((match) => match[0].slice(1, -1));
      const contexts = [...new Set(names)].filter((name) => !name.endsWith('/ 定向诊断'));
      if (contexts.length) return contexts;
    } catch (error) {
      core.warning(`无法读取 ${pr.base.ref}:${path}：${error.message}`);
    }
  }
  throw new Error(`无法确定 base 分支 \`${pr.base.ref}\` 的 NPU CI 门禁。`);
}

module.exports = async function defaultNpuCiStatus({ github, context, core }) {
  const pr = context.payload.pull_request;
  if (!pr) return;
  const contexts = await branchContexts({ github, context, core, pr });
  const checks = contexts.map((name) => [name, name]);
  const paths = changedPaths(await changedFiles({ github, context, pr }));
  if (paths.length > 0 && paths.every(isDocumentationPath)) {
    await writeStatuses({ github, context, pr, checks, state: 'success',
      description: '仅修改文档或 LICENSES，跳过 NPU CI', targetUrl: pr.html_url });
    await createComment({ github, context, pr,
      body: `PR \`${pr.head.sha}\` 仅修改 .md、docs/ 或 LICENSES/，${contexts.length} 项 NPU CI 门禁直接通过，不占用 NPU。` });
    return;
  }

  const statuses = await github.paginate(github.rest.repos.listCommitStatusesForRef, {
    ...context.repo, ref: pr.head.sha, per_page: 100,
  });
  const latest = new Map();
  for (const status of [...statuses].sort((a, b) =>
    new Date(b.created_at) - new Date(a.created_at) || Number(b.id || 0) - Number(a.id || 0))) {
    if (!latest.has(status.context)) latest.set(status.context, status);
  }
  const stale = [...latest.entries()].filter(([name, status]) =>
    name.startsWith('NPU CI / ') && !contexts.includes(name) && status.state === 'pending' &&
    !/\/actions\/runs\/\d+/.test(status.target_url || ''));
  await writeStatuses({ github, context, pr, checks: stale.map(([name]) => [name, name]), state: 'success',
    description: `不适用：base 分支 ${pr.base.ref} 的 NPU CI 不含此检查`, targetUrl: pr.html_url });
  // Statuses belong to a commit, so another PR/base may have marked this head
  // as Markdown-only. A non-Markdown diff must invalidate those shortcuts.
  const baseChanged = context.payload.action === 'edited';
  const pendingChecks = checks.filter(([name]) => {
    const status = latest.get(name);
    return baseChanged || !status ||
      (status.state === 'success' && ['仅修改 Markdown，跳过 NPU CI', '仅修改文档或 LICENSES，跳过 NPU CI'].includes(status.description));
  });
  if (!pendingChecks.length && [...latest.values()].some((status) =>
    contexts.includes(status.context) && /\/actions\/runs\/\d+/.test(status.target_url || ''))) {
    core.notice('当前 head commit 已经有 NPU CI 执行状态，保留执行结果。');
    return;
  }
  const commitUpdated = context.payload.action === 'synchronize';
  const description = baseChanged
    ? '未执行：PR 目标分支已变化，请重新触发 /run-npu-ci quick'
    : (commitUpdated ? '未执行：commit 已变化，请重新触发 /run-npu-ci quick' : '未执行：请仓库 Admin 评论 /run-npu-ci quick');
  await writeStatuses({ github, context, pr, checks: pendingChecks, state: 'pending', description,
    targetUrl: pr.html_url });
  await createComment({ github, context, pr, body: [
    `NPU CI 暂未执行：\`${pr.head.sha}\`。`, '',
    commitUpdated ? '- PR head commit 已变化，旧 commit 的 NPU CI 结果不再生效' : '',
    baseChanged ? '- PR 目标分支已变化，旧 base 的 NPU CI 结果不再生效' : '',
    `- CI 定义分支：\`${pr.base.ref}\``,
    `- 合入要求：当前 commit 必须通过 ${contexts.length} 个 NPU CI 分项状态`,
    '- 执行方式：A2/A5 并行；平台内部顺序执行，前置失败时后续项跳过',
    '- 触发命令：仓库 Admin 评论 `/run-npu-ci quick` 或 `/run-npu-ci full`',
  ].filter(Boolean).join('\n') });
};
