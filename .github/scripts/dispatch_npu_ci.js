'use strict';

const { requireAuthorizedCiUser, normalizeOps, decodeFile } = require('./npu_ci_policy');

function parseCommand(body) {
  const command = (body || '').trim().match(/^\/run-npu-ci(?:\s+([\s\S]*))?$/);
  if (!command) return null;
  const result = { ciMode: 'quick', ops: '', requestedRef: '' };
  const seen = new Set();
  for (const part of (command[1] || '').trim().split(/\s+/).filter(Boolean)) {
    let key;
    if (part === 'quick' || part === 'full') {
      key = 'ciMode';
      result.ciMode = part;
    } else if (part.startsWith('ops=')) {
      key = 'ops';
      if (!part.slice(4).trim()) throw new Error('ops 参数不能为空。');
      result.ops = normalizeOps(part.slice(4));
    } else if (part.startsWith('ci_ref=') || part.startsWith('branch=')) {
      key = 'requestedRef';
      result.requestedRef = part.slice(part.indexOf('=') + 1);
      if (!result.requestedRef) throw new Error('CI 定义分支不能为空。');
    } else {
      throw new Error(`不支持的 NPU CI 参数：${part}`);
    }
    if (seen.has(key)) throw new Error(`NPU CI 参数 ${key} 只能指定一次。`);
    seen.add(key);
  }
  return result;
}

module.exports = async function dispatchNpuCi({ github, context, core }) {
  const command = parseCommand(context.payload.comment.body);
  if (!command) {
    core.notice('评论包含 /run-npu-ci，但命令格式不正确，跳过。');
    return;
  }
  const users = await requireAuthorizedCiUser({ github, context, username: context.actor });
  if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) {
    // A rerun preserves the original comment actor. Check the rerun requester
    // before creating a new workflow_dispatch (whose attempt starts at 1).
    const rerunBy = process.env.NPU_CI_TRIGGERING_ACTOR || '';
    if (!users.some((user) => user.toLowerCase() === rerunBy.toLowerCase())) {
      throw new Error(`${rerunBy || '(空用户名)'} 无权重新运行 NPU CI 评论分发：须列入默认分支的 .github/npu-ci-users.json。`);
    }
  }
  const prNumber = context.payload.issue.number;
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: prNumber });
  if (pr.state !== 'open') throw new Error(`PR #${prNumber} 已关闭。`);
  if (pr.draft) throw new Error(`PR #${prNumber} 仍是草稿，请先标记为 ready for review。`);
  const targetRef = command.requestedRef || pr.base.ref;
  await github.rest.repos.getBranch({ ...context.repo, branch: targetRef });
  const path = '.github/workflows/ci.yml';
  const { data } = await github.rest.repos.getContent({ ...context.repo, path, ref: targetRef });
  if (!/^\s*workflow_dispatch\s*:/m.test(decodeFile(data, path))) {
    throw new Error(`分支 \`${targetRef}\` 的 ci.yml 没有声明 workflow_dispatch。`);
  }
  await github.rest.actions.createWorkflowDispatch({
    ...context.repo, workflow_id: 'ci.yml', ref: targetRef,
    inputs: { pr_number: String(prNumber), ci_mode: command.ciMode, ops: command.ops, requested_by: context.actor },
  });
  core.notice(`已将 PR #${prNumber} 的 NPU CI 分发到 \`${targetRef}\`，head=${pr.head.sha}，模式=${command.ciMode}。`);
};
