'use strict';

const {
  REQUIRED_CHECKS, SCOPED_CHECK, requireAuthorizedCiUser, normalizeOps, normalizeAtkOps, normalizeDeviceId,
  normalizeCiModified, changedFiles, changedPaths, isDocumentationPath,
  requireCiModificationApproval, createComment, writeStatuses, decodeFile,
} = require('./npu_ci_policy');

function parseCommand(body) {
  const command = (body || '').trim().match(/^\/run-npu-ci(?:\s+([\s\S]*))?$/);
  if (!command) return null;
  const result = { ciMode: 'quick', ops: '', atkOps: '', deviceId: '', ciModified: false, requestedRef: '' };
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
    } else if (part.startsWith('atk-ops=')) {
      key = 'atkOps';
      if (!part.slice(8)) throw new Error('atk-ops 参数不能为空，请提供逗号分隔的算子名，或使用 atk-ops=null 跳过。');
      result.atkOps = normalizeAtkOps(part.slice(8));
    } else if (part.startsWith('device-id=')) {
      key = 'deviceId';
      if (!part.slice(10)) throw new Error('device-id 参数不能为空，请提供非负整数。');
      result.deviceId = normalizeDeviceId(part.slice(10));
    } else if (part.startsWith('ci_modified=')) {
      key = 'ciModified';
      if (!part.slice(12)) throw new Error('ci_modified 只接受 true 或 false。');
      result.ciModified = normalizeCiModified(part.slice(12));
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
  if (result.ciMode === 'full' && result.ops) {
    throw new Error('full 与 ops= 定向诊断不能同时使用；请使用 full，或 quick ops=...。');
  }
  if (result.ops && result.atkOps) {
    throw new Error('ops= 定向编译诊断与 atk-ops= 不能同时使用；需要 ATK 验证时请使用 /run-npu-ci quick atk-ops=...。');
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
  const files = await changedFiles({ github, context, pr });
  const ciModified = changedPaths(files).some((path) =>
    !isDocumentationPath(path) && /^(?:ci|\.github)(?:\/|$)/.test(path));
  if (ciModified) {
    try {
      const usernames = [context.actor];
      if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) usernames.push(process.env.NPU_CI_TRIGGERING_ACTOR || '');
      await requireCiModificationApproval({ github, context, usernames, confirmed: command.ciModified });
    } catch (error) {
      await writeStatuses({ github, context, pr, checks: command.ops ? [SCOPED_CHECK] : REQUIRED_CHECKS,
        state: 'failure', description: 'CI 文件变更未经管理员 ci_modified=true 确认', targetUrl: pr.html_url });
      await createComment({ github, context, pr, body: `NPU CI 拒绝分发。\n\n- 原因及修改建议：${error.message}` });
      throw error;
    }
  }
  const targetRef = command.requestedRef || pr.base.ref;
  await github.rest.repos.getBranch({ ...context.repo, branch: targetRef });
  const path = '.github/workflows/ci.yml';
  const { data } = await github.rest.repos.getContent({ ...context.repo, path, ref: targetRef });
  if (!/^\s*workflow_dispatch\s*:/m.test(decodeFile(data, path))) {
    throw new Error(`分支 \`${targetRef}\` 的 ci.yml 没有声明 workflow_dispatch。`);
  }
  await github.rest.actions.createWorkflowDispatch({
    ...context.repo, workflow_id: 'ci.yml', ref: targetRef,
    inputs: {
      pr_number: String(prNumber), ci_mode: command.ciMode, ops: command.ops,
      atk_ops: command.atkOps, device_id: command.deviceId,
      ci_modified: String(command.ciModified), requested_by: context.actor,
    },
  });
  core.notice(`已将 PR #${prNumber} 的 NPU CI 分发到 \`${targetRef}\`，head=${pr.head.sha}，模式=${command.ciMode}。`);
};
