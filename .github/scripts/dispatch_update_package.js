'use strict';

const { requireAuthorizedCiUser } = require('./npu_ci_authorization');

function parseCommand(body) {
  const command = (body || '').trim().match(/^\/update-package(?:\s+(?:ci_ref|branch)=([^\s]+))?$/);
  return command ? { ref: command[1] || '' } : null;
}

module.exports = async function dispatchUpdatePackage({ github, context, core }) {
  const command = parseCommand(context.payload.comment.body);
  if (!command) {
    core.notice('使用 /update-package，可选 ci_ref=<CI定义分支>。');
    return;
  }
  await requireAuthorizedCiUser({ github, context, username: context.actor });
  if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) {
    await requireAuthorizedCiUser({ github, context, username: process.env.NPU_CI_TRIGGERING_ACTOR || '' });
  }
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: context.payload.issue.number });
  if (pr.state !== 'open') throw new Error('PR 已关闭；请在 Actions 手动刷新目标分支底包。');
  const ref = command.ref || pr.base.ref;
  await github.rest.repos.getBranch({ ...context.repo, branch: ref });
  await github.rest.actions.createWorkflowDispatch({
    ...context.repo, workflow_id: 'update-package.yml', ref,
    inputs: { base_ref: pr.base.ref, requested_by: context.actor },
  });
  const url = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/workflows/update-package.yml`;
  await github.rest.issues.createComment({ ...context.repo, issue_number: pr.number,
    body: `已请求刷新 A2/A5 的 \`${pr.base.ref}\` Daily 底包缓存。只下载并校验，不编译、不运行 NPU 测试。\n\n[查看刷新任务](${url})` });
};

module.exports.parseCommand = parseCommand;
