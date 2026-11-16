"use strict";

const USERS_PATH = '.github/npu-ci-users.json';

function requireOpenPullRequest(pr) {
  if (pr.state !== 'open' || pr.merged || pr.merged_at) {
    throw new Error(`PR #${pr.number} 已关闭或已合入，不能触发 NPU CI。`);
  }
  if (pr.draft) throw new Error(`PR #${pr.number} 仍是草稿，请先标记为 ready for review。`);
}

function requireProtectedBase({ pr, branch, workflowRef = pr.base.ref }) {
  requireOpenPullRequest(pr);
  if (workflowRef !== pr.base.ref || branch.protected !== true) {
    throw new Error('NPU CI 只允许在 PR 当前受保护的目标分支执行，不支持其他 CI 定义分支。');
  }
}

function requireDispatchIdentity({ pr, inputs, required = false, rerun = false }) {
  // The PR head and target branch identify the request. The target branch
  // commit may advance while a run is queued or being rerun; the device job
  // merges the current branch and lets Git report any real conflict.
  const expected = { expected_head_sha: pr.head.sha, expected_base_ref: pr.base.ref };
  if (rerun && Object.keys(expected).some((name) => !inputs[name])) {
    throw new Error('首次手动触发未保存可复用的完整版本输入，不能重新执行 prepare；请新建 Run workflow。已成功 prepare 的 Re-run failed jobs 可继续使用原版本。');
  }
  for (const [name, current] of Object.entries(expected)) {
    if ((required && !inputs[name]) || (inputs[name] && inputs[name] !== current)) {
      throw new Error(`NPU CI 触发快照 ${name} 缺失或已变化，请针对当前 head/base 重新触发。`);
    }
  }
}

function selectDispatchInputs(text, inputs) {
  // Current and maintenance workflows use this block-style YAML layout.
  // Read only input names, so each branch can keep its own CI interface.
  const lines = text.split(/\r?\n/);
  const start = lines.findIndex((line) => /^  workflow_dispatch:\s*(?:\{\s*\})?\s*(?:#.*)?$/.test(line));
  if (start < 0) throw new Error('目标分支的 CI workflow 没有可分发的 workflow_dispatch 入口。');
  const declared = new Set();
  let readingInputs = false;
  for (const line of lines.slice(start + 1)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    if (!/^ {4}/.test(line)) break;
    if (/^    inputs:\s*(?:#.*)?$/.test(line)) {
      readingInputs = true;
      continue;
    }
    if (readingInputs) {
      if (!/^ {6}/.test(line)) break;
      const match = line.match(/^      ['"]?([A-Za-z_][A-Za-z0-9_-]*)['"]?:/);
      if (match) declared.add(match[1]);
    }
  }
  return Object.fromEntries(Object.entries(inputs).filter(([name]) => declared.has(name)));
}

async function loadCiUsers({ github, context }) {
  let ref = context.payload.repository?.default_branch;
  if (!ref) ({ data: { default_branch: ref } } = await github.rest.repos.get(context.repo));
  if (!ref) throw new Error('无法确定仓库默认分支，拒绝触发 NPU CI。');
  // The PR and workflow-dispatch ref must never grant themselves permission.
  const { data } = await github.rest.repos.getContent({ ...context.repo, path: USERS_PATH, ref });
  if (!data || Array.isArray(data) || data.type !== 'file' || !data.content) {
    throw new Error(`${USERS_PATH} 不是可读取的权限文件。`);
  }
  const config = JSON.parse(Buffer.from(data.content, data.encoding || 'base64').toString('utf8'));
  if (!Array.isArray(config.users) || !config.users.length || config.users.some((user) =>
    typeof user !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9-]*$/.test(user))) {
    throw new Error(`${USERS_PATH} 必须包含非空的合法 GitHub 用户名 users 数组。`);
  }
  return [...new Map(config.users.map((user) => [user.toLowerCase(), user])).values()];
}

async function requireAuthorizedCiUser({ github, context, username }) {
  const users = await loadCiUsers({ github, context });
  if (typeof username !== 'string' || !users.some((user) => user.toLowerCase() === username.toLowerCase())) {
    throw new Error(`${username || '(空用户名)'} 无权触发 NPU CI：须列入默认分支的 ${USERS_PATH}。`);
  }
  return users;
}

async function authorizeCiRequest({ github, context, requestedBy = context.actor }) {
  // GitHub reruns continue the original authorized request. The device job
  // independently checks that its PR state and prepared commits still match.
  if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) return;
  await requireAuthorizedCiUser({ github, context, username: requestedBy });
}

async function hasCiChanges({ github, context, pr }) {
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo, pull_number: pr.number, per_page: 100,
  });
  if (!Number.isInteger(pr.changed_files) || pr.changed_files < 0 || files.length !== pr.changed_files ||
      new Set(files.map((file) => file.filename)).size !== files.length || files.some((file) =>
        typeof file.filename !== 'string' || !file.filename ||
        (file.status === 'renamed' && !file.previous_filename) ||
        (file.previous_filename !== undefined && typeof file.previous_filename !== 'string'))) {
    throw new Error(`PR #${pr.number} 修改文件列表不完整，无法确认 CI 文件变更；请缩小 PR 或修复文件列表读取后重试。`);
  }
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  requireOpenPullRequest(current);
  if (current.head.sha !== pr.head.sha || current.base.ref !== pr.base.ref ||
      current.changed_files !== pr.changed_files) throw new Error(`PR #${pr.number} 在检查过程中已变化，请重新触发。`);
  return files.some((file) => [file.filename, file.previous_filename].filter(Boolean).some((path) =>
    !/\.md$/i.test(path) && /^(?:ci|\.github)(?:\/|$)/.test(path)));
}

async function requireCiAuthorAdmin({ github, context, pr }) {
  const username = pr.user?.login;
  if (!username) throw new Error('无法确定 PR 作者，拒绝执行 CI 文件变更。');
  let permission;
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({ ...context.repo, username });
    permission = data.permission;
  } catch (error) {
    throw new Error(`无法确认 PR 作者 @${username} 的仓库 Admin 身份：${error.message}。请恢复权限读取后重试。`);
  }
  if (permission !== 'admin') {
    throw new Error(`PR 作者 @${username} 不是仓库 Admin；只有 Admin 提起的 PR 可以修改 ci/ 或 .github/ 的非 Markdown 文件。请由仓库 Admin 提交这些 CI 改动。`);
  }
}

module.exports = { USERS_PATH, loadCiUsers, requireAuthorizedCiUser, authorizeCiRequest,
  requireOpenPullRequest, requireProtectedBase, requireDispatchIdentity, selectDispatchInputs,
  hasCiChanges, requireCiAuthorAdmin };
