"use strict";

const { createHash } = require('crypto');
const USERS_PATH = '.github/npu-ci-users.json';

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
  const users = await requireAuthorizedCiUser({ github, context, username: requestedBy });
  const subjects = [requestedBy];
  if (Number(process.env.NPU_CI_RUN_ATTEMPT || '1') > 1) {
    const rerunBy = process.env.NPU_CI_TRIGGERING_ACTOR || '';
    if (!users.some((user) => user.toLowerCase() === rerunBy.toLowerCase())) {
      throw new Error(`${rerunBy || '(空用户名)'} 无权重新运行 NPU CI：须列入默认分支的 ${USERS_PATH}。`);
    }
    subjects.push(rerunBy);
  }
  return [...new Set(subjects)];
}

function normalizeCiModified(value = 'false') {
  if (value === true || value === 'true') return true;
  if (value === false || value === 'false' || value === '') return false;
  throw new Error('ci_modified 只接受 true 或 false；CI 文件变更须由管理员添加 ci_modified=true。');
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
  if (current.head.sha !== pr.head.sha || current.base.sha !== pr.base.sha || current.base.ref !== pr.base.ref ||
      current.changed_files !== pr.changed_files) throw new Error(`PR #${pr.number} 在检查过程中已变化，请重新触发。`);
  return files.some((file) => [file.filename, file.previous_filename].filter(Boolean).some((path) =>
    !/\.md$/i.test(path) && /^(?:ci|\.github)(?:\/|$)/.test(path)));
}

async function requireCiModificationApproval({ github, context, usernames, confirmed }) {
  if (!confirmed) throw new Error('PR 修改了 ci/ 或 .github/，必须由名单内仓库 Admin 使用 /run-npu-ci quick ci_modified=true 确认后重新触发。');
  for (const username of [...new Set(usernames)]) {
    let permission;
    try {
      const { data } = await github.rest.repos.getCollaboratorPermissionLevel({ ...context.repo, username });
      permission = data.permission;
    } catch (error) {
      throw new Error(`无法确认 @${username} 的仓库 Admin 身份：${error.message}。请恢复权限读取后由管理员带 ci_modified=true 重新触发。`);
    }
    if (permission !== 'admin') throw new Error(`@${username} 不是仓库 Admin，不能确认 CI 文件变更；请由名单内管理员带 ci_modified=true 重新触发。`);
  }
}

function buildCiApprovalKey({ prNumber, baseRef, baseSha, headSha }) {
  return createHash('sha256').update(JSON.stringify([prNumber, baseRef, baseSha, headSha])).digest('hex').slice(0, 12);
}

function hasApprovedRun({ statuses, contexts, context, pr }) {
  const marker = `ci-ok=${buildCiApprovalKey({ prNumber: pr.number, baseRef: pr.base.ref,
    baseSha: pr.base.sha, headSha: pr.head.sha })}`;
  const latest = new Map();
  for (const status of [...statuses].sort((a, b) =>
    new Date(b.created_at) - new Date(a.created_at) || Number(b.id || 0) - Number(a.id || 0))) {
    if (!latest.has(status.context)) latest.set(status.context, status);
  }
  const current = contexts.map((name) => latest.get(name));
  const runUrl = current[0]?.target_url || '';
  const prefix = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/`;
  return runUrl.startsWith(prefix) && /^\d+$/.test(runUrl.slice(prefix.length)) &&
    current.every((status) => status?.target_url === runUrl && status.description?.split(/\s+/).includes(marker));
}

module.exports = { USERS_PATH, loadCiUsers, requireAuthorizedCiUser, authorizeCiRequest,
  normalizeCiModified, hasCiChanges, requireCiModificationApproval, buildCiApprovalKey, hasApprovedRun };
