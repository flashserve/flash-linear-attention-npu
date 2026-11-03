'use strict';

const { createHash } = require('crypto');

// Keep the status names shared by default-status and manual preparation.
const REQUIRED_CHECKS = [
  ['NPU CI / A2+A5 / 01 环境、wheel 与运行时契约', '环境、wheel 与运行时契约'],
  ['NPU CI / A2+A5 / 02 全量 OPP 构建', '全量 OPP 构建'],
  ['NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局', 'torch_custom wheel 与 OPP 布局'],
  ['NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配', 'OPP 安装与 PyTorch 适配'],
  ['NPU CI / A2+A5 / 05 GDR Example/ST', 'GDR Example/ST'],
  ['NPU CI / A2+A5 / 06 报告与 commit 校验', '报告与 commit 校验'],
];
const SCOPED_CHECK = ['NPU CI / A2+A5 / 定向诊断', '定向诊断'];
const USERS_PATH = '.github/npu-ci-users.json';

function decodeFile(file, path) {
  if (!file || Array.isArray(file) || file.type !== 'file' || !file.content) {
    throw new Error(`${path} 不是可读取的文件。`);
  }
  return Buffer.from(file.content, file.encoding || 'base64').toString('utf8');
}

async function defaultBranch({ github, context }) {
  const supplied = context.payload.repository?.default_branch;
  if (supplied) return supplied;
  const { data } = await github.rest.repos.get(context.repo);
  if (!data.default_branch) throw new Error('无法确定仓库默认分支。');
  return data.default_branch;
}

async function loadCiUsers({ github, context }) {
  // Never read the allowlist from the PR, workflow_dispatch ref, or local checkout.
  const ref = await defaultBranch({ github, context });
  const { data } = await github.rest.repos.getContent({ ...context.repo, path: USERS_PATH, ref });
  const config = JSON.parse(decodeFile(data, USERS_PATH));
  if (!Array.isArray(config.users) || !config.users.length ||
      config.users.some((user) => typeof user !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9-]*$/.test(user))) {
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

async function assertUnchangedPullRequest({ github, context, pr }) {
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  if (current.head.sha !== pr.head.sha || current.base.sha !== pr.base.sha ||
      current.changed_files !== pr.changed_files) {
    throw new Error(`PR #${pr.number} 在检查过程中已变化，请重新触发。`);
  }
}

async function isMarkdownOnly({ github, context, pr }) {
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo, pull_number: pr.number, per_page: 100,
  });
  // GitHub limits this endpoint to 3,000 files; do not trust a truncated list.
  if (!Number.isInteger(pr.changed_files) || pr.changed_files < 0 ||
      files.length !== pr.changed_files || new Set(files.map((file) => file.filename)).size !== files.length) {
    throw new Error(`PR #${pr.number} 修改文件列表不完整，无法判断是否仅修改 Markdown。`);
  }
  const markdownOnly = files.length > 0 && files.every((file) =>
    typeof file.filename === 'string' && /\.md$/i.test(file.filename) &&
    (file.status !== 'renamed' || (typeof file.previous_filename === 'string' && /\.md$/i.test(file.previous_filename))) &&
    (!file.previous_filename || /\.md$/i.test(file.previous_filename)));
  await assertUnchangedPullRequest({ github, context, pr });
  return markdownOnly;
}

function normalizeOps(value = '') {
  if (typeof value !== 'string') throw new Error('ops 参数必须是字符串。');
  const items = value.trim() ? value.split(',').map((item) => item.trim()) : [];
  if (items.some((item) => !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(item))) {
    throw new Error('ops 参数必须是以逗号分隔的合法非空算子名。');
  }
  return [...new Set(items)].sort().join(',');
}

function buildRequestKey({ ciMode, ops, baseSha, workflowSha }) {
  const digest = createHash('sha256').update(JSON.stringify([ciMode, ops, baseSha, workflowSha])).digest('hex').slice(0, 16);
  return `${ops ? 'scoped' : 'all'}-${digest}`;
}

async function createComment({ github, context, pr, body }) {
  return github.rest.issues.createComment({
    ...context.repo,
    issue_number: pr.number,
    body: `<!-- npu-ci:pr=${pr.number}:sha=${pr.head.sha} -->\n${body}`,
  });
}

async function writeStatuses({ github, context, pr, checks, state, description, targetUrl }) {
  await Promise.all(checks.map(([statusContext, label]) => github.rest.repos.createCommitStatus({
    ...context.repo, sha: pr.head.sha, state, context: statusContext,
    description: typeof description === 'function' ? description(label) : description,
    target_url: targetUrl,
  })));
}

module.exports = {
  REQUIRED_CHECKS, SCOPED_CHECK, USERS_PATH, decodeFile, defaultBranch,
  loadCiUsers, requireAuthorizedCiUser, assertUnchangedPullRequest, isMarkdownOnly,
  normalizeOps, buildRequestKey, createComment, writeStatuses,
};
