'use strict';

const { createHash } = require('crypto');
const { requireAuthorizedCiUser } = require('./npu_ci_authorization');

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

async function assertUnchangedPullRequest({ github, context, pr }) {
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  if (current.head.sha !== pr.head.sha || current.base.sha !== pr.base.sha || current.base.ref !== pr.base.ref ||
      current.changed_files !== pr.changed_files) {
    throw new Error(`PR #${pr.number} 在检查过程中已变化，请重新触发。`);
  }
}

function isDocumentationPath(path) {
  return /\.md$/i.test(path) || /^(?:docs|LICENSES)(?:\/|$)/.test(path);
}

async function changedFiles({ github, context, pr }) {
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo, pull_number: pr.number, per_page: 100,
  });
  // GitHub limits this endpoint to 3,000 files; do not trust a truncated list.
  if (!Number.isInteger(pr.changed_files) || pr.changed_files < 0 ||
      files.length !== pr.changed_files || new Set(files.map((file) => file.filename)).size !== files.length) {
    throw new Error(`PR #${pr.number} 修改文件列表不完整，请缩小 PR 或修复 GitHub 文件列表读取后重新触发。`);
  }
  for (const file of files) {
    if (typeof file.filename !== 'string' || !file.filename ||
        (file.status === 'renamed' && !file.previous_filename) ||
        (file.previous_filename !== undefined && typeof file.previous_filename !== 'string')) {
      throw new Error(`PR #${pr.number} 修改文件路径不完整，无法安全计算编译和 ATK 范围，请重新触发。`);
    }
  }
  await assertUnchangedPullRequest({ github, context, pr });
  return files;
}

function changedPaths(files) {
  // Both sides of a rename affect the execution plan.
  return [...new Set(files.flatMap((file) => [file.filename, file.previous_filename].filter(Boolean)))];
}

async function headTree({ github, context, pr }) {
  const { data } = await github.rest.git.getTree({ ...context.repo, tree_sha: pr.head.sha, recursive: '1' });
  if (data.truncated || !Array.isArray(data.tree)) {
    throw new Error('PR 目录树不完整，无法确定算子编译和 ATK 范围，请缩小 PR 或修复目录树读取。');
  }
  return data.tree;
}

function operatorRoots(tree) {
  const roots = new Set();
  for (const entry of tree) {
    const match = entry.path?.match(/^(fla\/ops\/ascendc\/.+)\/op_host\/CMakeLists\.txt$/);
    if (match && !match[1].split('/').includes('tests')) roots.add(match[1]);
  }
  // A nested op_host is a build root only with its own CMakeLists.txt.
  // Match the closest such root, including independently built internal operators.
  return [...roots].sort((a, b) => b.length - a.length || a.localeCompare(b));
}

function allAtkOperators(tree) {
  const operators = new Set();
  for (const entry of tree) {
    const match = entry.path?.match(/^tests\/atk\/([^/]+)\//);
    if (match && match[1] !== 'common') operators.add(match[1]);
  }
  if (!operators.size) throw new Error('共享 ATK 文件发生修改，但 PR 中没有可识别的 ATK 算子目录，请补齐 tests/atk/<算子>/。');
  return [...operators].sort();
}

function classifyChanges(files, { ciMode = 'quick', ops = '', atkOps = '', deviceId = '', allAtkOps = [], opRoots = [] } = {}) {
  const paths = changedPaths(files);
  const documentationOnly = paths.length > 0 && paths.every(isDocumentationPath);
  let fullBuild = ciMode === 'full';
  let torchModified = false;
  let sharedAtkModified = false;
  const requestedOps = normalizeOps(ops);
  const requestedAtkOps = normalizeAtkOps(atkOps);
  const skipAtk = requestedAtkOps === 'null';
  const compiledOps = new Set(requestedOps.split(',').filter(Boolean));
  const testedOps = new Set(skipAtk ? [] : requestedAtkOps.split(',').filter(Boolean));
  for (const path of paths) {
    if (isDocumentationPath(path)) continue;
    if (/^(?:ci|\.github)(?:\/|$)/.test(path)) continue;
    if (/^(?:common|cmake)(?:\/|$)/.test(path)) { fullBuild = true; continue; }
    if (path.startsWith('fla/ops/ascendc/')) {
      if (requestedOps) continue;
      // Discover roots in the fixed PR tree, rather than assuming group depth.
      // Shared files and removed operators without an op_host require a full build.
      const root = opRoots.find((candidate) => path.startsWith(`${candidate}/`));
      if (root) {
        const op = root.split('/').at(-1);
        compiledOps.add(op);
        testedOps.add(op);
      } else fullBuild = true;
      continue;
    }
    if (path.startsWith('tests/atk/')) {
      const parts = path.slice('tests/atk/'.length).split('/');
      if (parts.length > 1 && parts[0] !== 'common') testedOps.add(parts[0]);
      else sharedAtkModified = true;
      continue;
    }
    if (/^torch_custom(?:\/|$)/.test(path)) { torchModified = true; continue; }
    if (/^(?:examples|tests)(?:\/|$)/.test(path)) continue;
    fullBuild = true;
  }
  for (const op of compiledOps) testedOps.add(op);
  if (sharedAtkModified && !skipAtk) for (const op of allAtkOps) testedOps.add(op);
  // An OPP-only overlay would otherwise leave the Daily adapter installed.
  if (compiledOps.size && torchModified) fullBuild = true;
  // The legacy ops= request is an isolated 01-02 diagnostic. It compiles only
  // explicitly named operators and never silently skips a requested ATK run.
  if (requestedOps) fullBuild = false;
  const buildKind = fullBuild ? 'full' : compiledOps.size ? 'incremental' : torchModified ? 'torch' : 'base';
  return {
    documentation_only: documentationOnly,
    full_build: fullBuild,
    build_ops: requestedOps || (fullBuild ? '' : normalizeOps([...compiledOps].join(','))),
    atk_ops: requestedOps || skipAtk ? '' : normalizeOps([...testedOps].join(',')),
    build_kind: buildKind,
    device_id: normalizeDeviceId(deviceId),
    shared_atk_modified: !requestedOps && !skipAtk && sharedAtkModified,
  };
}

async function inspectChanges({ github, context, pr, ...options }) {
  const files = await changedFiles({ github, context, pr });
  const first = classifyChanges(files, options);
  const needsOperatorTree = !normalizeOps(options.ops || '') && changedPaths(files).some((path) =>
    !isDocumentationPath(path) && path.startsWith('fla/ops/ascendc/'));
  if (!needsOperatorTree && !first.shared_atk_modified) return first;
  const tree = await headTree({ github, context, pr });
  const opRoots = needsOperatorTree ? operatorRoots(tree) : [];
  const allAtkOps = first.shared_atk_modified ? allAtkOperators(tree) : [];
  await assertUnchangedPullRequest({ github, context, pr });
  return classifyChanges(files, { ...options, opRoots, allAtkOps });
}

function normalizeDeviceId(value = '') {
  const text = String(value);
  if (!text) return '';
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(Number(text))) {
    throw new Error('device-id 必须是非负整数，例如 device-id=0；请按 runner 可用 NPU 编号重新触发。');
  }
  return String(Number(text));
}

function normalizeOps(value = '') {
  if (typeof value !== 'string') throw new Error('ops 参数必须是字符串。');
  const items = value.trim() ? value.split(',').map((item) => item.trim()) : [];
  if (items.some((item) => !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(item))) {
    throw new Error('ops 参数必须是以逗号分隔的合法非空算子名。');
  }
  return [...new Set(items)].sort().join(',');
}

function normalizeAtkOps(value = '') {
  if (typeof value === 'string' && value.trim() === 'null') return 'null';
  const ops = normalizeOps(value);
  if (ops.split(',').includes('null')) {
    throw new Error('atk-ops=null 必须单独使用，不能与算子名混合；请仅使用 null 跳过 ATK，或提供算子列表。');
  }
  return ops;
}

function buildRequestKey({ ciMode, ops, baseRef, baseSha, workflowSha, plan }) {
  const digest = createHash('sha256').update(JSON.stringify([ciMode, ops, baseRef, baseSha, workflowSha, plan])).digest('hex').slice(0, 16);
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
  REQUIRED_CHECKS, SCOPED_CHECK, decodeFile, defaultBranch,
  requireAuthorizedCiUser, assertUnchangedPullRequest, changedFiles, changedPaths,
  isDocumentationPath,
  classifyChanges, inspectChanges, normalizeDeviceId,
  normalizeOps, normalizeAtkOps, buildRequestKey, createComment, writeStatuses,
};
