"use strict";

const CODEOWNERS_PATH = '.github/CODEOWNERS';
const STATUS_CONTEXT = 'Code Owner Approval / 2 designated reviewers';
const REQUIRED_APPROVALS = 2;
const EXPECTED_CODEOWNERS = Object.freeze([
  'weinachuan',
  'weiwei-612',
  'chen-linxin',
  'Ensley0304',
  'zhangshuolei-hfut',
]);

const permissionRank = Object.freeze({
  admin: 4,
  maintain: 3,
  push: 2,
  triage: 1,
  pull: 0,
  none: -1,
});

function normalizeLogin(login) {
  return typeof login === 'string' ? login.trim().toLowerCase() : '';
}

function parseCodeowners(text) {
  if (typeof text !== 'string' || !text.trim()) {
    throw new Error(`${CODEOWNERS_PATH} 为空，无法确定 Code Owner 审批名单。`);
  }

  let globalOwners = null;
  const allOwners = new Map();
  for (const [index, rawLine] of text.split(/\r?\n/).entries()) {
    const line = rawLine.replace(/\s+#.*$/, '').trim();
    if (!line || line.startsWith('#')) continue;

    const parts = line.split(/\s+/);
    if (parts.length < 2) {
      throw new Error(`${CODEOWNERS_PATH}:${index + 1} 没有 Code Owner。`);
    }
    const pattern = parts[0];
    const owners = parts.slice(1);
    const lineOwners = [];
    for (const ownerToken of owners) {
      if (!/^@[A-Za-z0-9][A-Za-z0-9-]*$/.test(ownerToken)) {
        throw new Error(`${CODEOWNERS_PATH}:${index + 1} 包含非用户形式的 Code Owner：${ownerToken}。`);
      }
      const login = ownerToken.slice(1);
      const normalized = normalizeLogin(login);
      lineOwners.push(login);
      if (!allOwners.has(normalized)) allOwners.set(normalized, login);
    }
    if (pattern === '*') globalOwners = lineOwners;
  }

  if (!globalOwners || !globalOwners.length) {
    throw new Error(`${CODEOWNERS_PATH} 缺少有效的全仓 '*' Code Owner 规则。`);
  }

  const expected = new Set(EXPECTED_CODEOWNERS.map(normalizeLogin));
  const actualGlobal = new Set(globalOwners.map(normalizeLogin));
  if (actualGlobal.size !== expected.size || [...expected].some((login) => !actualGlobal.has(login))) {
    throw new Error(`全仓 Code Owner 必须且只能是：${EXPECTED_CODEOWNERS.join('、')}。`);
  }
  if (allOwners.size !== expected.size || [...allOwners.keys()].some((login) => !expected.has(login))) {
    throw new Error('所有路径级 Code Owner 都必须属于指定用户，不能引入其他账号或团队。');
  }
  return [...EXPECTED_CODEOWNERS];
}

function computeActiveApprovals(reviews, allowedOwners) {
  if (!Array.isArray(reviews)) throw new Error('reviews 必须是数组。');
  const allowed = new Set(allowedOwners.map(normalizeLogin));
  const states = new Map();
  const ordered = reviews
    .map((review, index) => ({ review, index }))
    .sort((left, right) => {
      const leftTime = Date.parse(left.review.submitted_at || left.review.submittedAt || '') || 0;
      const rightTime = Date.parse(right.review.submitted_at || right.review.submittedAt || '') || 0;
      return leftTime - rightTime || left.index - right.index;
    });

  for (const { review } of ordered) {
    const login = normalizeLogin(review.user?.login || review.author?.login);
    if (!allowed.has(login)) continue;
    const state = String(review.state || '').toUpperCase();
    if (state === 'APPROVED') {
      states.set(login, true);
    } else if (state === 'CHANGES_REQUESTED' || state === 'DISMISSED') {
      states.set(login, false);
    }
  }
  return allowedOwners.filter((login) => states.get(normalizeLogin(login)) === true);
}

function requireReviewerWritePermissions(permissions) {
  for (const [login, permission] of Object.entries(permissions)) {
    if ((permissionRank[String(permission).toLowerCase()] ?? -1) < permissionRank.push) {
      throw new Error(`Code Owner @${login} 当前仓库权限为 ${permission || 'none'}，至少需要 write 权限。`);
    }
  }
}

function getTargetBranchRef(pr) {
  const ref = pr?.base?.ref;
  if (typeof ref !== 'string' || !ref.trim()) {
    throw new Error('无法确定 PR 当前目标分支，不能读取 CODEOWNERS。');
  }
  return ref;
}

function truncateDescription(value) {
  return String(value).replace(/\s+/g, ' ').trim().slice(0, 140);
}

async function setStatus({ github, context, core, sha, prUrl, state, description }) {
  await github.rest.repos.createCommitStatus({
    ...context.repo,
    sha,
    state,
    context: STATUS_CONTEXT,
    description: truncateDescription(description),
    target_url: prUrl,
  });
  core.info(`${STATUS_CONTEXT}: ${state} - ${description}`);
}

async function run({ github, context, core }) {
  const eventPr = context.payload.pull_request;
  if (!eventPr) {
    core.info('事件中没有 pull_request，跳过 Code Owner approval 检查。');
    return;
  }

  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: eventPr.number,
  });
  if (pr.state !== 'open' || pr.merged || pr.merged_at) {
    core.info(`PR #${pr.number} 已关闭或合入，跳过状态写入。`);
    return;
  }

  try {
    const { data: codeownersFile } = await github.rest.repos.getContent({
      ...context.repo,
      path: CODEOWNERS_PATH,
      // PR base.sha can remain at the original merge base for long-lived PRs.
      // Read the current target branch so workflow/CODEOWNERS upgrades apply.
      ref: getTargetBranchRef(pr),
    });
    if (!codeownersFile || Array.isArray(codeownersFile) ||
        codeownersFile.type !== 'file' || !codeownersFile.content) {
      throw new Error(`${CODEOWNERS_PATH} 不是目标分支中的可读取文件。`);
    }
    const codeownersText = Buffer.from(
      codeownersFile.content,
      codeownersFile.encoding || 'base64',
    ).toString('utf8');
    const allowedOwners = parseCodeowners(codeownersText);

    const permissionEntries = await Promise.all(allowedOwners.map(async (login) => {
      const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
        ...context.repo,
        username: login,
      });
      return [login, data.permission || 'none'];
    }));
    requireReviewerWritePermissions(Object.fromEntries(permissionEntries));

    const reviews = await github.paginate(github.rest.pulls.listReviews, {
      ...context.repo,
      pull_number: pr.number,
      per_page: 100,
    });
    const approvedOwners = computeActiveApprovals(reviews, allowedOwners);

    const { data: current } = await github.rest.pulls.get({
      ...context.repo,
      pull_number: pr.number,
    });
    if (current.head.sha !== pr.head.sha || current.base.sha !== pr.base.sha) {
      core.info(`PR #${pr.number} 在检查过程中更新，等待新事件重新计算。`);
      return;
    }

    let state;
    let description;
    if (current.draft) {
      state = 'pending';
      description = '草稿 PR 不计入门禁；两名指定 Code Owner approval 后通过。';
    } else if (approvedOwners.length >= REQUIRED_APPROVALS) {
      state = 'success';
      description = `已获得 ${approvedOwners.length} 名指定 Code Owner approval。`;
    } else {
      state = 'pending';
      description = `指定 Code Owner approval ${approvedOwners.length}/${REQUIRED_APPROVALS}；其他账号 approval 不计入。`;
    }
    await setStatus({
      github, context, core, sha: current.head.sha, prUrl: current.html_url, state, description,
    });
  } catch (error) {
    try {
      await setStatus({
        github,
        context,
        core,
        sha: pr.head.sha,
        prUrl: pr.html_url,
        state: 'failure',
        description: `审批策略检查失败：${error.message}`,
      });
    } catch (statusError) {
      core.warning(`无法写入 Code Owner approval 失败状态：${statusError.message}`);
    }
    core.setFailed(error.message);
  }
}

module.exports = {
  CODEOWNERS_PATH,
  STATUS_CONTEXT,
  REQUIRED_APPROVALS,
  EXPECTED_CODEOWNERS,
  parseCodeowners,
  computeActiveApprovals,
  requireReviewerWritePermissions,
  getTargetBranchRef,
  run,
};
