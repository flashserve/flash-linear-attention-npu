'use strict';

const { createHash } = require('node:crypto');

const TYPES = ['bug', 'enhancement', 'question'];
const FIELDS = ['Issue 类型', '涉及分支', '涉及算子或模块'];
const REQUIRED = {
  bug: ['环境信息', '最小复现', '完整报错', '期望行为'],
  enhancement: ['背景', '问题', '期望实现的功能', '验收要求'],
  question: ['背景', '问题', '期望实现的功能'],
};
const MARKER = '<!-- fla-npu-issue-policy:v1 -->';
const NEEDS_INFO = 'needs-info';
const EXEMPT_LABELS = ['committer-promotion', 'requirements-index'];

// Ignore headings in examples and fenced reproduction code. Keep the code itself
// as section content, since it is often the entire reproduction.
function sectionsOf(body) {
  const sections = new Map();
  const duplicates = new Set();
  let heading = null;
  let fence = null;
  for (const line of String(body || '').replace(/<!--[^]*?-->/g, '').split(/\r?\n/)) {
    const fenceMatch = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (fence) {
      if (heading) sections.get(heading).push(line);
      if (fenceMatch && fenceMatch[1][0] === fence[0] &&
          fenceMatch[1].length >= fence.length && /^\s*[`~]+\s*$/.test(line)) fence = null;
      continue;
    }
    if (fenceMatch) {
      fence = fenceMatch[1];
    } else {
      const match = line.match(/^\s{0,3}(#{1,3})\s+(.+?)\s*#*\s*$/);
      if (match) {
        heading = match[1].length === 3 ? match[2] : null;
        if (heading) {
          if (sections.has(heading)) duplicates.add(heading);
          else sections.set(heading, []);
        }
        continue;
      }
    }
    if (heading) sections.get(heading).push(line);
  }
  return { sections: new Map([...sections].map(([key, lines]) => [key, lines.join('\n').trim()])), duplicates };
}

function emptyAnswer(value) {
  const text = String(value || '').replace(/```[^\n]*\n?|~~~[^\n]*\n?/g, '').trim();
  return !text || /^(?:_No response_|TODO|TBD|待补充|待填写|\.{3}|…+)$/i.test(text);
}

function selections(value) {
  return [...new Set(String(value || '').split(/[,，、\n]/)
    .map((item) => item.trim().replace(/^-\s+/, '').replace(/^`([^`]+)`$/, '$1'))
    .filter(Boolean))];
}

function branchLabel(branch) {
  if (branch === 'main') return 'main';
  const name = `branch:${branch}`;
  return name.length <= 50 ? name : `branch:${branch.slice(0, 32)}:${createHash('sha256').update(branch).digest('hex').slice(0, 10)}`;
}

function validateIssue(body, branches) {
  const { sections, duplicates } = sectionsOf(body);
  const errors = [];
  const type = sections.get(FIELDS[0]);
  const required = TYPES.includes(type) ? REQUIRED[type] : [];
  for (const field of FIELDS) {
    if (emptyAnswer(sections.get(field))) errors.push(`请填写“${field}”。`);
  }
  if (!TYPES.includes(type)) errors.push(`“Issue 类型”必须是 ${TYPES.join(' / ')} 中的一项。`);
  const selectedBranches = selections(sections.get(FIELDS[1]));
  if (!selectedBranches.length) errors.push('至少选择一个实际存在的涉及分支。');
  if (branches) {
    for (const branch of selectedBranches) {
      if (!branches.includes(branch)) errors.push(`涉及分支 ${JSON.stringify(branch)} 不存在，请核对仓库分支列表。`);
    }
  }
  for (const field of required) {
    const value = sections.get(field);
    if (emptyAnswer(value)) errors.push(`请填写“${field}”，不能保留空白或占位文字。`);
    if (field === '验收要求' && /(?:按\s*\.{3}|[：:]\s*\.{3}|TODO|待补充)/i.test(value || '')) {
      errors.push('请将“验收要求”中的 ... / TODO / 待补充替换为实际验收口径。');
    }
  }
  for (const field of [...FIELDS, ...required]) {
    if (duplicates.has(field)) errors.push(`“${field}”重复，请只保留一个同名三级标题。`);
  }
  return { errors, type, branches: selectedBranches, modules: selections(sections.get(FIELDS[2])) };
}

function desiredLabels(result, availableLabels) {
  const available = new Set(availableLabels);
  return [...new Set([
    result.type,
    ...result.branches.map(branchLabel),
    ...result.modules.map((name) => name.startsWith('op:') ? name : `op:${name}`)
      .filter((name) => available.has(name)),
  ])];
}

function managedLabel(name) {
  // v26.* and needs-version belong to the existing version scheduler.
  return TYPES.includes(name) || name === 'main' || name.startsWith('branch:') || name.startsWith('op:');
}

function feedback(errors, repo) {
  const root = `https://github.com/${repo.owner}/${repo.repo}`;
  if (!errors.length) return `${MARKER}\nIssue 规范校验已通过，类型、涉及分支与算子标签已同步。`;
  return [MARKER, 'Issue 信息尚不完整，请**编辑原 Issue 正文**补齐后等待自动复查：', '',
    ...errors.map((error) => `- ${error}`), '',
    '正文需包含以下三级标题（示例值请换成实际内容）：', '',
    '```markdown', '### Issue 类型', 'bug', '', '### 涉及分支', 'main', '',
    '### 涉及算子或模块', 'chunk_kda_fwd', '```', '',
    `- [提交规范](${root}/blob/HEAD/docs/agents/Issue提交规范.md)`,
    `- [可选分支](${root}/branches)`,
    '- 类型与涉及分支必须明确填写；涉及分支不等于承诺修复版本。',
    '- API 提交者无需拥有打标签权限，仓库 Action 会在校验通过后添加标签。',
  ].join('\n');
}

async function run({ github, context, core }) {
  const rawNumber = context.payload.issue?.number ?? context.payload.inputs?.issue_number;
  if (!/^\d+$/.test(String(rawNumber || '')) || Number(rawNumber) < 1) throw new Error('缺少合法的 Issue 编号。');
  const ref = { ...context.repo, issue_number: Number(rawNumber) };
  // Events can queue while the author is editing; always read the latest body and labels.
  const { data: issue } = await github.rest.issues.get(ref);
  const names = issue.labels.map((label) => typeof label === 'string' ? label : label.name);
  if (issue.pull_request || issue.state !== 'open' || names.some((name) => EXEMPT_LABELS.includes(name))) {
    core.notice('跳过 PR、已关闭 Issue 或使用独立模板的治理 Issue。');
    return;
  }
  const branches = await github.paginate(github.rest.repos.listBranches, { ...context.repo, per_page: 100 });
  const result = validateIssue(issue.body, branches.map((branch) => branch.name));
  const allLabels = await github.paginate(github.rest.issues.listLabelsForRepo, { ...context.repo, per_page: 100 });
  const available = new Set(allLabels.map((label) => label.name));
  async function ensureLabel(name) {
    if (available.has(name)) return;
    try {
      await github.rest.issues.createLabel({ ...context.repo, name, color: name === NEEDS_INFO ? 'fbca04' : '1d76db',
        description: name === NEEDS_INFO ? 'Issue 信息不完整，请编辑正文补齐' : 'Issue 声明的涉及分支（非修复版本排期）' });
    } catch (error) {
      // Other issues may create this repository label concurrently.
      if (error.status !== 422) throw error;
      await github.rest.issues.getLabel({ ...context.repo, name });
    }
    available.add(name);
  }
  if (result.errors.length) {
    await ensureLabel(NEEDS_INFO);
    if (!names.includes(NEEDS_INFO)) await github.rest.issues.addLabels({ ...ref, labels: [NEEDS_INFO] });
  } else {
    const wanted = desiredLabels(result, [...available]);
    for (const name of wanted) await ensureLabel(name);
    const missing = wanted.filter((name) => !names.includes(name));
    if (missing.length) await github.rest.issues.addLabels({ ...ref, labels: missing });
    // Remove only classification labels, never replace the whole label collection.
    for (const name of names.filter((name) => name === NEEDS_INFO || (managedLabel(name) && !wanted.includes(name)))) {
      try { await github.rest.issues.removeLabel({ ...ref, name }); }
      catch (error) { if (error.status !== 404) throw error; }
    }
  }
  const comments = await github.paginate(github.rest.issues.listComments, { ...ref, per_page: 100 });
  const previous = comments.find((comment) => comment.user?.login === 'github-actions[bot]' &&
    comment.user?.type === 'Bot' && comment.body?.startsWith(MARKER));
  const body = feedback(result.errors, context.repo);
  if (previous && previous.body !== body) {
    await github.rest.issues.updateComment({ ...context.repo, comment_id: previous.id, body });
  } else if (!previous && result.errors.length) {
    await github.rest.issues.createComment({ ...ref, body });
  }
  if (result.errors.length) core.setFailed('Issue 信息不完整，具体缺项已写入机器人评论。');
  else core.notice('Issue 校验通过，标签已同步。');
  return result;
}

module.exports = { TYPES, FIELDS, REQUIRED, MARKER, NEEDS_INFO, sectionsOf, selections, branchLabel,
  validateIssue, desiredLabels, managedLabel, feedback, run };
