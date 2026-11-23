#!/usr/bin/env node
'use strict';

const { readFileSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const { parseArgs } = require('node:util');
const { FIELDS, TYPES, sectionsOf, selections, validateIssue } = require('../../.github/scripts/issue_policy');

function prepareBody(body, options) {
  const values = [options.type, options.branch, options.module];
  const { sections } = sectionsOf(body);
  const prefix = [];
  FIELDS.forEach((field, index) => {
    const selected = values[index];
    if (!selected) return;
    if (sections.has(field)) {
      if (JSON.stringify(selections(sections.get(field)).sort()) !== JSON.stringify(selections(selected).sort())) {
        throw new Error(`参数与正文中的“${field}”冲突，请修改正文或参数。`);
      }
    } else prefix.push(`### ${field}\n\n${selected}`);
  });
  return [...prefix, body].join('\n\n');
}

function main(argv = process.argv.slice(2), gh = (args, input) => execFileSync('gh', args, {
  input, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
})) {
  const { values } = parseArgs({ args: argv, options: {
    repo: { type: 'string', default: 'flashserve/flash-linear-attention-npu' },
    title: { type: 'string' }, type: { type: 'string' }, branch: { type: 'string' },
    module: { type: 'string' }, 'body-file': { type: 'string' },
    'dry-run': { type: 'boolean', default: false }, help: { type: 'boolean', default: false },
  } });
  if (values.help) {
    console.log('node scripts/github/create_issue.js --title "标题" --type bug --branch main --module ci --body-file issue.md [--repo OWNER/REPO] [--dry-run]');
    console.log(`类型：${TYPES.join(', ')}。多个分支或模块用逗号分隔。需 Node.js 20+，实际提交需已登录的 GitHub CLI。`);
    return;
  }
  if (!values.title?.trim() || !values['body-file']) throw new Error('--title 和 --body-file 为必填参数。');
  if (!/^[\w.-]+\/[\w.-]+$/.test(values.repo)) throw new Error('--repo 必须是 OWNER/REPO。');
  const body = prepareBody(readFileSync(values['body-file'], 'utf8'), values);
  let result = validateIssue(body);
  if (result.errors.length) throw new Error(result.errors.join('\n'));
  if (values['dry-run']) {
    console.log(body);
    console.log('\n本地字段校验通过；dry-run 未联网核对分支，也未创建 Issue。');
    return;
  }
  const branches = gh(['api', '--paginate', `repos/${values.repo}/branches`, '--jq', '.[].name']).trim().split(/\r?\n/);
  result = validateIssue(body, branches);
  if (result.errors.length) throw new Error(result.errors.join('\n'));
  // Submit metadata in the body: outside contributors may not be allowed to set labels.
  const url = gh(['issue', 'create', '--repo', values.repo, '--title', values.title, '--body-file', '-'], body).trim();
  console.log(url);
  console.log('Issue 已创建。请检查仓库的“侧边栏标签 / Issue 规范校验”结果；needs-info 表示仍需编辑正文。');
  return url;
}

if (require.main === module) {
  try { main(); }
  catch (error) {
    console.error(error.stderr?.toString().trim() || error.message);
    process.exitCode = 1;
  }
}

module.exports = { prepareBody, main };
