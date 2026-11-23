'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { mkdtempSync, writeFileSync, unlinkSync, rmdirSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const policy = require('../issue_policy');
const cli = require('../../../scripts/github/create_issue');

function body(type = 'bug', branch = 'main', module = 'chunk_kda_fwd') {
  return [
    `### Issue 类型\n\n${type}`, `### 涉及分支\n\n${branch}`, `### 涉及算子或模块\n\n${module}`,
    ...(policy.REQUIRED[type] || []).map((heading) => `### ${heading}\n\n实际内容与可核对的证据。`),
  ].join('\n\n');
}

for (const type of policy.TYPES) {
  test(`${type}: complete API body passes without pre-existing labels`, () => {
    assert.deepEqual(policy.validateIssue(body(type), ['main']).errors, []);
  });
  for (const heading of [...policy.FIELDS, ...policy.REQUIRED[type]]) {
    test(`${type}: missing ${heading} is rejected`, () => {
      const missing = body(type).replace(new RegExp(`### ${heading}[^]*?(?=\\n### |$)`), '');
      assert.ok(policy.validateIssue(missing, ['main']).errors.some((error) => error.includes(heading)));
    });
  }
}

test('a title/label-like free-form submission cannot substitute for metadata', () => {
  assert.ok(policy.validateIssue('[Bug]: bug on main', ['main']).errors.length >= 3);
});

test('unknown or conflicting type selections and branches are rejected', () => {
  assert.ok(policy.validateIssue(body('bug,enhancement'), ['main']).errors.length);
  assert.ok(policy.validateIssue(body('bug', 'missing'), ['main']).errors.some((error) => error.includes('不存在')));
});

test('arbitrary type strings cannot access object prototype properties', () => {
  for (const type of ['__proto__', 'constructor', 'toString']) {
    assert.ok(policy.validateIssue(body().replace('\n\nbug\n', `\n\n${type}\n`), ['main'])
      .errors.some((error) => error.includes('Issue 类型')));
  }
});

test('multiple real branches accept commas, backticks and repeated values', () => {
  const result = policy.validateIssue(body('bug', '`main`, stable/abi，main'), ['main', 'stable/abi']);
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.branches, ['main', 'stable/abi']);
});

test('fenced headings are reproduction text, not metadata', () => {
  const example = '\n\n```python\n### Issue 类型\nmalicious\n```';
  assert.deepEqual(policy.validateIssue(body() + example, ['main']).errors, []);
  assert.ok(policy.validateIssue('```md\n' + body() + '\n```', ['main']).errors.length);
  assert.ok(policy.validateIssue('~~~~md\n' + body() + '\n~~~~', ['main']).errors.length);
});

test('duplicate headings and missing form responses cannot pass', () => {
  assert.ok(policy.validateIssue(body() + '\n\n### 涉及分支\nmain', ['main']).errors.some((error) => error.includes('重复')));
  for (const placeholder of ['_No response_', 'TODO', '待补充', '...', '<!-- invisible -->']) {
    const value = body().replace(/### 环境信息[^]*?(?=\n### )/, `### 环境信息\n${placeholder}\n`);
    assert.ok(policy.validateIssue(value, ['main']).errors.some((error) => error.includes('环境信息')));
  }
});

test('unfilled acceptance defaults fail and real fenced reproduction passes', () => {
  const unchanged = body('enhancement').replace(/### 验收要求[^]*/, '### 验收要求\n- 精度：按 ... 判定\n- 性能：... 平台');
  assert.ok(policy.validateIssue(unchanged, ['main']).errors.some((error) => error.includes('验收要求')));
  const code = body().replace(/### 最小复现[^]*?(?=\n### )/, '### 最小复现\n```python\nprint(1)\n```\n');
  assert.deepEqual(policy.validateIssue(code, ['main']).errors, []);
});

test('branch labels cannot be confused with repair versions, including long names', () => {
  assert.equal(policy.branchLabel('v26.11.0'), 'branch:v26.11.0');
  const long = 'feature/' + 'x'.repeat(80);
  assert.ok(policy.branchLabel(long).length <= 50);
  assert.notEqual(policy.branchLabel(long), policy.branchLabel(long + 'y'));
  assert.equal(policy.managedLabel('v26.11.0'), false);
  assert.equal(policy.managedLabel('needs-version'), false);
});

function fixture({ issue = {}, comments = [], available = [], collision = false } = {}) {
  let current = { number: 23, body: body(), state: 'open', labels: [], ...issue };
  const calls = [];
  const record = (method, args) => { calls.push({ method, args }); };
  const rest = { repos: { listBranches() {} }, issues: {
    get: async (args) => { record('get', args); return { data: structuredClone(current) }; },
    listLabelsForRepo() {}, listComments() {},
    createLabel: async (args) => {
      record('createLabel', args);
      if (collision) throw Object.assign(new Error('already exists'), { status: 422 });
    },
    getLabel: async (args) => { record('getLabel', args); },
    addLabels: async (args) => {
      record('addLabels', args);
      for (const name of args.labels) if (!current.labels.includes(name)) current.labels.push(name);
    },
    removeLabel: async (args) => { record('removeLabel', args); current.labels = current.labels.filter((name) => name !== args.name); },
    createComment: async (args) => {
      record('createComment', args);
      comments.push({ ...args, id: 100, user: { login: 'github-actions[bot]', type: 'Bot' } });
    },
    updateComment: async (args) => {
      record('updateComment', args);
      comments.find((comment) => comment.id === args.comment_id).body = args.body;
    },
  } };
  const github = { rest, paginate: async (method) => {
    if (method === rest.repos.listBranches) return [{ name: 'main' }, { name: 'stable/abi' }, { name: 'v26.11.0' }];
    if (method === rest.issues.listLabelsForRepo) return available.map((name) => ({ name }));
    if (method === rest.issues.listComments) return comments;
    throw new Error('Unexpected pagination endpoint');
  } };
  const context = { repo: { owner: 'flashserve', repo: 'flash-linear-attention-npu' }, payload: { issue: { number: 23, body: 'stale event body' } } };
  const core = { notice: (message) => record('notice', message), setFailed: (message) => record('failed', message) };
  return { github, context, core, calls, comments, current, run: () => policy.run({ github, context, core }) };
}

test('server reads fresh API body and supplies labels for outside contributors', async () => {
  const f = fixture({ available: ['op:chunk_kda_fwd'] });
  await f.run();
  assert.deepEqual(f.current.labels.sort(), ['bug', 'main', 'op:chunk_kda_fwd'].sort());
  assert.equal(f.calls.some((call) => call.method === 'failed'), false);
});

test('reclassification removes stale owned tags and preserves scheduler and business tags', async () => {
  const f = fixture({ issue: { body: body('enhancement', 'stable/abi', 'chunk_kda_fwd'),
    labels: ['bug', 'main', 'branch:old', 'op:old', 'v26.11.0', 'needs-version', 'documentation', 'needs-info'] },
    available: ['op:chunk_kda_fwd'] });
  await f.run();
  assert.deepEqual(f.current.labels.sort(), ['enhancement', 'branch:stable/abi', 'op:chunk_kda_fwd',
    'v26.11.0', 'needs-version', 'documentation'].sort());
});

test('unknown module names do not create operator labels', async () => {
  const f = fixture({ issue: { body: body('bug', 'main', 'ci,op:invented') } });
  await f.run();
  assert.deepEqual(f.current.labels.sort(), ['bug', 'main']);
});

test('invalid submissions get one actionable comment; repeat runs are idempotent', async () => {
  const f = fixture({ issue: { body: 'free text', labels: ['v26.11.0', 'engineering'] } });
  await f.run();
  await f.run();
  assert.equal(f.calls.filter((call) => call.method === 'createComment').length, 1);
  assert.equal(f.calls.filter((call) => call.method === 'updateComment').length, 0);
  assert.ok(f.current.labels.includes('needs-info'));
  assert.ok(f.current.labels.includes('v26.11.0'));
  assert.ok(f.comments[0].body.includes('编辑原 Issue 正文'));
});

test('editing a previously invalid issue resolves the same comment and marker label', async () => {
  const f = fixture({ issue: { body: 'free text' } });
  await f.run();
  f.current.body = body();
  await f.run();
  assert.equal(f.current.labels.includes('needs-info'), false);
  assert.equal(f.calls.filter((call) => call.method === 'updateComment').length, 1);
  assert.ok(f.comments[0].body.includes('已通过'));
});

test('human-authored marker comments are never edited', async () => {
  const f = fixture({ issue: { body: 'free text' }, comments: [
    { id: 77, body: policy.MARKER, user: { login: 'outside-user', type: 'User' } },
  ] });
  await f.run();
  assert.equal(f.comments.length, 2);
  assert.equal(f.comments[0].body, policy.MARKER);
});

test('repository label creation races are verified before proceeding', async () => {
  const f = fixture({ collision: true });
  await f.run();
  assert.ok(f.calls.some((call) => call.method === 'getLabel'));
});

for (const issue of [{ state: 'closed' }, { pull_request: {} },
  { labels: ['committer-promotion'] }, { labels: ['requirements-index'] }]) {
  test(`skip independent or inactive issues: ${JSON.stringify(issue)}`, async () => {
    const f = fixture({ issue });
    await f.run();
    assert.deepEqual(f.calls.map((call) => call.method), ['get', 'notice']);
  });
}

test('manual dispatch works without an issues-event payload', async () => {
  const f = fixture();
  f.context.payload = { inputs: { issue_number: '23' } };
  await f.run();
  assert.ok(f.current.labels.includes('bug'));
  f.context.payload.inputs.issue_number = '23; malicious';
  await assert.rejects(f.run(), /合法/);
});

test('CLI refuses conflicting metadata instead of silently rewriting the author', () => {
  assert.throws(() => cli.prepareBody(body(), { type: 'question' }), /冲突/);
});

test('CLI validates remote branches and sends literal Markdown via stdin without labels', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'fla-issue-policy-'));
  const file = path.join(dir, 'issue.md');
  const calls = [];
  const originalLog = console.log;
  try {
    writeFileSync(file, body() + '\nLiteral: `code` $(not-a-shell-command)\n', 'utf8');
    console.log = () => {};
    const gh = (args, input) => {
      calls.push({ args, input });
      return args[0] === 'api' ? 'main\n' : 'https://github.com/example/repo/issues/1\n';
    };
    cli.main(['--title', 'Bug', '--body-file', file], gh);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].args.includes('--label'), false);
    assert.ok(calls[1].input.includes('$(not-a-shell-command)'));
    calls.length = 0;
    cli.main(['--title', 'Bug', '--body-file', file, '--dry-run'], gh);
    assert.equal(calls.length, 0);
    assert.throws(() => cli.main(['--title', 'Bug', '--body-file', file], () => 'other\n'), /不存在/);
  } finally {
    console.log = originalLog;
    unlinkSync(file);
    rmdirSync(dir);
  }
});
