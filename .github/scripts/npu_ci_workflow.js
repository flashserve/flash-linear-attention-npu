"use strict";

const auth = require('./npu_ci_authorization.js');
const WORKFLOW_PATH = '.github/workflows/ci.yml';
const SHA_PATTERN = /^[0-9a-f]{40}$/;
const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function parseCiCommand(body) {
  const command = String(body || '').trim().match(/^\/run-npu-ci(?:\s+([\s\S]*))?$/);
  if (!command) return null;
  let ciMode = 'quick', ops = '', modeProvided = false, opsProvided = false;
  for (const part of command[1] ? command[1].trim().split(/\s+/) : []) {
    if (part === 'quick' || part === 'full') {
      if (modeProvided) throw new Error('CI 模式只能指定一次。');
      ciMode = part;
      modeProvided = true;
    } else if (part.startsWith('ops=')) {
      if (opsProvided) throw new Error('ops 参数只能指定一次。');
      ops = part.slice('ops='.length);
      opsProvided = true;
    } else {
      throw new Error(`不支持的 NPU CI 参数：${part}`);
    }
  }
  const items = ops.split(',').map((item) => item.trim());
  if ((opsProvided && !ops.trim()) || (ops.trim() && items.some((item) =>
    !item || !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(item)))) {
    throw new Error('ops 参数必须是以逗号分隔的合法非空算子名。');
  }
  return { ciMode, ops };
}

async function isLatestCiComment({ github, context, prNumber, users }) {
  const commentId = context.payload.comment?.id;
  if (!Number.isSafeInteger(commentId) || commentId <= 0) {
    throw new Error('无法确定触发评论编号，拒绝替换现有 NPU CI。');
  }
  const allowed = new Set(users.map((user) => user.toLowerCase()));
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: prNumber, per_page: 100,
  });
  return !comments.some((comment) => {
    if (comment.id <= commentId || !allowed.has((comment.user?.login || '').toLowerCase())) return false;
    try { return Boolean(parseCiCommand(comment.body)); } catch { return false; }
  });
}

async function cancelAndWaitForRuns({ github, context, runs, matches,
  isCurrentRequest = async () => true, wait = pause, cancellationPolls = 73 }) {
  if (!await isCurrentRequest()) return false;
  if (runs.some((run) => !matches(run))) throw new Error('待取消的 NPU CI 身份不匹配，拒绝取消。');
  for (const run of runs) {
    try {
      await github.rest.actions.cancelWorkflowRun({ ...context.repo, run_id: run.id });
    } catch (error) {
      // A run can finish between listing and cancellation; verify below.
      if (error.status !== 409) throw error;
    }
  }
  let remaining = runs;
  for (let attempt = 0; remaining.length; attempt++) {
    if (!await isCurrentRequest()) return false;
    const current = await Promise.all(remaining.map(async (run) => {
      const { data } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: run.id });
      if (!matches(data)) throw new Error(`NPU CI 运行 #${run.id} 的身份发生变化，拒绝继续替换。`);
      return data;
    }));
    remaining = current.filter((run) => run.status !== 'completed');
    if (!remaining.length) break;
    if (attempt >= cancellationPolls - 1) {
      throw new Error(`NPU CI 尚未停止（${remaining.map((run) => `#${run.id}`).join('、')}）；未启动替代任务，请等待取消完成后重新评论。`);
    }
    await wait(5000);
  }
  return true;
}

async function replaceActiveNpuRuns({ github, context, pr, execution, statusContexts,
  isCurrentRequest = async () => true, wait = pause, cancellationPolls = 73 }) {
  const repository = `${context.repo.owner}/${context.repo.repo}`.toLowerCase();
  // Filter on the PR identity, not head/base/ref: a new request also replaces
  // older commits, modes and temporary tags for this PR.
  const matchingRun = (run) => Number.isSafeInteger(run.id) && run.id > 0 &&
    run.event === 'workflow_dispatch' && run.display_title === `NPU CI · PR #${pr.number}` &&
    run.repository?.full_name?.toLowerCase() === repository &&
    [WORKFLOW_PATH, `${WORKFLOW_PATH}@${run.head_branch}`].includes(run.path);
  const active = (await Promise.all(['queued', 'in_progress', 'waiting', 'pending', 'requested'].map((status) =>
    github.paginate(github.rest.actions.listWorkflowRuns, { ...context.repo,
      workflow_id: 'ci.yml', event: 'workflow_dispatch', status, per_page: 100 })))).flat();
  const runs = [...new Map(active.filter((run) => matchingRun(run) && run.status !== 'completed')
    .map((run) => [run.id, run])).values()];
  const runHeads = new Map();
  const rememberHead = (runId, head) => {
    if (!Number.isSafeInteger(runId) || runId <= 0) throw new Error('CI 运行登记包含无效 run ID。');
    if (runHeads.has(runId) && runHeads.get(runId) !== head) {
      throw new Error(`NPU CI #${runId} 的 PR commit 登记冲突，保留任务并停止分发。`);
    }
    runHeads.set(runId, head);
  };
  const runPrefix = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/`;
  if (runs.length) {
    // workflow run.head_sha is the workflow/base commit, not the PR head.
    // Record the association in a fresh bot comment before releasing the
    // dispatcher lock; this also works with unchanged maintenance workflows.
    const comments = await github.paginate(github.rest.issues.listComments, {
      ...context.repo, issue_number: pr.number, per_page: 100,
    });
    for (const comment of comments) {
      if (comment.user?.login !== 'github-actions[bot]') continue;
      const body = String(comment.body || '');
      const marker = new RegExp(`<!-- npu-ci-request:pr=${pr.number}:head=([a-f0-9]{40}):run=([1-9][0-9]*) -->`).exec(body);
      if (marker) rememberHead(Number(marker[2]), marker[1]);
      // Compatibility with runs started before request registration existed.
      const legacy = new RegExp(`<!-- npu-ci:pr=${pr.number}:sha=([a-f0-9]{40}) -->`).exec(body);
      if (legacy) {
        for (const run of runs) {
          if (body.includes(`${runPrefix}${run.id})`) || body.includes(`${runPrefix}${run.id}/`)) {
            rememberHead(run.id, legacy[1]);
          }
        }
      }
    }
    if (runs.some((run) => !runHeads.has(run.id))) {
      const statuses = await github.paginate(github.rest.repos.listCommitStatusesForRef, {
        ...context.repo, ref: pr.head.sha, per_page: 100,
      });
      for (const run of runs) {
        if (!runHeads.has(run.id) && statuses.some((status) =>
          status.context?.startsWith('NPU CI / ') &&
          [runPrefix + run.id, runPrefix + run.id + '/'].includes(status.target_url))) {
          rememberHead(run.id, pr.head.sha);
        }
      }
    }
    const unknown = runs.filter((run) => !runHeads.has(run.id));
    if (unknown.length) {
      throw new Error(`无法确认已有 NPU CI ${unknown.map((run) => `#${run.id}`).join('、')} 对应的 PR commit；保留任务且不重复分发。请等待 prepare 写入运行信息后重试，或在 Actions 中确认并停止遗留任务。`);
    }
  }
  // Identical PR head with a changed target/base is a new merge snapshot too.
  let existing = runs.filter((run) => runHeads.get(run.id) === pr.head.sha &&
    run.head_sha === execution.workflowSha && run.head_branch === execution.ref)
    .sort((left, right) => right.id - left.id)[0];
  if (!await cancelAndWaitForRuns({ github, context, runs: runs.filter((run) => run.id !== existing?.id), matches: matchingRun,
    isCurrentRequest, wait, cancellationPolls })) return false;
  if (!await isCurrentRequest()) return false;
  if (existing && runs.length > 1) {
    const { data: kept } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: existing.id });
    if (!matchingRun(kept) || kept.head_sha !== execution.workflowSha || kept.head_branch !== execution.ref) {
      throw new Error('保留的 NPU CI 运行身份变化，停止分发。');
    }
    // A cancelled duplicate may have finalized after the keeper completed.
    // In that case run again, so stale duplicate statuses cannot be final.
    if (kept.status === 'completed') existing = undefined;
  }
  const { data: currentPr } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  requireSamePullRequest(pr, currentPr);
  if (existing) return { existingRunId: existing.id };
  if (!Array.isArray(statusContexts) || !statusContexts.length ||
      new Set(statusContexts).size !== statusContexts.length || statusContexts.some((name) =>
        typeof name !== 'string' || !name.startsWith('NPU CI / '))) {
    throw new Error('无法确定本次 NPU CI 的检查项，拒绝重置状态。');
  }
  // Old branches may reuse a previous successful or pending result. A fresh
  // request deliberately invalidates that shortcut without changing their CI.
  // Wait for cancellation first so an old finalize cannot overwrite this reset.
  const targetUrl = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  await Promise.all(statusContexts.map((statusContext) => github.rest.repos.createCommitStatus({
    ...context.repo, sha: pr.head.sha, state: 'pending', context: statusContext,
    description: '新的 NPU CI 请求已受理，等待启动', target_url: targetUrl,
  })));
  return true;
}

async function confirmDispatchedRun({ github, context, pr, execution, response, wait = pause,
  visibilityPolls = 16, cancellationPolls = 73 }) {
  const runId = response?.data?.workflow_run_id;
  if (response?.status !== 200 || !Number.isSafeInteger(runId) || runId <= 0) {
    throw new Error('GitHub 未返回有效的 NPU CI 运行编号；无法确认任务，拒绝重复分发，请检查 Actions 运行列表。');
  }
  const targetUrl = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${runId}`;
  const repository = `${context.repo.owner}/${context.repo.repo}`.toLowerCase();
  const matches = (run) => run?.id === runId && run.event === 'workflow_dispatch' &&
    run.display_title === `NPU CI · PR #${pr.number}` && run.head_sha === execution.workflowSha &&
    run.head_branch === execution.ref && run.repository?.full_name?.toLowerCase() === repository &&
    [WORKFLOW_PATH, `${WORKFLOW_PATH}@${execution.ref}`].includes(run.path);
  let verifiedRun;
  try {
    for (let attempt = 0; attempt < visibilityPolls; attempt++) {
      let run;
      try {
        ({ data: run } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: runId }));
      } catch (error) {
        if (error.status !== 404) throw error;
      }
      if (run) {
        if (!matches(run)) throw new Error('新任务的仓库、PR、workflow、ref 或 commit 与本次请求不匹配');
        verifiedRun = run;
        if (run.status === 'completed') return runId;
        const runs = await github.paginate(github.rest.actions.listWorkflowRuns, { ...context.repo,
          workflow_id: 'ci.yml', event: 'workflow_dispatch', head_sha: execution.workflowSha, per_page: 100 });
        const visible = runs.find((item) => item.id === runId);
        if (visible) {
          if (!matches(visible)) throw new Error('运行列表中的新任务身份与本次请求不匹配');
          return runId;
        }
      }
      if (attempt < visibilityPolls - 1) await wait(2000);
    }
    throw new Error('等待新任务进入 Actions 运行列表超时');
  } catch (error) {
    let cancellation = '';
    // Only cancel an identity we verified, never an unrelated run returned in
    // an invalid response. Hold the dispatch queue until cancellation settles.
    if (verifiedRun) {
      try {
        await cancelAndWaitForRuns({ github, context, runs: [verifiedRun], matches, wait, cancellationPolls });
        cancellation = '；该任务已取消，不会继续运行';
      } catch (cancelError) {
        cancellation = `；取消确认失败：${cancelError.message}，请在 Actions 中确认已停止`;
      }
    }
    throw new Error(`无法确认新 NPU CI #${runId}：${error.message}${cancellation}。运行链接：${targetUrl}。不会重复分发。`);
  }
}

function requireSamePullRequest(pr, current) {
  auth.requireOpenPullRequest(current);
  if (current.head.sha !== pr.head.sha || current.base.sha !== pr.base.sha ||
      current.base.ref !== pr.base.ref) {
    throw new Error(`PR #${pr.number} 的 head 或目标分支已变化，请针对当前版本重新触发。`);
  }
}

async function readWorkflow({ github, context, ref }) {
  let file;
  try {
    ({ data: file } = await github.rest.repos.getContent({ ...context.repo, path: WORKFLOW_PATH, ref }));
  } catch (error) {
    throw new Error(`无法读取 ${ref} 的 ${WORKFLOW_PATH}：${error.message}。请保留可执行的 CI workflow 后重新触发。`);
  }
  if (!file || Array.isArray(file) || file.type !== 'file' || !file.content) {
    throw new Error(`${ref} 的 ${WORKFLOW_PATH} 不是可读取的文件。`);
  }
  return Buffer.from(file.content, file.encoding || 'base64').toString('utf8');
}

async function findMergeCommit({ github, context, pr, wait }) {
  // GitHub computes refs/pull/<n>/merge asynchronously. Never use an older
  // preview, even if its workflow text happens to look usable.
  for (let attempt = 0; attempt < 4; attempt++) {
    const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
    requireSamePullRequest(pr, current);
    if (current.mergeable === false) {
      throw new Error(`PR #${pr.number} 存在合并冲突，无法执行当前 PR 的 CI workflow；请解决冲突后重新触发。`);
    }
    if (SHA_PATTERN.test(current.merge_commit_sha || '')) {
      let commit;
      try {
        ({ data: commit } = await github.rest.repos.getCommit({ ...context.repo, ref: current.merge_commit_sha }));
      } catch (error) {
        if (![404, 409].includes(error.status)) throw error;
        commit = {};
      }
      const parents = (commit.parents || []).map((parent) => parent.sha);
      if (commit.sha === current.merge_commit_sha && parents.length === 2 &&
          parents[0] === pr.base.sha && parents[1] === pr.head.sha) {
        return commit.sha;
      }
    }
    if (attempt < 3) await wait(2000);
  }
  throw new Error(`PR #${pr.number} 的合并预览尚未对应当前 base/head，拒绝执行旧版 CI；请等待 GitHub 更新合并结果后重新触发。`);
}

async function selectWorkflowExecution({ github, context, pr, baseBranch, files, inputs,
  wait = pause }) {
  // A CI change can affect hosted control scripts as well as the YAML itself.
  // Source-only PRs keep the target branch's existing workflow and need no tag.
  if (!auth.containsCiChanges(files)) {
    const text = await readWorkflow({ github, context, ref: baseBranch.commit.sha });
    return { ref: pr.base.ref, source: 'base', workflowSha: baseBranch.commit.sha,
      inputs: auth.selectDispatchInputs(text, { ...inputs, workflow_source: 'base' }) };
  }

  const mergeSha = await findMergeCommit({ github, context, pr, wait });
  const text = await readWorkflow({ github, context, ref: mergeSha });
  const selectedInputs = auth.selectDispatchInputs(text, { ...inputs,
    expected_workflow_sha: mergeSha, workflow_source: 'pr-merge' });
  const required = ['pr_number', 'requested_by', 'expected_head_sha', 'expected_base_ref',
    'expected_base_sha', 'expected_workflow_sha', 'workflow_source'];
  if (required.some((name) => !selectedInputs[name])) {
    throw new Error(`PR #${pr.number} 修改了 ci/ 或 .github/ 的非 Markdown 文件，但其 ${WORKFLOW_PATH} 缺少 workflow_source 或固定版本输入；请将完整的 PR workflow 分发契约同步到该文件后重新触发。`);
  }
  return { ref: auth.buildPrWorkflowTag(pr.number, mergeSha), source: 'pr-merge',
    workflowSha: mergeSha, inputs: selectedInputs };
}

async function createWorkflowTag({ github, context, ref, sha }) {
  const permissionAdvice = '请在仅允许 main 分支的 npu-ci-dispatch Environment 中配置 secret NPU_CI_DISPATCH_TOKEN，授予本仓库 Contents 和 Workflows 写权限，并允许 npu-ci/ tags。';
  try {
    await github.rest.git.createRef({ ...context.repo, ref: `refs/tags/${ref}`, sha });
    return;
  } catch (error) {
    if (error.status === 422) {
      let existing;
      try {
        ({ data: existing } = await github.rest.git.getRef({ ...context.repo, ref: `tags/${ref}` }));
      } catch (readError) {
        throw new Error(`无法创建或核对 NPU CI workflow tag ${ref}：${error.message}；读取 tag：${readError.message}。${permissionAdvice}`);
      }
      if (existing.object?.type === 'commit' && existing.object.sha === sha) return;
      throw new Error(`NPU CI workflow tag ${ref} 已存在且指向其他对象，拒绝覆盖；请管理员检查该 tag。`);
    }
    throw new Error(`无法创建 NPU CI workflow tag：${error.message}。${permissionAdvice}`);
  }
}

async function dispatchNpuWorkflow({ github, refGithub = github, context, pr, baseBranch, files, inputs, wait,
  isCurrentRequest = async () => true, beforeDispatch = async () => true,
  visibilityPolls = 16, cancellationPolls = 73 }) {
  const execution = await selectWorkflowExecution({ github, context, pr, baseBranch, files, inputs, wait });
  if (!await isCurrentRequest()) return { ...execution, skipped: true };
  let referenceReady = false;
  try {
    const decision = await beforeDispatch(execution);
    if (decision?.existingRunId) {
      return { ...execution, skipped: true, reason: 'duplicate', runId: decision.existingRunId };
    }
    if (!decision || !await isCurrentRequest()) return { ...execution, skipped: true };
    if (execution.source === 'pr-merge') {
      await createWorkflowTag({ github: refGithub, context,
        ref: execution.ref, sha: execution.workflowSha });
      referenceReady = true;
      const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
      requireSamePullRequest(pr, current);
    }
    if (!await isCurrentRequest()) {
      return { ...execution, skipped: true };
    }
    const dispatch = () => github.rest.actions.createWorkflowDispatch({ ...context.repo, workflow_id: 'ci.yml',
      ref: execution.ref, inputs: execution.inputs, headers: { 'X-GitHub-Api-Version': '2026-03-10' } });
    let response;
    try {
      response = await dispatch();
    } catch (error) {
      if (execution.source !== 'pr-merge' || ![404, 422].includes(error.status)) throw error;
      // Cleanup may finish between reusing a tag and dispatching it. Recover
      // only when the ref is actually absent, and retry at most once.
      let referenceMissing = false;
      try {
        await refGithub.rest.git.getRef({ ...context.repo, ref: `tags/${execution.ref}` });
      } catch (refError) {
        if (refError.status !== 404) throw error;
        referenceMissing = true;
      }
      if (!referenceMissing) throw error;
      referenceReady = false;
      await createWorkflowTag({ github: refGithub, context,
        ref: execution.ref, sha: execution.workflowSha });
      referenceReady = true;
      const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
      requireSamePullRequest(pr, current);
      if (!await isCurrentRequest()) return { ...execution, skipped: true };
      response = await dispatch();
    }
    // Keep confirmation outside the dispatch retry catch: a get/list failure
    // after creation must never create another workflow run.
    execution.runId = await confirmDispatchedRun({ github, context, pr, execution, response,
      wait, visibilityPolls, cancellationPolls });
    try {
      await github.rest.issues.createComment({ ...context.repo, issue_number: pr.number, body: [
        `<!-- npu-ci-request:pr=${pr.number}:head=${pr.head.sha}:run=${execution.runId} -->`,
        `NPU CI 已启动，固定 PR commit：\`${pr.head.sha}\`。`,
        `[查看本次任务](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${execution.runId})`,
      ].join('\n') });
    } catch (error) {
      // A run without durable PR-head registration cannot be deduplicated by
      // the next request on an old branch. Stop it instead of creating overlap.
      const { data: run } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: execution.runId });
      await cancelAndWaitForRuns({ github, context, runs: [run],
        matches: (current) => current.id === execution.runId && current.event === 'workflow_dispatch' &&
          current.head_sha === execution.workflowSha && current.head_branch === execution.ref &&
          current.repository?.full_name?.toLowerCase() === `${context.repo.owner}/${context.repo.repo}`.toLowerCase(),
        wait, cancellationPolls });
      throw new Error(`无法登记 NPU CI #${execution.runId} 的 PR commit，任务已取消；请恢复评论写入后重试：${error.message}`);
    }
  } catch (error) {
    // Another request may already be using this immutable ref. Leave it in
    // place on failure; the same PR snapshot can safely reuse it on retry.
    if (referenceReady) error.message += `（固定 workflow tag ${execution.ref} 已保留，可供相同版本重试使用。）`;
    throw error;
  }
  return execution;
}

module.exports = { WORKFLOW_PATH, parseCiCommand, isLatestCiComment, replaceActiveNpuRuns,
  selectWorkflowExecution, dispatchNpuWorkflow };
