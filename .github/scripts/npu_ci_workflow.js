"use strict";

const auth = require('./npu_ci_authorization.js');
const WORKFLOW_PATH = '.github/workflows/ci.yml';
const SHA_PATTERN = /^[0-9a-f]{40}$/;

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
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)) }) {
  // Any non-Markdown file under ci/ or .github/ can change how the workflow
  // prepares or executes the PR. Select the fixed PR merge workflow for the
  // whole CI control surface, not only when ci.yml itself changed.
  const modifiesCi = auth.containsCiChanges(files);
  if (!modifiesCi) {
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
    throw new Error(`PR #${pr.number} 修改了 ci/ 或 .github/ 下的 CI 控制文件，但 PR workflow 缺少 workflow_source 或固定版本输入；请将完整的 PR workflow 分发契约同步到 ${WORKFLOW_PATH} 后重新触发。`);
  }
  return { ref: auth.buildPrWorkflowTag(pr.number, mergeSha), source: 'pr-merge',
    workflowSha: mergeSha, inputs: selectedInputs };
}

async function createWorkflowTag({ github, context, ref, sha }) {
  const permissionAdvice = '请配置 Actions secret NPU_CI_DISPATCH_TOKEN，授予本仓库 Contents 和 Workflows 写权限，并允许 npu-ci/ tags。';
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

async function dispatchNpuWorkflow({ github, refGithub = github, context, pr, baseBranch, files, inputs, wait }) {
  const execution = await selectWorkflowExecution({ github, context, pr, baseBranch, files, inputs, wait });
  let referenceReady = false;
  try {
    if (execution.source === 'pr-merge') {
      await createWorkflowTag({ github: refGithub, context,
        ref: execution.ref, sha: execution.workflowSha });
      referenceReady = true;
      const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
      requireSamePullRequest(pr, current);
    }
    const dispatch = () => github.rest.actions.createWorkflowDispatch({ ...context.repo, workflow_id: 'ci.yml',
      ref: execution.ref, inputs: execution.inputs });
    try {
      await dispatch();
    } catch (error) {
      if (execution.source !== 'pr-merge' || ![404, 422].includes(error.status)) throw error;
      // Cleanup may finish between reusing a tag and dispatching it. Recover
      // only when the ref is actually absent, and retry at most once.
      try {
        await refGithub.rest.git.getRef({ ...context.repo, ref: `tags/${execution.ref}` });
      } catch (refError) {
        if (refError.status !== 404) throw error;
        referenceReady = false;
        await createWorkflowTag({ github: refGithub, context,
          ref: execution.ref, sha: execution.workflowSha });
        referenceReady = true;
        const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
        requireSamePullRequest(pr, current);
        await dispatch();
        return execution;
      }
      throw error;
    }
  } catch (error) {
    // Another request may already be using this immutable ref. Leave it in
    // place on failure; the same PR snapshot can safely reuse it on retry.
    if (referenceReady) error.message += `（固定 workflow tag ${execution.ref} 已保留，可供相同版本重试使用。）`;
    throw error;
  }
  return execution;
}

module.exports = { WORKFLOW_PATH, selectWorkflowExecution, dispatchNpuWorkflow };
