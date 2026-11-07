"use strict";

const auth = require('./npu_ci_authorization.js');
const WORKFLOW_PATH = '.github/workflows/ci.yml';
const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function requireSamePullRequest(pr, current) {
  auth.requireOpenPullRequest(current);
  // A target branch tip can advance while dispatching or rerunning CI.
  if (current.head.sha !== pr.head.sha || current.base.ref !== pr.base.ref) {
    throw new Error(`PR #${pr.number} 的 head 或目标分支已变化，请重新触发。`);
  }
}

async function readWorkflow({ github, context, ref }) {
  const { data: file } = await github.rest.repos.getContent({ ...context.repo, path: WORKFLOW_PATH, ref });
  if (!file || Array.isArray(file) || file.type !== 'file' || !file.content) {
    throw new Error(`${ref} 的 ${WORKFLOW_PATH} 不是可读取的文件。`);
  }
  return Buffer.from(file.content, file.encoding || 'base64').toString('utf8');
}

async function findMergeCommit({ github, context, pr, wait }) {
  // The preview must match the head and the base tip returned together by
  // GitHub. A later base advance does not invalidate this immutable snapshot.
  for (let attempt = 0; attempt < 4; attempt++) {
    const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
    requireSamePullRequest(pr, current);
    if (current.mergeable === false) {
      throw new Error(`PR #${pr.number} 存在合并冲突，请解决冲突后重新触发 NPU CI。`);
    }
    if (/^[0-9a-f]{40}$/.test(current.merge_commit_sha || '')) {
      let commit;
      try {
        ({ data: commit } = await github.rest.repos.getCommit({ ...context.repo, ref: current.merge_commit_sha }));
      } catch (error) {
        if (![404, 409].includes(error.status)) throw error;
      }
      if (commit?.sha === current.merge_commit_sha && commit.parents?.length === 2 &&
          commit.parents[0]?.sha === current.base.sha && commit.parents[1]?.sha === pr.head.sha) {
        return { sha: commit.sha, baseSha: current.base.sha };
      }
    }
    if (attempt < 3) await wait(2000);
  }
  throw new Error(`PR #${pr.number} 的合并预览尚未对应当前 base/head，请等待 GitHub 更新合并结果后重新触发。`);
}

async function selectWorkflowExecution({ github, context, pr, baseBranch, files, inputs, wait = pause }) {
  let ref = pr.base.ref, source = 'base', workflowSha = baseBranch.commit.sha;
  let baseSha = workflowSha;
  if (auth.containsCiChanges(files)) {
    const merge = await findMergeCommit({ github, context, pr, wait });
    workflowSha = merge.sha;
    baseSha = merge.baseSha;
    ref = auth.buildPrWorkflowTag(pr.number, workflowSha);
    source = 'pr-merge';
  }
  const text = await readWorkflow({ github, context, ref: workflowSha });
  // Maintenance branches keep their own interface, even if it predates the
  // optional workflow_source and expected_* inputs. Never require an upgrade.
  return { ref, source, workflowSha, inputs: auth.selectDispatchInputs(text, {
    ...inputs, expected_base_sha: baseSha, expected_workflow_sha: workflowSha, workflow_source: source,
  }) };
}

async function createWorkflowTag({ github, context, ref, sha }) {
  try {
    await github.rest.git.createRef({ ...context.repo, ref: `refs/tags/${ref}`, sha });
  } catch (error) {
    if (error.status !== 422) throw error;
    const { data: existing } = await github.rest.git.getRef({ ...context.repo, ref: `tags/${ref}` });
    if (existing.object?.type !== 'commit' || existing.object.sha !== sha) {
      throw new Error(`NPU CI workflow tag ${ref} 已存在且指向其他对象，拒绝覆盖。`);
    }
  }
}

async function dispatchNpuWorkflow({ github, refGithub = github, context, pr, baseBranch, files, inputs, wait }) {
  // This module is loaded from the default branch by issue_comment. Do not
  // read or dispatch PR workflow code until its author has been authorized.
  if (auth.containsCiChanges(files)) await auth.requireCiAuthorAdmin({ github, context, pr });
  const execution = await selectWorkflowExecution({ github, context, pr, baseBranch, files, inputs, wait });
  if (execution.source === 'pr-merge') {
    await createWorkflowTag({ github: refGithub, context, ref: execution.ref, sha: execution.workflowSha });
  }
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  requireSamePullRequest(pr, current);
  await github.rest.actions.createWorkflowDispatch({ ...context.repo, workflow_id: 'ci.yml',
    ref: execution.ref, inputs: execution.inputs });
  return execution;
}

module.exports = { WORKFLOW_PATH, selectWorkflowExecution, dispatchNpuWorkflow };
