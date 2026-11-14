'use strict';

const definition = require('./npu_ci_checks.json');
const DEFINITION_PATH = '.github/scripts/npu_ci_checks.json';

function contextNames(config) {
  const entries = config?.required;
  if (!Array.isArray(entries) || !entries.length || entries.some((entry) =>
    typeof entry?.context !== 'string' || !entry.context.startsWith('NPU CI / ') ||
    entry.context.trim() !== entry.context) ||
    new Set(entries.map((entry) => entry.context)).size !== entries.length) {
    throw new Error('NPU CI 检查项定义必须包含非空、无重复的 required context 列表。');
  }
  return entries.map((entry) => entry.context);
}

const contexts = contextNames(definition);
const prepareChecks = contexts.map((context) =>
  [context, context.split(' / ').at(-1).replace(/^\d+\s+/, '')]);

async function readBranchContexts({ github, context, pr }) {
  // Read the target's fixed base commit, never evaluate code from that branch.
  if (!/^[0-9a-f]{40}$/.test(pr.base.sha || '')) throw new Error('无法确定目标分支的固定 commit。');
  const readText = async (path) => {
    const { data } = await github.rest.repos.getContent({ ...context.repo, path, ref: pr.base.sha });
    if (!data || Array.isArray(data) || data.type !== 'file' || !data.content) {
      throw new Error(`${path} 不是可读取的文件。`);
    }
    return Buffer.from(data.content, data.encoding || 'base64').toString('utf8');
  };
  try {
    return contextNames(JSON.parse(await readText(DEFINITION_PATH)));
  } catch (error) {
    // Old release branches still define their contexts directly in this YAML.
    if (error.status !== 404) throw error;
  }
  const text = await readText('.github/workflows/npu-ci-default-status.yml');
  const legacy = [...new Set([...text.matchAll(/'NPU CI \/ [^']+'/g)].map((match) => match[0].slice(1, -1)))];
  if (!legacy.length) throw new Error('目标分支没有可识别的 NPU CI 检查项。');
  return legacy;
}

module.exports = {
  required: definition.required, contexts, diagnostic: definition.diagnostic,
  contractTest: definition.contract_test, prepareChecks, readBranchContexts,
};
