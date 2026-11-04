'use strict';

async function requireAuthorizedCiUser({ github, context, username }) {
  if (typeof username !== 'string' || !username) {
    throw new Error('无法确定 NPU CI 触发人；需要仓库 Admin 权限。');
  }
  let permission;
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({ ...context.repo, username });
    permission = data.permission || 'none';
  } catch (error) {
    throw new Error(`无法确认 @${username} 的仓库 Admin 权限：${error.message}。请恢复权限读取后重试。`);
  }
  if (permission !== 'admin') {
    throw new Error(`${username} 无权触发 NPU CI：需要仓库 Admin 权限，当前权限为 ${permission}。`);
  }
}

module.exports = { requireAuthorizedCiUser };
