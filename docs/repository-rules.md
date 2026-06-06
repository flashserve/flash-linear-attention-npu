# 仓库规则

## 分支创建

仅以下 GitHub 账号允许在主仓库创建分支：

- `juyangokok`
- `weinachuan`
- `weiwei-612`（Sun Weiwei）
- `woey`
- `zhangshuolei-hfut`
- `chen-linxin`

其他成员请 fork 本仓库到个人仓库，并从个人仓库向主仓库提交 Pull Request。

`.github/workflows/repository-rules.yml` 中的 `仓库规则 / 分支创建权限检查` 会在主仓库出现未授权新分支时自动删除该分支，并提示提交者走 fork + PR 流程。

## 合入检视

PR 合入前需要上述维护账号中的 2 个账号在当前 head commit 上检视通过。`仓库规则 / 维护者检视门禁` workflow 会检查有效审批数，并排除 PR 提交人自己的审批。

`.github/CODEOWNERS` 将全仓默认归属到维护账号组，便于 GitHub 自动请求检视。仓库管理员应将 `仓库规则 / 维护者检视门禁` 配置为 `main` 分支必需状态检查。

## NPU CI 门禁

NPU 资源有限，`NPU CI` 不会由 PR 新建、重开或 push 新 commit 自动触发。PR 合入前必须让当前 head commit 具备成功的 `NPU CI / 手动验证` 状态；当 PR 更新 commit 后，旧 commit 上的 CI 成功状态自动失效，需要维护者重新触发。

维护者可以通过两种方式触发：

1. 在 GitHub `Actions` 页面选择 `NPU CI`，点击 `Run workflow`，填写 PR 编号、`quick/full` 模式和可选 `ops`。
2. 在 PR 评论区发送命令：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

如果当前 commit 已经通过 `NPU CI / 手动验证`，重复触发会被跳过，不会再次占用 NPU。

仓库管理员应将 `NPU CI / 手动验证` 也配置为 `main` 分支必需状态检查。

NPU CI 的 self-hosted runner、Docker 镜像、`--privileged`、触发方式和排障步骤见 [`Fla-npu仓CI部署教程.md`](Fla-npu仓CI部署教程.md)。

## 强行合入

`weinachuan` 的强行合入权限需要在 GitHub 分支保护中配置 PR bypass allowance。仓库管理员可使用以下脚本应用 `main` 分支保护：

```sh
GITHUB_TOKEN=<admin-token> scripts/github/apply_branch_protection.sh main
```

该 token 需要具备仓库 administration 写权限。脚本会要求 `仓库规则 / 维护者检视门禁` 与 `NPU CI / 手动验证` 通过，并配置 2 个 approval、CODEOWNERS review、stale review dismiss 和 `weinachuan` bypass。
