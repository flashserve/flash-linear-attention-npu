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

PR 合入前仍需要 2 个 approval。该要求由 GitHub 分支保护原生能力处理，不再设置自定义检视 status。仓库管理员应在分支保护中配置 required approvals 为 `2`，并保留 stale review dismiss 和 last push approval。

`.github/CODEOWNERS` 用于 GitHub 原生 Code Owners review。全仓默认请求维护账号检视，ABI 敏感路径会在 `CODEOWNERS` 末尾覆盖为仅 `@weinachuan`。

## ABI 兼容性

如果 PR 修改了算子 `def`、`aclnn` 接口入参类型、返回值、必选/可选属性，或修改了 `torch` 接口入参类型等可能导致 ABI 不一致的内容，仍必须由 `weinachuan` 检视确认。该要求通过 GitHub 原生 Code Owners review 生效，不再由自定义检视 workflow 自动阻塞。

需要重点关注的 ABI 风险区域：

- `fla/ops/**/op_host/*_def.cpp` 中的 `Input` / `Output` / `Attr` / dtype / format / required / optional 等定义
- `fla/ops/**/op_host/op_api/aclnn_*.h` 和 `aclnn_*.cpp` 中的 aclnn 接口签名、入参类型和返回相关声明
- `torch_custom/fla_npu/npu_custom.yaml` 等 torch schema 文件中的 `func` 定义
- `torch_custom/fla_npu/op_plugin/ops/opapi/**` 中的 torch 适配接口签名

如果 PR push 了新 commit，相关 ABI 检视也应重新确认。

## NPU CI 门禁

NPU 资源有限，`NPU CI` 不会由 PR 新建、重开或 push 新 commit 自动执行。PR 新建、重开或 push 新 commit 时，会自动给当前 head commit 写入 `NPU CI / 手动验证 pending`，描述为“未执行”，并在 PR 评论区提示可请求仓库 Admin 权限账号触发。PR 合入前必须让当前 head commit 具备成功的 `NPU CI / 手动验证` 状态；当 PR 更新 commit 后，旧 commit 上的 CI 成功状态自动失效，需要仓库 Admin 权限账号重新触发。

仓库 Admin 权限账号可以通过两种方式触发：

1. 在 GitHub `Actions` 页面选择 `NPU CI`，点击 `Run workflow`，填写 PR 编号、`quick/full` 模式和可选 `ops`。
2. 在 PR 评论区发送命令：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

如果当前 commit 已经通过 `NPU CI / 手动验证`，重复触发会被跳过，不会再次占用 NPU。如果同一 PR 的同一 commit 已经有 NPU CI 处于排队或运行中，重复评论只会更新机器人评论为“已在运行”，不会启动新的 runner job。runner 宿主机还会用 `/tmp/fla-npu-ci-npu-<id>.lock` 对物理 NPU 加锁，避免多个任务抢同一张卡。

仓库管理员应将 `NPU CI / 手动验证` 也配置为 `main` 分支必需状态检查。

NPU CI 的 self-hosted runner、Docker 镜像、`--privileged`、触发方式和排障步骤见 [`Fla-npu仓CI部署教程.md`](Fla-npu仓CI部署教程.md)。

## 强行合入

`weinachuan` 的强行合入权限需要在 GitHub 分支保护中配置 PR bypass allowance。仓库管理员可使用以下脚本应用 `main` 分支保护：

```sh
GITHUB_TOKEN=<admin-token> scripts/github/apply_branch_protection.sh main
```

该 token 需要具备仓库 administration 写权限。脚本会要求 `NPU CI / 手动验证` 通过，并配置 2 个 approval、Code Owners review、stale review dismiss、Admin 也必须遵守分支保护，以及仅 `weinachuan` 具备 bypass。
