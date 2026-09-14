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

NPU 资源有限，`NPU CI` 不会由 PR 新建、重开或 push 新 commit 自动执行。发生这些事件时，默认状态 workflow 会给当前 head commit 写入以下 7 个 pending 状态，描述为“未执行”，并在 PR 评论区提示可请求仓库 Admin 权限账号触发：

1. `NPU CI / A2+A5 / 01 环境、wheel 与运行时契约`
2. `NPU CI / A2+A5 / 02 全量 OPP 构建`
3. `NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局`
4. `NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配`
5. `NPU CI / A2+A5 / 05 GDR Example/ST`
6. `NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装`
7. `NPU CI / A2+A5 / 07 报告与 commit 校验`

一次触发固定并行验证 A2（`ascend910b`）和 A5（`ascend950`）。每个平台内部按顺序执行前 6 个分项；任一分项失败时，后续分项标记为未执行，不再继续运行。可执行分项失败时，关键诊断会给出 `CI_STAGE=<分项>`、CI 模式和目标 SOC 对应的 `bash ci/run_ci_container.sh` 复现命令。两个 self-hosted job 只上传带平台、SOC、commit、Actions run 和 attempt 身份的分项及精度结果；GitHub-hosted 汇总 job 校验两平台结果后发布 7 个 aggregate context，其中第 7 项负责报告完整性和 PR commit 一致性。7 个 context 必须来自同一次 A2+A5 Actions run；任一平台缺席、任一已执行分项失败、后续分项未执行、精度未通过或报告身份不匹配，都不能满足门禁。

PR 合入前，当前 head commit 的 7 个 NPU 状态和自动执行的 `CI 契约测试` 必须全部成功；当 PR 更新 commit 后，旧 commit 上的 CI 成功状态自动失效，需要仓库 Admin 权限账号重新触发 NPU CI。

仓库 Admin 权限账号可以通过两种方式触发：

1. 在 GitHub `Actions` 页面选择 `NPU CI`，点击 `Run workflow`，填写 PR 编号和 `quick/full` 模式；正式门禁请求不填写 `ops`，仅在定向诊断时填写 `ops`。
2. 在 PR 评论区发送命令：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

带 `ops=` 的触发是编译定向诊断：只顺序执行第 01 项环境契约和第 02 项指定算子 OPP 构建，不执行 GDR 精度或后续安装分项；它只发布 `NPU CI / A2+A5 / 定向诊断`，不会写入或覆盖 01-07 正式门禁状态。用于合入门禁的运行必须省略 `ops`。

如果当前 commit 的 7 个 aggregate context 已由同一次 A2+A5 run 通过，重复触发会被跳过，不会再次占用 NPU。同一 PR 的触发通过 workflow concurrency 串行化；同一 commit 已有 A2+A5 NPU CI 处于排队或运行中时，重复评论只会更新机器人评论为“已在运行”，不会启动新的 runner job。runner 宿主机还会用 `/tmp/fla-npu-ci-npu-<id>.lock` 对物理 NPU 加锁，避免多个任务抢同一张卡。

仓库管理员应将上述 7 个 context 和 `CI 契约测试` 都配置为 `main` 分支必需状态检查。分项 context 使用带 `A2+A5` 的独立名称，避免旧版状态或 A2-only 历史成功状态被误认为已经覆盖当前门禁。`CI 契约测试` 在普通 GitHub-hosted runner 上自动执行 Python、Node 状态发布逻辑及分阶段调度测试，用于保护 CI 代码本身。

NPU CI 的 self-hosted runner、Docker 镜像、`--privileged`、触发方式和排障步骤见 [`Fla-npu仓CI部署教程.md`](Fla-npu仓CI部署教程.md)。

## 强行合入

`weinachuan` 的强行合入权限需要在 GitHub 分支保护中配置 PR bypass allowance。仓库管理员可使用以下脚本应用 `main` 分支保护：

```sh
GITHUB_TOKEN=<admin-token> scripts/github/apply_branch_protection.sh main
```

该 token 需要具备仓库 administration 写权限。脚本会要求上述 7 个 NPU CI 分项状态和 `CI 契约测试` 都通过，并配置 2 个 approval、Code Owners review、stale review dismiss、Admin 也必须遵守分支保护，以及仅 `weinachuan` 具备 bypass。
