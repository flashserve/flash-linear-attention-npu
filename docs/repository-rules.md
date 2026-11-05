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
- `torch_custom/fla_npu/csrc/src/stable_*.cpp` 中的 dispatcher schema、适配函数及 ACLNN 调用
- `torch_custom/fla_npu/csrc/include/stable/**` 中的公共 ABI、boxed 参数与 tensor 描述符契约
- `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py` 中的公共 Python 签名及参数转换

如果 PR push 了新 commit，相关 ABI 检视也应重新确认。

## NPU CI 门禁

检查项的 context 名称、顺序和阶段映射统一维护在 `.github/scripts/npu_ci_checks.json`；触发、默认状态、结果发布、异常收敛和分支保护配置均读取这份定义。非默认目标分支按固定 base commit 读取其定义，尚未迁移的旧分支兼容读取原 workflow 中的 context。默认状态检查或写入异常时，workflow 失败，并尽力将当前及历史 NPU 状态写为 failure，使旧成功结果失效。

NPU 资源有限，`NPU CI` 不会由 PR 新建、重开或 push 新 commit 自动执行。发生这些事件时，默认状态 workflow 会给当前 head commit 写入以下 6 个 pending 状态，描述为“未执行”，并在 PR 评论区提示可请求 CI 触发名单中的账号触发：

1. `NPU CI / A2+A5 / 01 环境、wheel 与运行时契约`
2. `NPU CI / A2+A5 / 02 全量 OPP 构建`
3. `NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局`
4. `NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配`
5. `NPU CI / A2+A5 / 05 GDR Example/ST`
6. `NPU CI / A2+A5 / 06 报告与 commit 校验`

一次触发固定并行验证 A2（`ascend910b`）和 A5（`ascend950`）。每个平台内部按顺序执行前 5 个分项；任一分项失败时，后续分项标记为未执行，不再继续运行。任一平台 job 失败后，Actions 会取消仍在排队或运行的另一平台 job，避免继续占用 NPU；汇总结果仍按失败收敛。可执行分项失败时，关键诊断会给出 `CI_STAGE=<分项>`、CI 模式和目标 SOC 对应的 `bash ci/run_ci_container.sh` 复现命令。两个 self-hosted job 只上传带平台、SOC、commit、Actions run 和 attempt 身份的分项及精度结果；GitHub-hosted 汇总 job 校验两平台结果后发布 6 个 aggregate context，其中第 6 项负责报告完整性和 PR commit 一致性。6 个 context 必须来自同一次 A2+A5 Actions run；任一平台缺席、任一已执行分项失败、后续分项未执行、精度未通过或报告身份不匹配，都不能满足门禁。

A5 job 使用 `self-hosted,linux,npu,ascend950,flash-linear-attention-npu-a5` 标签，不限制 CPU 架构，可由 x64 或 ARM64 原生 A5 runner 承接。一次触发仍只有 A2、A5 两个设备 job；A5 只选择一台匹配 runner，不增加第三项架构门禁。每台 A5 需预置 `fla-npu-ci:9.1.0-950` 的本机原生架构镜像，并使用配套 CANN 9.1.0、Python 3.12、PyTorch/torch_npu 2.12.0；不得用 QEMU 或跨架构镜像替代原生 NPU 验证。

A2/A5 统一使用 Git 获取源码：先直连 checkout prepare 固定的 base commit，失败后调用主机 `CI_PROXY_SCRIPT` 的 `proxyon` 并重试；随后获取目标分支与 PR head 的 Git ref，并在固定 base 上本地合并 PR。fetch 同样支持直连失败后启用代理重试。设备端的版本校验位于源码获取之后，只用本地 Git 核对 ref、固定 base/head commit 和合并结果祖先，不调用 GitHub API。PR 是否仍 open、非草稿、未合并，以及是否切换目标分支，由 GitHub-hosted prepare 和 finalize 联网检查；设备端离线检查不能判断这些 GitHub 元数据。

PR 合入前，当前 head commit 的 6 个 NPU 状态和自动执行的 `CI 契约测试` 必须全部成功；当 PR 更新 commit 后，旧 commit 上的 CI 成功状态自动失效，需要 CI 触发名单中的账号重新触发 NPU CI。

新版 CI 在复用成功结果或等待已有任务前，还会读取该任务的运行信息，确认目标分支及 base commit 与本次请求一致；无法确认时启动新验证。最终发布结果前再次检查 PR 状态和 head/base，运行期间目标分支切换或 base commit 更新会使本次结果失效。

默认分支的 [`.github/npu-ci-users.json`](../.github/npu-ci-users.json) 独立维护可触发用户名；仓库读写权限不直接赋予触发权。默认评论分发和新版手动执行均校验初次触发人的名单，PR 自身修改名单不会授予权限。

Actions 重跑权限由 GitHub 平台控制。新版 CI 重跑仍检查 PR 状态及固定 head/base/workflow 版本。目标 workflow 支持完整四个 `expected_*` 输入时，评论触发自动携带快照，版本未变化时可使用 `Re-run all jobs`。新版 Actions 手动首次默认不填快照；此类运行重新执行 prepare（`Re-run all jobs` 或 prepare 失败后重跑）会被拒绝，应新建 `Run workflow`，避免取到新的 head。`Re-run failed jobs` 若复用已成功的 prepare，则可正常重试，设备仍离线检查固定版本，finalize 再检查当前 PR 元数据；旧 workflow 的重跑行为保持原样。已创建的 Actions run 重跑仍使用原 workflow，不会切换到修复后的流程。需要验证 PR 中的 `.github/workflows/ci.yml` 修改时，使用下述评论入口创建新的 PR workflow 运行。

默认评论入口和新版 CI 会检查非 Markdown 的 `ci/` 或 `.github/` 变更：只有 PR 创建者（`pr.user.login`）具有该仓库 Admin 权限才允许执行，否则门禁直接失败。PR 作者无需位于 CI 触发名单，初次触发人仍须在名单中：名单内只读用户在新版 CI 中可以触发 Admin 作者的 PR，名单内 Admin 不能替非 Admin 作者放行 CI 修改。文件检测读取完整列表并检查重命名两侧；列表不完整、作者权限读取失败均拒绝。纯 CI 文件变更检测本身不新增编译规则。

新版 CI 设备上的 A2/A5 一律使用当前 PR 合并结果中的 `ci/`，不再检出或叠加可信 CI 目录。默认分支的评论分发器先完成触发名单、PR 作者 Admin、目标分支及固定版本检查，再选择实际 workflow 来源：

- 未修改 `.github/workflows/ci.yml` 时，继续使用 PR 目标分支的 workflow；仅修改 `ci/` 也走这条路径。未更新 CI 的旧目标分支保持兼容。
- 修改 `.github/workflows/ci.yml` 时，使用固定 base/head 的 PR merge commit，创建指向该 commit 的临时 tag `npu-ci/pr-<编号>/<40 位 merge SHA>`，并从这个 tag 分发 PR 自身的 workflow。PR YAML 中必须声明 `workflow_source` 和完整 `expected_*` 输入，分发值为 `workflow_source=pr-merge`；缺少契约时明确失败，提示先补齐当前 PR 的 workflow。设备任务开始前的 checkout、代理回退等 Actions 步骤也会采用 PR 修改。

默认分发器使用 `npu-ci-dispatch` Environment，通过其中的可选 secret `NPU_CI_DISPATCH_TOKEN` 创建临时引用；该凭据需要仓库 Contents 和 Workflows 写权限。管理员须预先创建此环境，将部署规则设为自定义规则，仅允许 branch 类型的 `main`，不允许任何 tag，再添加 secret。不要将同名凭据放入仓库级或组织级 Actions secrets：PR workflow 通过主仓库 tag 执行，可能显式引用这些凭据。分发器不会通过 inputs 或 job 环境将此 token 转交给 PR 运行，设备和汇总 job 不使用此 Environment。未配置凭据时尝试当前 workflow 的 `GITHUB_TOKEN`，权限不足会明确报错。新的分发器须先合入默认分支一次，之后即可在 CI 修改合入前验证 PR YAML；仅更新 PR 不能改变默认分支正在执行的分发器。

默认分支的 `NPU CI 临时 workflow 引用清理` 在 `NPU CI` 完成后清理对应 tag。它只校验 GitHub 运行信息和精确 tag 目标，不检出或执行 PR 代码；同一 tag 仍有其他运行排队或执行时保留引用。清理与 PR 后续是否新增 commit 无关，不删除其他 tag。

若创建 tag 后分发失败，保留该固定引用供重试，避免删除并发请求正在使用的 tag。放弃重试时，管理员确认没有相关运行后可删除错误消息中给出的精确 tag。

默认评论入口和新版手动执行的权限要求如下，CI 名单不会授予 GitHub 平台权限：

| 触发入口 | 所需权限 |
| --- | --- |
| PR 评论 `/run-npu-ci` | 默认分支 CI 名单；不要求仓库 write 权限 |
| 新版 CI 的 Actions 页面、CLI 或 REST 手动分发 | 默认分支 CI 名单，并具备仓库 write 权限 |

GitHub 要求[手动运行 workflow 的用户具有仓库 write 权限](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)；REST 的 fine-grained token 还须具有 `Actions: write`，classic PAT 或 OAuth token 须具有 `repo` scope，见[分发 API 权限要求](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)。涉及 CI 文件修改时，默认评论入口和新版 workflow 均要求 PR 作者具有仓库 Admin 权限。

1. 在 GitHub `Actions` 页面选择 `NPU CI`，将 `Branch` 设为 PR 当前目标分支，点击 `Run workflow`，填写 PR 编号和 `quick/full` 模式；正式门禁请求不填写 `ops`，仅在定向诊断时填写 `ops`。
2. 在 PR 评论区发送命令：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

默认评论入口和新版 CI 仅接受 open、未合并且非草稿的 PR，目标分支必须受 GitHub 分支保护。默认分支上的 `NPU CI 评论分发` 校验请求后，按上述规则选择目标分支或固定 PR merge tag 上的 `.github/workflows/ci.yml`，不支持用户通过 `ci_ref` 或 `branch` 覆盖。Actions 页面、CLI 和 REST 的普通手动分发选择 PR 当前目标分支；验证 PR 修改后的 YAML 应使用评论入口，由分发器创建固定引用。

评论分发读取所选 workflow 声明的输入，仅发送其支持的参数。未修改 CI YAML 的旧分支无需更新 CI 或声明新版契约即可触发；修改 CI YAML 的 PR 则须满足 `pr-merge` 契约。分发器自动附带支持的 `expected_*` 版本快照，由 prepare 校验，用户无需手填。

名单及 CI 修改作者 Admin 规则在默认评论入口校验，旧分支仍执行自己的权限检查和 CI 流程。例如旧 workflow 仍要求 `requested_by` 为 Admin 时，默认入口通过后也须满足该要求；不会自动获得新版名单、快照或重跑行为。

带 `ops=` 的触发是编译定向诊断：只顺序执行第 01 项环境契约和第 02 项指定算子 OPP 构建，不执行 GDR 精度或后续安装分项；它只发布 `NPU CI / A2+A5 / 定向诊断`，不会写入或覆盖 01-06 正式门禁状态。用于合入门禁的运行必须省略 `ops`。

如果当前 commit 的 6 个 aggregate context 已由同一次 A2+A5 run 通过，且模式、算子范围和 workflow 版本匹配，重复触发会跳过设备执行。同一 PR 通过 workflow concurrency 串行化，`cancel-in-progress: false`，重复评论不会取消正在运行的任务；排队请求进入 prepare 后再做去重。GitHub 默认保留一个运行和一个排队请求，较新的排队请求可能替换较早的请求。每次进入 prepare 并生成触发反馈时新建评论，汇总仅更新本次 `comment_id`，不覆盖前一次运行的评论。runner 宿主机还会用设备锁避免多个任务抢同一张卡。

从 PR 到 A2/A5 的事件、YAML 来源、Secret 范围和每个 CI 文件的用途见 [`NPU-CI流程与文件说明.md`](NPU-CI流程与文件说明.md)。

仓库管理员应将上述 6 个 context 和 `CI 契约测试` 都配置为 `main` 分支必需状态检查。分项 context 使用带 `A2+A5` 的独立名称，避免旧版状态或 A2-only 历史成功状态被误认为已经覆盖当前门禁。`CI 契约测试` 在普通 GitHub-hosted runner 上自动执行 Python、Node 状态发布逻辑及分阶段调度测试，用于保护 CI 代码本身。

NPU CI 的 self-hosted runner、Docker 镜像、`--privileged`、触发方式和排障步骤见 [`Fla-npu仓CI部署教程.md`](Fla-npu仓CI部署教程.md)。

## 强行合入

`weinachuan` 的强行合入权限需要在 GitHub 分支保护中配置 PR bypass allowance。仓库管理员可使用以下脚本应用 `main` 分支保护：

```sh
GITHUB_TOKEN=<admin-token> scripts/github/apply_branch_protection.sh main
```

该 token 需要具备仓库 administration 写权限。脚本会要求上述 6 个 NPU CI 分项状态和 `CI 契约测试` 都通过，并配置 2 个 approval、Code Owners review、stale review dismiss、Admin 也必须遵守分支保护，以及仅 `weinachuan` 具备 bypass。
