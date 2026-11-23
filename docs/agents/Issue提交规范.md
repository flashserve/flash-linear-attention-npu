# Issue 提交规范

本规则适用于网页以及 Agent 通过 GitHub CLI、MCP、REST/GraphQL API 创建或修改的普通 Issue。不能用自由格式提交绕过网页表单的必填项；标题前缀和已有标签不能代替正文信息。

## 提交前

1. 读取目标仓库默认分支中的本规则和对应 `.github/ISSUE_TEMPLATE/` 表单。
2. 查询目标仓库的分支和现有标签，明确类型、涉及分支和算子或模块。不能从本地工作分支推断涉及分支，也不能默认填写 `main`；有多个涉及分支时使用逗号分隔。
3. 正文必须有且仅有一个 `### Issue 类型`、`### 涉及分支`、`### 涉及算子或模块`。类型为 `bug`、`enhancement` 或 `question`，涉及分支必须真实存在于目标仓库。
4. 按对应表单填写以下必填三级标题及实际内容，不保留空白、`_No response_`、`TODO`、`待补充` 或默认验收占位文字。

| 类型 | 必填正文标题 |
| --- | --- |
| `bug` | 环境信息、最小复现、完整报错、期望行为 |
| `enhancement` | 背景、问题、期望实现的功能、验收要求 |
| `question` | 背景、问题、期望实现的功能 |

精度或性能异常没有错误栈时，在“完整报错”写明实际异常结果和证据。需求的验收要求必须给出可核对的口径；无关维度说明不适用及原因，不编造实测结论。提交前检查正文、复现和日志，避免公开 token、账号、内网地址等敏感信息。

## 涉及分支与标签

- 正文是分类依据，仓库 Action 在校验通过后添加一个类型标签；不依赖外部提交者拥有打标签权限。
- 涉及 `main` 时复用 `main` 标签，其他分支使用 `branch:<分支名>`；过长名称保留前缀并附加稳定哈希，标签总长不超过 50 字符。多个涉及分支会生成多个标签。
- 算子名可写 `chunk_kda_fwd` 或 `op:chunk_kda_fwd`，多个算子或模块用英文/中文逗号或换行分隔。自动匹配仓库已有的 `op:*` 标签；公共模块可写 `runtime`、`build`、`ci`、`docs` 等，不凭自由文本创建算子标签。
- 类型标签、`main`、`branch:*`、`op:*` 按正文同步；修改分类时，相关旧标签会被移除。维护者调整这些分类也应同步修改正文。
- `v26.*` 是 **Bug 的目标修复版本排期**，由现有 `version-tags.yml` 与维护者 `/version` 命令管理。不能把“问题出现在 v26.x 分支”直接转换成“承诺在该版本修复”。本校验不修改版本标签、`needs-version` 或其他业务标签。

## 推荐提交入口

需 Node.js 20+；实际提交另需已登录的 GitHub CLI。先将类型对应的正文写入 UTF-8 文件，再运行：

```sh
node scripts/github/create_issue.js \
  --repo flashserve/flash-linear-attention-npu \
  --title "[Bug]: 问题摘要" \
  --type bug --branch main --module chunk_kda_fwd \
  --body-file issue.md --dry-run
```

确认正文与实际情况一致后去掉 `--dry-run` 创建。脚本复用仓库校验器，实际提交前联网核对分支；dry-run 只检查本地字段，不联网。正文已有元数据标题时，参数必须与正文一致。脚本通过标准输入传递正文，无需自行拼接 shell 命令或转义 Markdown。

使用其他工具时生成相同的正文。例如 Bug：

```markdown
### Issue 类型
bug

### 涉及分支
main

### 涉及算子或模块
chunk_kda_fwd

### 环境信息
粘贴经检查的 scripts/collect_public_env.py 输出。

### 最小复现
提供实际执行的最小代码、命令与输入参数。

### 完整报错
提供实际异常结果或完整错误栈。

### 期望行为
说明预期输出与实际结果的差异。
```

示例中的内容需替换为真实信息。校验器只检查结构、必填内容和分支存在性，信息真实性及验收口径仍需提交者和维护者核对。

## 提交后

1. 等待默认分支的 `Issue 规范校验` 完成，查询 Issue 标签与机器人评论；不能只凭创建 API 返回成功就认定规范校验通过。
2. 出现 `needs-info` 时编辑**原 Issue 正文**补齐。只在评论区回复不会修改正文，也不会触发本校验。
3. 校验通过后自动同步分类标签、移除 `needs-info`，并将原有的提示评论更新为通过。重复检查不重复发提示；人工评论即使含有机器人标记也不会被修改。

本阶段仅标记和提醒，不自动关闭或重开 Issue，不占用 NPU runner。通过 `issues` 的 opened、edited、reopened、labeled、unlabeled、transferred 事件检查当前 Issue，不批量扫描历史记录；已关闭的 Issue、PR、带 `committer-promotion` 或 `requirements-index` 标签的独立治理 Issue 跳过。

`GITHUB_TOKEN` 创建或更新的 Issue 通常不会再次触发 Actions。此类自动化应使用可触发事件的 GitHub App/PAT，或由有权限者手动运行 `Issue 规范校验` 并填写 Issue 编号。手动入口同样只执行默认分支的规则。工作流必须合入默认分支后才会开始生效，fork 仓库默认不运行。
