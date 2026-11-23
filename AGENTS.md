# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件保留开发流程来源、跨任务强制约束和任务路由；具体方法、阶段输出和完成条件按任务读取 `docs/agents/` 中对应文件。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 本仓开发流程来源

- 在本仓执行开发任务时，只使用本文件和 `docs/agents/` 中定义的开发流程。
- 仓内代码和其他技术文档按本文件与 `docs/agents/` 规定的阶段作为接口证据、工程资料或实现参考。
- 如其他流程说明与本文件或 `docs/agents/` 冲突，以本文件和 `docs/agents/` 为准。

## Agent 提交 Issue 的强制要求

- 创建或更新 Issue 前必须读取 [`docs/agents/issue-submission.md`](docs/agents/issue-submission.md) 和当前仓库对应的 Issue 模板。网页、REST API、GraphQL、`gh` CLI、MCP 等入口一律遵守，不能因工具不强制而省略标签。
- 必须按真实用途选择且仅选择一个主分类标签：缺陷用 `bug`，需求或改进用 `enhancement`，Committer 晋升用 `committer-promotion`；纯咨询保留 `question`，不得用咨询或晋升分类绕过缺陷、需求的要求。
- `bug` / `enhancement` 必须同时带至少一个实际需要合入的目标分支标签，例如 `main`、`v26.11.0`，多目标逐一添加。目标必须有任务依据且远端分支真实存在，禁止为通过校验随意默认 `main`。纯咨询和晋升申请不强制分支标签。
- API 创建请求必须显式传入 `labels`，CLI 必须显式传入 `--label`。禁止先发布无标签 Issue 再等待机器人或他人补齐；缺少分类、目标分支依据或标签写入权限时，保留草稿并说明缺项。
- 创建或更新后必须读回远端 Issue 的实际 `labels`，核对主分类及全部目标分支标签。缺失时修复同一 Issue 并再次读回；未确认标签齐全前，不得报告提交完成。完整流程和失败处理见上述提交规范。

## 任务路由

| 任务类型 | 执行流程或必读内容 |
| --- | --- |
| 新增算子或新公开接口 | 按顺序执行 `docs/agents/01-接口确认.md` → `02-标杆生成.md` → `03-方案设计.md` → `04-算子开发.md` → `05-算子测试.md` |
| 修改既有算子的接口或功能 | 先读取当前接口、CPU 标杆、设计、实现和测试，再按 `docs/agents/01-接口确认.md` 的“既有算子修改路由”确定起始阶段 |
| 修改既有算子的内部实现或优化性能 | 读取当前算子的设计、实现和测试，按 `docs/agents/03-方案设计.md` → `04-算子开发.md` → `05-算子测试.md` 执行 |
| 修改或新增算子测试，包括 ATK 用例 | `docs/agents/05-算子测试.md`、`tests/atk/README.md` 和当前算子的 ATK README；涉及 CPU 标杆对齐、输入值域、精度规则或精度失败时，同时按下一行进入精度路由 |
| 对齐 CPU 标杆、校准精度值域或定位精度问题 | 先读取 `docs/agents/reference/精度对比与定位.md` 选择场景，再按该文件指向的执行方法操作 |
| 修改公共组件、公共 ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 创建或更新 Issue，包括为 PR 创建关联 Issue | [`docs/agents/issue-submission.md`](docs/agents/issue-submission.md)、当前 Issue 模板和 `docs/repository-rules.md` |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README，并采用 Triton 对应的实现约束 |

前一阶段的结论发生变化时，从最早受影响的阶段重新执行后续阶段。

目录索引、阶段输入输出和按任务阅读顺序见 `docs/agents/README.md`。
