# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件只保留开发流程来源和任务路由；具体方法、阶段输出和完成条件按任务读取 `docs/agents/` 中对应文件。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 本仓开发流程来源

- 在本仓执行开发任务时，只使用本文件和 `docs/agents/` 中定义的开发流程。
- 仓内代码和其他技术文档按本文件与 `docs/agents/` 规定的阶段作为接口证据、工程资料或实现参考。
- 如其他流程说明与本文件或 `docs/agents/` 冲突，以本文件和 `docs/agents/` 为准。

## 任务路由

| 任务类型 | 执行流程或必读内容 |
| --- | --- |
| 新增算子或新公开接口 | 按顺序执行 `docs/agents/01-interface-confirmation.md` → `02-reference-generation.md` → `03-solution-design.md` → `04-operator-development.md` → `05-operator-testing.md` |
| 修改既有算子的接口或功能 | 先读取当前接口、CPU 标杆、设计、实现和测试，再按 `docs/agents/01-interface-confirmation.md` 的“既有算子修改路由”确定起始阶段 |
| 修改既有算子的内部实现或优化性能 | 读取当前算子的设计、实现和测试，按 `docs/agents/03-solution-design.md` → `04-operator-development.md` → `05-operator-testing.md` 执行 |
| 修改或新增算子测试，包括 ATK 用例 | `docs/agents/05-operator-testing.md`、`tests/atk/README.md` 和当前算子的 ATK README |
| 修改公共组件、公共 ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README，并采用 Triton 对应的实现约束 |

前一阶段的结论发生变化时，从最早受影响的阶段重新执行后续阶段。

目录索引、阶段输入输出和按任务阅读顺序见 `docs/agents/README.md`。
