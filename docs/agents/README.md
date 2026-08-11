# AI 开发指南

本目录沉淀给 AI coding agent 和开发者使用的基础知识、开发方法、优化方法、验证流程和经验。根目录 `AGENTS.md` 只负责全仓强制规则和任务路由；进入具体任务后，再按本页选择需要阅读的文档。

## 首次进入仓库

先读 [`foundation.md`](foundation.md)，建立项目组件、调用链、L2/L0、Tiling、workspace、OPP 和 wheel 的共同心智模型。构建、安装和测试命令以根目录 `README.md` 为准。

## 按任务阅读

| 任务 | 必读文档 |
|---|---|
| 新增或修改 Ascend C 算子 | [`operator-development.md`](operator-development.md) -> [`operator-coding-standard.md`](operator-coding-standard.md) -> [`operator-checklist.md`](operator-checklist.md) -> [`validation.md`](validation.md) |
| 算子性能设计或优化 | [`operator-coding-standard.md`](operator-coding-standard.md) -> [`operator-optimization/README.md`](operator-optimization/README.md) 按依赖类型和目标 SOC 路由 -> [`validation.md`](validation.md) |
| 修改 Python runtime、wheel、OPP 或兼容路径 | [`torch-npu-decoupled-architecture.md`](torch-npu-decoupled-architecture.md) -> [`validation.md`](validation.md) |
| 定位精度、ABI、生成代码或跨 SOC 问题 | [`lessons.md`](lessons.md) 和对应任务文档 |
| 整理交付和测试结果 | [`operator-checklist.md`](operator-checklist.md) -> [`validation.md`](validation.md) |

具体算子的 README 和设计文档只说明该算子的接口、语义、实现和验证，不作为其他算子可以直接复制的通用优化规则。

## 编写原则

- 这里记录可复用的方法论和经验，不记录个人机器、内网路径、临时目录、账号或 token。
- 新增经验时优先写触发条件、判断方法、推荐处理方式，避免只写口号。
- 通用优化方法按依赖模型、技术类别和 SOC 能力归档，不按具体算子组织。
- 具体算子的变量名、固定窗口数字、代码路径和性能结果不得反向写成全仓通用规则。
- 与仓库规则、PR 模板、CI 机制有关的事实，以根目录和 `.github/` 下的现有文件为准。
