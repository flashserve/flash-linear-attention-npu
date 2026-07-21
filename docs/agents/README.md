# AI 开发指南

本目录用于沉淀给 AI coding agent 阅读的开发原理、方法论、验证流程和经验总结。根目录 `AGENTS.md` 保持为短入口；需要深入开发时，再阅读本目录中与任务相关的文档。

## 使用方式

- 新增、替换或重要修改算子前，先读 [`../operator-development-standard.md`](../operator-development-standard.md)；功能、平台、调用通路、编码、测试和文档交付以该规范为准。
- 修改算子实现前，先读 [`operator-development.md`](operator-development.md)。
- 修改 `fla_npu` Python runtime、wheel 打包、OPP 安装、依赖版本门禁、legacy `torch.ops.npu` 兼容路径、多卡 device guard、stream、autograd、alias/mutation 或图编译适配前，先读 [`torch-npu-decoupled-architecture.md`](torch-npu-decoupled-architecture.md)；该文档也包含构建/运行组件关系和基础术语表。
- 设计验证方案或整理测试结果前，先读 [`validation.md`](validation.md)。
- 遇到精度、ABI、生成代码、跨 SOC 或提交范围问题时，先读 [`lessons.md`](lessons.md)。

## 编写原则

- 这里记录可复用的方法论和经验，不记录个人机器、内网路径、临时目录、账号或 token。
- 新增经验时优先写触发条件、判断方法、推荐处理方式，避免只写口号。
- 与具体算子强绑定的能力说明放在算子 `README.md`，设计方案放在 `docs/design.md`，全部公开 API 和调用示例放在统一 `docs/api.md`，再从这里链接过去。
- 与算子准入有关的要求以 `docs/operator-development-standard.md` 为准；与仓库规则、PR 模板、CI 机制有关的事实，以根目录和 `.github/` 下的现有文件为准。
