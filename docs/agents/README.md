# AI 开发指南

本目录用于沉淀给 AI coding agent 阅读的开发原理、方法论、验证流程和经验总结。根目录 `AGENTS.md` 保持为短入口；需要深入开发时，再阅读本目录中与任务相关的文档。

## 使用方式

- 修改算子实现前，先读 [`operator-development.md`](operator-development.md)。
- 设计验证方案或整理测试结果前，先读 [`validation.md`](validation.md)。
- 遇到精度、ABI、生成代码、跨 SOC 或提交范围问题时，先读 [`lessons.md`](lessons.md)。

## 编写原则

- 这里记录可复用的方法论和经验，不记录个人机器、内网路径、临时目录、账号或 token。
- 新增经验时优先写触发条件、判断方法、推荐处理方式，避免只写口号。
- 与具体算子强绑定的细节优先放在该算子的 README 或设计文档中，再从这里链接过去。
- 与仓库规则、PR 模板、CI 机制有关的事实，以根目录和 `.github/` 下的现有文件为准。
