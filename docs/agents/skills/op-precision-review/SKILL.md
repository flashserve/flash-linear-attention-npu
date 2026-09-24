---
name: op-precision-review
description: Ascend C 算子精度静态检视。用于对算子（transformer/cv/nn/math 四仓）做代码白盒检视或 PR 精度检视，按 lv0-lv3 分级和 13 类细粒度根因定位问题，输出可核对到文件行号的检视意见，并按漏警率/虚警率验收检视结果。
metadata:
  short-description: 算子精度静态检视（lv0-lv3 分级）
---

# 算子精度静态检视

按"漏警优先"的原则对 Ascend C 算子做精度静态检视：把代码/PR 改动映射到 lv0-lv3 四个级别和
13 类细粒度根因，输出可核对、可验收的检视意见。只输出检视意见，不修改代码。

## 何时使用

- 对算子实现做整体白盒检视（新增算子、较大重构、送检前的自查）。
- 对 PR diff 做精度检视（判断改动是否引入精度/功能风险）。
- 对历史精度问题做回归检视（用已知 issue 集验收检视能力）。

不适用于：纯性能优化评估、编码风格巡检、文档校对。

## 检视分级

| 级别 | 根因类别 | 判定所需材料 | 典型手法 |
| --- | --- | --- | --- |
| lv0 | 索引/计数、NaN/Inf/除0、空值/null、数值溢出/位宽 | 只需 scalar 数据流与计算逻辑 | 顺着标量变量的定义-使用链核对 |
| lv1 | shape/校验缺失、模板/分支缺失、dtype/format | 算子接口文档的每个参数说明与"约束"条目 | 文档约束 ↔ 代码拦截双向核对 |
| lv2 | 内存容量/超限、越界/地址偏移、bias/辅助输入 | AscendC API 文档 + 硬件手册（片上资源规格） | 逐个 API/片上资源核对用法与容量 |
| lv3 | 同步/事件时序、tiling/分核错误、累加顺序/舍入 | 算子详设；无详设时按参考流程自行推导 | 生成分核/内存/流水报告后做三项分析 |

级别只决定"需要读什么材料"，不决定问题严重性；同一处代码可以同时命中多个类别。

## 工作流

1. **取范围**：确认被检视对象（仓、算子、PR diff、基线 commit）与目标级别。
   默认执行 lv0-lv2；被检视算子是复杂算子（多核、多 Stage、跨核握手、融合）时叠加 lv3。
2. **建映射**：把被检视文件按层归类（`op_host`、`op_kernel`、`op_api`、调用层），
   列出本次改动涉及的函数/分支，作为后续逐条检视的清单。
   先排除环境与构建类错误：过滤构建缺依赖配置、算子未命中当前 OPP 时会报
   `aclnnStatus=561103` / `Config_Error(EZ1013): ... the JSON configuration file of operator ...
   cannot be found` 一类信息，这属于依赖/安装问题（排查顺序见
   [`../../reference/04-operator-development/engineering-structure.md`](../../reference/04-operator-development/engineering-structure.md) §3.1、§5.4），
   不要按精度根因统计。
3. **逐级检视**：按 lv0 → lv1 → lv2 → lv3 执行，每级读对应 reference：
   - 类别定义、范围与分级依据：[`references/01-scope-and-categories.md`](references/01-scope-and-categories.md)
   - lv0 / lv1 / lv2 检查清单：[`references/02-lv0-lv2-checklist.md`](references/02-lv0-lv2-checklist.md)
   - lv3 五步白盒流程：[`references/03-lv3-whitebox-flow.md`](references/03-lv3-whitebox-flow.md)
4. **写意见**：每条按 [`references/04-output-format.md`](references/04-output-format.md) 的格式输出，
   必须带文件、行号、代码片段、触发条件、影响和修改方向。
5. **自检**：对每条意见自问"这条能被被检视方用文件行号复现吗"；不能复现的降级为"待确认"或不写。
   同一处根因只报一次，合并同类项。
6. **验收（可选）**：有 issue 标注集时，用 [`scripts/review_score.py`](scripts/review_score.py)
   统计逐级别漏警率与虚警，判定是否达标；分工与验收规则见
   [`references/05-acceptance-and-roles.md`](references/05-acceptance-and-roles.md)。

## 硬性要求

1. **只输出检视意见**，不改代码、不给总结、不写"未发现问题项"。仓库要求格式时沿用根 `AGENTS.md`
   的代码检视输出规范，并在其中补 `精度级别` 与 `细粒度根因` 两行（见 `04-output-format.md`）。
2. **每条意见必须可核对**：文件名 + 行号 + 代码片段 + 触发条件 + 影响 + 修改方向，缺一不可。
3. **不得用"看不懂"作为结论**：材料不足时写成"缺少 X（详设/AscendC API 文档/硬件手册），无法判定 Y"，
   并列出需要补充的材料。
4. **不得靠降级手段消解问题**：不把"收窄输入 range""跳过用例""放宽阈值""屏蔽比较区域"当作修改建议。
5. **额外发现要保留**：issue 未记录但确实存在的问题按额外意见输出，并标注"额外发现"，作为 bonus 指标，
   不因为不在 issue 中而丢弃。
6. **公开输出脱敏**：结论中不写服务器、账号、绝对路径、临时目录、日志路径、token。
7. 输出语言默认中文；引用代码片段保持原文。

## 责任田扩展

`02-lv0-lv2-checklist.md` 与 `03-lv3-whitebox-flow.md` 是按类别维护的清单，各算子责任田可以在对应类别下追加
本领域的常见问题点（格式见 05-acceptance-and-roles.md），追加内容必须能落到具体代码模式，不写泛化提醒。
