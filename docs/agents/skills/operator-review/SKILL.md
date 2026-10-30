---
name: operator-review
description: 对 flash-linear-attention-npu 的算子、PR diff 或新增算子资产做四层静态检视：功能正确性、工程结构（含已发布 Python 接口兼容性）、DFX 拦截与报错、文档正确性与易用性。需要结合仓内工程样例、历史 issue 分类和可执行静态检查时使用。
metadata:
  short-description: 算子四层静态检视与 issue 回归覆盖
---

# 算子四层静态检视

按“功能正确性 → 工程结构 → DFX → 文档”顺序检视，漏报优先。skill 只输出检视意见，不修改代码；
用户明确要求修复时，再按检视结果另行实施。

## 输入与范围

支持四种范围：

| 范围 | 示例 |
| --- | --- |
| 单算子当前实现 | `--op chunk_kda_fwd` |
| PR 相对主线 diff | `--diff-base origin/main` |
| 指定文件 | `--path <file> --path <dir>` |
| 全仓存量审计 | `--all` |

开始前确认目标 commit、算子名、调用层和验证平台。PR 检视以 diff 为主，同时把同一算子中会被本次
改动破坏的旧路径纳入范围；全仓审计与 PR 门禁分开报告。

## 四层检查

### 1. 功能正确性

读取 [`references/functional-scope-and-categories.md`](references/functional-scope-and-categories.md)、
[`references/functional-lv0-lv2-checklist.md`](references/functional-lv0-lv2-checklist.md) 和
[`references/functional-lv3-whitebox-flow.md`](references/functional-lv3-whitebox-flow.md)。

先加载 [`references/issue-rule-map.md`](references/issue-rule-map.md) 与
[`references/issue-catalog.md`](references/issue-catalog.md)。所有当前已 `COMPLETED` 的 issue 都必须出现在
分类表中并映射至少一个根因规则；用
[`scripts/check_issue_coverage.py`](scripts/check_issue_coverage.py) 做机器校验。检视当前算子时，
优先核对带 `op:<算子名>` 标签的历史 issue，再检查跨算子公共规则。

功能检视至少核对：

- 接口、CPU 标杆、设计、实现和测试是否描述同一套数学语义；
- shape/rank/dtype/layout/varlen/optional output 的每个分支是否有唯一且可达的实现；
- 分核、地址、workspace、同步、事件、累加序和 tail 是否与设计一致；
- 历史 issue 中的触发条件是否仍可能通过另一入口、另一布局或另一平台复现。

### 2. 工程结构

读取 [`references/engineering-structure-review.md`](references/engineering-structure-review.md)，再读取
[`engineering-example/README.md`](references/engineering-example/README.md)。
`engineering-example/` 是完整工程 reference，内含
[`engineering-structure.md`](references/engineering-example/engineering-structure.md)
规则和逐文件样例。

重点检查目录职责、必需文件、平台目录、调用层归属、测试位置、冗余文件和构建依赖。新增文件必须
能在工程样例中找到对应角色；不能解释职责的文件应先删除或归档。

只要 diff 涉及 `fla_npu.ops.ascendc` 公开入口、Stable-ABI schema、Python wrapper、参数或返回
tuple，都必须执行公开接口兼容性检查。以改动前的已发布 tag/基线提交为准，逐项比较入口名、参数名与
顺序、positional/keyword-only 属性、是否必选、默认值、类型和语义、返回槽位，以及 stable、ctypes、
legacy 入口是否继续提供同一调用形状。只证明当前代码、当前 YAML、当前文档彼此一致，不构成兼容性证据。
版本化契约的完整快照见
[`references/public-api-contract.v26.9.1.yaml`](references/public-api-contract.v26.9.1.yaml)。
契约必须覆盖 `_ASCENDC_OPS` 中全部未日落公开入口，并与 `docs/op_list.md` 的算子责任面一致；
不得只保留当前 diff 涉及的几个入口。

### 3. DFX 设计

读取 [`references/dfx-error-review.md`](references/dfx-error-review.md)。

aclnn 与 tiling 的公开约束必须在 host 侧拦截，并通过 `op_common/log/log.h` 的结构化宏向调用方
返回实际值、期望值和原因。仅返回 `161002`、仅打印日志但未 report、使用 `CHECK_RET`/`CHECK_COND`
丢失上下文，均按 DFX 问题报告。核对代码拦截、返回码、报错文本、负向测试和文档约束是否闭环。

### 4. 文档正确性与易用性

读取 [`references/documentation-review.md`](references/documentation-review.md)。

检查 README、`docs/api.md`、`docs/design.md`、示例和根级索引。文档格式、公开入口、Shape 语义、
支持范围、已知限制、返回码、示例命令和实际代码必须一致；按“文档约束 → 代码拦截 → 返回码 →
负向测试”双向核对。

## 可执行检查

在仓库根执行：

```bash
python3 docs/agents/skills/operator-review/scripts/check_issue_coverage.py
python3 docs/agents/skills/operator-review/scripts/static_review.py \
  --repo . --op <op_name>
python3 docs/agents/skills/operator-review/scripts/static_review.py \
  --repo . --diff-base origin/main --fail-on-warning
```

Issue 分类快照需要更新时执行：

```bash
python3 docs/agents/skills/operator-review/scripts/sync_issue_corpus.py
python3 docs/agents/skills/operator-review/scripts/check_issue_coverage.py
```

`static_review.py --format json` 可供 CI 或其他 agent 消费。脚本负责可机械判定的事项；脚本零告警
不等于功能正确，功能、流水和文档语义仍必须按 references 白盒核对。

## 完成条件

只有同时满足以下条件才能结束检视：

1. issue corpus 覆盖全部 `COMPLETED` issue，且每条至少有一个规则；
2. 目标算子的相关 issue 规则逐条检查，结论能落到文件行号；
3. 工程结构、DFX、文档检查完成；
4. diff 涉及公开接口时，已与改动前的已发布基线逐项比较 Python 签名、默认值、返回值和各后端入口；
5. 每条意见说明问题分类、位置、问题、可能导致的结果和可执行修改方法；
6. 不把“未发现问题”写成问题项，不用无问题总结代替具体检视条目。

输出格式见 [`references/review-output-format.md`](references/review-output-format.md)。如仓库根
`AGENTS.md` 对格式有更严格规定，以根规范为准。
