# 历史 issue 检视规则

[`issue-catalog.md`](issue-catalog.md) 列出当前全部 `COMPLETED` issue 的分类，
[`issue-corpus.json`](issue-corpus.json) 是由
[`sync_issue_corpus.py`](../scripts/sync_issue_corpus.py) 生成的机器可读快照。

分类以 issue 标题和标签为主要证据，并至少分配一个规则。`NOT_PLANNED` 不纳入“已明确解决”
覆盖；对应记录保留在 JSON 中便于审计。新增 completed issue 后必须更新快照并通过
`check_issue_coverage.py`。

## 功能规则

| Rule | 检视内容 |
| --- | --- |
| `FUNC-SYNC` | 核内/跨核同步成对、event/flag 生命周期、ready/free 反向握手、多 Stage WAR/RAW、并发窗口；race 类优先 racecheck |
| `FUNC-PRECISION` | 公式、dtype、累加序、尾块、特殊值、NaN/Inf、CPU 标杆与 Stage 追溯 |
| `FUNC-DETERMINISM` | 分核顺序、归约顺序、workspace 初值、重复调用 bitwise 一致性 |
| `FUNC-MEMORY` | UB/L1/L0/GM 容量、地址区间、workspace 别名、未初始化读取、越界、泄漏 |
| `FUNC-VARLEN` | `cu_seqlens`/`chunk_indices` 配对、单调性、上下界、空序列、tail/partial chunk |
| `FUNC-SHAPE` | rank、dim、head 映射、scalar/empty tensor、shape 推导和平台范围 |
| `FUNC-DTYPE-FORMAT` | dtype 组合、format/layout、非连续 view、混合精度转换点 |
| `FUNC-OPTIONAL-OUTPUT` | 可空输出组合、输出 mask、返回槽位、原地/非原地语义 |
| `FUNC-STATE` | 初始/最终状态、chunk 间状态传播、gate、重计算保存值 |
| `FUNC-REGRESSION` | 历史 bug 的通用回归：保留原 shape/layout/dtype/平台/调用链复现并检查根因 |

## DFX 规则

| Rule | 检视内容 |
| --- | --- |
| `DFX-VALIDATION` | 文档公开约束是否有 host 侧拦截，非法组合是否可达 |
| `DFX-MESSAGE` | 是否使用结构化 report、是否包含参数/实际值/期望值/平台/模式原因、返回码是否一致 |

## 工程规则

| Rule | 检视内容 |
| --- | --- |
| `ENG-STRUCTURE` | 目录职责、文件唯一事实、入口分层、命名、ABI、冗余文件 |
| `ENG-BUILD` | CMake、依赖闭包、wheel/OPP、安装布局、导入链和构建参数 |
| `ENG-COMPATIBILITY` | SoC/架构路径、版本/能力探测、回退路径、平台目录和产物配套 |

## 文档、测试、CI 与性能规则

| Rule | 检视内容 |
| --- | --- |
| `DOC-FORMAT` | README/API/design/example 的格式、链接、版本配套和公开脱敏 |
| `DOC-CONSTRAINT` | 文档约束与代码拦截、返回码、负向用例双向一致 |
| `TEST-COVERAGE` | 主路径、泛化、边界、负向、平台、布局、回归和测试断言 |
| `CI-WORKFLOW` | CI 触发、commit 绑定、报告、标签、发布和自动刷新流程 |
| `PERF-REGRESSION` | 同一场景基线、目标、profiling 口径、回退和性能不退化 |
| `FEATURE-DELIVERY` | 新算子/新入口的完整交付闭环；仍按四层检查落地内容 |

## 使用方式

1. 运行 `static_review.py --op <op>` 得到该算子的历史 issue 编号；
2. 对每个编号读取 issue 标题、复现与修复说明；
3. 按规则 reference 检查当前代码是否还可能出现同类问题；
4. 新根因先加入规则表，再更新分类脚本和 corpus；
5. 不允许用“当前全量测试通过”替代对历史 issue 根因的逐条检查。
