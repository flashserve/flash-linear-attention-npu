<!--
示例文件：tests/atk/op_name/README.md

规范来源：目录与文件命名按本仓 tests/atk/README.md；内容质量要求参考 ATK 仓
skill/atk-quality-guard（SKILL.md、references/op-engineering.md、references/atk_user_guide.md、
templates/test_case_design_report.md）。skill 只是内容指导，不改变本仓命名。

注意事项：
  1. 三个源文件命名固定为仓内命名：<算子>.yaml（参数空间）、gen_<算子>.py（参数修正）、
     executor_<算子>.py（执行插件）。
  2. YAML 只声明参数空间，gen 只做修正；不要把 dtype 列表、shape 取值、attr 候选写进 gen。
  3. 用例必须使用算子真实的 inputs / attr / outputs，禁止 low_precision_marker / fp32_marker /
     case_spec 之类的占位参数。
  4. 用例 JSON 是 `atk case` 的产物，默认落在执行目录下的 result/<算子>/json/all_<算子>.json；
     需要固化的精度/性能/内存检测用例再按本仓要求拆成三份。不要拿旧 JSON 当本次结果。
  5. 精度失败不要改判、不要动算子源码：先区分"用例设计问题"与"算子实现问题"（精度问题报算子开发人员）。
  6. 不提交 atk_output/、result/、xlsx、profiling/sanitizer 日志等测试产物（固化用例 JSON 除外）。
-->

# OpName ATK 工程（示例）

## 命名与规范来源

| 维度 | 以谁为准 |
| --- | --- |
| 目录结构、文件命名、三份固化用例 JSON | 仓库根 `tests/atk/README.md`（算子目录不放测试）：`<算子>.yaml` + `gen_<算子>.py` + `executor_<算子>.py` |
| YAML 字段要求、约束修正方法、用例覆盖要求 | ATK 仓 `skill/atk-quality-guard`（内容指导，不改命名） |
| 算子输入/属性/输出限制 | 算子 [`README.md`](../../../fla/ops/ascendc/ops_classify/op_name/README.md)（唯一定义来源） |

**用例参数必须是算子真实签名**：yaml、gen、executor、固化 JSON 里出现的参数名，必须与算子 README 的
输入表、属性表逐项对应（本示例：`x`、`g`、`a_log`、`initial_state`、`cu_seqlens`、`chunk_indices`、
`layout`、`chunk_size`、`scale`、`epsilon`、`return_saved`，输出 `y`、`state`、`x_norm`）。
禁止用 `low_precision_marker` / `fp32_marker` / `case_spec` 这类占位参数承载参数空间、shape/dtype 档位
或用例元数据；那类写法会让用例参数表与算子签名脱钩，检视时看不出真实的 dtype × shape × Attr 组合。

## 文件职责

| 文件 | 职责 | 规范要点 |
| --- | --- | --- |
| `op_name.yaml` | 声明参数空间：dtypes 候选、`shapes.dim_values` 离散取值、`ranges`、`random_types`、attr 候选值 | `dim_values` 必须用 `values` 列表（禁止 `range`）；小数写 `1.0e-5`；`dtype_numbers` 先填占位 `1`；`standard.acc: mixed_tolerance_bm`；`shape_distributions: [[0, 1.0]]` |
| `gen_op_name.py` | 把随机参数修正到不触发算子硬校验的合法范围（dtype / shape / attr） | 注册名 = YAML `generate`；每条 C++ assert 一条修正；维度索引精确；int32 溢出看护用 `OVERFLOW_GUARD_ENABLED` 保留溢出档位 |
| `executor_op_name.py` | 执行插件：按 `self.device` 分支，NPU 走真实入口、CPU 走 PyTorch golden；变长元数据归一只写一份、两路共用 | 注册名 = YAML `api_type`；CPU golden 低精度先转 fp32 再 cast 回；只返回 `with_output=True` 时的真实输出 |
| `result/op_name/json/all_op_name.json` | `atk case` 生成的用例（不提交到本示例目录，路径仅作说明） | 生成后先核对日志里的 `save case json file:`，避免误用旧 JSON |
| `atk_op_name.json` | 固化精度用例：覆盖 dense/varlen、layout、dtype、`return_saved` 两档 | 每个可达逻辑分支至少 1 条，每个精度用例至少 3 个固定种子 |
| `atk_op_name_perf.json` | 固化性能用例：用户在算子开发开始时提供的模型 case | 保留原始 shape、dtype、属性和目标 SoC；只用 NPU DUT，不做精度比较 |
| `atk_op_name_mss.json` | 固化内存检测用例：按全部可达 TilingKey 人工构造 | 每个可达 TilingKey 至少 1 条最小代表用例 |
| `test_case_design_report.md` | 用例设计覆盖评估报告（8 维度，结论前置） | 基于 skill 模板生成，占位符全部替换；降级模式下填设计值并标注 |

## 输入限制

与算子 [`README.md`](../../../fla/ops/ascendc/ops_classify/op_name/README.md) 保持一致：

| 项 | 取值 |
| --- | --- |
| layout | `BSND` / `BNSD` / `TND` / `NTD`（只解释输入，输出 `y` 固定 BSND/TND） |
| rank | `BSND`/`BNSD` 的 `x` 为 4 维 `[B,T,H,D]`/`[B,H,T,D]`；`TND`/`NTD` 为 3 维打包 token；`g` 比 `x` 少 D 维，`initial_state` 去掉 token 轴后与 `x` 的 B/H 一致 |
| dtype | `x`/`y` BF16/FP16；`g` A5 支持 BF16/FP32、A2/A3 只支持 BF16；`a_log` FP32；`x_norm` FP32 |
| D | 只支持 128（约束生成器只改末维，不外推） |
| chunk_size | 64 / 128 |
| scale / epsilon | 均为正数；`epsilon` 只用于 L2 归一化 |
| 变长 | `cu_seqlens`（INT64，从 0 开始、单调不减、末元素等于总 token 数）与 `chunk_indices`（INT64，两列）必须成对；varlen 下 `B=1` |
| 可选输出 | 由 `return_saved` 表达：`false` 只出 `y`，`true` 额外出 `state`、`x_norm`；`output_mode` 是 L2 内部档位，不是用例 attr |

变长用例的两个注意点：

1. `cu_seqlens` 由 ATK 随机生成，必须先归一成合法前缀和；`chunk_indices` 按归一后的 `cu_seqlens`
   与 `chunk_size` 重算，保证两者一致（归一逻辑放在 `executor_op_name.py`，NPU 与 CPU 两路共用）。
2. 归一后 `chunk_indices` 的行数由序列切分结果决定，JSON 里的 `shape` 只表示"两列 INT64 的列表形态"。

## 最小工作顺序

1. 读算子源码/文档，提取 shape、dtype、attr 约束，产出**约束清单表格**（来源标注到 C++ 源码）。
2. 写 `op_name.yaml`，只声明候选空间（含真实 inputs/attr 名，不含占位参数）。
3. 写 `gen_op_name.py`，每条约束一条修正。
4. `python -c "import atk; print(atk.__version__)"` 检测 ATK 可用性；可用则先 dry-run：
   `timeout 60 atk case -f op_name.yaml -p gen_op_name.py -dt 1 -en 2`，通过后回填 `dtype_numbers`
   正式生成（本仓统一入口：`bash tests/atk/run_test_cpu.sh -op=op_name -scope=gen_cases`）。
5. 生成后检查前几条 JSON，再用 `run_test_cpu.sh -op=op_name -scope=accuracy` 做精度冒烟。
6. 按模板产出 `test_case_design_report.md`，8 维度逐项填满后再固化三份用例 JSON。

检查点：约束清单齐全（每条有 gen 对应）→ YAML 语法/字段合法 → dry-run 通过 → JSON 路径确认 →
精度结果确认 → 覆盖报告 8 维度无占位符。

## 三份固化用例与映射

三份 JSON 来源不同、不能互相替代：精度用例来自生成器筛选补充；性能用例来自用户模型 case；
`_mss.json` 按全部可达 TilingKey 人工构造。算子 ATK README 必须建立三类映射并给出实际选择证据：

| 映射 | 本示例登记 |
| --- | --- |
| 逻辑分支 → 精度 case id | dense BSND / BNSD / varlen TND、`return_saved` false/true、bf16/fp16 → `atk_op_name.json` 的 0/1/2 |
| 模型 case → 性能 case id | B=1,H=16,T=4096,D=128,BF16,BSND → `atk_op_name_perf.json` 的 300 |
| 可达 TilingKey → `_mss.json` case id | chunk64 + dense → 200；chunk128 + varlen → 201 |

## 与仓内历史用例的关系

仓内一部分存量算子用 `low_precision_marker` + `fp32_marker` + `case_spec` 承载参数空间，executor 再从
`case_spec` 里解析 shape/dtype/档位。这是历史形态，有两个副作用：

1. 用例参数表与算子签名脱钩，固化 JSON 里看不到真实 shape/dtype/Attr 组合，检视时无法核对覆盖；
2. dtype/shape 档位由 executor 二次构造，ATK 生成器声明的参数空间和固化 JSON 不再是同一份事实。

新增算子一律按本示例用真实 inputs/attr/outputs；存量算子按其自身 README 登记迁移计划，迁移前不要混用
两种写法。
