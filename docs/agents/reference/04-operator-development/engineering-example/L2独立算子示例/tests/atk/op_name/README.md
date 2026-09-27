<!--
示例文件：tests/atk/op_name/README.md

规范来源：ATK 仓 skill/atk-quality-guard（SKILL.md、references/op-engineering.md、
references/atk_user_guide.md、templates/test_case_design_report.md）

注意事项：
  1. 用例工程按算子建目录，目录/文件名用 snake_case；三个源文件命名固定：
     test_<op>.yaml（参数空间）、<op>_constraint.py（约束修正）、execute_<op>.py（执行插件）。
  2. YAML 只声明参数空间，constraint 只做修正；不要把 dtype 列表、shape 取值、attr 候选写进 constraint。
  3. 用例 JSON 是 `atk case` 的产物，默认落在执行目录下的
     result/<yaml 名>/json/all_<yaml 名>.json；不要手写 JSON，也不要拿旧 JSON 当本次结果。
  4. 精度失败不要改判、不要动算子源码：先区分"用例设计问题"与"算子实现问题"（精度问题报算子开发人员）。
  5. 本 README 只写本算子的输入限制、约束清单、覆盖结论与验收记录；通用要求以 skill 与 ATK 仓文档为准。
  6. 不提交 atk_output/、result/、xlsx、profiling/sanitizer 日志等测试产物（固化用例 JSON 除外）。
-->

# OpName ATK 工程（示例）

## 文件职责

| 文件 | 职责 | 规范要点 |
| --- | --- | --- |
| `test_op_name.yaml` | 声明参数空间：dtypes 候选、`shapes.dim_values` 离散取值、`ranges`、`random_types`、attr 候选值、`reduction` | `dim_values` 必须用 `values` 列表（禁止 `range`）；小数写 `1.0e-5`；`dtype_numbers` 先填占位 `1`；`standard.acc: mixed_tolerance_bm_v2`；`shape_distributions: [[0, 1.0]]` |
| `op_name_constraint.py` | 把随机参数修正到不触发算子硬校验的合法范围（dtype / shape / attr） | 注册名 = YAML `generate`；每条 C++ assert 一条修正；维度索引精确；int32 溢出看护用 `OVERFLOW_GUARD_ENABLED` 保留溢出档位 |
| `execute_op_name.py` | 执行插件：按 `self.device` 分支，NPU 走真实入口、CPU 走 PyTorch golden | 注册名 = YAML `api_type`；CPU golden 低精度先转 fp32 再 cast 回；只返回 `with_output=True` 时的结果 |
| `result/test_op_name/json/all_test_op_name.json` | `atk case` 生成的用例（不提交到本示例目录，路径仅作说明） | 生成后先核对日志里的 `save case json file:`，避免误用旧 JSON |
| `test_case_design_report.md` | Phase C 用例设计覆盖评估报告（8 维度，结论前置） | 基于 skill 模板生成，占位符全部替换；降级模式下填设计值并标注 |
| `atk_op_name.json` / `_perf.json` / `_mss.json` | 本仓附加要求：把生成并筛选后的用例固化成三份（精度 / 性能 / 内存检测），来源不同、不可互相替代 | 精度来自 `atk case` 产物筛选补充；性能来自用户模型 case；`_mss.json` 按全部可达 TilingKey 人工构造 |

## 输入限制

与算子 [`README.md`](../../../fla/ops/ascendc/ops_classify/op_name/README.md) 保持一致：

| 项 | 取值 |
| --- | --- |
| layout | `BSND` / `BNSD` / `TND` / `NTD`（只解释输入） |
| dtype | `x` BF16/FP16；`g` A5 支持 BF16/FP32、A2/A3 只支持 BF16 |
| D | 只支持 128（约束生成器只改末维，不外推） |
| chunk_size | 64 / 128 |
| 变长 | `cu_seqlens` + `chunk_indices` 必须成对；`cu_seqlens` 从 0 开始、单调不减、末元素等于总 token 数 |
| 原地开关 | 若算子声明写回 state，用例必须同时覆盖"写回"与"不写回"两档，并在此写出对应 case id |

## 最小工作顺序（按 skill）

1. 读算子源码/文档，提取 shape、dtype、attr 约束，产出**约束清单表格**（来源标注到 C++ 源码）。
2. 写 `test_op_name.yaml`，只声明候选空间。
3. 写 `op_name_constraint.py`，每条约束一条修正。
4. `python -c "import atk; print(atk.__version__)"` 检测 ATK 可用性；可用则先 dry-run：
   `timeout 60 atk case -f test_op_name.yaml -p op_name_constraint.py -dt 1 -en 2`，通过后回填 `dtype_numbers` 正式生成。
5. 生成后检查前几条 JSON，再用 `atk pytorch ... --task accuracy`（或 `atk node -b cpu task`）做冒烟。
6. Phase C：按模板产出 `test_case_design_report.md`，8 维度逐项填满后再决定是否进入 Phase B。

检查点：约束清单齐全（每条有 constraint 对应）→ YAML 语法/字段合法 → dry-run 通过 → JSON 路径确认 → 精度报告生成 → 覆盖报告 8 维度无占位符。

## 与本仓 tests/atk/README.md 现状的差异

| 本示例（ATK skill 要求） | 仓内既有算子（历史命名） |
| --- | --- |
| `test_<op>.yaml` | `<op>.yaml` |
| `<op>_constraint.py` | `gen_<op>.py` |
| `execute_<op>.py` | `executor_<op>.py` |
| 参数空间写在 YAML（`dim_values`/`ranges`/`random_types`/attr 候选） | 部分算子把参数空间放在 case_spec 里由生成器展开 |
| 用例 JSON 由 `atk case` 生成到 `result/` | 固化 `atk_<op>.json` 等三份 JSON |

新增算子按本示例（skill）执行；存量算子的命名与用例包结构待仓库统一后再批量迁移，迁移前不要混用两套命名。
