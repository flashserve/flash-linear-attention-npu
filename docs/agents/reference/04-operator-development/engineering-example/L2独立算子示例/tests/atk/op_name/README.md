<!--
示例文件：tests/atk/op_name/README.md

注意事项：
  1. 每个算子一个 ATK 目录，必备：README.md、atk_<算子>.json、atk_<算子>_perf.json、
     atk_<算子>_mss.json、<算子>.yaml、gen_<算子>.py、executor_<算子>.py；可选 scripts/。
  2. 三份 JSON 来源不同、不能互相替代：精度用例来自 gen 生成后筛选补充；性能用例来自用户模型 case；
     _mss.json 按全部可达 TilingKey 人工构造（每个 key 至少一条）。
  3. 本 README 必须建三类映射：逻辑分支 -> 精度 case id、模型 case -> 性能 case id、
     TilingKey -> _mss case id，并在"实际选择证据"列写 host tiling UT 或运行时记录。
  4. 精度只用 ATK 原生判据：`--task accuracy` + yaml 的 mixed_tolerance_bm，不在 executor 里自造阈值；
     反向/异常用例用 `--task run` + case 里的 expected_return_code。
  5. 环境准备、统一脚本、执行阶段见 tests/atk/README.md；本文件只写本算子的输入限制与覆盖结论。
  6. 不提交 atk_output/、result/、xlsx、profiling/sanitizer 日志。
  7. 原地（in-place）形态：控制写回的开关（如 inplace_final_state / output_mode）必须在用例里显式覆盖
     两档——写回档校验调用方张量被更新且返回值与被写回的是同一张量；不写回档校验调用方张量逐位不变、
     结果由返回值承载。`requires_grad=True` 被拒绝、version counter 推进这类契约属于调用层回归，
     放在 `tests/stable_abi/regression_mutation_contract.py`，不在 ATK 里重复。
-->

# OpName ATK 工程（示例骨架）

## 输入限制

与算子 [`README.md`](../../../fla/ops/ascendc/ops_classify/op_name/README.md) 保持一致：

| 项 | 取值 |
| --- | --- |
| layout | `BSND` / `BNSD` / `TND` / `NTD`（只解释输入） |
| dtype | `x` BF16/FP16；`g` BF16/FP32 |
| D | 只支持 128 |
| chunk_size | 64 / 128 |
| 变长 | 需要 `cu_seqlens` + `chunk_indices`，从 0 开始、单调不减、末元素等于总 token 数 |
| output_mode | `0`（none）/ `1`（save），由用例显式指定 |
| 原地开关 | 若算子声明写回 state（`inplace_final_state` 之类），用例必须同时覆盖"写回"与"不写回"两档，并在 README 里写出对应 case id |

## 精度拓扑

```text
ATK accuracy task
|-- CPU FP64 golden（executor 的 run_cpu 路径，输出转 FP32）
`-- NPU DUT（fla_npu.ops.ascendc.op_name）
```

CPU 标杆、输入生成方式和混合容差在校准记录里固化；本文件只记录用例规模与覆盖。精度标准只使用
`<算子>.yaml` 里的 `cv_fused_double_benchmark` 配置。

## 文件职责

| 文件 | 必须写什么 |
| --- | --- |
| `atk_op_name.json` | 逻辑分支覆盖用例：每个 layout、每个 dtype 组合、chunk 64/128、fixed/varlen、tail/partial、`output_mode` 两档、负向用例（非法 chunk_size、D≠128、只给一个可选输出） |
| `atk_op_name_perf.json` | 用户模型 case，保留原始 shape/dtype/属性；只跑 NPU，不做精度对比 |
| `atk_op_name_mss.json` | 按全部可达 TilingKey 构造；每个 key 至少一条，覆盖 slot 复用与 Save 档搬出路径 |
| `op_name.yaml` | ATK 生成配置，shape/dtype 必须满足算子 README 与 tiling 校验 |
| `gen_op_name.py` | 生成精度候选用例（不生成 perf/mss） |
| `executor_op_name.py` | `build_inputs`、CPU 标杆、`run_cpu`、`run_npu`、`FunctionApi` |

本目录同时给出三份 JSON 的最小示例（每个文件只放 1–2 条用例，真实交付需要按上表补全覆盖）：
`atk_op_name.json`（精度，2 条结构分支）、`atk_op_name_perf.json`（模型 case，1 条）、
`atk_op_name_mss.json`（按档位/TilingKey，2 条）。生成后再按分支映射筛选、补充边界与异常用例。

异常/负向用例的表达方式跟随相邻算子（`tests/atk/README.md`「测试动作」与各自 executor 的判定），
本示例不额外发明字段。

## TilingKey 覆盖表（示例，需按实际实现填写并给出证据）

| TilingKey 场景 | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 × fp32 × none | `x=BF16, g=FP32, output_mode=0` | `case_0` | `case_17`（tail） | `case_200` | A2/A3/A5 | host tiling UT + 运行时打印 tilingKey |
| bf16 × fp32 × save | 同上，`output_mode=1` | `case_3` | `case_19`（varlen） | `case_201` | A2/A3/A5 | 同上 |
| ... | `fp16 × bf16 × ...` 等其余组合 | ... | ... | ... | ... | ... |

## 验收结果记录（示例，跑完填写）

| 项 | 内容 |
| --- | --- |
| 被测版本 | 算子/适配层 commit 与 OPP 安装包版本 |
| 目标 SoC | A2 / A3 / A5（分别记录） |
| 精度 | 用例总数、执行成功数、失败数、失败分类（ERROR / 数值 / 无效区 / 标杆语义） |
| 确定性 | 重复次数与逐位比较结论 |
| 内存检查 | mssanitizer 工具、是否命中 sanitizer 版本 kernel、结论 |
| 性能 | 模型 case → 基线/目标/实测/比值/结论（逐 case，不用平均值掩盖） |
| 回归 | 本次改动涉及的原有场景及其结论 |
