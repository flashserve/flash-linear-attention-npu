# RecurrentGatedDeltaRule ATK 工程

本目录提供 `recurrent_gated_delta_rule` 的 ATK 单算子工程，包含 `executor_recurrent_gated_delta_rule.py`、`gen_recurrent_gated_delta_rule.py`、`recurrent_gated_delta_rule.yaml`、`atk_recurrent_gated_delta_rule.json`。测试通过稳定入口 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 调用算子。

## 输入约束

- `query/key` 的 shape 为 `(T, Nk, Dk)`，`value/out` 的 shape 为 `(T, Nv, Dv)`。
- `beta` 的 shape 为 `(T, Nv)`；`stateRef` 为原地输入输出，shape 为 `(BlockNum, Nv, Dv, Dk)`。
- `query/key/value/beta/out` 为 BF16；`stateRef` 支持 BF16 和 FP32。
- `actualSeqLengths` 为 `(B+1,)` 的 INT32 张量，首元素是不参与计算的无效前缀长度，所有元素之和等于 `T`。
- `ssmStateIndices` 为 `(T,)` 的 INT32 张量，取值范围为 `[0, BlockNum)`。
- `g` 如提供，shape 为 `(T, Nv)`；`gk` 如提供，shape 为 `(T, Nv, Dk)`。两者均为 FP32，且至少提供一个。
- `numAcceptedTokens` 如提供，shape 为 `(B,)`，每项取值范围为 `[1, actualSeqLengths[i+1]]`。
- `Nv` 必须是 `Nk` 的整数倍；`Nk <= 256`、`Nv <= 256`、`Dk <= 512`、`Dv <= 512`，且 `Dk`、`Dv` 必须是 16 的整数倍；每条有效序列长度不超过 8。
- 泛化矩阵覆盖连续 state、transpose 非连续 state、head padding、block padding、复合 padding 和 storage offset；同时覆盖 Q/K/V/gate/metadata 的合法非连续视图。

## 标杆来源

`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta/golden.py`；`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/README.md`

CPU 标杆、输入构造、`run_cpu`、`run_npu` 和 `FunctionApi` 均在本目录的 `executor_recurrent_gated_delta_rule.py` 中实现；公共文件只提供基础工具函数。CPU golden 按 batch 分段执行高精度递推，覆盖 GQA 头映射、`g/gk` 衰减、接受 token 对初始状态索引的选择，以及逐 token 状态写回。

ATK 比较以下两个结果：

- `out`：`(T, Nv, Dv)` 的注意力输出；无效前缀没有公开数值语义，比较前统一置零。
- `stateRef`：算子执行后原地更新的完整状态张量。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。提交的 JSON 用例使用 `soc=all`，同一矩阵用于 A2、A3、A5。

## 默认用例

`atk_recurrent_gated_delta_rule.json` 由 `gen_recurrent_gated_delta_rule.py` 确定性生成，固定包含 146 条正向用例。每条 `case_spec` 都携带详细设计 `design_id`，映射关系如下：

| ATK Case ID | 详细设计 ID | 数量 | 覆盖范围 |
|---:|---|---:|---|
| 0-15 | RGDR-P001-P016 | 16 | gate 组合、BF16/FP32 state、A2/A3/A5 归约路径 |
| 16-31 | RGDR-P017-P032 | 16 | 变长、prefix、零长度逻辑序列、accepted token |
| 32 | RGDR-P033 | 1 | `Dk=Dv=128` 的对齐维度、UB profile |
| 33-48 | RGDR-P034-P049 | 16 | `Nk/Nv=1..256`、GVA group size |
| 49-64 | RGDR-P050-P065 | 16 | state stride/layout、稀疏/逆序/重复 index |
| 65-80 | RGDR-P066-P081 | 16 | 平台、调用通路代表 shape、非连续输入、连续调用与回归 |
| 81-96 | RGDR-G001-G016 | 16 | GVA head 映射与 group size |
| 97-112 | RGDR-G017-G032 | 16 | GVA gate、state dtype、可追踪 head 数据 |
| 113-128 | RGDR-G033-G048 | 16 | GVA varlen、prefix、accepted 与稀疏 block |
| 129 | RGDR-G049 | 1 | GVA 对齐维度、UB 分片（`Dk=Dv=128`） |
| 130-145 | RGDR-G050-G065 | 16 | GVA state layout、index、非连续输入与复杂视图 |

ATK DUT 统一通过 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 执行。`design_routes` 只保留详细设计中 P/A/D/F 的来源追溯，不表示本 JSON 已执行 aclnn C++ example 或快速拉起通路；这些通路仍由各自专用测试验证。

正向 JSON 只包含 `Dk`、`Dv` 为 16 的整数倍的合法 shape；维度泛化分组仅保留 `Dk=Dv=128`。算子原有支持范围仍为 `Dk/Dv<=512`，固定功能与平台回归用例可覆盖更大合法维度。非 16 倍数的维度属于非法输入，由 `fla_npu` wrapper 和 op_host tiling 共同拦截；对应拦截由 op_host UT 覆盖。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```

提交的 146 条 JSON 可通过以下命令重新生成并复核：

```bash
python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --summary
```

统一脚本的 `gen_cases` 默认传入 `-dt 100 -en 0`。需要通过 ATK case generator 生成完整矩阵时，显式设置：

```bash
GEN_CASES_DTYPE_NUMBERS=146 \
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```
