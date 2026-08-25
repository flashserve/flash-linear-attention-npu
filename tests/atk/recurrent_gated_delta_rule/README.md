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
- `Nv` 必须是 `Nk` 的整数倍；`Nk/Nv <= 256`，`Dk/Dv <= 512`，每条有效序列长度不超过 8。

## 标杆来源

`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta/golden.py`；`fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/README.md`

CPU 标杆、输入构造、`run_cpu`、`run_npu` 和 `FunctionApi` 均在本目录的 `executor_recurrent_gated_delta_rule.py` 中实现；公共文件只提供基础工具函数。CPU 标杆按 batch 分段执行递推，覆盖 GQA 头映射、`g/gk` 衰减、接受 token 对初始状态索引的选择，以及逐 token 状态写回。高精度标杆使用 FP64 计算，同精度标杆保留原始输入量化和输出 dtype。

ATK 比较以下两个结果：

- `out`：`(T, Nv, Dv)` 的注意力输出；无效前缀没有公开数值语义，比较前统一置零。
- `stateRef`：算子执行后原地更新的完整状态张量。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。提交的 JSON 用例使用 `soc=all`，同一矩阵用于 A2、A3、A5。

## 默认用例

`atk_recurrent_gated_delta_rule.json` 由 `gen_recurrent_gated_delta_rule.py` 生成，固定包含以下 10 条用例：

| Case ID | 场景 | 关键覆盖 |
|---:|---|---|
| 0 | 最小 BF16 state | 单 token、`g` |
| 1 | 基础 FP32 state | 双 token、`g` |
| 2 | 非整块尾维 | `Dk=80`、`Dv=96`、仅 `gk` |
| 3 | 变长 GQA | 双 batch、`g+gk` |
| 4 | 接受 token | 三 batch、GQA、FP32 state |
| 5 | 无效前缀 | 前缀长度 2、变长、`g+gk` |
| 6 | 最大 MTP | 单序列 8 token、`Dv=256` |
| 7 | 零长度 batch | 中间 batch 长度为 0 |
| 8 | 非连续多头 `gk` | `Nk=4`、`Nv=8`、FP32 state、transpose 非连续布局 |
| 9 | 最大状态维度 | `Dk=Dv=512`、`g+gk` |

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。提交的 JSON 固定包含上述 10 条用例，可通过 `python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --summary` 复核。
