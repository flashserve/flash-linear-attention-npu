# RecurrentGatedDeltaRule ATK 工程

本目录验证统一的 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 接口。A2/A3 使用
`op_kernel/arch22`，A5 使用 `op_kernel/arch35`；两个架构都由同一个 host V2
接口、Python adapter 和 CPU 标杆驱动。兼容的 V1 host ABI 仍保留，但 kernel
内部不再维护单独的 V1 实现。

## 接口约束

- Q/K 为 `[T,HK,128]`，V/out 为 `[T,HV,128]`，公开 dtype 为 BF16。
- `HV/HK` 必须是 `[1,16]` 范围内的正整数。
- beta、可选标量 gate g 和 State 分别支持 BF16/FP32；三者 dtype 共同组成
  8 种 TilingKey。
- `actual_seq_lengths` 是 `[prefix,length_0,...,length_B-1]` INT32 张量，元素和为 T。
- `ssm_state_indices` 支持 `[T]` 或 `[B,W]` INT32；二维表的 W 必须覆盖每个
  request 的有效长度。
- `gk` 仅保留 ABI 槽位，必须为 `None`。Python、ACLNN 和 tiling 都会拒绝非空值。
- Q/K/V 可使用最内特征轴连续的合法正 stride 视图；State 可使用 K 轴连续、
  不重叠的合法正 stride 视图。
- 融合开关覆盖 Q/K L2 归一化、raw gate、beta sigmoid 和允许负特征值。

CPU 标杆位于 `reference_recurrent_gated_delta_rule.py`，按 batch、value head 和
token 使用 FP32 显式递推，并覆盖二维 State 索引、accepted token 和全部融合逻辑。
比较结果为有效 token 的 out 与完整的原地 State。

## A2/A3 矩阵

`gen_arch22_cases.py` 确定性生成：

| 文件 | 数量 | 用途 |
|---|---:|---|
| `atk_recurrent_gated_delta_rule_arch22.json` | 200 | 正向精度与功能 |
| `atk_recurrent_gated_delta_rule_arch22_mss.json` | 50 | 确定性和内存 |
| `atk_recurrent_gated_delta_rule_arch22_perf.json` | 8 | dtype 组合性能 |

矩阵覆盖 8 种 dtype、12 种合法融合开关组合、GVA、变长、prefix、accepted
token、合法非连续视图、连续调用和跨 8-token 窗口回绕。

```bash
cd tests/atk/recurrent_gated_delta_rule
python gen_arch22_cases.py
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/accuracy \
  bash scripts/run_arch22_validation.sh accuracy
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/determinism \
  bash scripts/run_arch22_validation.sh determinism
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/memory \
  bash scripts/run_arch22_validation.sh memory
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/performance \
  bash scripts/run_arch22_validation.sh performance
```

## A5 矩阵

`gen_recurrent_gated_delta_rule.py` 保留原有泛化矩阵并增加 `RGDR-V001` 到
`RGDR-V023`，共 223 条。V2 专项覆盖无 gate、融合开关、二维 State 索引、
合法非连续视图以及 State/g/beta 的 8 种 dtype TilingKey。

```bash
python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --summary
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule \
  -npu_device_id=0 -scope=accuracy
```

YAML 的 SOC 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`。精度统一
使用 ATK 原生 `mixed_tolerance_bm`；不能通过缩小输入值域、调整阈值或删除失败
用例处理精度问题。完整接口和内部方案见算子目录的 `docs/api.md` 与
`docs/design.md`。

## A5 验收记录（2026-11-05）

- 被测代码：`flaserve/main@7b4f00a` 加本分支 RecurrentGDN A5 合并改动；
  CPU 标杆、A5 223 条精度 JSON 和 8 条 TilingKey JSON 与该代码同源。
- 环境：Ascend950PR，CANN `9.2.0-beta.2`，ATK `26.9.8`，隔离 wheel
  `flash_linear_attention_npu_a5-26.10.0+rgdn.working.dev7b4f00a`。
- 精度：`223/223` 通过，失败 0；覆盖一维/二维 State 索引、accepted
  token、非连续 Q/K/V/State、8 种 dtype TilingKey 和全部合法融合开关。
- 边界：`HV/HK=16` 通过 Python adapter 和 ATK；`HV/HK=17` 被 adapter
  拒绝，所有验收 JSON 均无超出 `[1,16]` 的用例。
- 确定性：8 条 TilingKey 用例、每条 50 次，`8/8` 通过。
- 内存检测：mssanitizer memcheck `8/8` 通过，所有 RecurrentGatedDeltaRule
  kernel 报告 `No error detected`。
- 性能：`atk_recurrent_gated_delta_rule_perf.json` 的 1 条用例成功执行，
  ATK profiler 平均设备时间为 `18.4066 us`；本轮无性能基线或性能目标变更。
