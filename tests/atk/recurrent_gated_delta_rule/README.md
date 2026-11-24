# RecurrentGatedDeltaRule ATK 工程

本目录验证统一的 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 接口。A2/A3 使用
`op_kernel/arch22`，A5 使用 `op_kernel/arch35`，A6 使用 `op_kernel/arch92`。
这些架构共用完整的 RecurrentGDN 参数契约、Python adapter 和 CPU 标杆。ACLNN 仅保留原名
`aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` / `aclnnRecurrentGatedDeltaRule`，
参数覆盖原 V2 功能。

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

## A6 矩阵

A6（`ascend960dt`，`__NPU_ARCH__ == 9201`）只走 arch92 的 RGDRV2，因此只复用 V2 用例族：

```bash
python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --a6 \
  --output tests/atk/recurrent_gated_delta_rule/atk_recurrent_gated_delta_rule_a6.json
```

产物 `atk_recurrent_gated_delta_rule_a6.json` 为 `RGDR-V001`~`RGDR-V023` 共 23 条，
`platform_focus` 与 `soc` 均改写为 `ascend960dt`（该取值已列入
`recurrent_gated_delta_rule.yaml` 的 `soc` 合法枚举）；A2/A3 与 A5 用例族不在 A6 上执行。

Python 稳定入口 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 通过共用的 adapter
调用原名 ACLNN 接口，支持 `A_log`、`dt_bias` 和融合开关；A6 矩阵沿用同一参数契约。

## 验证脚本的环境变量

`scripts/run_arch22_validation.sh` 使用以下环境变量：

| 环境变量 | 默认值 | 作用域与消费者 | 非法值行为 |
|:---|:---|:---|:---|
| `OMP_NUM_THREADS` | `1` | 仅脚本子进程；ATK 精度比对与 CPU 标杆 | 不做校验，由 OpenMP 运行时自行处理 |
| `MKL_NUM_THREADS` | `1` | 仅脚本子进程；ATK 精度比对与 CPU 标杆 | 不做校验，由 MKL 运行时自行处理 |
| `NPU_DEVICE_ID` | `0` | 脚本读取后传给 ATK 的 `-npu_device_id` | 非整数时 ATK 报错退出 |
| `ATK_OUTPUT_ROOT` | `./arch22_results` | 脚本读取，作为结果输出目录 | 目录不可创建时脚本因 `set -e` 退出 |

前两者固定为 1：ATK 精度比对要求确定性，多线程 BLAS 的归约顺序变化会引入 1e-6 级抖动。
它们只作用于本脚本进程，不继承给调用方，也不影响算子运行时行为；回退方式是取消 export
后重新执行脚本。

YAML 的 SOC 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950` 和 `ascend960dt`。精度统一
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

## ACLNN 接口收敛精度回归（2026-11-14）

- 代码基线：`add_refine@dd54e688` 加本轮接口收敛变更；使用 A2 工作区已有的 513 条独立 CPU 双标杆矩阵。
- 仅保留 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` /
  `aclnnRecurrentGatedDeltaRule`，采用原 V2 的完整参数表；Stable adapter 和 C++ 示例已同步。
  描述符校验从独立头文件移入 ACLNN cpp，计算、tiling、CPU 标杆、输入分布和原生精度阈值保持一致。
- 重新构建并安装 A2 wheel，实际加载的 opapi 和 4 个 kernel 文件与本轮构建结果一致；
  安装后的头文件与 adapter 参数匹配，动态库只导出上述两个符号。

| 用例范围 | 执行成功 | 原生双标杆通过 | 原生双标杆未通过 |
| --- | --- | --- | --- |
| 0–479 | 480/480 | 464 | 16 |
| 480–512 | 33/33 | 33 | 0 |
| 合计 | 513/513 | 497 | 16 |

未通过 case_id 为 42、106、145、153、156、178、215、223、232、298、366、375、
379、391、437、473，与改前的原生精度报告一致，未新增失败。
另选取包含这 16 条在内的 30 条代表用例，在改前 V2 与本轮接口上使用相同输入执行：
attention 及完整 State pool 的字节哈希 30/30 一致；无效 prefix 在比较前按原规则清零。
该一致性检查不替代原生精度标准，本轮原生双标杆仍未全通过。

C++ 两段式示例编译及执行通过。CPU beta、query dtype 错误、一维索引长度错误、非空 gk、
缺失融合 gate 输入等 5 条负向检查均保留包含参数名的报错，且 State 数据与版本计数保持不变。
