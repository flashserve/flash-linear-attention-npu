# RecurrentGatedDeltaRule ATK 工程

## A2/A3 arch22 验证

稳定入口为 `fla_npu.ops.ascendc.recurrent_gated_delta_rule`。A2/A3 的两个 ACLNN host 接口共用 `op_kernel/arch22/recurrent_gated_delta_rule.h`；所有融合开关关闭时也使用同一实现。`gk` 是保留参数，必须拒绝，不能作为合法精度用例。

### 输入和标杆

- Q/K 为 `[T,HK,128]`，V/out 为 `[T,HV,128]`，均为 BF16，最内轴连续，允许合法行 padding。
- beta 和可选标量 gate g 为 `[T,HV]`，分别支持 BF16/FP32。State 为 `[BlockNum,HV,128,128]`，支持 BF16/FP32，以及内轴连续、非重叠的合法正 stride 视图。
- `HV/HK` 必须是 1 到 16 的整数。实际序列长度、索引和 accepted-token 参数沿用接口文档定义。
- 融合开关覆盖 Q/K L2 归一化、原始 gate 变换、beta sigmoid、允许负特征值。最后一项只能和 beta sigmoid 一起开启。
- CPU 标杆在 `reference_recurrent_gated_delta_rule.py` 中按 FP32 执行递推；executor 独立执行对应的归一化、gate 和 beta 前处理。输入值域沿用原用例生成器，精度标准保持 YAML 中 ATK 原生的 `mixed_tolerance_bm`。
- 比较 out 和原地更新的完整 State；无效前缀 out 没有数值语义，比较前统一置零。

### 用例矩阵

当前 A2/A3 规格文件为 `recurrent_gated_delta_rule_arch22.yaml`；旧 YAML 仅保留历史规格。

运行 `python gen_arch22_cases.py` 可确定性生成以下矩阵：

| 文件 | 数量 | 用途 |
|---|---:|---|
| `atk_recurrent_gated_delta_rule_arch22.json` | 200 | 正向精度；8 种 State/beta/g dtype 组合、12 种合法融合开关组合、GVA、变长、prefix、accepted token、合法非连续视图及连续调用 |
| `atk_recurrent_gated_delta_rule_arch22_mss.json` | 50 | 确定性和内存；覆盖全部 dtype/开关组合，包含跨 8-token 窗口回绕 |
| `atk_recurrent_gated_delta_rule_arch22_perf.json` | 8 | 性能；覆盖全部 dtype 组合 |

新矩阵的设计 ID 为 `RGDR-A001` 到 `RGDR-A200`，保留原矩阵的数值 profile 和精度阈值。最后 8 条补充 9、16、17 token 和含零长度序列的窗口回绕场景。原 `gen_recurrent_gated_delta_rule.py` 及旧 JSON 保留用于历史用例追溯；含 gk 的旧矩阵不代表当前 A2/A3 合法规格。

### 执行方式

先安装与源码一致的 wheel 并加载配套 CANN 环境。ATK 必须提供原生 `mixed_tolerance_bm`；本次验证使用 ATK 26.9.8。切换工具版本不改变标准、阈值或输入范围。

```bash
cd tests/atk/recurrent_gated_delta_rule
python gen_arch22_cases.py
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/accuracy bash scripts/run_arch22_validation.sh accuracy
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/determinism bash scripts/run_arch22_validation.sh determinism
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/memory bash scripts/run_arch22_validation.sh memory
NPU_DEVICE_ID=0 ATK_OUTPUT_ROOT=./arch22_results/performance bash scripts/run_arch22_validation.sh performance
```

如通过 `ASCEND_RT_VISIBLE_DEVICES` 只暴露一张卡，`NPU_DEVICE_ID` 使用进程可见编号 0。确定性每条重复 50 次；内存使用 mssanitizer memcheck，以 `--kernel-name=RecurrentGatedDeltaRule` 匹配目标算子的全部模板并开启泄漏检查；性能使用设备 profiler，正式耗时读取 `op_summary` 的 `Task Duration(us)`，不使用 Python wall time。

### 结果检查

精度报告应为 200 条执行成功且精度达标。确定性和内存报告各为 50 条通过；内存还需检查 sanitizer 原始日志。性能交付保留 profiler CSV 和用例映射。若出现精度失败，必须保存输出并用 CT 可视化定位，不能缩小输入范围、调整阈值或删除失败用例。

完整接口和实现约束见算子目录的 `docs/api.md`、`docs/design.md`。测试结果以转测交付中的原始报告为准。

精度脚本显式传入 `--gm_init_flag 512`；可用 `GM_INIT_MB` 配置工具初始化参数，避免默认策略尝试填充接近整卡的显存。该选项只影响测试前的 GM 初始化，不改变算子输入值域、shape 或精度标准。

## A5 arch35 V2

A5 独立 V2 算子的稳定入口为
`fla_npu.ops.ascendc.npu_recurrent_gated_delta_rule_v2`。该实现使用
三组 State bank 和 4-token tile；每个 task 只启动一次 `ComputeOne`，并根据
token-per-batch 选择 V 到 MTE2 的释放时机。`gk` 为保留槽位，任何非空值都会被拒绝。

新增用例位于 `gen_recurrent_gated_delta_rule.py` 生成的 `RGDR-V` 系列，覆盖无 gate、
Q/K L2Norm、raw gate、beta sigmoid、二维 State 索引、accepted token、prefix、零长度序列、
GVA、BF16/FP32 State/g/beta，以及合法的非连续 Q/K/V 和 State 视图。ATK 精度标准仍使用
YAML 中配置的原生 `mixed_tolerance_bm`。

Issue 679 的 36 组典型 shape 已完成 A5 精度与性能采集。36 组输出均与基准二进制一致；
性能结论以 profiler 的 `Task Duration(us)` 为准。A2/A3 与整体 Example ST 的补充验证由
仓库 NPU CI 执行。
