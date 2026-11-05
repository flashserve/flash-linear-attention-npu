# Recurrent Gated Delta Rule V2 开发验证

## 架构评审修改后的最终状态

2026-11-03 按架构评审意见完成以下收敛，后文各节保留的是此前逐轮实验记录；本节数据和结论
代表当前工作区最终版本：

- 使用 `ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL` 和 `GET_TPL_TILING_KEY` 对 State、scalar `g`、
  beta 的 BF16/FP32 组合，以及 Q/K L2Norm、raw gate、beta sigmoid、allow negative eigenvalue
  四个功能位生成 128 个实际 kernel 模板。当前实现固定要求 scalar `g` 输入，Q/K L2Norm、
  raw gate、beta sigmoid 和 allow negative eigenvalue 路径均为 `if constexpr`，VF 内没有运行时 `if`。
- 删除 `ComputePipelineTokenDispatch` 等薄分派函数。VF 的 mask、RegTensor 和标量变量统一在
  函数开头声明，主体只有一个按 State row 展开的循环；源码不包含 `LocalMemBar`。VF 不再
  include 或调用 `common/kernel_utils/vector/regbase.hpp`，load/cast/store 均直接使用 MicroAPI。
- `gk` 仅保留公开 ABI 输入槽。stable、ctypes、ACLNN 和 tiling 对非空值统一报错，kernel
  不再包含 Gk GM/UB、模板参数、搬运或计算路径。
- 所有支持的 State dtype、Q/K/V 外层 stride 和 gate 模式统一使用 4-token input/output
  ping-pong 与三个 State bank；Q/K/V 尾轴 stride 固定为 1，State bank 按声明 dtype
  保存 recurrence state，不再存在 `commonModelPath` 或 `usePipeline_` 条件路径。
- work 边界只保留 MTE3→MTE2 的 State bank 生命周期交接；该同步不包含 `PIPE_S`，不存在
  Scalar 等待 Vector。
- stable ABI 和 ctypes 的公开 `npu_recurrent_gated_delta_rule` 均直接调用 V2；V1 C++ 算子源码
  未修改。

最终 A5 精度回归覆盖 223 个组合，结果为 `FAILED []`；输出最大绝对误差 `7.62939453125e-06`，
State 最大绝对误差 `0.0001220703125`。专门覆盖同一 AIV 连续处理第二个 work 的 `RGDR-P034`
输出/State 最大绝对误差分别为 `1.9073486328125e-06` 和 `9.313225746154785e-09`。

issue 679 的 36 个模型 shape 在可见设备 0 上使用同一进程、同一输入、每个实现预热 5 次，
profile 25 次并丢弃前 5 次，按 `msprof op_summary` 的 `Task Duration(us)` 中位数比较：

| 指标 | 最终结果 |
| --- | ---: |
| 36 case 的 V2/V1 中位数 | 0.9796 |
| 36 case 的 V2/V1 均值 | 1.0070 |
| V2 更快的 case | 32/36 |
| 最小/最大 V2/V1 | 0.8990 / 1.6806 |
| 最大模型 case（V1 / V2） | 5049.992 / 4980.635 us |

四个回退 case 都位于小 shape 的 TPB=1/2；TPB=4/8 的 18 个 case 全部更快。短序列
`B=1, T=8, HK=2, HV=4` 的 msprof V2 Task Duration 中位数为 `15.9525 us`，精度为
output `0`、State `7.450580596923828e-09`。本版 V2 对修改前 profile 的 36-case 逐 shape
耗时比中位数为 `1.0028x`，未观察到移除 regbase inline 封装后的可辨别性能变化。

`msprof` 中代表性多 tile case 的 Vector、Scalar、MTE2、MTE3 累计活跃时间之和为 AIV
活跃墙钟的 `1.32x`，最大模型 case 为 `1.45x`，确认各 pipe 存在重叠；该证据与源码中的
“MTE2 预取下一 tile、Vector 消费当前 tile、MTE3 写回上一结果”顺序一致。最终 profile 位于
`msprof_dtype_vf_model36/PROF_*/mindstudio_profiler_output/op_summary_*.csv`；短序列 profile 位于
`msprof_dtype_vf_short/PROF_*/mindstudio_profiler_output/op_summary_*.csv`。

另实验了把 `SequenceStart/SequenceEnd` 的长度前缀扫描合并为一次。精度不变，但 36 case
相对上述最终 V2 的逐 case event 耗时中位数为 `1.0029x`，仅 4/36 变快，收益未超过波动且
整体轻微回退，因此未采用。

### `ComputeToken` 双发射调度（2026-11-03）

按无依赖指令相邻的原则重排单 token VF：Q/K 的 load、cast、平方、归约和归一化形成两条并行
依赖链；gate 与 beta 预处理在各自数学顺序不变的前提下交错；每个 State row 的两半 decay、dot、
outer update、State add 和 output dot 成对发射。双链临时量复用后续阶段尚未使用或已经结束生命期的
`dot0/dot1/out0/out1/queryWork/keyWork`，避免为调度额外增加长期存活的 Vector 寄存器。跨 token
recurrence、FP32 运算顺序和 MTE 流水均未改变。

A5 最终回归仍为 223/223 通过，output/State 最大绝对误差分别为
`7.62939453125e-06` 和 `0.0001220703125`。性能基线和修改版都只运行 V2；36 个模型 shape 各做
两轮 msprof，每轮包含 5 次预热和 20 次采样，逐 shape 合并两轮的 40 个正式样本取中位数：

| 指标 | 双发射版 / 修改前 V2 |
| --- | ---: |
| `Task Duration(us)` 逐 shape 比值中位数 / 均值 | 1.00034 / 0.99964 |
| `aiv_vec_time(us)` 逐 shape 比值中位数 | 0.99817 |
| 双发射版 Task Duration 更快的 shape | 15/36 |
| 最大模型 shape（修改前 / 双发射版） | 4966.334 / 4966.146 us |

Vector 活跃时间下降约 0.18%，但 Task Duration 的整体变化未超过重复测量波动，因此本轮只能确认
指令调度结构和 Vector 局部收益，不能宣称端到端性能提升。修改前和最终版的原生 V2-only
`op_summary` 分别归档在 `perf_result/op_summary_v2_before_dual_issue_*.csv` 和
`perf_result/op_summary_v2_dual_issue_final_*.csv`；远端 profile 位于
`msprof_before_dual_issue_v2_model36*` 和 `msprof_dual_issue_reuse_v2_model36*`。

### 单 VF 与首 tile 即时计算（2026-11-03）

将原有 Compute、初始 State load、最终 State store 和 output conversion 四个源码级 VF 合并为
唯一的 `RecurrentGatedDeltaRuleVf`。计算模式在同一个 VF 中直接完成 output dtype 转换和写入；
BF16 初始 State 由首 token 的计算模式直接 load/cast。BF16 最终 State 仍以编译期
`VfMode::STORE_STATE` 特化调用同一 VF 定义，避免让下一 token 的三 State bank 计算等待上一
token 的 MTE3。源码静态检查只有一个 `__simd_vf__`，且不包含 `PipeBarrier`、`LocalMemBar`
或旧的 `attnFloat_` 中转区。

下一 tile 的 `CopyInputTile` 从 token 循环前移动到 token0 VF 发射后，使首 tile MTE2 完成后
立即发射 Vector，再并行预取第二 tile。`CopyStridedRows` 内的 `PipeBarrier<PIPE_V>()` 已删除；
保留的 MTE2/V Mutex 用于保护逐 row 复用的 strided staging，和外层 input ping-pong buffer
Mutex 的所有权范围不同，不能一并删除。

A5 `Ascend950PR` 上重新构建、安装并执行 223 个开发回归组合，结果为 `FAILED []`，output 和
State 最大绝对误差分别为 `7.62939453125e-06`、`0.0001220703125`。额外 smoke 覆盖 FP32、
BF16、raw gate、二维 State 索引及 Q/K/V 非连续视图，全部通过。该记录是开发回归，不替代
ATK `all` 对 accuracy、performance、determinism 和 mssanitizer 的正式验收。

性能比较只运行 V2，并在同一张物理卡 1 上对修改前后分别执行 36 个 issue 679 shape；每个
shape 预热 5 次、正式采样 20 次。逐 shape 中位耗时再汇总如下：

| 指标 | 修改前 V2 | 单 VF/即时计算 V2 | 新版/修改前 |
| --- | ---: | ---: | ---: |
| 36 shape 中位数 | 724.476 us | 725.008 us | 1.00188 |
| TPB=1 比值中位数 | - | - | 1.00190 |
| TPB=2 比值中位数 | - | - | 0.99984 |
| TPB=4 比值中位数 | - | - | 1.00292 |
| TPB=8 比值中位数 | - | - | 1.00174 |

逐 shape 比值范围为 `[0.98791, 1.02038]`，新版在 11/36 个 shape 更快；整体约 `+0.19%`，
属于测量波动，不能宣称端到端性能提升或下降。V2-only 原始结果归档在
`perf_result/op_summary_v2_single_vf_before_same_device1_20261103214404.csv` 和
`perf_result/op_summary_v2_single_vf_after_20261103213537.csv`。

`B=2,T=8,Hq=2,Hv=4` 的上板 PipeTimeline 显示 core0 veccore0 首 tile MTE2 在
`4.003636 us` 开始、`5.212121 us` 结束，Vector 在 `5.220606 us` 开始，第二 tile MTE2
随后在 `5.232121 us` 开始并持续到 `5.916363 us`。因此 Vector 在首 tile 搬运结束后约
`0.0085 us` 即开始，并与第二 tile MTE2 重叠，不再等待第二 tile 搬完。原始
`visualize_data.bin`、`trace.json` 和 `OpBasicInfo.csv` 位于
`perf_result/msprof_op_pipetimeline_v2_single_vf_b2_t8_hq2_hv4_20261103_2137/`。

为观察单核连续处理多个 head work 的完整流水，远端临时将 `vectorCoreNum` 固定为 1，对
`B=4`、每序列 32 token、`Hq=2`、`Hv=4` 的 case 采集上板 PipeTimeline。`OpBasicInfo.csv`
记录 `Block Dim=1`、`Task Duration=856.732056 us`；trace 只包含 `core0.veccore0`，其中 16 段
Vector 对应 `B*Hv=16` 个 head work。完整 OPPROF（包括源码调试信息对应的二进制和原始 timeline）
位于 `perf_result/msprof_op_pipetimeline_v2_single_aiv_b4_s32_hq2_hv4_20261103_2302/`，
`visualize_data.bin` SHA256 为
`3709fd6246c7656544b64b26ad06a024b7ccf723953d1bb3511d3b82bb39f731`。采集后已恢复远端
tiling 源码、重新构建安装正式多核版本，并通过 `B=2,T=8,Hq=2,Hv=4` smoke；单 AIV 限制
未写入本地算子源码。

## A5 性能优化记录

验证日期为 2026-11-02 至 2026-11-03，设备为 A5 `Ascend950PR`（主要使用物理卡 6，Static Tensor
模型矩阵使用物理卡 4），软件环境为
CANN 9.2.0 beta 2。V1 和 V2 使用相同设备、进程内输入、dtype、预热次数和采样次数。
正式比较口径是 `msprof op_summary` 的 `Task Duration(us)` 中位数；NPU Event 仅作调用链观察。
目标 kernel 分别为
`aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule` 和
`aclnnRecurrentGatedDeltaRuleV2A5_RecurrentGatedDeltaRuleV2_RecurrentGatedDeltaRuleV2`。

本轮只修改 V2：

- scalar gate 在 UB 中保持 compact tile，不再广播为 `[X,128]`；gate 的 `exp/softplus`
  只处理 compact 标量，VF 读取标量后在寄存器中广播。
- dense FP32 State 热路径使用三个独立 FP32 bank 轮转；BF16 State 和 strided Q/K/V
  继续使用通用静态 buffer 路径，并使 `K=V=128` 的 FP32 State 采用完整 `vStep=128`。
- 删除每个 head 在计算前对全部 State slot 的重复扫描；initial slot 在 work 初始化时校验，
  write slot 在每个 token 实际使用前校验一次。
- V2 UB 改为 Static Tensor 绝对地址管理，不再创建 `TPipe/TQue/TBuf`；A5 核内跨流水
  同步改为固定 ID 的 Mutex `Lock/Unlock`，输出热路径按 V→MTE3 stage-chain 复用同一 ID。

### 方案 2.8 与流水图落地审计

本轮重新解析 `info/流水图.drawio`，并逐项对照方案 2.8。优化前的 Static Tensor 版本
并没有完整实现该图：输入只有单 buffer，Q/K/V 先整 tile 转 FP32，State 只有一个计算 bank，
且 attention output 逐 token 搬出。最终 dense FP32 State 热路径如下：

| 流水图要求 | 最终实现 |
| --- | --- |
| tile 含 4 token | `TOKEN_TILE_4=4`，尾 tile 只处理有效 token |
| Q/K/V/beta/g ping-pong | 两套原始 dtype 静态 tensor；当前 tile 计算时由 MTE2 预取下一 tile |
| 一个 token 一次 VF | `ComputeToken` 一次完成 cast、可选 norm/gate/beta、recurrence 和 output |
| State A/B/C 轮转 | 三个 64 KiB FP32 bank，token 结果按 A→B→C→A 轮转 |
| State 逐 token 写回 | VF 完成后 MTE3 直接从对应 State bank 写回，三 bank 避免覆盖未完成搬运 |
| output 整 tile 写回 | 同一 tile 的 4 个 output 写入一个 ping/pong buffer，tile 完成后一次 MTE3 搬出 |

时间线与图一致：T0 搬初始 State 和 tile0；T1 开始 tile0 token1 的 VF，同时预取 tile1；
T2-T4 顺序执行其余 token，MTE3 与 Vector 重叠写回上一 token State；tile0 完成后整 tile
写 output。T5 起消费 tile1，并复用已经预取的另一套 input/output buffer。同一 work 的后续
tile 不重新读取 State、`A_log` 或 `dt_bias`；图中每个 tile 属于不同 4-token batch 时才各自
初始化 State。

最终源码不含 `VEC_SCOPE`，旧宏 VF 的 `#if 0` 死代码也已删除。VF 是 engine 头文件内的单 token
融合 `ComputeToken(...)`，dtype 转换直接使用底层 MicroAPI，不再额外维护只被 engine 使用的
vector 头文件或引用公共 regbase 封装。Scalar 不等待 Vector；保留的 `PIPE_S` Mutex 只用于
通用 strided gather offset 的 S→V 依赖。V/MTE2、V/MTE3 Mutex 都对应 UB 生产者/消费者或
bank 复用的真实依赖，最后一次 MTE3 发出后不再建立无用途的 MTE3→V drain。

本轮短序列同卡结果（物理卡 6，110 条目标 kernel）为：

| 实现阶段 | V2 中位数 | 相对 14.1175 us |
| --- | ---: | ---: |
| 原 Static Tensor + stage-chain | 14.1175 us | 1.0000 |
| 4-token/三 State bank 完整流水 | 13.1835 us | 0.9338 |
| 常用模型分支专用 VF | 13.0820 us | 0.9266 |
| 删除末尾无依赖 drain | 13.0635 us | 0.9253 |

最终版本相对本轮起点下降 7.47%。最终 36 个 issue 679 shape 使用同一卡、相同输入和
`warmup=5, loops=20`；V1 取同轮 full-package profile，V2 取最终 V2-only profile：

| TPB | V2/V1 中位数 | 范围 |
| ---: | ---: | ---: |
| 1 | 0.9501 | [0.8720, 0.9787] |
| 2 | 0.9781 | [0.9665, 0.9896] |
| 4 | 0.9843 | [0.9766, 0.9932] |
| 8 | 0.9848 | [0.9784, 0.9932] |

36 个比值中位数 `0.9786`、均值 `0.9726`，范围 `[0.8720, 0.9932]`，V2 在 36/36
case 中快于 V1。收益主要集中在 TPB=1；TPB=4/8 时每 token recurrence 的约 9 us Vector
热链占主导，流水只能隐藏 MTE2/MTE3，不能消除递推计算，因此整体仍未达到 0.8x。
对应 profile 为 `msprof_pipeline_short_nodrain`、`msprof_model_v2_pipeline_device6` 和
`msprof_model_v2_nodrain_device6`。

### VF 合并与命名清理后的流水、性能复测

2026-11-03 将 VF 直接并入 `recurrent_gated_delta_rule_v2_engine.h`，删除
`recurrent_gated_delta_rule_v2_vector.h`，并将流水相关常量、函数和 tensor 改为职责命名。
这次只改变源码组织和标识符，不改变 VF、Static Tensor 地址布局、Mutex ID 或计算顺序。
重新构建的 OPP 包中只包含 engine 头文件，确认没有遗留 include。

短序列使用 `--instr-profiling=on --aic-metrics=PipeUtilization` 采集 11 条目标 kernel，
`Task Duration(us)` 中位数为 `13.012 us`，相对合并前 `13.0635 us` 为 `0.9961x`。稳定调用的
AIV 活跃中位数为 `12.710 us`，各流水占用中位数如下：

| 流水 | 占用时间 | 占 AIV 活跃时间 |
| --- | ---: | ---: |
| Vector | 9.036 us | 70.9% |
| Scalar | 3.184 us | 25.0% |
| MTE2 | 1.561 us | 12.3% |
| MTE3 | 5.411 us | 42.5% |

各流水占用相加明显大于 AIV 总活跃时间，证明 Vector、MTE2、MTE3 并非串行执行。时间线 JSON
中每次稳定 kernel 都先出现 State/tile 的 MTE2 事件，随后出现 VEC，并在同一 kernel 窗口内
连续出现 State/output 的 MTE3 事件；这与方案 2.8 的“预取下一 tile、当前 token 计算、上一
结果写回”顺序一致。msprof 事件不携带 tile/token 语义标签，因此具体的 tile0/token1 名称仍由
源码控制流审计确认，不能仅凭 profile 自动命名。

36 个 issue 679 shape 使用与历史相同的 `warmup=5, loops=20` 口径复测两轮。首轮受短任务
调度波动影响，V2/V1 比值中位数为 `0.9869`，32/36 更快；同卡立即重复后结果如下：

| TPB | V2/V1 中位数 | 范围 | 更快 case |
| ---: | ---: | ---: | ---: |
| 1 | 0.9634 | [0.7272, 0.9804] | 9/9 |
| 2 | 0.9776 | [0.9473, 0.9905] | 9/9 |
| 4 | 0.9845 | [0.9771, 0.9904] | 9/9 |
| 8 | 0.9859 | [0.9783, 0.9918] | 9/9 |

重复轮 36 个比值中位数为 `0.9794`、均值为 `0.9684`，范围 `[0.7272, 0.9918]`，V2 在
36/36 case 中快于 V1。相对合并和命名清理前的最终 V2，逐 shape 比值中位数为 `1.0006`，
即整体持平；范围为 `[0.8330, 1.0148]`，短任务的轮间波动显著，不能把单轮离群点解释为
源码组织变化带来的收益或回退。对应 profile 为 `msprof_pipeline_merged_instr`、
`msprof_model_v2_merged_device6` 和 `msprof_model_v2_merged_repeat_device6`。

### 短序列开发基准

输入为 `B=1, T=8, HK=2, HV=4, K=V=128`，BF16 Q/K/V/beta、FP32 State/g，
V2 新增开关关闭。每个 profile 包含一次 parity 调用、10 次预热和 20 次测量，筛选出的
V1/V2 目标 kernel 各有 31 条记录。

| 版本 | V1 中位数 | V2 中位数 | V2/V1 |
| --- | ---: | ---: | ---: |
| 优化前 | 11.925 us | 18.905 us | 1.5853 |
| compact gate + 三 State bank | 11.834 us | 14.847 us | 1.2546 |
| 再移除重复 State slot 扫描 | 11.841 us | 14.596 us | 1.2327 |
| V2 `cce-auto-sync=off` | 11.841 us | 14.603 us | 1.2333 |

最终 V2 相对优化前下降 22.76%，但短序列仍比 V1 慢 23.33%。关闭 auto-sync 前后
V2 分别为 `14.596 us` 和 `14.603 us`，差异为 `+0.05%`，属于测量波动。最终 profile 路径：

- V1：`msprof_short_opt2_v1/PROF_*/mindstudio_profiler_output/op_summary_*.csv`
- V2 auto-sync=on：`msprof_short_opt2_v2/PROF_*/mindstudio_profiler_output/op_summary_*.csv`
- V2 auto-sync=off：`msprof_short_v2_autosync_off/PROF_*/mindstudio_profiler_output/op_summary_*.csv`

### Issue 679 模型典型 Shape

模型矩阵来自 issue 679：`BS={32,64,72}`、`HK=16`、`HV={32,48,128}`、
`TOKEN_PER_BATCH={1,2,4,8}`、`K=V=128`，共 36 个 case。公平比较采用 V1/V2
共同支持的 BF16 Q/K/V/beta、FP32 State/g 路径，V2 新增开关关闭。每个 case 每个实现
预热 5 次、测量 20 次，下表为测量区间的 `Task Duration(us)` 中位数。

| BS | HV | TPB | V1 us | V2 us | V2/V1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 1 | 48.683 | 55.369 | 1.1373 |
| 32 | 32 | 2 | 136.585 | 130.683 | 0.9568 |
| 32 | 32 | 4 | 325.621 | 319.160 | 0.9802 |
| 32 | 32 | 8 | 564.865 | 556.260 | 0.9848 |
| 32 | 48 | 1 | 92.025 | 91.371 | 0.9929 |
| 32 | 48 | 2 | 268.646 | 263.043 | 0.9791 |
| 32 | 48 | 4 | 483.159 | 477.284 | 0.9878 |
| 32 | 48 | 8 | 842.578 | 834.725 | 0.9907 |
| 32 | 128 | 1 | 461.715 | 452.413 | 0.9799 |
| 32 | 128 | 2 | 749.046 | 735.660 | 0.9821 |
| 32 | 128 | 4 | 1273.557 | 1262.512 | 0.9913 |
| 32 | 128 | 8 | 2244.997 | 2231.753 | 0.9941 |
| 64 | 32 | 1 | 168.980 | 162.202 | 0.9599 |
| 64 | 32 | 2 | 376.780 | 370.555 | 0.9835 |
| 64 | 32 | 4 | 645.688 | 633.870 | 0.9817 |
| 64 | 32 | 8 | 1133.365 | 1116.225 | 0.9849 |
| 64 | 48 | 1 | 333.191 | 324.499 | 0.9739 |
| 64 | 48 | 2 | 565.420 | 554.012 | 0.9798 |
| 64 | 48 | 4 | 954.251 | 948.285 | 0.9937 |
| 64 | 48 | 8 | 1690.089 | 1670.771 | 0.9886 |
| 64 | 128 | 1 | 915.093 | 901.411 | 0.9850 |
| 64 | 128 | 2 | 1477.893 | 1466.356 | 0.9922 |
| 64 | 128 | 4 | 2539.474 | 2520.868 | 0.9927 |
| 64 | 128 | 8 | 4488.191 | 4459.762 | 0.9937 |
| 72 | 32 | 1 | 222.389 | 203.927 | 0.9170 |
| 72 | 32 | 2 | 431.329 | 419.418 | 0.9724 |
| 72 | 32 | 4 | 724.833 | 714.111 | 0.9852 |
| 72 | 32 | 8 | 1275.088 | 1256.523 | 0.9854 |
| 72 | 48 | 1 | 379.645 | 369.812 | 0.9741 |
| 72 | 48 | 2 | 634.511 | 622.338 | 0.9808 |
| 72 | 48 | 4 | 1072.369 | 1066.460 | 0.9945 |
| 72 | 48 | 8 | 1900.320 | 1883.508 | 0.9912 |
| 72 | 128 | 1 | 1026.566 | 1012.276 | 0.9861 |
| 72 | 128 | 2 | 1661.951 | 1648.354 | 0.9918 |
| 72 | 128 | 4 | 2852.954 | 2836.968 | 0.9944 |
| 72 | 128 | 8 | 5044.495 | 5015.358 | 0.9942 |

36 个比值的中位数为 `0.9853`，均值为 `0.9870`，范围为 `[0.9170, 1.1373]`；
V2 在 35/36 个 case 中快于 V1，但没有 case 达到 issue 要求的 `0.8x`。完整 profile：
`msprof_model_v2_autosync_off/PROF_000001_20261102180715581_00844432NFEHMKEB/mindstudio_profiler_output/op_summary_20261102180728.csv`。

同一矩阵与 auto-sync=on 的 V2 数据逐 case 比较，off/on 中位数为 `0.9997`，均值为
`0.9995`，范围为 `[0.9777, 1.0252]`；20/36 个 case 的 off 更快，32/36 个 case
落在 ±1% 内。整体性能无显著变化，小 shape 的最大波动约为 2.5%。

### Static Tensor 与 Mutex

最终实现不包含 `TPipe`、`TQue`、`TBuf`、`InitBuffer`、`SetFlag` 或 `WaitFlag`。Q/K/V、
gate、beta、State 预取、State/out 写回和计算临时量均直接用
`LocalTensor(TPosition::VECCALC, absoluteOffset, length)` 构造。UB 地址区间由 host tiling
统一预算，kernel 使用同一物理地址图。Mutex ID 使用 0–4，避开系统预留的 28–31。
Kernel 不建立 V→S 同步：Q/K norm、raw gate 和非 128 维 beta 均在 Vector
寄存器或 UB 内完成标量广播和消费，Scalar 不回读 Vector 结果。

短序列在物理卡 6 上按相同输入和 profile 参数复测：

| 实现 | V2 中位数 | 相对 TQue 版本 |
| --- | ---: | ---: |
| TQue + SetFlag/WaitFlag，auto-sync=off | 14.603 us | 1.0000 |
| Static Tensor + 双流水 fence Mutex 初版 | 18.894 us | 1.2938 |
| Static Tensor + stage-chain Mutex | 14.117 us | 0.9667 |

初版在每个 token 上分别执行完整 V→MTE3 和 MTE3→V fence，Mutex 指令过多，因此放弃。
最终版让 V 和 MTE3 在同一 Mutex ID 上交替 `Lock/Unlock`，相对 TQue 版本下降 3.33%。
最终短序列 profile：
`msprof_short_v2_static_chain/PROF_000001_20261102202121697_02219220BCQPLQAB/mindstudio_profiler_output/op_summary_20261102202141.csv`。

36 个模型 shape 在物理卡 4 上使用同一进程、相同输入、每 case 预热 5 次并测量 20 次。
Static V2/V1 比值中位数为 `0.9769`，均值为 `0.9951`，范围为 `[0.9062, 1.3603]`；
V2 在 31/36 个 case 中更快。按 token 数分组如下：

| TPB | V2/V1 中位数 | 范围 | V2 更快 |
| ---: | ---: | ---: | ---: |
| 1 | 0.9740 | [0.9375, 1.3603] | 6/9 |
| 2 | 0.9675 | [0.9062, 1.0535] | 7/9 |
| 4 | 0.9794 | [0.9728, 0.9961] | 9/9 |
| 8 | 0.9772 | [0.9712, 0.9892] | 9/9 |

完整 profile：
`msprof_model_v2_static_chain_device4/PROF_000001_20261102202436892_02299146PKRPNBDJ/mindstudio_profiler_output/op_summary_20261102202449.csv`。
旧 36-shape profile 位于卡 6，因此不使用跨卡绝对耗时判定 Static Tensor 的收益；本轮实现收益
以卡 6 的短序列同卡前后结果为准，模型矩阵用于判断最终实现相对 V1 的当前表现。

### 删除 Vector 到 Scalar 等待

Static Tensor 初版仍在 Q/K norm、raw gate 和通用 beta 路径中用 Mutex 建立 V→S，
由 Scalar 回读 Vector 写入 UB 的标量。最终实现删除该方向的 Mutex 和 scalar UB：Q/K norm
在 Vector 寄存器内完成 reduce、sqrt、倒数及广播；raw gate 在 Vector 寄存器内广播
`A_log`/`dt_bias` 并完成 exp/softplus；通用 beta 路径从 UB 直接广播到 Vector 寄存器。
删除 scalar UB 后，host UB 预算同步减去 64B。

物理卡 0 上分别安装删除前、删除后的 V2-only 包，每个 case 独立进程执行 10 次预热和
100 次测量。下表取 `msprof op_summary` 中 110 条目标 kernel 的
`Task Duration(us)` 中位数：

| Case | 删除前 | 删除后 | 删除后/删除前 |
| --- | ---: | ---: | ---: |
| 短序列，新增开关关闭 | 14.1415 us | 14.1175 us | 0.9983 |
| `RGDR-V004`，QK norm + beta sigmoid | 9.6415 us | 8.2910 us | 0.8599 |
| `RGDR-V005`，beta sigmoid + allow negative | 6.9100 us | 6.9170 us | 1.0010 |
| `RGDR-V007`，raw gate + dt bias | 8.4875 us | 8.1685 us | 0.9624 |

`RGDR-V005` 走 K=128 融合路径，原实现也不执行通用 beta 的 V→S 回读，因此结果持平；
QK norm 和 raw gate 是本次删除等待的直接收益路径。对应 profile 目录为
`msprof_{vtos,no_vtos}_{short,qk,beta,raw}`。

issue 679 的 36 个模型 shape 也在同一物理卡 0 上对删除前后各 profile 一次，每个 shape
5 次预热、20 次测量。逐 shape 比值中位数为 `0.9998`、均值为 `1.0019`，20/36 个
case 更快。首轮 `BS=32, HV=48, TPB=1` 出现 `1.1106x` 离群值；将该 shape 独立扩大到
10 次预热和 100 次测量后，删除前后分别为 `105.7205 us` 和 `105.7900 us`，比值
`1.0007`，确认常规模型路径没有可复现回归。完整矩阵位于
`msprof_model_{vtos,no_vtos}_device0`，离群复测位于
`msprof_{vtos,no_vtos}_model_worst_repeat`。

最终同步审计结果：

- 不存在 `SetFlag`、`WaitFlag`、`HardEvent::V_S`、`MUTEX_V_S` 或 V→S Mutex；Scalar
  不读取 Vector 产生的 UB 值。
- `GetValue` 只读取 GM 中的 sequence/state metadata、`A_log` 和 `dt_bias`。
- MTE2↔V Mutex 保护输入 staging 和 State prefetch 的 UB 复用；V↔MTE3 Mutex 保护
  output stage-chain，单独的 V→MTE3 Mutex 保护转置完成。这些都是跨流水真实依赖。
- 唯一包含 `PIPE_S` 的同步是 strided gather offset 初始化后的 S→V，方向符合约束且每个
  kernel 初始化最多执行一次。`PIPE_V` barrier 均位于同流水写后读或写后写依赖之间。

复现命令：

```bash
msprof --output=msprof_model_v2_autosync_off --task-time=l0 --ai-core=on \
  --application="python tests/atk/recurrent_gated_delta_rule/benchmark_v1_v2_a5.py \
  --mode profile --impl both --warmup 5 --loops 20"

msprof --output=msprof_model_v2_static_chain_device4 --task-time=l0 --ai-core=on \
  --application="python tests/atk/recurrent_gated_delta_rule/benchmark_v1_v2_a5.py \
  --mode profile --impl both --warmup 5 --loops 20"
```

### 精度回归

V2 关闭 auto-sync 后重新执行专属矩阵 `RGDR-V001` 至 `RGDR-V015`，结果为 15/15
通过。最大绝对误差保持为 `out=4.76837158e-07`、`stateRef=5.96046448e-08`；
与开启 auto-sync 时一致。改为 Static Tensor 和 stage-chain Mutex 后再次得到相同的 15/15
结果与最大误差。Python adapter 单元测试为 25/25 通过。

### 删除错误继承的 safe-gate 逻辑

V2 的 raw scalar gate 只实现方案定义的
`g_eff = -exp(A_log) * softplus(g + dt_bias)`，不再包含从 KDA 继承的 safe-gate
模板参数、运行时分支、下界参数或 tiling 读取。tiling 结构中的两个 `compatibility*`
字段只是保持既有记录偏移的无语义槽位；host 不赋值，kernel 不读取，也不构成公开接口。

删减分支后，大 value-head 用例暴露了原实现中跨 work 的 UB 竞争：一个 core 处理第二个
head 时，下一 head 的 MTE2 State 预取可能覆盖上一 head 仍由 MTE3 写回的 State bank。
`HandoffToNextWork` 因此改为 MTE3 到 MTE2 的 Mutex 交接。该同步只发生在同一 core
切换 head 时，不进入 token 内层循环。

A5 上安装同一次构建生成的 V1+V2 包后，运行时 host 库与 10 个 kernel `.o` 均与构建
产物一致。215 个精度 case 全部通过，最大绝对误差为 `out=7.62939453e-06`、
`state=1.22070312e-04`。三个 raw-gate case `RGDR-V007` 至 `RGDR-V009` 的最大 state
误差不超过 `3.72529030e-09`，最大 output 误差为 `4.76837158e-07`。

短序列 parity 为 `out=0`、`state=7.45058060e-09`。issue 679 的 36 个模型 shape 使用
event 计时，每 case 预热 5 次、10 次循环、5 组样本；V2 在 34/36 个 case 中快于 V1，
V2/V1 中位数为 `0.9791`、均值为 `0.9839`。该结果与修改前约 `0.9813` 的中位比一致，
跨 head handoff 修复未造成整体性能退化。

### PR #997 VF 对照与单 AIV 长序列优化

参考 recurrent KDA PR #997 的 V-first VF 后，确认主要差异不是 VF 提交次数，而是 State 行间
指令级并行、`beta*k` 的 token 级复用以及 `MulAddDst` 的使用。GDN V2 保持一个源码 VF、
单 head/single batch work、4-token 输入双缓冲和三 State bank，不引入 scratch、
`LocalMemBar` 或额外 VF。保留的改动为：

- 相邻两行 State 同时推进，交错安排彼此独立的 load、mul 和 reduce。
- 每 token 预计算 `updateKey=beta*k`，行循环用 `MulAddDst` 更新 State。
- prediction 和 output dot-product 使用 `mul + MulAddDst`，保持 FP32 中间结果。
- 连续输出共用一个 `UnalignRegForStore`，只在 token 末执行一次 `StoreUnAlignPost`。

单 AIV 调试 case 为 `B=4, S=32, Hq=2, Hv=4, K=V=128`，BF16 Q/K/V/beta、
FP32 gate/State。`PipeUtilization` 对比如下：

| Metric | 基线 | 最终 | 最终/基线 |
| --- | ---: | ---: | ---: |
| Task Duration | 857.968994 us | 685.653015 us | 0.799158 |
| AIV total cycles | 1,413,472 | 1,129,422 | 0.799041 |
| Vector time | 831.534546 us | 659.830322 us | 0.793509 |
| MTE2 time | 53.110909 us | 52.750301 us | 0.993210 |
| MTE3 time | 251.495148 us | 253.827271 us | 1.009273 |

最终 `PipeTimeline` 三次 Task Duration 为 `676.979004/677.072021/686.007996 us`，
中位数 `677.072021 us`；基线三次中位数为 `872.587036 us`，比值 `0.775936`。
三次最终结果的总体 CV 为 `0.62%`。Vector event sum 稳定在约 `651.7 us`，MTE2/MTE3
未减少，确认收益来自 VF 而非搬运量变化。

曾实验将每两行输出通过 `Interleave/Gather` 收集到完整寄存器再统一 cast/store；精度通过，
但 AIV cycles 从 `1,111,507` 回退到 `1,160,614`，因此未保留。最终源码 SHA256 为
`2d17a512c0245830adda05c51930f30b3afc3a54c9eb892c22576db1acf6a16e`。

最终候选的 223 个 CPU/NPU 精度 case 全部通过，最大绝对误差为
`out=7.62939453125e-06`、`state=1.220703125e-04`，与优化前一致。完整
`PipeUtilization.csv`、三份 `visualize_data.bin`、trace 和原始 dump 位于
`perf_result/msprof_op_pipetimeline_v2_single_aiv_b4_s32_hq2_hv4_optimized_20261104/`。
单 AIV 仅用于暴露流水；归档后已恢复正式 `vectorCoreNum=min(AIV, B*Nv)` tiling。

恢复正式多核后，在物理卡 1 上只运行 V2，按每 shape 预热 5 次、采集 20 次的口径复测
36 个 issue 679 模型 shape。重复轮的 shape 中位数中位数为 `722.848 us`，修改前 V2 为
`725.008 us`；逐 shape 新版/修改前 V2 比值中位数 `0.997077`、均值 `0.996513`，24/36
个 shape 更快。按 tokens-per-batch 分组，比值中位数依次为
`1.001546/1.001599/0.993594/0.994443`。首轮 case 30/31 的异常高值在同卡立即复测后恢复
为 `1060.801/1863.991 us`，均略低于修改前 V2 的 `1068.543/1869.660 us`，不是可复现
回退。两轮原始 summary 为 `perf_result/op_summary_v2_vf_optimized_20261104010624.csv` 和
`perf_result/op_summary_v2_vf_optimized_repeat_20261104010848.csv`。

### 结论和后续候选

compact scalar gate、三 State bank、单次 slot 校验和本轮 VF 行间并行均有设备收益并保留。
单 AIV 长序列 case 的 Vector 时间下降 20.65%，Task Duration 达到约 0.8x；瓶颈仍是
State recurrence 与逐 token State 写回。正式多核模型矩阵用于确认其它支持场景没有不可接受
的下降，不能用单 AIV 调试数据替代模型性能结论。
