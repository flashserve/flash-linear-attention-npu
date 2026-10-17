# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 支持多时间窗口长序列；A5 每段仍不超过 8；显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

tests/reference/recurrent_kda_reference.py; fla/ops/ascendc/kda/recurrent_kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。


## 2026-10-16 A2/A3 内部优化验证

本轮调整 A2/A3 公共 kernel 的事件保留、head 直接寻址、残差 block 复用、
有效 Beta 缓存、A_log 预处理和限定单 token 配置的下一 head raw 输入预取。
接口、CPU 标杆、host tiling、UB 分配总量、guard、state 布局和长度约束不变。
A5 的 arch35 kernel 及入口分派未修改。

### 精度与构建

- 原性能工程的 4 个 base case（id=0/2/4/6，B=1/4/16/64）精度全部通过，沿用原输入、CPU 标杆和阈值。
- 独立补充工程的原始 200 个 case 执行成功 200/200、精度通过 200/200，
  沿用原生 cv_fused_double_benchmark 标准，未修改 case、golden 或阈值。
  覆盖 BSND/TND、稠密/变长/空序列、两种 state 布局、BF16/FP32 state、
  V=128/256、最长 8 token、slot 索引、accepted tokens 及非连续输入。
- 200-case 回归显式设置 FLA_NPU_STABLE_ABI=ctypes，调用已有 ACLNN Python 路径。
  默认 Stable Python 包装器的非 inplace 分支存在未复制 initial_state 的既有问题；
  本轮未修改跨架构共享包装器，不能将上述结果解释为默认 Stable 路径全量通过。
- ascend910b 单算子构建、安装通过；安装的 72 个 kernel 对象与最终普通包逐一 SHA256 一致。
- ascend910_93 单算子构建通过；没有 A3 设备精度或性能实测结论。
- A5 源码分派和实现保持不变；没有 A5 设备回归结论。

补充 200-case JSON 的 SHA256：
b2cf9488216b431ecaff661f389b7ca36b5276ba15a1011215bbad146bcf4650。
该独立工程与本目录默认单 case 工程不同。

### msopprof 性能验收

最终性能验收使用同一空闲 Ascend 910B3 上的 msopprof BasicInfo。
直接复用原 base executor 的固定种子输入构造、非连续 QKV 视图和调用参数，
每个 batch 先调用 20 次，再采集后续 20 次；两版本使用相同脚本。
参数为 launch-count=20、launch-skip-before-match=20、warm-up=0、kill=off，
统计 OpBasicInfo CSV 中目标 RecurrentKda 的 Task Duration(us) 均值。
每次运行前检查无其他进程占用；一次受到其他任务干扰的采样已排除，
未将该轮较高基线时延计入收益。对照仍为仅修复 EventID 的精度合格版本。

| B | 对照时延/us | 优化后时延/us | 时延下降 |
| ---: | ---: | ---: | ---: |
| 1 | 38.651000 | 38.228000 | 1.09% |
| 4 | 93.255000 | 88.570000 | 5.02% |
| 16 | 304.468997 | 288.426999 | 5.27% |
| 64 | 1101.775012 | 1076.481995 | 2.30% |

下面的 ATK 重复测量作为独立复测记录；两种采样方式不混合统计或比较绝对时延。

同型号另一设备上的 B=1 补充复测采用三轮交替顺序，包含首轮全部样本，
每版本共 60 条记录；对照均值 39.075667 us，优化均值 40.010000 us，
时延回退 2.39%。因此这里的四 case 性能验收仅对应原验证设备，
不宣称 B=1 在所有同型号设备都有稳定收益。

### ATK 设备性能复测

实测平台为 Ascend 910B3。对照为 f623f2b0 的公共 kernel 仅将五类缓存手工事件
由 FetchEventID 改为 AllocEventID，保留原扫描、广播、重复 Beta 及输入装载。
该对照的 4 个 base case 精度均通过；原始未修复版本出现 NaN，不作为精度合格基线。

沿用 ATK performance_device 和 performance_data=20,20,10，提取 CANN op_summary
中 RecurrentKda 的 Task Duration(us)。对照与优化各执行两轮，
每 case 每轮 20 条有效 kernel 记录，下表为各 40 条记录的均值。
未使用 Python wall time。每个 case 的两轮优化均值都低于两轮对照均值。

| case id | B | 对照时延/us | 优化后时延/us | 时延下降 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 29.895575 | 28.536600 | 4.55% |
| 2 | 4 | 83.050200 | 77.872050 | 6.23% |
| 4 | 16 | 293.863900 | 272.460575 | 7.28% |
| 6 | 64 | 1121.287400 | 1060.948725 | 5.38% |

性能模型为 T=1、H=HV=96、K=V=128、BF16 QKV/state、FP32 gate/Beta，
开启 Q/K L2Norm、raw safe gate 和 Beta sigmoid。以上为 A2 实测，
不能据此填写 A3 或 A5 的实测性能。


### 内存异常检查及边界

插桩包构建使用 op_debug_level=1、dump_cce,sanitizer，编译命令含 sanitizer 选项；
对象含 sanitizer 符号，日志中确认实际启动并完成 RecurrentKda 检查。

- base case 0 的 block 0：racecheck、memcheck 均未检测到错误，
  覆盖同核连续 head 及新增 raw 输入预取 bank；该结果不代表全部 block 或全部用例通过。
- 补充 case 1：racecheck、memcheck 均未检测到错误。
- 补充 case 111：racecheck 无报告，memcheck 在残差 Muls 语句报告 4 条 UB 写入问题。
- 补充 case 189：memcheck 无报告，racecheck 在 K-first state 写回报告 4 条 GM WAW。
- 仅修复 EventID 的精度合格对照在 case 111/189 的相同语句复现相同类型和数量的报告。
  这些报告不是本轮优化独有；尚未完成真实缺陷与工具建模限制的最终判定，
  因此本轮不宣称 sanitizer 全量通过。
- 内存检查完成后恢复普通包，临时编译参数恢复，原始日志和聚合对照摘要保留用于复核。


## 2026-10-17 Packed raw 双缓冲与跨 head 预取

本轮对照为上一轮已验证的优化工作树，包含 EventID 保留、head 直接寻址、
Brcb、Beta 缓存和 B=1 单 token 手工输入预取，不与更早的 EventID-only 对照混用。
对照公共 kernel SHA256：
8f0dea2f771696cf1c9575f46410b1d6edb18e71a2b1a8435737eb551510c84f。
最终公共 kernel SHA256：
26e6353c5f32f1069747dcbafcb04c6c687999b7a211d9d5cf63459b84fbf22f。

### 实现、UB 与精度

- A2/A3 将 Q/K/V/gate 合并为一个 packed raw 队列，使用两个物理槽；
  持有当前消费槽时提交同序列下一拥有的 head，之后释放当前槽，以队列 ready/free 保护复用。
  每槽覆盖最多 8 token；当前 head 的 FP32 输入仍为单份。
- host 为 A2/A3 计入两份 raw 容量，并按实际 NV 计 Beta 容量，tmpBuff 只申请具名分区。
  B=1 的运行时 tiling 为 ubCalSize=196352、ubRestBytes=85504、vStep=64，
  stateOutBufferNum=2、attnOutBufferNum=2。每个 AIV 核申请 174912 B，
  相对本轮对照减少 19392 B；raw 两槽各 10240 B，guard 仍为 2048 B。
  tmpBuff 从 115136 B 收紧至 85504 B，旧手工 raw bank 一并替换。
- sanitizer 发现多 token 双 state 输出队列尾部同槽写回缺少完成等待。
  最终版本仅在尾部最后两个 state 槽相同时，成对分配、Set/Wait、释放 MTE3_S 事件，
  保持快照和数学序列不变；单 token 不执行该等待。详见设计 14.8。
- 最终普通版本原四 base 精度 4/4，独立工程原 200-case 执行成功、精度均为 200/200；
  保持原输入、CPU golden、阈值和 JSON。200-case 显式使用 FLA_NPU_STABLE_ABI=ctypes，
  默认 Stable 包装器的既有非 inplace 问题不属于上述通过结论。
- 两个新增 MTP 用例精度 2/2：B=1、T=8、H=48、HV=96、K=128、V=256，
  分别为 BF16 gate/Beta/state、V-first，以及 FP16 gate、FP32 Beta/state、K-first。
  新增用例仅使用独立 spec 恢复入口，输入生成、计算和双标杆逻辑复用原工程。
- ascend910b、ascend910_93、ascend950 均仅构建 recurrent_kda，编译通过；
  A2 安装并实测。A3/A5 只有编译和静态检查，没有设备精度或性能结论。
  arch35、入口分派、公开 API、tiling 结构未改；A5 保留原输入预算、对齐 NV 和余量规则。

### 最终版本 msopprof 性能

同一空闲 Ascend 910B3，复用原四 base 的固定种子、非连续输入和全部属性；
每轮预热 20 次、采样 20 次，统计目标 RecurrentKda 的 Task Duration(us)。
基线和最终版本各两轮，每 case 各 40 条有效记录，采集频率均为 1800 MHz。
调优中间版本及一次无有效记录的采集通道失败不计入本表，不使用 Python wall time。

| B | 本轮基线/us | 最终版本/us | 时延下降 |
| ---: | ---: | ---: | ---: |
| 1 | 40.145000 | 35.428500 | 11.75% |
| 4 | 88.205000 | 80.732000 | 8.47% |
| 16 | 286.957999 | 261.047498 | 9.03% |
| 64 | 1077.148007 | 962.701494 | 10.62% |

ATK performance_device=20,20,10 的 CANN op_summary 独立交叉测量如下，
每版本每 case 各 20 条有效记录；与 msopprof 分别统计，四 case 均提升。

| B | 本轮基线/us | 最终版本/us | 时延下降 |
| ---: | ---: | ---: | ---: |
| 1 | 27.489500 | 26.289550 | 4.37% |
| 4 | 79.169600 | 75.546350 | 4.58% |
| 16 | 262.898250 | 246.273950 | 6.32% |
| 64 | 1034.968650 | 960.045200 | 7.24% |

### 槽位与流水取证

最终源码的独立 workspace 轨迹覆盖 B=1 的全部 40 个 AIV 核。
core 0 的 head 0/40/80 使用 raw 地址 0/10240/0，确认切换和回绕；
core 16 的 head 16/56 使用 0/10240。每个 head 都有提交、消费、释放记录，
没有在热路径增加硬事件或 pipe barrier。轨迹版本的 out/state 与普通版本逐位一致。
轨迹仅用于验证，正式包保持原系统 workspace，没有算法专用 GM 中转。

普通编译 kernel 的 TimelineDetail 仿真在 core 0 记录到 head 80 raw DMA
与 VMLA/VMUL 等指令交叠；head 40 的 raw DMA 没有观察到 V 运算交叠。
因此只确认存在跨 head 搬运/计算重叠，不宣称每个 head 的 DMA 都被隐藏。
该仿真用于指令排程取证，真实时延使用上面的硬件 BasicInfo。
示例：head 80 raw DMA 区间 16.656000–17.029000 us，与 VMLA 的 16.546000–16.742000 us 区间交叠。

### Sanitizer 范围、结果和限制

插桩包 opc 参数含 op_debug_level=1、dump_cce,sanitizer，CANN 编译命令含
--cce-enable-sanitizer；核对了实际 base 对象符号、安装对象来源及目标 kernel Start/Finish。

| 范围 | racecheck | memcheck |
| --- | --- | --- |
| 原四 base，block 0 | 4 次目标 kernel 均无错误 | 4 次目标 kernel 均无错误 |
| 原 200-case 代表 id=1/111/189，block 0 | 3 次均无错误 | 3 次均无错误 |
| 新增两个 T=8/HV=96 MTP，block 0 | 2 次均无错误 | 2 次均无错误 |
| B=1 base，全核 | 无错误 | 无错误 |

上述共 20 次目标 kernel 检查均实际启动、完成，无 inactive target 或记录丢失提示。
尾部等待修正消除了本轮发现的 GM WAW；旧版 case 111 的 4 条 UB memcheck
报告在最终版本未复现，不能将旧版报告与最终结果混用。

额外 initcheck 检查 B=1/B=64 的 block 0，各有 16 条 PRIVATE 8 字节读取报告，
均指向 DataCopyPad 调用处，没有报告 UB/GM 未初始化读取；旧版 B=1 复现 18 条同类
PRIVATE 报告。尚未完成该类报告的最终归因，不能称 sanitizer 全工具通过。
synccheck 在 B=1/B=64 及两个新增 MTP 的 block 0 未报告错误，
仍分别有 44/2284/44/44 条冗余 set/wait 警告；旧版 B=1 复现 64 条同类警告。
保留原始日志和聚合摘要，不把既有报告直接判为工具误报。

验证后恢复普通包和编译参数；最终普通包的 72 个 kernel、OP API、op_tiling
与已完成精度和性能验证的普通候选逐一 SHA256 相同，抽查普通对象不含 sanitizer 符号。


## Beta raw 双缓冲快速实验（2026-10-17，未采用）

基线：909cda68，保留 packed raw 双缓冲和跨 head 预取。
此前被否决的 FP32 Q/K/V/gate 双缓冲及跨 head 预处理已回退。
本候选仅针对 A2/A3：Beta raw 两槽轮换并跨 sequence 预取；
FP32 Beta 保留单份，guard 保持 2048 B。B=1 每核实际分配 177984 B，
比基线增加 3072 B。A5 kernel 逐字节一致，A5 host 仍按单 Beta raw 槽计费。

recurrent_kda 单算子构建通过；72 个目标 kernel 对象及 API/tiling 库与安装结果一致。
原 4 个 base + T=4 新增 4 个 base 精度 8/8；补充精度 200/200。
沿用原输入、CPU 标杆和精度标准。按照本轮要求未执行 sanitizer。

同卡 msopprof BasicInfo，20 次预热后采样 20 次，目标 kernel 的
Task Duration(us) 均值，耗时变化正值代表回退。频率：['1800']。

| T | B | 基线 / μs | Beta 双缓冲 / μs | 耗时变化 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 34.933 | 35.858 | +2.65% |
| 1 | 4 | 78.677 | 85.253 | +8.36% |
| 1 | 16 | 252.305 | 269.583 | +6.85% |
| 1 | 64 | 992.053 | 996.638 | +0.46% |
| 4 | 1 | 72.975 | 72.883 | -0.13% |
| 4 | 4 | 205.883 | 206.218 | +0.16% |
| 4 | 16 | 747.837 | 758.658 | +1.45% |
| 4 | 64 | 2891.387 | 2938.955 | +1.65% |

恢复基线包后，T=1、B=4 复测 20 次均值为 81.000 μs，
Beta 候选为 85.253 μs，仍慢 5.25%。
八模型没有获得可采用的整体收益，本候选放弃。
代码与安装包均恢复到 909cda68；实验补丁、构建包、原始精度和性能证据单独保留。
本轮未创建 commit。


## A2/A3 L2Norm epsilon 修复（2026-10-17）

A2/A3 在 Q/K 平方和开方前加入 1e-6，计算 x/sqrt(sum(x*x)+1e-6)；
仅 Q 在归一化后乘 scale。CPU 标杆同步改为显式公式，不再使用 F.normalize。
A5 kernel、kernel 分派入口和 host 保持不变。

新增独立回归入口 test_l2norm_epsilon.py。该脚本用一个非零 K 坐标构造
可独立演算的标量 recurrence，分别检查 CPU 标杆、NPU 输出及最终 state，
并拒绝 NaN/Inf。小输入为 BF16 可精确表示的 2^-10，正常输入为 1；
Q/K 分别为零、同时为零、分别小范数、同时小范数和混合行均有覆盖。
T=1/4、BSND/TND、V-first/K-first、FP32 state 组合共 64 个开启用例，
再加 8 个关闭 L2Norm 的对照。RTOL=0.02、ATOL=0.01 沿用原 base。

运行示例：

    python tests/atk/recurrent_kda/test_l2norm_epsilon.py --device 0 --output l2norm_result.json

旧包反例验证可加 --expect-mismatch。固定容差下旧包被检出 32/72 个偏差，
与无 epsilon 解析结果可区分的 32 个用例一致；修复后 72/72 全部通过。
此项确认缺陷可被测试检出，不以零值用例单独通过代表完整归一化语义正确。

构建：ascend910b 单算子构建、安装通过，72 个目标对象与 API/tiling 库
均与修复包匹配；ascend910_93 单算子构建通过，产出 72 个对象，
没有 A3 硬件精度或性能实测。A5 kernel、分派入口与 host 逐字节未改动，
本轮没有 A5 设备回归结论。

精度：原 4 个 T=1 base 和 4 个 T=4 base 共 8/8 通过；
补充原 200-case JSON 未变，修正 CPU 归一化公式后 200/200 通过。
补充测试沿用现有 ACLNN/ctypes 路径，未扩大为默认 Stable 包装器的全路径结论。
新增 72/72 边界组合通过，CPU 标杆与独立解析 recurrence 72/72 一致；
旧包检出 32 个预期偏差。所有边界检查同时核对 out/state 和非有限值。
原八模型 PTA、补充 200-case executor 和仓内 ATK 三份 CPU 标杆已同步修正。
按本轮要求未执行 sanitizer。

性能使用同一健康 A2 设备的 msopprof BasicInfo；
每模型先运行 20 次，再采样 20 次，统计 RecurrentKda Task Duration(us) 均值。
T=1、B=4/64 各追加一轮，每版本合计 40 条；其余每版本 20 条。
全部原始样本均纳入统计，频率均为 1800 MHz；正值代表变慢。

| T | B | 每版本样本数 | 修复前 / μs | 修复后 / μs | 耗时变化 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 20 | 36.049 | 35.133 | -2.54% |
| 1 | 4 | 40 | 81.210 | 79.204 | -2.47% |
| 1 | 16 | 20 | 251.833 | 245.357 | -2.57% |
| 1 | 64 | 40 | 953.103 | 973.879 | +2.18% |
| 4 | 1 | 20 | 72.063 | 71.517 | -0.76% |
| 4 | 4 | 20 | 204.193 | 205.167 | +0.48% |
| 4 | 16 | 20 | 746.730 | 746.901 | +0.02% |
| 4 | 64 | 20 | 2886.289 | 2893.522 | +0.25% |

- T=1、B=4 第二轮：79.731 → 80.736 μs，耗时变化 +1.26%。
- T=1、B=64 第二轮：965.891 → 945.841 μs，耗时变化 -2.08%。
两组复测均与首轮的变化方向相反，说明本次测量存在轮间波动；
合并均值不作为稳定性能收益的证明。T=1、B=64 合并均值慢 2.18%，
T=4 各组变化绝对值不超过 0.76%。数值语义修复保留，
未据此宣称全部模型获得性能提升。


## A2/A3 Q/K 批量 L2Norm（2026-10-17）

性能基线为已修复 epsilon 的 24b44ae1。
公共 kernel 将一个 head 的有效 Q/K 行分别批量平方和归约，
范数分别保存在现有 deltaInUb、attnInUb，再共享一次 V_S 同步。
Scalar 倒数、逐行 Muls 和独立后置 Q scale 保持不变。
Q 平方值归约完毕后通过 PIPE_V 保护 scratch 复用；不跨 head 合并 FP32 输入。
L2Norm 内 V_S 从每 head 的 2L 次变为 1 次，其余 gate/Beta 同步不变。
InitLocalBuffers 与基线逐字节一致，不增加 UB 或改变 guard/vStep；
A5 kernel、host tiling 和 kernel 分派入口保持不变。

精度：原 8 个 base 通过 8/8，补充原 200-case 通过 200/200，
既有零值/小范数等 72 个边界组合通过 72/72，同时检查 out/state 和非有限值。
本轮未改变三份 CPU 标杆、输入、JSON、seed 或混合容差；
补充 200-case 继续使用既有 ACLNN/ctypes 路径。
A2 单算子构建安装通过，安装的 72 个对象、API 和 tiling 库与本轮构建一致。
A3 单算子编译通过，产出 72 个对象，没有 A3 硬件实测；
A5 文件保持不变，没有 A5 设备回归结论。按本轮要求未运行 sanitizer。

msopprof 最终比较基线为 24b44ae1。两版本各执行两轮，每模型每轮
20 次预热后采样 20 次，共 40 个目标 kernel Task Duration(us) 样本；
同一健康 A2 设备、相同脚本、输入和属性，频率均为 1800 MHz。
全部原始样本纳入均值，未使用 Python wall time，未剔除不利轮次。

| T | B | 基线 / μs | 批量版本 / μs | 耗时下降 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 34.187 | 32.760 | 4.17% |
| 1 | 4 | 76.823 | 73.310 | 4.57% |
| 1 | 16 | 254.037 | 240.489 | 5.33% |
| 1 | 64 | 959.504 | 897.495 | 6.46% |
| 4 | 1 | 72.409 | 70.038 | 3.27% |
| 4 | 4 | 205.063 | 195.788 | 4.52% |
| 4 | 16 | 745.317 | 709.238 | 4.84% |
| 4 | 64 | 2896.288 | 2752.410 | 4.97% |

八组 40 样本均值均低于基线，耗时下降 3.27%～6.46%。
第2轮 T=1、B=1 耗时回退 0.59%；其余 15 个“轮次×模型”比较均更快。
因此结果支持本机当前八模型的平均耗时改善，但 B=1/T=1 存在轮间波动，
不宣称每轮均提升或 A3/A5 已取得实测性能收益。
运行时文件访问核验确认实际打开的 RecurrentKda 对象 SHA256 与候选包一致，
并再次确认当前安装的 72 个对象、API 和 tiling 库与本轮构建匹配。
候选普通包已保留安装；本轮未创建新 commit。

## A2/A3 长序列支持（2026-10-17）

本轮基线为 e2890ad6。A2/A3 的 8 仅表示内部输入窗口容量，不再限制 dense
单序列或 varlen 分段长度。保留 T<=8 快路径，长路径按 head、V tile、时间窗口、token
顺序递推，当前 V tile 的 FP32 state 跨窗口驻 UB，不从 BF16 快照重新读取。
UB、guard 和系统 workspace 的分配保持现状。A5 kernel、分派入口、tiling struct
保持原样，新增 host 容量检查仅对 A2/A3 生效。现有 uint32 tiling 容量和合法地址范围仍须满足。

新增执行器 executor_recurrent_kda_longseq.py 和固定用例包 atk_recurrent_kda_longseq.json，
共 41 个结构配置、每配置 3 个固定种子，共 123 个精度 case。标杆沿用当前 CPU
recurrent 公式，扩展 FP64 工作 dtype 用于真正的高精度 benchmark；原 FP32 分支在
9 个代表性场景上与 e2890ad6 的 out/state 逐元素完全一致。原有输入与精度阈值未放宽。

| 分支 | 长序列 case id |
| --- | --- |
| T=9/15/16/17/31/32/33/64/128/256/512/1024，BSND/TND，state 两种布局及 dtype，V=128/256，门控与归一化组合 | 0–71 |
| dense 多 batch、cu_seqlens 可选、单 head | 72–83 |
| 同一 core 连续多个 head | 84–86 |
| 非连续 Q/K/V/state | 87–92 |
| 混合长短/空分段、有效 token 前缀与 padding tail | 93–98 |
| packed/二维 state 索引、accepted token 跨窗口、重复 slot、原位与非原位 | 99–113 |
| state 输出与原位开关 | 114–119 |
| 缺省初始 state | 120–122 |

长序列 ATK 精度采用既有补充测试的原生 cv_fused_double_benchmark：
max_re_ratio=5、avg_re_ratio=1.5、root_mean_squared_ratio=1.5、
number_count_radio=4。ATK 丢弃 non_param case_spec 时，由 case_id 恢复固定配置，
并校验其余实际传入的结构参数与种子，避免默认参数掩盖覆盖缺失。
非原位 state 输出只比较实际写入槽；原位模式比较完整 pool 并保留未写入槽的检查，
非原位调用同时检查 initial_state 未修改。padding tail 不参与精度比较。

TilingKey 仍为 0；短/长运行时分支分别由原 base 与新增 case 覆盖。
atk_recurrent_kda_longseq_mss.json 人工选取 0、84、93、102、108 五个最小代表，
覆盖窗口 tail、同 core 多 head、混合空分段、归一化、重复槽与 accepted token。

可在已加载测试环境后执行：

~~~sh
python tests/atk/recurrent_kda/executor_recurrent_kda_longseq.py --generate tests/atk/recurrent_kda/atk_recurrent_kda_longseq.json
atk node --backend npu --devices <id> node --backend cpu task -c tests/atk/recurrent_kda/atk_recurrent_kda_longseq.json --task accuracy -p tests/atk/recurrent_kda/executor_recurrent_kda_longseq.py -sp -to 2000
~~~

本轮 A2 精度结果：8/8 base、200/200 补充、72/72 既有边界、
123/123 新增长序列全部通过。新增长序列使用原生 CPU FP32 控制组与 CPU FP64
benchmark；补充 200-case 沿用 ACLNN/ctypes 路径。
A2、A3、A5 单算子构建通过；当前只有 A2 设备执行结果，没有 A3/A5 硬件回归结论。

本轮性能和 sanitizer 已完成采集。相对 e2890ad6，T=4/B=1 耗时降低3.47%，
其余七个 base 的耗时增加0.68%至6.12%，不能宣称短序列性能收益。
执行追踪确认同核多 head 的双 raw slot 轮转、跨窗口驻留和非零分段起点。
racecheck、memcheck 的代表用例未报告目标算子错误；initcheck 仍有 PRIVATE 报告，
其中同一短用例在候选和基线均为16条；synccheck 有冗余 set/wait 警告。
因此本轮尚未形成 sanitizer 全部通过的正式验收结论，开发期验证记录继续保留。
A5 仅完成源码一致性和编译核验，未做硬件回归。普通包已恢复，本轮未提交。
