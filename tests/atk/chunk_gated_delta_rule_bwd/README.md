# ChunkGatedDeltaRuleBwd ATK 测试说明

本目录验证 `chunk_gated_delta_rule_bwd` 融合接口。NPU 节点只调用一次公开融合接口；
CPU 节点使用同一组冻结输入，独立串联四个阶段的 CPU 标杆，比较最终有效梯度。

## 标杆与输出

CPU 标杆依次计算 `ChunkGdnBwdIntra`、`ChunkFwdH`、
`ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize`。阶段边界按照
NPU 接口的数据类型量化，避免把 NPU 中间结果作为标杆。

有效输出顺序为 `dq, dk, dv, d_beta, d_g`；传入 `initial_state` 时额外比较
`dh0`。其中 `d_beta` 和 `d_g` 固定使用 BSND `[B,T,HV]`，其余有效输出跟随
对应输入布局。保留接口输出 `d_a_log` 和 `d_dt_bias` 当前为空，不参与数值比较。

## 覆盖范围

- Ascend 950，输入主类型为 BF16，`K=V=128`，`chunk_size=64`。
- 覆盖 `BNSD/BSND/NTD/TND`，定长与变长，以及完整块、尾块和多块。
- 覆盖 `G=HV/HK` 的 `1/2/3/4`。
- 覆盖 BF16/FP32 标量输入、初始状态、末状态梯度和 `state_v_first`。
- 覆盖 Q/K L2Norm backward、beta sigmoid backward 及保留输入。
- `use_exp2=true`、`use_gate_in_kernel=false`。

## 用例矩阵

`gen_chunk_gated_delta_rule_bwd.py` 定义 12 个确定性 profile。矩阵以较小 shape
覆盖接口分支和四阶段组合语义，避免大 shape 使 CPU 标杆成为测试瓶颈。

## 执行

```bash
GEN_CASES_DTYPE_NUMBERS=12 bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_bwd -scope=gen_cases
bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_bwd -soc=ascend950 -npu_device_id=0 -scope=accuracy
```


## Issue #795 典型 shape 验证

本次新增范围使用独立用例生成入口，保留原小 shape 矩阵。生成四个算子各两例：
`B=1, HK=16, HV=128, K=V=128, chunk_size=64`，主类型 BF16，g/beta FP32，
BNSD；定长 `T=11264`，变长 `T=16384`、16 段各 1024。其余用例待后续提供。

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/gen_gva_typical.py --output-dir /tmp/gva_typical
# 在对应 tests/atk/<op> 目录，沿用 CPU 单标杆及 mixed_tolerance_bm：
atk node --name npu_dut --backend npu --devices 0 \
  node --name cpu_golden --backend cpu task \
  -c /tmp/gva_typical/atk_<op>_gva_typical.json --task accuracy --bm_device cpu \
  -p ./executor_<op>.py --gm_init_flag 4096 -to 14400
```

2026-11-04 验证：Ascend950PR（设备4），CANN 9.2.0-beta.2，torch/torch_npu 2.12.0，
ATK 提交 45cc4061（单 CPU worker、solo pool，避免 PyTorch prefork 等待）。
CPU 单标杆及原 mixed_tolerance_bm 阈值不变，四个算子各两例全部通过（8/8）。
case_spec 必须以 string 属性传入；non_param 会被 ATK 从 executor kwargs 排除，
不能据此声称验证了指定变长分段。

A5 Host/Kernel/wheel 构建及安装产物 MD5 校验通过；A2 的 intra/finalize Host/Kernel
编译及 2 处 ABI 检查通过。构建脚本的多 SoC 参数被拒绝，因此分别编译 A5/A2；
本轮不代表 A2/A3 硬件回归。实际 runtime leader 辅助函数的 G=5..16、17/31/32/64
映射检查通过；NPU 精度仅覆盖上表两个 G=8 典型 shape。

200 MiB 按额外 workspace 申请量核对，覆盖四个阶段的系统和用户 workspace，
排除输入、输出及既有中间张量存储。逐项申请量和本次新增量见下方
[Workspace 核对记录](#workspace-核对记录2026-11-19)。

2026-11-04 的 A5 完整调用使用 BNSD 输入，计入 transpose/contiguous、四节点及输出转换。
预热 5 次、采样 20 次，表中取中位数，使用普通包和设备 4。

| 典型 shape | NPU event (ms) | 同步 wall (ms) | peak allocated (MiB) |
| --- | ---: | ---: | ---: |
| 定长 T=11264 | 12.721 | 12.793 | 5089.000 |
| 变长 T=16384，16×1024 | 18.095 | 18.150 | 7350.261 |

2026-11-17 对照以 Ascend 950 上 `25c0cc3d` 的完整融合反向 NPU Event 和 NVIDIA H20
上 FLA 的完整反向 CUDA Event 为准。两侧各预热 5 次、采样 20 次，取中位数，均启用
初态、末状态梯度和 Q/K L2Norm。对照固定 `HK=16`，覆盖 GVA 1～16 的定长和变长。
A5 与 H20 的状态输入分别为 BF16 和 FP32，变长对照使用 16 段不等长序列。

| GVA 8 对照 | A5 NPU Event P50 (µs) | H20 CUDA Event P50 (µs) | H20 耗时 / A5 耗时 |
| --- | ---: | ---: | ---: |
| 定长 T=11264 | 13887.57 | 12907.92 | 0.929× |
| 变长 T=16384 | 20154.57 | 17778.34 | 0.882× |

该版本 32 组中有 30 组达到 0.8×。GVA 1 定长和变长分别为 0.631× 和 0.714×。

MSS 同步问题于 2026-11-05 完成基线定界。独立构建原始提交
`2fb57c34750a046594c903586e970be5e3ca9d7b`，使用独立安装环境和同一冻结典型
G4 输入（HK=32，其余同定长典型 shape）：普通运行成功，MSS 下复现等待。
底层 plog 同时记录 AI Core 非法指令地址异常（错误码 256 / runtime 507015），
因此报告中的 Vector 等待是执行中断后的现象，不能据此修改同步流水。

在相同 `-O2 -g --sanitizer`、单 TilingKey=102 的基线对照中，仅增加
`-fno-jump-tables` 后，内核完整执行、输出有限且 synccheck 无错误。
本轮使用 CANN 9.2.0-beta.2 / MSS 26.2.0；检测构建已有
`--cce-enable-sanitizer` 和调试信息，但缺少禁用跳转表的选项。此结果与
[CANN 官方 Ascend 950 检测构建说明](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/debug/op_debug_prof.md)
一致（核对文档 blob `fc2ef34a1e6d969b2dd0c59c3fbd995fc25fd890`）。本问题
定界为当前工具链的跳转表插桩兼容问题，此次 GVA 扩展未引入该异常。
最终修复未修改算子源码或共享 CANN 安装。

本次最小修复限定于 MSS 检测包的构建参数：

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_gdn_bwd_intra \
  FLA_NPU_BUILD_ARGS='-j16 -O2 --ops-compile-options -fno-jump-tables' \
  python scripts/build_wheel.py -g --sanitizer
```

当前版本不裁剪 TilingKey，完整构建并安装至独立检测环境，安装产物 MD5 核验通过。
synccheck 使用 `--block-id=0 --kernel-name=ChunkGdnBwdIntra`；只有内核及程序完整
结束、存在有效检测记录且无错误时才记为通过。MSS 检测包的耗时包含插桩编译，
不作为性能数据。原普通包的 8/8 精度和性能记录仍按上文解释。

| 当前版本典型输入 | 普通运行 | synccheck | memcheck | CPU 混合容差精度 |
| --- | --- | --- | --- | --- |
| G4 定长，T=11264，HK=32 | 通过 | 通过 | 未扩展 | 未扩展 |
| G8 定长，T=11264，HK=16 | 通过 | 通过 | 通过 | 通过 |
| G8 变长，T=16384，16×1024，HK=16 | 通过 | 通过 | 通过 | 通过 |

检测目标均为 intra，MSS 只检查 block 0；memcheck 按工具说明设置
`PYTORCH_NO_NPU_MEMORY_CACHING=1`。同步记录分别为 G4/G8 定长各 225168 条、
G8 变长 327568 条，均包含 `KERNEL_FINISH` 和服务正常退出记录。
两例精度沿用原 ATK CPU 标杆及 mixed_tolerance_bm，比较完整 `w/u/dv_local`，
通过 2/2；未增加其他 shape、属性或硬件平台验证。

当前检测 wheel SHA256：
`bb18cdf37dd145c76db8c31ee9930122d7dbfc1c0b233c2ca5213b4b2436a52f`；
基线禁用跳转表 wheel SHA256：
`f8319632b86efbbdfb1d0c30e69653227a42d0cf7b54f009ec0df99570fb6635`。

## Workspace 核对记录（2026-11-19）

200 MiB 的统计范围为 Intra、FwdH、DHU、Finalize 的系统及用户 workspace。
输入、输出和既有中间张量存储按约定排除。以各阶段申请量求和作为复用前的上界，
并单列本次需求新增量。此口径取代此前按父链显存峰值差值判断 200 MiB 的结论。
父链 executor 的 `GetWorkspaceSize` 返回值还包含中间张量存储，核对时使用各子阶段
tiling 中 `GetWorkspaceSizes(1)` 设置的字节数。

实测版本为 `4759e2c079d479a2a18fc053c0c31d2c27ae4da0`，Ascend 950、
28 个 AIC，CANN 9.2.0-beta.2。独立诊断 Host 库在各阶段设置 workspace 后打印
系统、既有用户、本次新增和总申请量。GVA 4、8 的定长及变长共 4 次真实 ACLNN
`GetWorkspaceSize` 查询均成功。此次核对读取申请量，未启动 kernel。
其余比例按已核对的申请公式展开。

下表使用 ATK 典型矩阵，`B=1, K=V=128, BT=64`，
`HK=min(16,128//G), HV=HK*G`。定长 `T=11264`，变长为 16 段各 1024 token。
所有数值单位均为 MiB，1 MiB = 1024² 字节。

| GVA | HK | HV | 定长 workspace 合计 | 变长 workspace 合计 | 定长本次新增 | 变长本次新增 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 16 | 151.501 | 173.376 | 0 | 0 |
| 2 | 16 | 32 | 151.501 | 174.064 | 0 | 0 |
| 3 | 16 | 48 | 167.001 | 174.751 | 0 | 0 |
| 4 | 16 | 64 | 163.126 | 174.751 | 0 | 0 |
| 5 | 16 | 80 | 176.314 | 178.251 | 3.5 | 3.5 |
| 6 | 16 | 96 | 170.501 | 178.251 | 3.5 | 3.5 |
| 7 | 16 | 112 | 178.251 | 178.251 | 3.5 | 3.5 |
| 8 | 16 | 128 | 176.876 | 178.251 | 3.5 | 3.5 |
| 9 | 14 | 126 | 176.876 | 178.251 | 3.5 | 3.5 |
| 10 | 12 | 120 | 175.501 | 178.251 | 3.5 | 3.5 |
| 11 | 11 | 121 | 176.189 | 178.251 | 3.5 | 3.5 |
| 12 | 10 | 120 | 175.501 | 178.251 | 3.5 | 3.5 |
| 13 | 9 | 117 | 175.501 | 178.251 | 3.5 | 3.5 |
| 14 | 9 | 126 | 176.876 | 178.251 | 3.5 | 3.5 |
| 15 | 8 | 120 | 175.501 | 178.251 | 3.5 | 3.5 |
| 16 | 8 | 128 | 176.876 | 178.251 | 3.5 | 3.5 |

GVA 8 的逐阶段查询结果如下，每阶段系统 workspace 均为 16 MiB。

| 阶段 | 定长既有用户 workspace | 变长既有用户 workspace | 定长本次新增 | 变长本次新增 |
| --- | ---: | ---: | ---: | ---: |
| Intra | 0 | 0 | 0 | 0 |
| FwdH | 49.876221 | 51.251221 | 0 | 0 |
| DHU | 35 | 35 | 0 | 0 |
| Finalize | 24.5 | 24.5 | 3.5 | 3.5 |

本次新增只来自 Finalize 的两个 FP32 累计区，
`2*C*65536=3670016` 字节，C=28 时为 3.5 MiB。
矩阵内四阶段 workspace 合计的最大上界为 178.251221 MiB，满足 200 MiB 约束。

## main 合并后的 A5 典型回归（2026-11-19）

代码提交 `6e91ecbd` 合并 main `e5a58fdc`，保留主线 BSND/TND 直接访问路径，
并为 GVA>4 的 Finalize 归约补齐对应的 Q/K 读取、rstd 索引和梯度写回。
本节所涉 workspace 申请公式与上述核对版本一致。

Ascend 950、CANN 9.2.0-beta.2 下的 A5 O3 包构建和安装通过，
实际加载的 Host 库及 21 个 kernel 对象均与构建产物一致。
Stable 离线门禁 53 项通过。沿用既有 ATK 输入生成方式和精度判据，以下五例通过。

| GVA | 模式 | 布局 | T | Q/K norm | 精度 |
| --- | --- | --- | ---: | --- | --- |
| 4 | 定长 | BSND | 11264 | 关闭 | 通过 |
| 4 | 变长，16×1024 | TND | 16384 | 关闭 | 通过 |
| 8 | 定长 | BSND | 11264 | 关闭 | 通过 |
| 8 | 变长，16×1024 | TND | 16384 | 关闭 | 通过 |
| 8 | 定长 | BSND | 11264 | 开启 | 通过 |

## A2 布局归一化与恢复修复抽样回归（2026-11-18）

Ascend 910B3，BF16，`K=V=128`、`chunk_size=64`。从既有 140 条冻结矩阵使用
随机种子 `20261118` 抽取 20 条，覆盖 BNSD 5 条、BSND 8 条、NTD 3 条、TND 4 条。
使用 CPU FP32 benchmark 与 FP64 golden，沿用 `cv_fused_double_benchmark` 的
`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5`。

恢复 `8d27f1fd` 撤掉的实现修复后，20 条执行均成功、无超时，精度由 16 条通过提升至
18 条通过。来源 case `133/104` 的空序列 `dh0` 已通过；来源 case `1/44` 的 `dG`
仍失败，平均相对误差比分别为 `1.503182364`、`1.505164869`，超过原阈值 `1.5`。
这两条用例的 12 个有效梯度输出恢复前后逐元素完全一致，最大绝对差为 0。
本轮 host、tiling 动态库与全部 21 个 kernel 的运行产物一致性检查通过。

## A2 全量精度与 CT 复检（2026-11-18）

在同一份恢复修复的代码和已核对的安装产物上，沿用上述双标杆和阈值，执行既有
三组各 140 条冻结矩阵、12 条基础用例和 14 条补充用例，共 446 条。

| 用例组 | 总数 | 单轮通过 | 单轮失败 |
| --- | ---: | ---: | ---: |
| keys_0 | 140 | 131 | 9 |
| keys_1000 | 140 | 135 | 5 |
| keys_2000 | 140 | 138 | 2 |
| 基础用例 | 12 | 12 | 0 |
| 补充用例 | 14 | 13 | 1 |
| 合计 | 446 | 429 | 17 |

446 条运行均成功且无超时。17 条失败均为 `d_g`：16 条平均相对误差比超限，
1 条最大相对误差比超限。

对全部 17 条失败用例分别执行 50 轮 `accuracy_lt`，共 850 组样本。每条用例均核对
50 个不同种子，且每轮 NPU、CPU FP32 benchmark、CPU FP64 golden 的种子一致。
CT Tool v0.9.1 执行 `ct dual analyze <accuracy_lt_report.xlsx>`，17 条全部通过；
99 个有效输出的 297 项统计指标均通过，每项均有 50 个有效样本。
复检的 ATK 单轮判定为 779 条通过、71 条失败；CT 统计结论与单轮结果分别记录，
未修改标杆、输入值域或精度阈值。

上述 A2 精度结果对应原 layout 分支的 `8689b067` 及其安装产物。移植到 `upstream/main` 的 `56984e51` 后，已完成离线 Stable 门禁、HostProbe、Intra 参数拒绝与 head 映射回归、CPU 标杆回归；新基线尚未重新编译或执行 NPU 精度测试。
