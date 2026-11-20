# ChunkGatedDeltaRuleBwd ATK 测试说明

本目录验证 `chunk_gated_delta_rule_bwd` 融合接口。NPU 节点只调用一次公开融合接口；
CPU 节点使用同一组冻结输入，独立串联四个阶段的 CPU 标杆，比较最终有效梯度。

## 标杆与输出

CPU 标杆依次计算 `ChunkGdnBwdIntra`、`ChunkFwdH`、
`ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize`。阶段边界按照
NPU 接口的数据类型量化，避免把 NPU 中间结果作为标杆。

有效输出顺序为 `dq, dk, dv, d_beta, d_g`；传入 `initial_state` 时额外比较
`dh0`。其中 `d_beta` 和 `d_g` 固定使用 BSN `[B,T,HV]`，其余有效输出跟随
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
