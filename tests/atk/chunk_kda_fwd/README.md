# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。
- `q/k/v` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/BFLOAT16`；`beta` 支持 `FLOAT/BFLOAT16`。
- `K/V` 只支持两档且必须同档：`K=V=64` 或 `K=V=128`；混合档（如 `K=64,V=128`）与其它
  取值（含 `V=256`）返回 `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖同类拦截。
- `chunk_size` 只支持 `64`，其它取值（含 `128`）在参数校验阶段返回
  `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖该类拦截；rank-4 变长输入要求 `B=1`，
  逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## 精度拓扑

精度标准统一使用 `mixed_tolerance_bm`。显式 CPU 节点调用 executor 的 FP64 路径生成
唯一 golden，显式 NPU 节点只运行 DUT。CPU golden 输出转换为 FP32，NPU 输出保持算子
原始 dtype，由 ATK 按混合容差规则比较。

```text
ATK accuracy task
|-- CPU FP64 golden (output FP32)
`-- NPU DUT
```

## 运行

在仓库根目录准备好 ATK、CANN、OPP 和 Python 包环境后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

只验证前 5 个 case：

```bash
CASE_START=0 CASE_END=5 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

确定性回归会重复运行 MSS 用例并逐位比较全部可见输出。MSS 包含 #440 的 63-token 单序列
尾块场景，通过 96 个 value heads 放大并行调度覆盖，同时启用 `disable_recompute=true`、最终状态
和全部中间量：

```bash
DC_LOOP_NUMS=100 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=determinism
```

性能、确定性、mssanitizer 和用例生成均通过统一脚本的对应 `-scope` 执行。


## 核内游标回归（#1164）

与 PR #1194 在同一 main 基线 `3b24b56d` 上比较。使用原生 ACLNN 空表入参，
覆盖真正的 kernel fallback；常规 Python 包装层自动补表行为保持兼容。
数学计算、调度、buffer、同步、ABI 和 TilingKey 均未修改。

A2 原始精度套件 48/48 通过。独立 CPU oracle 覆盖 1658 组配置、
4787524 次查询，包括重复、前进跳跃、回退、越界后重查、空序列、grid stride
和 A2 RotateWork。原生无表与显式 canonical 表的有效输出逐位一致；
覆盖千级序列、V1 的 64/128 维，以及总 token=524288、4684 段、4 heads、
K=V=128 的 V2 场景。原生 V1 在同规模下的 attn/Aqk/Akk 有效输出亦逐位一致。

游标分别定义在 V1、Prepare、Finalize 已有头文件中；CPU oracle 采用临时编译验证，
不新增公共头文件或测试文件。

性能使用相同 A2、输入和采集参数，串行测量；总 token=524288，
序列数覆盖 1/128/512/1024/4684，4 heads，K=V=128，chunk_size=64。
每例预热 5 次后采集 1 次，不报告均值或稳定性统计。下表为 4684 段结果，单位 us。
host 与整次调用来自 msprof API 时间线，kernel 来自 op_summary 的 Task Duration(us)。
整次调用范围为 GetWorkspaceSize 开始至随后的同步结束，包含采集开销；
描述符构造、输入生成和 workspace 分配不在统计范围内。

| 方案 | host GetWorkspaceSize | kernel | 整次调用 |
| --- | ---: | ---: | ---: |
| PR #1194 host 建表 | 821.910 | 25242.325 | 26281.720 |
| kernel 游标 | 584.860 | 25313.806 | 26115.380 |

V2 的 kernel 数值为同一次调用内 Prepare、FwdH、Finalize 三个顺序 task 的耗时之和；各 pipe 存在 overlap，不相加。

原生 V1 同规模对比：

| 方案 | host GetWorkspaceSize | kernel | 整次调用 |
| --- | ---: | ---: | ---: |
| PR #1194 host 建表 | 516.640 | 625710.942 | 626705.000 |
| kernel 游标 | 287.540 | 621361.955 | 622053.790 |

V1 与 V2 分别按各自接口开关测量，仅比较同一入口的两种寻址实现。

完整段数扫描：下表为游标相对 PR #1194 的时延变化百分比，负值表示更快。

| 入口 | 序列数 | GetWorkspaceSize 变化 | kernel 变化 | 整次调用变化 |
| --- | ---: | ---: | ---: | ---: |
| fwd_v2 | 1 | -96.13% | +2.06% | +1.60% |
| fwd_v2 | 128 | -96.48% | -0.28% | -2.00% |
| fwd_v2 | 512 | -96.58% | -0.50% | -2.31% |
| fwd_v2 | 1024 | -33.22% | -0.32% | -0.86% |
| fwd_v2 | 4684 | -28.84% | +0.28% | -0.63% |
| fwd_v1 | 1 | -89.38% | -1.51% | -1.83% |
| fwd_v1 | 128 | -91.10% | -1.59% | -2.26% |
| fwd_v1 | 512 | -90.56% | -2.03% | -2.54% |
| fwd_v1 | 1024 | -40.44% | -1.21% | -1.38% |
| fwd_v1 | 4684 | -44.34% | -0.70% | -0.74% |

本轮单序列 V2 的整次调用增加 1.60%；4684 段 V2 减少 0.63%。host 降时延不能直接等同整次调用加速；本轮为单次观测，不代表稳定性统计。

### 文件组织调整复验（2026-11-24）

游标实现归入各算子已有头文件，未新增公共头文件或测试文件。
A2、A5 的三个相关算子及依赖均编译通过，Stable-ABI 调用点检查通过。
A2 全量精度用例重新执行通过：Fwd 48 条、Finalize 200 条，Prepare 198 条直接通过，
另 2 条既有数值误差用例各执行 50 轮并通过 CT 双标杆复检；原始混合容差结果保留。
原生空表补充用例及 524288 token、4684 段的四个入口均与显式表有效输出逐位一致。
V1、Prepare、Finalize 三份游标分别通过 1658 组配置、4787524 次查询的 CPU oracle。
调整后 A2 的 19 个 kernel 二进制与调整前逐字节一致，上述性能数据对应相同二进制。
