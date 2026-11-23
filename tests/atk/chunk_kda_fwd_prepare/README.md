# chunk_kda_fwd_prepare ATK 测试

本目录按 ATK 公开输入格式验证 `chunk_kda_fwd_prepare`。NPU 节点只调用稳定入口
`fla_npu.ops.ascendc.chunk_kda_fwd_prepare`，不加载 `torch_npu` dispatcher。

## 输入与输出

YAML 和三份冻结 JSON 都直接声明以下 21 个公开输入，顺序与 executor 一致：

```text
q, k, v, g, beta,
scale, layout, chunk_size, epsilon,
use_qk_l2norm_in_kernel, use_gate_in_kernel,
use_beta_sigmoid_in_kernel, allow_neg_eigval, safe_gate,
lower_bound, use_exp2,
a_log, dt_bias, cu_seqlens, chunk_indices, backward_mode
```

用例中不使用 marker tensor、序列化 `case_spec` 或其他隐藏控制字段。
`a_log`、`dt_bias`、`cu_seqlens` 和 `chunk_indices` 按场景传空或真实数据。

接口固定保留 13 个输出槽位：

```text
gk, aqk, akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

`backward_mode` 只控制真实搬出，不改变输出槽位数：

| 模式 | 真实输出 |
| --- | --- |
| `none` | `gk/aqk/w/u/kg/qg_scaled` |
| `recompute` | `none` 模式输出，加 `akk/q_hat/k_hat/q_rstd/k_rstd/beta_eff` |
| `save` | 全部 13 个输出，额外保存 `qg` |

CPU 节点使用 FP64 独立标杆计算；NPU 节点检查 13 个槽位的可选性、shape、dtype
和有限值。性能任务在 `with_output=false` 时只检查接口合同，不执行同步、有限值扫描
或 D2H 搬运。

## 用例资产

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_prepare.json` | 200 | 精度、功能、布局、dense/varlen、tail、GVA 和输出模式 |
| `atk_chunk_kda_fwd_prepare_perf.json` | 10 | 10 个模型大 shape，连续 BNSD、纯推理输出 |
| `atk_chunk_kda_fwd_prepare_mss.json` | 432 | 每个可达 TilingKey 一条，用于确定性及后续内存检查 |

200 条精度用例是 200 个不同结构，而不是少量结构重复 seed。它们包含 100 条均匀
分布和 100 条正态分布，并覆盖模板属性的全部合法二元组合。432 条 TilingKey 用例由
以下轴的合法笛卡尔积构成：

```text
2 gate dtype x 2 beta dtype x 2 norm mode x
3 beta mode x 3 gate mode x 2 exp mode x 3 output mode = 432
```

其中三个输出模式各 144 条。内存检查当前暂停，`_mss.json` 只表示用例已准备，不能
据此声称 sanitizer 已通过。

性能文件的 10 条记录保留精度集合中的逻辑 ID `56..65`，但在该文件内的选择位置仍
是 `0..9`。重点用例名称分别包含逻辑 ID `0063` 和 `0064`，对应性能文件位置 7 和 8：

- `B=1, HK=96, HV=96, T=8192`，dense BNSD，目标 2700 us；
- `B=1, HK=96, HV=96, T=16384`，varlen BNSD，目标 5600 us。

执行前应按 case name 核对目标，不能把 ATK 重新生成后的物理顺序当作稳定 ABI。

## 生成

`dtype_numbers: 100` 与 `shape_distributions: [[0, 1.0]]` 组合后由 ATK 生成
200 条用例。目标 A5 环境中的 ATK 26.8.8 只注册 `mixed_tolerance_bm`，因此 YAML
使用该原生标准作为版本兼容选择。

先用官方 ATK 生成链验证 YAML 与 generator：

```bash
cd tests/atk/chunk_kda_fwd_prepare
atk case \
  -f ./chunk_kda_fwd_prepare.yaml \
  -p ./gen_chunk_kda_fwd_prepare.py \
  -en 0 -s 20260904
```

三份冻结 JSON 是本目录用例的唯一来源。生成器只为 `atk case` 提供精度集，
不会覆盖这三份已验证的输入。检查数量及用例名是否重复：

```bash
python3 ./gen_chunk_kda_fwd_prepare.py --summary
```

`atk case` 可以合法重排 case 或重写 ID；校验时按 case name 比较 21 个公开输入的
语义，不要求生成文件与冻结文件逐字节相同。

## 精度

加载当前构建的 A5 wheel 和 custom OPP 后，先跑单条，再跑全量：

```bash
atk node --backend npu --devices 0 -o ./atk_output/accuracy \
  node --backend cpu task \
  -c ./atk_chunk_kda_fwd_prepare.json \
  --task accuracy \
  -p ./executor_chunk_kda_fwd_prepare.py \
  -s 0 -e 1 -sp -to 60

atk node --backend npu --devices 0 -o ./atk_output/accuracy_full \
  node --backend cpu task \
  -c ./atk_chunk_kda_fwd_prepare.json \
  --task accuracy \
  -p ./executor_chunk_kda_fwd_prepare.py \
  -s 0 -e 200 -sp -to 60
```

单条用例执行超过 60 秒即判定超时。只有最终报告同时满足总任务 200、执行失败 0、
精度结论通过，才能声明全量精度通过；不能只依据 shell 返回码。

## 性能

性能以 `msopprof` 中目标 kernel 的 duration 为准，不使用 Python wall time。先按名称
确认性能 JSON 中的位置，再用单算 ATK `run` 命令下发；例如重点位置 7：

```bash
msopprof \
  --application="atk node --backend npu --devices 0 task -c ./atk_chunk_kda_fwd_prepare_perf.json --task run -p ./executor_chunk_kda_fwd_prepare.py -s 7 -e 8 -sp -to 60" \
  --output=./atk_output/profile_case_7 \
  --aic-metrics=BasicInfo \
  --launch-count=1 \
  --warm-up=0 \
  --kill=off
```

位置 8 同理。报告需记录实际 shape、case name 和 kernel duration，并分别与
2700 us、5600 us 目标比较。

## 确定性

确定性使用 `_mss.json` 的 432 个 TilingKey，通过 ATK 26.8.8 的
`accuracy_dc` 任务逐位比较，每条重复 50 轮：

```bash
atk node --backend npu --devices 0 task \
  -c ./atk_chunk_kda_fwd_prepare_mss.json \
  --task accuracy_dc \
  -p ./executor_chunk_kda_fwd_prepare.py \
  -s 0 -e 432 -sp -to 60
```

单条仍使用 60 秒上限。内存检查恢复后需另行构建 sanitizer 包并分别运行 memcheck、
racecheck、initcheck 和 synccheck，不能复用普通优化包的结论。

## 静态检查

```bash
python3 -m py_compile \
  tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py \
  tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py
```


## 核内游标回归（#1164）

与 PR #1194 在同一 main 基线 `3b24b56d` 上比较。使用原生 ACLNN 空表入参，
覆盖真正的 kernel fallback；常规 Python 包装层自动补表行为保持兼容。
数学计算、调度、buffer、同步、ABI 和 TilingKey 均未修改。

A2 原始精度套件 200/200 执行成功，198 条直接通过。
case_id=123、195（需要复检）在两种实现中均出现相同的原生 mixed_tolerance_bm
失败；两条都是 dense 用例。保留原始 range、数据分布和阈值，保存输出并通过
CT viz 排查后，对每条固定结构执行 accuracy_lt 50 组不同随机数值。
以原有 FP64 CPU golden 为真值、PR #1194 的同精度输出为 benchmark，
游标版所有输出均与 benchmark 逐位一致，CT L2 和 ct dual analyze 统计均通过。
原始 mixed_tolerance_bm 失败结果保留；复检结论表示本次寻址优化无精度退化。

补充无表/显式表对比覆盖空序列、尾块、不同 head 数、最多 4096 段，
以及总 token=524288、4684 段、4 heads、K=V=128；有效输出逐位一致。

性能使用相同 A2、输入和采集参数，串行测量；总 token=524288，
序列数覆盖 1/128/512/1024/4684，4 heads，K=V=128，chunk_size=64。
每例预热 5 次后采集 1 次，不报告均值或稳定性统计。下表为 4684 段结果，单位 us。
host 与整次调用来自 msprof API 时间线，kernel 来自 op_summary 的 Task Duration(us)。
整次调用范围为 GetWorkspaceSize 开始至随后的同步结束，包含采集开销；
描述符构造、输入生成和 workspace 分配不在统计范围内。

| 方案 | host GetWorkspaceSize | kernel | 整次调用 |
| --- | ---: | ---: | ---: |
| PR #1194 host 建表 | 331.040 | 16124.262 | 16670.500 |
| kernel 游标 | 115.800 | 15897.458 | 16185.230 |

完整段数扫描：下表为游标相对 PR #1194 的时延变化百分比，负值表示更快。

| 入口 | 序列数 | GetWorkspaceSize 变化 | kernel 变化 | 整次调用变化 |
| --- | ---: | ---: | ---: | ---: |
| prepare | 1 | -92.17% | +0.38% | -1.87% |
| prepare | 128 | -93.49% | -1.29% | -3.45% |
| prepare | 512 | -94.60% | -1.74% | -4.25% |
| prepare | 1024 | -63.55% | -1.21% | -2.79% |
| prepare | 4684 | -65.02% | -1.41% | -2.91% |

### 文件组织调整复验（2026-11-24）

游标实现归入各算子已有头文件，未新增公共头文件或测试文件。
A2、A5 的三个相关算子及依赖均编译通过，Stable-ABI 调用点检查通过。
A2 全量精度用例重新执行通过：Fwd 48 条、Finalize 200 条，Prepare 198 条直接通过，
另 2 条既有数值误差用例各执行 50 轮并通过 CT 双标杆复检；原始混合容差结果保留。
原生空表补充用例及 524288 token、4684 段的四个入口均与显式表有效输出逐位一致。
V1、Prepare、Finalize 三份游标分别通过 1658 组配置、4787524 次查询的 CPU oracle。
调整后 A2 的 19 个 kernel 二进制与调整前逐字节一致，上述性能数据对应相同二进制。
