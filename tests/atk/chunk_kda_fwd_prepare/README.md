# chunk_kda_fwd_prepare ATK 测试

本目录按 ATK 公开输入格式验证 `chunk_kda_fwd_prepare`。使用 `npu` 后端调用稳定入口
`fla_npu.ops.ascendc.chunk_kda_fwd_prepare`，并与独立 CPU 标杆比较。

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

### 安装 main Daily 包

先按根目录 [README](../../../README.md) 准备与 CANN/Python 匹配的运行环境，
并按 [ATK 官方说明](https://gitcode.com/Ascend/ATK) 在同一 Python 环境安装 ATK。
以下命令从仓库根目录执行；使用本目录的最新测试脚本，无需本地编译算子。
ATK 及其依赖必须与当前 Python 版本匹配，不要混入另一 Python 小版本的 site-packages。

```bash
source <CANN安装目录>/ascend-toolkit/set_env.sh
FLA_TIER=a2  # A5 改成 a5；pip 自动匹配 aarch64/x86_64
python -m pip install --upgrade --pre --no-deps --no-cache-dir --no-index \
  --find-links "https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/flash-linear-attention-npu-${FLA_TIER}/latest/" \
  "flash-linear-attention-npu-${FLA_TIER}==0.*"
python -m pip show "flash-linear-attention-npu-${FLA_TIER}" atk
python -m atk --version

FLA_PACKAGE=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("fla_npu").submodule_search_locations)))')
source "$FLA_PACKAGE/opp/vendors/fla_npu_transformer/bin/set_env.bash"
npu-smi info
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
```

`0.*` 选择 main Daily，避免 `latest` 页面同时存在维护分支 Daily 时选择另一分支。
`--no-deps` 不替代 README 的依赖准备步骤。记录实际安装版本和 wheel 哈希；两个架构
发布进度可能不同，不应把“最新可下载包”称为“当前 main 源码重构建”。

### 验证 Stable wrapper

选择健康且空闲的设备，输出目录必须不存在，避免读到上一轮的报告。以下默认验证全部
200 个冻结精度用例；最小排障时可加 `--start 0 --end 1`。

```bash
ATK_RUN=$(mktemp -d)
python tests/atk/chunk_kda_fwd_prepare/scripts/run_accuracy.py \
  --backend npu --device 0 --output "$ATK_RUN/npu"
```

命令经 ATK `accuracy` 任务执行，使用 `mixed_tolerance_bm`；脚本不修改输入、
CPU 标杆或精度标准。ATK 单用例超时参数为 120 秒，外层作业仍应设置总时限，以处理
设备故障导致的底层调用阻塞。`npu` executor 将测试字段 `a_log` 传给公共 wrapper 的
`A_log` 参数，测试 JSON 的字段名不变。

本测试入口选择 `--backend pyaclnn` 会立即报错，并提示改用 `--backend npu`。
`pyaclnn` 直接调用 C ACLNN 接口时，需要另行适配每个算子的库搜索、参数顺序及类型；
该入口仅支持公共 Stable wrapper 的调用方式。

每次运行保存 `command.json`（真实命令）、`cases.json`、`summary.json` 和
`accuracy/atk_output/**/report/*.xlsx`。运行副本把正常用例的空 `expected_error_msg`
改为 `null`，防止空正则匹配任意异常。脚本检查进程退出码、报告唯一性、case ID 完整性、
总数、通过数、执行失败数和最终精度结论；缺报告或少跑用例均失败。只有全部 200 个用例
精度通过才能声明本集合通过，不能只看日志中的 `atk task success`。

### 验证记录（2026-11-04）

以下使用下载的 main Daily wheel、CANN 9.2、Python 3.12 和 ATK 26.9.8，
测试源码基于 `a13871e8` 加本目录的调用适配修复；未重新编译算子。

| 平台 | main Daily 版本 | torch / torch_npu | Stable wrapper (`npu`) |
| --- | --- | --- | --- |
| A5 / x86_64 / Ascend950 | `0.dev20261103+main.fb3d47d` | 2.13.0 / 2.13.0 | 200/200 PASS |
| A2 / aarch64 / Ascend910B2 | `0.dev20261104+main.8f6c77f` | 2.12.0 / 2.12.0 | 198/200，精度未通过 |

每轮检查 200 个 case、2,096 个实际输出，覆盖 `none/recompute/save` 三种输出模式
和 `BNSD/BSND/NTD/TND` 四种布局。两份 Daily 到测试基线间，本算子的 kernel、
ACLNN 接口、Stable adapter 和冻结精度集合没有改动。

A2 失败项为 case 123 的 `w/u`、case 195 的 `akk/w/u`。在另一张健康设备
单例复跑均复现。此前还通过专用 `pyaclnn` 适配器作过诊断，两条路径的 18 个实际输出
逐张量完全一致；该诊断路径现已从本目录移除。该现象属于当前 A2 包与既有精度标准
不符，具体算子数值根因待定位。本轮未放宽容差或修改标杆，不声明 A2 全量精度通过。

本记录仅覆盖该算子的 Stable wrapper 精度调用，不代表 performance、DC、MSS、A3 或 overlay
安装场景已完成验收。

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
