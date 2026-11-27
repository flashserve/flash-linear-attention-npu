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
`a_log` 是用例字段和 executor 内部字段名；调用 Stable 公共接口时使用关键字
`A_log=inputs.a_log`，与 `chunk_kda_fwd_prepare(..., A_log=None, ...)` 的签名一致。

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

按 [公共环境准备说明](../README.md#运行前准备) 激活包含 ATK 和完整 FLA wheel 的
Python 环境并加载 CANN/vendor 脚本后，可从仓库根目录使用统一入口：

```bash
# 最小调用回归，只执行第一条 case；不代表全量精度验收。
CASE_START=0 CASE_END=1 \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd_prepare -soc=ascend950 -npu_device_id=0 -scope=accuracy

# 全量精度：移除外层 shell 中已有的 case 范围限制。
unset CASE_START CASE_END ACCURACY_START ACCURACY_END
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd_prepare -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

入口使用 `--backend npu` 和本目录 executor，通过 FLA Stable wrapper 加载算子；
无需为此切换编译后端或设置 `ATK_CUSTOM_OPP_PATH`。脚本会检查 ATK 报告中的精度结果。

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

### Stable 调用关键字回归（2026-11-04）

在源码 `a13871e8` 上修正 executor 的 `A_log` 调用关键字后，使用上述统一脚本的
`CASE_START=0 CASE_END=1` 命令，在 A5 / CANN 9.2 / Python 3.12 / torch 2.13.0 /
torch_npu 2.13.0 / ATK 26.9.8 上完成修改前后对照。两次均保留默认 GM 初始化，
且未设置 `ATK_CUSTOM_OPP_PATH`。

被测完整 wheel 为 `0.dev20261103+main.fb3d47d`，与上述源码的本算子实现及测试资产一致。
原 executor 的报告为 0/1，统一脚本退出码为 1；修正后为 1/1 Pass，6 个可见输出全部
通过比较，脚本退出码为 0。这是调用契约的最小回归，不代替 200 条精度用例、性能、
确定性、MSS 或 A2/A3 验收；本轮未重新编译算子。

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
