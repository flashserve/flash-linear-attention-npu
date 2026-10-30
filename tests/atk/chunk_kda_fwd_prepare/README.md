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
