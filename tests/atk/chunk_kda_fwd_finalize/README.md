# chunk_kda_fwd_finalize ATK 测试

本目录只依赖八个公开输入：

```text
qg_scaled, aqk, v_new, h, cu_seqlens, chunk_indices,
output_layout, state_v_first
```

冻结 JSON 是 ATK 原生直接输入；没有 marker tensor、序列化的
`case_spec` 或隐藏测试开关。`qg_scaled/Aqk` 使用 Prepare 的
head-major BF16 形状，packed 时 rank-3；`v_new/h` 使用 FwdH 的
rank-4/rank-5 BF16 形状，packed 时首维仍为 1。`Aqk`
已经乘过 scale，CPU 和 NPU 都不再重复缩放。

## 资产

| 文件 | 用例 | 覆盖 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_finalize.json` | 200 | 25 个边界/shape/变长结构 × 4 个输出 layout × 2 种 state 轴顺序，覆盖两个 tiling 模板实例 |
| `atk_chunk_kda_fwd_finalize_perf.json` | 10 | 模型大 shape、dense/varlen，供单算 profiling |
| `atk_chunk_kda_fwd_finalize_mss.json` | 12 | 4 个输出 layout × 2 种 state 轴顺序的尾块输入，以及 AIV 搬运模板的 dense/packed、KV/VK 确定性输入 |

精度包括 T 为 `1/15/16/17/31/32/33/63/64/65` 的边界、
多 chunk、严格递增变长序列、显式/自动 canonical chunk indices、
不同 HV、`B=2` dense，以及 `B=384,HV=5,T=17` 和由 382 个单
token 序列加 33/64 行序列组成的 AIV 搬运模板 dense/varlen 边界。这些
用例同时覆盖不足 4 个 head 的尾组、显式 `chunk_indices` 及
`Aqk` 的 1/2/3/4 个 16-row 数据块。
packed 组中 `qg_scaled/Aqk` 是 rank-3，
`v_new` 是 rank-4。独立 Finalize 无 HK 输入，不把这些 case
记作 GVA 映射验证；需要在三阶段组合链路另测 HK<HV。

## Tiling 模板覆盖

| `USE_AIV_INPUT_MOVER` | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 预期核类型 |
| --- | --- | --- | --- | --- | --- | --- |
| `false` | A2/A3；或 A5 的 head 被拆分；或最小单核负载不足 8 个 chunk | 64-71 | 0-63、72-87、92-191 | 0-7 | A2/A3/A5 | AIC-only |
| `true` | A5、每个 work item 包含完整 value head，且最小单核负载至少 8 个 chunk | 192-199 | 88-91、192-199 | 8-11 | A5 | MIX AIC 1:2 |

case 88-91 是 dense 的 17 行尾块；case 192-199 在同一变长输入中
包含 1/33/64 行序列并覆盖四种 layout。两组均使用非 4 倍数 HV，
因此会进入第二个 head group 并验证不足 4 个 head 的尾组。上述
模板组合同时通过 `ASCENDC_TPL_KERNEL_TYPE_SEL` 声明核类型，不在
Kernel 内按数值 TilingKey 分支。当前提交的 runtime profile 中，性能
case 2 命中 `_0`/cube 实例，耗时 908.545 us；性能 case 3 命中
`_1_mix_aic`/mix 实例，耗时 1794.889 us，AIC/AIV block 为
28/56。性能结论以性能 JSON 对应的大 shape profiling 为准。

CPU 节点把四个直接输入恢复为 BF16，使用 FP32 两项矩阵乘并在求和
后按 BF16 输出舍入；返回 FP32 承载 BF16 结果供 ATK 原生
`mixed_tolerance_bm` 比较。NPU 节点用本 executor 的窄 aclnn
直调适配器验证设备实现。公开
`fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 稳定入口由共享 Python
wrapper 提供，其接口验证不属于本 ATK 目录。

## 生成与静态核对

```bash
python3 tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  --output-dir tests/atk/chunk_kda_fwd_finalize --summary

cd tests/atk/chunk_kda_fwd_finalize
atk case -f ./chunk_kda_fwd_finalize.yaml \
  -p ./gen_chunk_kda_fwd_finalize.py -en 0 -s 20260914
```

`dtype_numbers: 200` 与 `shape_distributions: [[0,1.0]]` 在目标 ATK
生成 200 条；生成器覆盖 YAML 临时形状，并在导出前检查八个输入
顺序。冻结 JSON 不因 `atk case` 的物理 case ID 重新排序而覆盖。

```bash
python3 -m py_compile \
  tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py
```

## 精度

加载与待测算子配套的 custom OPP、ATK 和 Python runtime 后：

```bash
cd tests/atk/chunk_kda_fwd_finalize
atk node --backend npu --devices 0 -o ./atk_output/accuracy \
  node --backend cpu task \
  -c ./atk_chunk_kda_fwd_finalize.json \
  --task accuracy -p ./executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 200 -sp -to 60
```

单条超过 60 秒判定超时。只有最终报告确认总任务 200、执行失败
0、精度结论通过，才能声称全量通过；不能仅依据 shell 退出码。
任意 case 未通过时先使用 `--save_data output` 保留实值，再定位
shape/索引/尾块问题或执行精度复检，不能改输入 range 或阈值掩盖失败。

## 性能与确定性

大 shape 只跑 NPU 单算。先核对目标 case name，再在单条 ATK
`run` 外包 `msopprof`；性能只读取目标 kernel 的 duration，不使用
Python wall time。例如性能集第 2 条（从 0 开始）：

```bash
msopprof \
  --application="atk node --backend npu --devices 0 task -c ./atk_chunk_kda_fwd_finalize_perf.json --task run -p ./executor_chunk_kda_fwd_finalize.py -s 2 -e 3 -sp -to 60" \
  --output=./atk_output/profile_case_2 \
  --aic-metrics=BasicInfo --launch-count=1 --warm-up=0 --kill=off
```

确定性使用 `_mss.json` 的十二条输入，覆盖八种输出布局/状态组合和
AIV 搬运模板的 dense/packed、KV/VK 四条分支，逐位比较实际
可见结果；不能把准备好 `_mss.json` 说成内存检查已通过。当前内存
检查暂停，恢复后须先构建 sanitizer 对象，并确认运行时真正加载。

```bash
atk node --backend npu --devices 0 task \
  -c ./atk_chunk_kda_fwd_finalize_mss.json \
  --task accuracy_dc -p ./executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 12 -sp -to 60
```
