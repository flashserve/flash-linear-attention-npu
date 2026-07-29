# GDR 适配层 fla-org Triton-Ascend `solve_tri` 泛化验证报告

## 1. 结论

PR #249 的 `examples/chunk_gated_delta_rule_function.py` 仅将
`solve_tri` 回切到
[fla-org/flash-linear-attention `triton_ascend` backend](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/utils/backends/triton_ascend/solve_tril.py)
后，在 A2 上完成源码编包、原问题 shape 单层验证、100 层 20 step
压测、4 卡模型分片矩阵、内存和确定性验证。测试使用 fla-org
`main@9c8e42e762fce087c27b673af4922795d9edb85e`：

- 所有纳入结论的 forward、`dq`、`dk`、`dv`、`dg`、`dbeta` 均为
  finite，没有出现 NaN/Inf。
- 相同输入重复执行时，输出和全部被检查梯度均二进制一致。
- 100 层 × 20 step 的纯异步主体通过，20 个 step 的输出、输入梯度、
  grad norm 和全部参数梯度逐 step 一致。
- 对原始 `T=32768, H=16, BT=64` 变长用例，fla-org 与原 AscendC
  路径的输出相对 L2 误差为 `4.62254e-5`，最大绝对误差为
  `9.765625e-4`。
- 补测 `T=32768, H=8, BT=64, BF16` 后，fla-org 核心耗时约
  `5.351 ms`，原 AscendC 核心约 `5.581 ms`，只提升 `1.04x`；计入
  两边适配辅助算子后约为 `5.364 ms` 与 `5.972 ms`，只提升
  `1.11x`。
- 将 solve 输入 `A` 也改成 BF16 时，fla-org Triton-Ascend 在缩小到
  `T=1024, H=8, BT=64` 的零输入用例上仍无法完成；相同 BF16 输入的
  AscendC 路径可以正常运行。该环境下不能使用 fla-org 的 BF16 `A`
  specialization。

因此，fla-org Triton-Ascend `solve_tri` 的正确性和确定性满足本报告
覆盖的 adapter 调用链，但 H=8/H=16 主场景的性能收益均不可接受，
不能作为 `solve_tri` 性能问题的最终解决方案。当前切换仅适合继续做
后端对照和优化定位。

## 2. 代码变更

功能变更只发生在独立适配脚本：

1. 删除 `fla_npu.ops.ascendc.solve_tri` 导入。
2. 从 `FLA_ORG_ROOT` 显式加载
   `fla.ops.utils.backends.triton_ascend.solve_tril.solve_tril_npu`，不再
   经过本仓 `fla_npu.ops.triton` 的 solve 重导出。
3. 校验加载模块必须位于 `FLA_ORG_ROOT`，并在首个本仓 Triton kernel
   launch 前完成 namespace 切换。缺少 fla-org checkout 时直接报错，
   不静默 fallback。
4. KKT 的 FP32 输出 `A` 保持 FP32 输入 fla-org solve，只在 64 × 64
   合并输出阶段按 `output_dtype` 转回 BF16/FP16。
5. 变长场景直接传 device tensor `cu_seqlens` 和当前 chunk size 对应的
   `chunk_indices`；solve 路径不再把 metadata 拷回 host。
6. GDR 其余正反向算子和 causal-conv 调用保持不变。

## 3. 环境与测试口径

- 硬件：A2
- Python：3.10.20
- PyTorch：2.10.0
- torch-npu：2.10.0.post2
- triton-ascend：3.2.1
- fla-npu：PR #249 当前源码构建的 v26.6.0 wheel，不使用 release wheel
- fla-org：`main@9c8e42e762fce087c27b673af4922795d9edb85e`
- 主精度 dtype：BF16；补充训练 dtype：FP16
- chunk size：64
- 原问题用例：`T=32768, H=16, K=V=128`，使用
  `flash_gated_delta_rule_100layer_stress.py` 内置的 64 段原始
  `cu_seqlens`

测试命令未设置 launch blocking 或关闭 task queue 一类的强串行环境
变量。100 层压力脚本在层内、层间和 step 主体中均不调用
`torch.npu.synchronize()`；所有 step 提交完成后才统一做 finite 和
确定性结果回读。模型矩阵脚本仅在每个 case 结束时同步，用于读取
finite、显存和确定性统计。

压力日志和每个矩阵 JSON 均确认
`solve_module=fla.ops.utils.backends.triton_ascend.solve_tril`。

## 4. 模型 shape 来源与映射

公开配置中：

- [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/main/config.json)
  的 linear attention 使用 16 个 key heads、32 个 value heads，
  key/value head dim 均为 128。公开型号为 4B，因此本报告将用户所说的
  “3.5B”映射为最接近的公开 3.5B-class/Qwen3.5-4B。
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json)
  使用相同的 GDR head/dim 配置。
- [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json)
  同样使用 16/32 个 key/value heads 和 128/128 head dim。
- Qwen3-Next 的 GDN 结构和 state 处理可参考
  [Transformers 官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py)。

TP case 按 head 切分；SP/CP case 按 token 切分。4B 与 35B 的单层
GDR 本地 shape 相同，所以两者的单层显存相近；模型层数、MoE、dense
attention、模型权重和 optimizer state 不在本适配层报告的显存统计内。

## 5. 四卡泛化矩阵

表中 shape 为单 rank 的
`T / key_heads / value_heads / key_dim / value_dim`。显存是两次重复
执行期间所有 rank 的 `torch.npu.max_memory_allocated()` 最大值。
“确定性”表示相同输入的 output、final state（如有）和全部训练梯度
使用 `torch.equal` 比较均一致。

| 场景 | 本地 shape | 模式 | dtype | finite | 确定性 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | ---: |
| Qwen3.5-4B TP4 | 32768 / 4 / 8 / 128 / 128 | 训练、packed | BF16 | 通过 | 通过 | 1692.09 MiB |
| Qwen3.5-4B SP4 | 8192 / 16 / 32 / 128 / 128 | 训练、dense local shard | BF16 | 通过 | 通过 | 1692.03 MiB |
| Qwen3.5-35B-A3B TP4 | 32768 / 4 / 8 / 128 / 128 | 训练、packed | BF16 | 通过 | 通过 | 1692.09 MiB |
| Qwen3.5-35B-A3B CP4 | 8192 / 16 / 32 / 128 / 128 | 训练、按完整序列切分 | BF16 | 通过 | 通过 | 1692.05 MiB |
| Qwen3-Next TP4 | 32768 / 4 / 8 / 128 / 128 | 推理 prefill | BF16 | 通过 | 通过 | 969.07 MiB |
| Qwen3-Next SP4 | 8192 / 16 / 32 / 128 / 128 | 推理 continuation + final state | BF16 | 通过 | 通过 | 975.01 MiB |
| Qwen3-Next CP4 | 8192 / 16 / 32 / 128 / 128 | 推理、按完整序列切分 | BF16 | 通过 | 通过 | 969.03 MiB |
| GVA TP4 | 16384 / 4 / 8 / 128 / 256 | 训练、packed | BF16 | 通过 | 通过 | 1332.06 MiB |
| Qwen3.5-4B TP4 | 32768 / 4 / 8 / 128 / 128 | 训练、packed、loss scale=1024 | FP16 | 通过 | 通过 | 1692.09 MiB |

BF16 训练 case 的 grad norm 分别为 `2.26e-6` 或 `2.27e-6`。FP16
直接对全输出均值求导会下溢，因此补充 case 使用固定 1024 loss scale，
得到非零 grad norm `0.00232835`；该 case 的五个输入梯度仍全部 finite
且二进制一致。

## 6. 原始场景与压力测试

### 6.1 单层原始 shape

- 配置：1 层、2 step、关闭 checkpoint、BF16、原始
  `cu_seqlens`、`T=32768`、`H=16`、`K=V=128`
- forward、backward、grad norm：通过
- 两个 step 的 grad norm：均为 `4.1270933`
- 输出 MD5：均为 `ff1159fa57f230482407a2d3a08fe603`
- 输入梯度 MD5：均为 `51a90cc2d8806c3177c2bcc0511ef5cc`
- 峰值显存：`4427.14 MiB`

### 6.2 100 层 × 20 step

- 配置：100 层、20 step、activation checkpoint、causal-conv 正反向、
  causal-conv NTD 输出、GDR 正反向、BF16、原始 `cu_seqlens`
- forward/backward：20/20 step 全部 finite
- grad norm：20 个 step 均为 `40.915287`
- 相同输入重放：output、input grad、grad norm、全部参数梯度逐 step
  二进制一致
- 峰值显存：`21.217 GiB`
- 结果：PASS

## 7. `solve_tri` 精度与性能

### 7.1 精度

输入 `A` 由 BF16 的 `k/g/beta` 经过真实 `chunk_local_cumsum` +
`chunk_scaled_dot_kkt_fwd(output_dtype=torch.float32)` 生成，不使用独立
随机三角矩阵。按照实际 adapter 语义，传入 solve 的 `A` 是 FP32，
solve 输出为 BF16。比较结果如下：

| shape | fla-org finite | AscendC finite | 最大绝对误差 | 相对 L2 误差 |
| --- | --- | --- | ---: | ---: |
| T=32768, H=8, BT=64 | 是 | 是 | 0.0009765625 | 4.6058543e-5 |
| T=32768, H=16, BT=64 | 是 | 是 | 0.0009765625 | 4.6225377e-5 |

### 7.2 `msopprof` 性能

在相同输入上分别循环调用 20 次。以下时间取 `msopprof BasicInfo`
中对应 kernel 的平均 Task Duration，不使用 Python wall time。

| shape | 路径 | 组成 | 单次估算 |
| --- | --- | --- | ---: |
| H=8 | fla-org 核心 | `merge_16x16_to_64x64_inverse_kernel_npu` | 5350.82 us |
| H=8 | fla-org 适配路径 | 核心 + zeros-like 13.44 us | 5364.25 us |
| H=8 | AscendC 核心 | SolveTri | 5581.43 us |
| H=8 | 原 AscendC 适配层 | SolveTri + BF16 cast 49.61 us + logical-not 9.36 us + masked-fill 331.39 us | 5971.80 us |
| H=16 | fla-org 核心 | `merge_16x16_to_64x64_inverse_kernel_npu` | 10692.24 us |
| H=16 | fla-org 适配路径 | 核心 + zeros-like 23.01 us | 10715.25 us |
| H=16 | AscendC 核心 | SolveTri | 11285.32 us |
| H=16 | 原 AscendC 适配层 | SolveTri + BF16 cast 106.73 us + logical-not 9.25 us + masked-fill 716.30 us | 12117.61 us |

每条路径执行一次 warmup 和 20 次计时循环，因此 solve、输出初始化和
mask 均捕获 21 次。AscendC 日志中的同类型 cast 还包含一次输入构造
阶段的 cast；表中已排除该次输入构造，只统计 21 次 adapter cast。
上表按每种 kernel 的独立平均值组合，不代表完整 GDR 层耗时。

H=8 的变长 metadata 包含 544 个 chunk，Triton grid 为
`544 × 8 = 4352` 个 programs，与 profiler 记录的 Block Dim 一致；
没有发现重复展开 chunk/head 的调用错误。64 × 64 kernel 每个 program
内部执行四个 16 × 16 递推块和多次 FP32 IEEE dot，是当前耗时的主要
来源。

### 7.3 实验性 BF16 `A`

真实 adapter 保持 KKT 输出 `A=FP32`。为判断减少一半输入带宽是否能
改善 solve 性能，benchmark 额外支持让 KKT 直接写出 BF16 `A`，以及
使用 BF16 零输入隔离 solve：

- FP32 KKT 输出后执行 NPU `A.to(torch.bfloat16)` 时，进程停在
  `rtStreamSynchronize`，没有进入 solve。
- 改成 KKT 直接写 BF16 后，组合调用仍无法完成。
- 进一步绕过 KKT，在 `T=1024, H=8, BT=64` 上直接构造 BF16 零输入，
  单独调用 fla-org solve 仍无法完成，说明问题位于 fla-org BF16 `A`
  specialization，而不是大 shape 或 KKT 数值。
- 相同缩小输入的 AscendC solve 正常完成。

对可运行的 AscendC 路径补测完整 `T=32768, H=8, BT=64` BF16 零输入，
20 次 `msopprof BasicInfo` 平均结果为：

| 组成 | 单次耗时 |
| --- | ---: |
| AscendC SolveTri | 5552.48 us |
| logical-not | 8.25 us |
| masked-fill | 331.03 us |
| 合计 | 5891.76 us |

该路径不再包含 FP32→BF16 cast，但相比真实 FP32 `A` 适配路径的
`5971.80 us` 只减少约 `80 us`，改善约 `1.4%`。因此即使忽略
fla-org BF16 specialization 无法完成的问题，单纯把 `A` 改成 BF16
也不能解决当前性能瓶颈。

## 8. 复现方式

从 PR 当前源码构建 wheel：

```bash
FLA_NPU_SOC=ascend910b \
python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps \
  dist/flash_linear_attention_npu-*.whl
```

指定经过验证的 fla-org checkout：

```bash
export FLA_ORG_ROOT=/path/to/flash-linear-attention
git -C "$FLA_ORG_ROOT" rev-parse HEAD
# 本报告使用 9c8e42e762fce087c27b673af4922795d9edb85e
```

单层和 100 层压力测试。`--cu-seqlens` 为空时使用脚本内置的原始
32768-token offsets：

```bash
python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 1 \
  --steps 2 \
  --no-checkpoint \
  --replay-step-inputs \
  --md5

python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 100 \
  --steps 20 \
  --replay-step-inputs
```

四卡模型 shard case：

```bash
python -m torch.distributed.run \
  --nproc_per_node 4 \
  examples/chunk_gated_delta_rule_model_matrix.py \
  --case qwen35_35b_tp4_train \
  --determinism-runs 2

python examples/chunk_gated_delta_rule_model_matrix.py --list
```

`solve_tri` 精度比较与 profiler workload：

```bash
python examples/solve_tri_backend_benchmark.py \
  --backend compare \
  --tokens 32768 \
  --heads 8 \
  --chunk-size 64 \
  --dtype bf16

python examples/solve_tri_backend_benchmark.py \
  --backend fla-org \
  --tokens 32768 \
  --heads 8 \
  --chunk-size 64 \
  --dtype bf16 \
  --repeats 20

# 实验性 BF16 A；默认仍为真实 adapter 使用的 --a-dtype fp32
python examples/solve_tri_backend_benchmark.py \
  --backend ascendc \
  --tokens 32768 \
  --heads 8 \
  --chunk-size 64 \
  --dtype bf16 \
  --a-dtype bf16 \
  --input-source zeros \
  --repeats 20
```

## 9. 边界与未覆盖项

- 四卡测试验证的是各 rank 本地 GDR shard 的并发提交、数值、显存和
  确定性，不等价于完整模型 TP/SP/CP collective 正确性或端到端吞吐。
- packed CP 按完整序列切分，不需要跨 rank 传递 recurrent state。
  SP continuation case 验证了本地 initial/final state 的推理路径，但没有
  把前一 rank 的 final state 作为后一 rank 的 initial state 串成完整
  distributed pipeline。
- 当前 v26.6.0 `bwd_dhu` 不返回 `dh0`。因此跨 rank 连续序列的 SP/CP
  训练 state-gradient 语义仍不受此适配层支持；这与本次
  `solve_tri` 后端切换无关。
- 显存数字是 adapter workload 的 allocated peak，不包含完整模型权重、
  KV/cache 之外的模型状态、optimizer state 和通信 buffer。
- fla-org checkout 是运行时依赖，不包含在 fla-npu wheel 中；执行脚本
  前必须设置 `FLA_ORG_ROOT`。加载器不会 fallback 到本仓同名实现。
- 本报告只验证 A2；A3/A5 需要分别编包后补充平台验证。
