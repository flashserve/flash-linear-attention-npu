# GDR 适配层 AscendC `solve_tri` A2 优化验证报告

## 1. 结论

PR #249 已将 `examples/chunk_gated_delta_rule_function.py` 的
`solve_tri` 恢复为 `fla_npu.ops.ascendc`，并对 A2 的 64×64 FP32
kernel 做稀疏块合并优化。对真实 GDR KKT 输入
`T=32768, BT=64, BF16`，`msopprof BasicInfo` 结果为：

| heads | 原 AscendC | 优化后 AscendC | 时延降低 | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5581.43 us | 4442.20 us | 20.41% | 1.256x |
| 16 | 11285.32 us | 8871.40 us | 21.39% | 1.272x |

优化后的 AscendC 核心也比对照的 fla-org Triton-Ascend 实现快约
1.205x。每个参与 AIC 的 FP32 workspace 从 192 KiB 降到 64 KiB，
减少 66.7%。

最新 v26.6.0 源码本地编包后，单算子 15/15、原问题 shape 单层正反向、
100 层 × 20 step 纯异步压力、四卡 TP/SP/CP/GVA、BF16/FP16 和二进制
确定性检查均通过。没有使用 release wheel。

## 2. 优化方法

原 64×64 路径在一个 tile 内执行 10 次完整 `64×64×64` FP32 GEMM：
6 次用于四个 16×16 MCH 基块递推，4 次用于 16→32、32→64 两级 MXR
合并。虽然辅助矩阵的大部分区域为零，原实现仍计算完整 GEMM。

本次保持 MCH 的连续 64×64 GEMM，不拆成多个小 GEMM。A2 实测表明，
四次 16×16 GEMM 的调用、拼块和搬运开销大于省下的计算量。只对两级
MXR 合并按实际非零块计算：

1. 16→32：将两组独立块打包，每阶段执行一次
   `32×16 @ 16×32` FP32 GEMM。
2. 32→64：每阶段只执行一次 `32×32 @ 32×32` FP32 GEMM。
3. AIV 仅将有效的下三角合并结果写回运行中的逆矩阵。
4. workspace 改为四个连续 64×64 FP32 槽：运行逆矩阵、幂矩阵、
   GEMM 临时结果和 `-A`。

两级合并的 GEMM MAC 数从 `4 × 64³` 降为
`2 × (32×32×16) + 2 × 32³`，减少 90.625%；整个 64×64 solve 的
GEMM MAC 数减少 36.25%。

所有中间计算保持原生 FP32，显式 `SetHF32Mode(false)`，没有引入 HF32
乘法。AIC/AIV 的 ready/free 双向握手和每 tile workspace 复用边界保持
不变，没有新增 host 同步或 device-to-host 回读。

## 3. 适配层

独立适配脚本恢复为：

```python
from fla_npu.ops.ascendc import npu_solve_tri
```

KKT 仍输出 FP32 `A`，适配层按模型 dtype 转为 BF16/FP16 后调用
AscendC；kernel 内部再提升到 FP32，仅最终结果转换回输入 dtype。

布局分支为：

- packed varlen：去掉 batch 维后使用 TND，并传入已经准备好的 host
  `cu_seqlens` 和 `chunk_indices`。
- dense：保留 `[B,S,N,D]`，使用 BSND。

层内和层间没有 `torch.npu.synchronize()`。压力脚本只在全部 step 提交
后统一回读结果；`cu_seqlens` 的 host list 沿用每个 step 开始前已经
准备的 metadata，不在每一层重复 device-to-host。

`causal_conv1d` 和 `causal_conv1d_bwd` 仍由独立 autograd wrapper 调用，
正向输出使用 NTD。

## 4. 环境与口径

- 硬件：A2
- Python：3.10.20
- PyTorch：2.10.0
- torch-npu：2.10.0.post2
- triton-ascend：3.2.1
- fla-npu：PR #249 当前 v26.6.0 源码构建 wheel
- 主 dtype：BF16；补充 FP16
- 原问题：`T=32768, H=16, K=V=128, BT=64` 和原始 64 段
  `cu_seqlens`

没有设置 launch blocking、关闭 task queue 或其他强串行环境变量。
性能只采用 `msopprof BasicInfo` 的 SolveTri Task Duration，不用 Python
wall time。

## 5. 精度、数值与性能

### 5.1 单算子

`fla/ops/ascendc/gdn/recurrent_gdn/solve_tri/test/test.py` 共 15 个用例，
全部通过，覆盖：

- 16、32、64、128 chunk；
- FP16/BF16；
- MCH 大数值中间结果；
- BSND/BHTD/TND；
- 非 16 对齐尾块；
- TND 上三角和短尾无效区清零。

### 5.2 真实 KKT 输入精度

输入由 BF16 `k/g/beta` 经真实 `chunk_local_cumsum` 和
`chunk_scaled_dot_kkt_fwd(output_dtype=torch.float32)` 生成。
AscendC 适配按实际模型语义将 `A` 转为 BF16，输出也是 BF16。与
fla-org Triton-Ascend 参考结果比较：

| shape | 两端 finite | 最大绝对误差 | 相对 L2 |
| --- | --- | ---: | ---: |
| T=32768, H=8, BT=64 | 是 | 0.0009765625 | 4.6058543e-5 |
| T=32768, H=16, BT=64 | 是 | 0.0009765625 | 4.6225377e-5 |

### 5.3 `msopprof` 性能

下表只统计 SolveTri 核心，不包含 KKT、FP32→BF16 cast 和适配层无效区
清零。H=8 捕获 21 次、H=16 捕获 13 次，取全部 Task Duration 平均值。

| shape | fla-org Triton-Ascend | 原 AscendC | 优化后 AscendC |
| --- | ---: | ---: | ---: |
| T=32768, H=8, BT=64 | 5350.82 us | 5581.43 us | 4442.20 us |
| T=32768, H=16, BT=64 | 10692.24 us | 11285.32 us | 8871.40 us |

相对 fla-org，优化后 H=8/H=16 的时延分别降低 16.98%/17.03%。

作为优化选择依据，A2 上还验证了以下未采用方案：

| 方案，H=8 | SolveTri |
| --- | ---: |
| 仅 workspace stride 128→64 | 5513.63 us |
| MCH 拆为四次 16×16 GEMM | 6448.24 us |
| GM 拼块后执行 MCH | 7484.97 us |
| L1 直接拼块后执行 MCH | 6317.57 us |

这些方案说明 A2 上不应为 MCH 引入多个小 GEMM；稀疏优化应集中在两级
MXR 合并。

## 6. 原始场景和确定性

### 6.1 单层

- 1 层、2 step、关闭 checkpoint、BF16、原始 `cu_seqlens`
- `T=32768, H=16, K=V=128, BT=64`
- 两个 step 的 forward、backward、grad norm 全部 finite
- 两个 step 的 grad norm 均为 `4.1270857`
- output、input grad、grad norm 和全部参数梯度二进制一致
- 峰值显存：4429.14 MiB

### 6.2 100 层 × 20 step

- activation checkpoint、causal-conv 正反向、NTD 输出、GDR 正反向
- step 主体中无冗余 `torch.npu.synchronize()`
- 20/20 step 的 forward、backward 和 grad norm 全部 finite
- 20 个 step 的 grad norm 均为 `40.915497`
- output、input grad、grad norm 和全部参数梯度逐 step 二进制一致
- 峰值显存：21.219 GiB
- 结果：PASS

## 7. 四卡泛化

每个 case 对相同输入运行两次。“确定性”使用 `torch.equal` 比较输出、
final state（若有）和全部训练梯度。表中 shape 为单 rank 的
`T / key_heads / value_heads / key_dim / value_dim`。

| 场景 | 单 rank shape | dtype | finite | 确定性 | 峰值显存 |
| --- | --- | --- | --- | --- | ---: |
| Qwen3.5-4B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 1694.09 MiB |
| Qwen3.5-4B SP4 训练 | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 1692.53 MiB |
| Qwen3.5-35B-A3B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 1694.09 MiB |
| Qwen3.5-35B-A3B CP4 训练 | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 1692.55 MiB |
| Qwen3-Next TP4 prefill | 32768 / 4 / 8 / 128 / 128 | BF16 | 通过 | 通过 | 971.07 MiB |
| Qwen3-Next CP4 prefill | 8192 / 16 / 32 / 128 / 128 | BF16 | 通过 | 通过 | 969.53 MiB |
| GVA V=256 TP4 训练 | 16384 / 4 / 8 / 128 / 256 | BF16 | 通过 | 通过 | 1334.06 MiB |
| Qwen3.5-4B TP4 训练 | 32768 / 4 / 8 / 128 / 128 | FP16 | 通过 | 通过 | 1694.09 MiB |

本轮范围按模型不用 `initial_state` 的前提执行。模型权重、optimizer
state、MoE 和 dense attention 不在适配层显存统计内。

## 8. 执行方式

从最新 v26.6.0 源码构建和安装 wheel：

```bash
FLA_NPU_SOC=ascend910b \
python -m pip wheel --no-build-isolation --no-deps . -w dist

python -m pip install --force-reinstall --no-deps \
  dist/flash_linear_attention_npu-*.whl
```

单算子：

```bash
python fla/ops/ascendc/gdn/recurrent_gdn/solve_tri/test/test.py
```

单层和 100 层压力：

```bash
python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 1 --steps 2 --no-checkpoint --replay-step-inputs --md5

python examples/flash_gated_delta_rule_100layer_adapter_stress.py \
  --layers 100 --steps 20 --replay-step-inputs
```

四卡模型 case：

```bash
python -m torch.distributed.run --nproc_per_node 4 \
  examples/chunk_gated_delta_rule_model_matrix.py \
  --case qwen35_35b_tp4_train \
  --determinism-runs 2
```

精度对照和性能 workload：

```bash
export FLA_ORG_ROOT=/path/to/flash-linear-attention

python examples/solve_tri_backend_benchmark.py \
  --backend compare \
  --tokens 32768 \
  --heads 8 \
  --chunk-size 64 \
  --dtype bf16 \
  --a-dtype fp32 \
  --input-source kkt \
  --repeats 1
```

硬件闭环范围为 A2。A3/A5 未做本轮硬件实测，不将 A2 结果外推为其他
平台结论。本轮也没有构建 sanitizer 专用对象，因此内存部分只报告
单算子边界/无效区测试和端到端峰值显存，不作为 sanitizer 结论。
