# ChunkLocalCumsum 算子说明

## 算子功能

`ChunkLocalCumsum` 对输入张量 `g` 在时间维 `T` 上按固定 `chunk_size` 分块执行局部累加，输出形状与输入一致。

对每个 batch、token 和 head：

- 正向模式 `reverse=false`：
  `out[b, t, h] = sum(g[b, k, h])`，其中 `k` 从当前 chunk 起点累加到 `t`。
- 反向模式 `reverse=true`：
  `out[b, t, h] = sum(g[b, k, h])`，其中 `k` 从 `t` 累加到当前 chunk 终点前一个 token。
- 最终输出乘 `scale`；默认 `scale=1.0`，等价于 Triton 的 `scale=None`。

算子支持固定长度序列，也支持基于 `cu_seqlens` 和 `chunk_indices_out` 的 varlen 序列。

## 输入输出

| 名称 | 类型 | dtype | shape | 说明 |
| --- | --- | --- | --- | --- |
| `g` | input | `float32` | `[B, T, H]` | 输入张量 |
| `cu_seqlens` | optional input | `int64` | `[num_seq + 1]` | varlen 前缀和。固定长度模式传空 tensor |
| `chunk_indices_out` | optional input | `int64` | `[num_blocks, 2]` | Triton `chunk_indices_out[str(block_t)]` 解析后的 tensor，varlen block 到 `(seq_id, block_id)` 的映射。固定长度模式传空 tensor |
| `out` | output | `float32` | `[B, T, H]` | 输出张量，shape 与 `g` 一致 |

## 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | `int` | required | 局部 cumsum 的 chunk 长度，必须为 2 的幂 |
| `reverse` | `bool` | `false` | 是否执行反向局部累加 |
| `scale` | `float` | `1.0` | 输出缩放系数 |
| `head_first` | `bool` | `false` | 与 Triton 函数同名参数保持一致；当前 scalar 路径按 `[B, T, H]` 处理 |
| `output_dtype` | `string` | `"float32"` | 与 Triton 函数同名参数保持一致；当前仅支持 `float32` |

## 约束

- 当前实现仅支持 `float32` 输入输出。
- `g` 必须为 3 维 `[B, T, H]`，且各维为正数。
- `chunk_size` 必须为 2 的幂。
- varlen 模式下 `B` 必须为 1。
- varlen 模式下 `chunk_indices_out` 元素数必须为偶数，并按 `[seq_id, block_id]` 成对存储。
- `output_dtype` 仅支持 `float32`、`torch.float`、`torch.float32`。
- 固定长度模式下仍需向 ACLNN 接口传入空 `int64` tensor 作为 optional input，占位避免生成的 ACLNN glue 对 `nullptr` 报错。

## Tiling 逻辑

Host tiling 从 `g` 读取 `B/T/H`，并按 Triton 原算子公式计算：

```text
block_t = next_power_of_2((1 << 17) / (H * chunk_size))
num_blocks = ceil(T / block_t)
```

varlen 模式下：

```text
num_blocks = chunk_indices_out.element_count / 2
```

分核数：

```text
fixed:  block_dim = min(AIV 核数, ceil(B * T * H / 16))
varlen: block_dim = 1
```

fixed 模式按 16 个 `float` 一组的 64B GM cache line 粒度分核，保证同一条 cache line 只由一个 AIV core 写。varlen 模式当前采用保守单核调度，避免不对齐序列边界触发同类并发写冲突。

tiling 数据下发字段包括 `B/T/H/chunk_size/block_t/num_blocks/totalElements/isVarlen/reverse/headFirst/scale`。

## Kernel 实现逻辑

Kernel 只使用 AIV。

固定长度模式：

1. 按全局输出线性地址每 16 个 `float` 作为一个任务粒度分核。
2. 每个核处理一个或多个 64B cache line，避免多核同时写同一条 GM cache line。
3. 对每个输出 offset 反推 `(batch, local_t, h)`。
4. 按当前 token 所在 chunk 计算正向或反向累加。
5. 按需乘 `scale` 后写入 `out`。

varlen 模式：

1. 每个任务从 `chunk_indices_out` 读取 `(seq_id, local_block)`。
2. 通过 `cu_seqlens[seq_id]` 和 `cu_seqlens[seq_id + 1]` 得到当前序列的 `bos/eos/seq_len`。
3. 在该序列的本地坐标内按 `block_t` 遍历 token。
4. 对每个 head 执行 chunk 内正向或反向累加，并写回全局输出。

## 测试方法

Triton 标杆和 AscendC 对拍脚本位于 `examples/`：

```bash
cd /home/m00913889/codex04/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh
```

`run_precision_test.sh` 会生成 Triton golden、准备临时 custom OPP、编译 ACLNN 测试程序，并在 910B 上比较 AscendC 输出与 Triton 输出。
