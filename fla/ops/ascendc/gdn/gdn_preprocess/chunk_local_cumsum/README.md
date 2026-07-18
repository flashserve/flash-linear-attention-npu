# ChunkLocalCumsum

## 1. 功能概述

在每个时间 chunk 内对 gate 或任意尾部特征做局部前缀/后缀累加。该 Ascend C 实现替换 Triton 预处理路径，并支持定长/变长序列与 scale。

## 2. 数学定义

将 token 后的尾部维展平为 P：

```text
out[b,h,t,p] = scale * sum(k=chunk_start..t) g[b,h,k,p]       reverse=false
out[b,h,t,p] = scale * sum(k=t..chunk_end-1) g[b,h,k,p]       reverse=true
```

变长序列的 chunk_start/chunk_end 在每条逻辑序列内计算，不跨 `cu_seqlens` 边界。

## 3. 输入、输出和属性

本文使用的 Shape 符号统一引用[GDN 模型符号表](../../README.md#model-shape-symbols)，不在算子 README 中重复定义。

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `g` | 必选 | `[B,H_v,T] 或 [B,H_v,T,...]` | FP32 | head-first | 待累加 gate/特征 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度；aclnn 使用 host `aclIntArray`，Python 使用整数序列 |
| `chunk_indices_out` | 可选 | `[N_b,2] 或 [2*N_b]` | INT64 | ND | 变长序列块映射；aclnn 使用展平的 host `aclIntArray` |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | `与 g 相同` | FP32 | chunk-local 累加结果 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | 正整数且为 2 的幂 | 还须满足 `B_T >= chunk_size` |
| `reverse` | bool | `false` | `{false, true}` | false=前缀，true=后缀 |
| `scale` | double | `1.0` | - | 输出缩放 |
| `head_first` | bool | `true` | `{true}` | 当前必须 true |
| `output_dtype` | str | `float32` | `{"float32", "torch.float", "torch.float32"}` | 三个别名均表示 FP32 输出 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 输入输出仅 FP32；索引 INT64 |
| Format/Layout | head-first `[B,H_v,T,...]` |
| 模式 | 定长/变长序列、forward/reverse、任意连续尾部 P |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices_out` 按 sequence-major 保存内部处理块 `(seq_id, local_block_id)`；处理块长度 `B_T` 由 tiling 根据 `chunk_size` 和尾部维 `P` 计算，不等同于算子属性 `chunk_size`。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_local_cumsum` |
| aclnn | `aclnnChunkLocalCumsumGetWorkspaceSize` / `aclnnChunkLocalCumsum` |
| Ascend C `<<<>>>` | `chunk_local_cumsum<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_local_cumsum` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_local_cumsum/accuracy/test_chunk_local_cumsum.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_local_cumsum.json`；覆盖定长/变长序列、forward/reverse、任意连续尾部 P。
- 参考实现：`torch_chunk_local_cumsum_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 在相同 shape/dtype/layout、warmup 和迭代配置下对比 `fla_npu.ops.triton.chunk_local_cumsum`；Ascend C 必须更快，仓内主 example 已切换到 Ascend C。

## 7. 已知限制

- g rank 至少 3，所有维度为正；head_first 当前必须 true。
- chunk_size 必须为 2 的幂。令 `P` 为时间维后的连续尾部维乘积，host 计算
  `B_T = next_pow2(max(floor(2^17 / (P * chunk_size)), 1))`，必须满足 `B_T >= chunk_size`；
  交付矩阵覆盖 `P=1,chunk_size=64` 的变长序列尾块和 `P=16,chunk_size=64` 的 dense 尾块。
- output_dtype 仅支持 FP32 别名。
- 变长序列物理 B=1，两个 host 整数数组必须同时提供；cu_seqlens 首项为 0、末项为 T 且非递减。
- chunk_indices_out 必须按 sequence-major 完整列出内部处理块；每条序列需要
  `ceil(sequence_length / B_T)` 对索引，处理块长度 `B_T` 由 UB、chunk_size 和 P 共同决定，不能直接按 chunk_size 构造。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_local_cumsum python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_local_cumsum/accuracy/test_chunk_local_cumsum.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_local_cumsum/routes/`，均使用同一份 JSON 规格。
