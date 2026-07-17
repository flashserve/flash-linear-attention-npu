# ChunkLocalCumsum

## 1. 功能概述

在每个时间 chunk 内对 gate 或任意尾部特征做局部前缀/后缀累加。该 Ascend C 实现替换 Triton 预处理路径，并支持 fixed/varlen 与 scale。

## 2. 数学定义

将 token 后的尾部维展平为 P：

```text
out[b,h,t,p] = scale * sum(k=chunk_start..t) g[b,h,k,p]       reverse=false
out[b,h,t,p] = scale * sum(k=t..chunk_end-1) g[b,h,k,p]       reverse=true
```

varlen 的 chunk_start/chunk_end 在每条逻辑序列内计算，不跨 `cu_seqlens` 边界。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `g` | 必选 | `[B,H_v,T] 或 [B,H_v,T,...]` | FP32 | head-first | 待累加 gate/特征 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度，Device tensor |
| `chunk_indices_out` | 可选 | `[N_b,2] 或 [2*N_b]` | INT64 | ND | varlen 内部处理块映射，Device tensor |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | `与 g 相同` | FP32 | chunk-local 累加结果 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `无` | 2 的幂 |
| `reverse` | bool | `false` | 前缀/后缀 |
| `scale` | double | `1.0` | 输出缩放 |
| `head_first` | bool | `true` | 当前必须 true |
| `output_dtype` | str | `float32` | 仅 float32/torch.float/torch.float32 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 输入输出仅 FP32；索引 INT64 |
| Format/Layout | head-first `[B,H_v,T,...]` |
| 模式 | fixed/varlen、forward/reverse、任意连续尾部 P |

varlen 模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices_out` 按 sequence-major 保存内部处理块 `(seq_id, local_block_id)`；处理块长度 `B_T` 由 tiling 根据 `chunk_size` 和尾部维 `P` 计算，不等同于数学 chunk 长度 `C`。fixed 与 varlen、尾块与整块遵循同一数学定义。

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
- 用例规格：`tests/op_cases/chunk_local_cumsum.json`；覆盖 fixed/varlen、forward/reverse、任意连续尾部 P。
- 参考实现：`torch_chunk_local_cumsum_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 在相同 shape/dtype/layout、warmup 和迭代配置下对比 `fla_npu.ops.triton.chunk_local_cumsum`；Ascend C 必须更快，仓内主 example 已切换到 Ascend C。

## 7. 已知限制

- g rank 至少 3，所有维度为正；head_first 当前必须 true。
- chunk_size 必须为 2 的幂，且结合 P 后能满足 UB tile；交付矩阵覆盖 16/32/64/128。
- output_dtype 仅支持 FP32 别名。
- varlen 物理 B=1，两个 Device 索引必须同时提供；cu_seqlens 首项为 0、末项为 T 且非递减。
- chunk_indices_out 必须按 sequence-major 完整列出内部处理块；处理块长度 B_T 由 UB、C 和 P 共同决定，不能直接按 C 构造。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_local_cumsum python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_local_cumsum/accuracy/test_chunk_local_cumsum.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_local_cumsum/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；varlen 打包场景通常为 1 |
| `N` | varlen 逻辑序列数 |
| `T` | dense 序列长度或 varlen 打包后的 token 总数 |
| `H_v` | Value/Output/State head 数 |
| `C` | chunk_size |
| `P` | token 后连续尾部元素乘积 |
| `N_b` | varlen 内部处理块总数；由各序列 ceil(seq_len/B_T) 求和 |
| `N_b` | varlen 内部处理块总数；由各序列 ceil(seq_len/B_T) 求和 |
| `B_T` | 单个 varlen 处理块覆盖的 token 数，由 tiling 根据 UB、C、P 计算 |
