# ChunkScaledDotKkt

## 1. 功能概述

构造 GDN WY 表示的 chunk-wise 严格下三角 KKT 矩阵。它融合 K@K^T、gate 差指数和 beta 行缩放，输出与 key head 对齐的 A。

## 2. 数学定义

令 `r=t mod chunk_size`、`s=t-r+c`：

```text
A[b,h_k,t,c] = beta[b,h_k,t] * exp(clip(g[t]-g[s],-50,50))
        * dot(k[b,h_k,t,:], k[b,h_k,s,:])  if c < r
        0                                    otherwise
```

输出严格下三角，不含对角线；H_v>H_k 时当前 KKT 阶段读取 g/beta 的前 H_k 个 head。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `g` | 必选 | `[B,H_v,T]` | FP32 | BNS | 累积 gate |
| `beta` | 必选 | `[B,H_v,T]` | FP32 | BNS | 行缩放 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `A` | `[B,H_k,T,chunk_size]` | FP32 | 严格下三角 scaled KKT |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `64` | `{16, 32, 64, 128}` | chunk 长度 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | K 为 FP16/BF16；g/beta/A 为 FP32 |
| Format/Layout | head-first BNSD/BNS |
| 模式 | 定长/变长序列、GVA 输入、整块/尾块 |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_scaled_dot_kkt` |
| aclnn | `aclnnChunkScaledDotKktGetWorkspaceSize` / `aclnnChunkScaledDotKkt` |
| Ascend C `<<<>>>` | `chunk_scaled_dot_kkt<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_scaled_dot_kkt` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_scaled_dot_kkt/accuracy/test_chunk_scaled_dot_kkt.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_scaled_dot_kkt.json`；覆盖定长/变长序列、GVA 输入、整块/尾块。
- 参考实现：`torch_chunk_scaled_dot_kkt_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 在相同 shape/dtype/layout、warmup 和迭代配置下对比 `fla_npu.ops.triton.chunk_scaled_dot_kkt_fwd`；Ascend C 必须更快，仓内主 example 已切换到 Ascend C。

## 7. 已知限制

- chunk_size 仅支持 16/32/64/128。
- 必须满足 H_v % H_k == 0；A 的 head 维为 H_k。
- cu_seqlens/chunk_indices 必须同时提供或同时省略；变长序列物理 B 必须为 1。
- 变长序列累计长度必须覆盖 [0,T]，chunk_indices 必须按 sequence-major 完整列出每个 C 大小的 chunk。
- 指数差固定 clip 到 [-50,50]；H_v>H_k 时当前实现读取 g/beta 的前 H_k 个 head。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_scaled_dot_kkt python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_scaled_dot_kkt/accuracy/test_chunk_scaled_dot_kkt.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_scaled_dot_kkt/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；变长序列打包场景通常为 1 |
| `N` | 变长序列的逻辑序列数 |
| `T` | 定长序列长度或变长序列打包后的 token 总数 |
| `H_k` | Query/Key head 数 |
| `H_v` | Value/Output/State head 数 |
| `K` | Query/Key 单 head 特征维 |
| `chunk_size` | 每个 chunk 的 token 数 |
| `N_c` | 当前调用中的 chunk 总数 |
