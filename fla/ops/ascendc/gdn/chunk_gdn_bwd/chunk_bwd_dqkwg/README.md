# ChunkBwdDqkwg

## 1. 功能概述

Gated Delta Rule 分块反向链路的主梯度算子。它消费前向激活、chunk 状态和上游梯度，计算 `dQ`、`dK`、`dW` 与 `dG`，并支持 `H_v/H_k` 分组映射。

## 2. 数学定义

对 value head `h_v`，先映射 `h_k=floor(h_v/(H_v/H_k))`。算子按链式法则把
`dO`、`dH` 和 `dV` 对 chunk 内 score、状态项及门控项的贡献合并：

```text
dQ[h_k] += dS[h_v] @ K[h_k]
dK[h_k] += dS[h_v]^T @ Q[h_k] + dW[h_v] * beta/gate terms
dW[h_v]  = state/output branches reduced on V
dG[h_v]  = reverse cumulative reduction of gate-dependent terms
```

`dQ/dK` 在同一 key head 对应的多个 value head 上归约；尾块仅对有效 token 求值。
`g` 是沿 `T` 的 chunk-local 累积 gate，要求调用者提供与前向完全一致的值。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key，与 q 同形 |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local 累积 gate |
| `h` | 必选 | `[B,H_v,N_c,K,V]` | FP16/BF16 | ND | 前向保存的 chunk 状态 |
| `dox` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 输出梯度 |
| `dh` | 必选 | `[B,H_v,N_c,K,V]` | FP16/BF16 | ND | chunk 状态梯度 |
| `dv` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value 分支梯度 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平的 (seq_id,chunk_id) |
| `w` | 预留 | `-` | 与 q 一致 | ND | 当前必须为 None |
| `g_gamma` | 预留 | `-` | 与 g 一致 | ND | 当前必须为 None |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dq` | `[B,H_k,T,K]` | 与 q 一致 | Query 梯度；多 value head 贡献已归约 |
| `dk` | `[B,H_k,T,K]` | 与 k 一致 | Key 梯度；多 value head 贡献已归约 |
| `dw` | `[B,H_v,T,K]` | 与 q 一致 | WY 中间量 W 的梯度 |
| `dg` | `[B,H_v,T]` | 与 g 一致 | 累积 gate 的梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `scale` | float | `None` | - | Python 的 None 按 1.0 传入；注意力缩放需显式传 1/sqrt(K) |
| `chunk_size` | int | `64` | `{64, 128}` | chunk 长度 |
| `use_exp2` | bool | `false` | `{false}` | 预留，当前仅支持 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 预留，当前仅支持 false |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 主张量 FP16/BF16；gate 可为 FP32；输出跟随对应输入 |
| Format/Layout | BNSD；状态张量为 `[B,H_v,N_c,K,V]` |
| 模式 | fixed 与 varlen；varlen 的两个索引必须同时提供 |

varlen 模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。fixed 与 varlen、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_bwd_dqkwg` |
| aclnn | `aclnnChunkBwdDqkwgGetWorkspaceSize` / `aclnnChunkBwdDqkwg` |
| Ascend C `<<<>>>` | `chunk_bwd_dqkwg<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_bwd_dqkwg` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_bwd_dqkwg/accuracy/test_chunk_bwd_dqkwg.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_bwd_dqkwg.json`；覆盖 fixed 与 varlen；varlen 的两个索引必须同时提供。
- 参考实现：`仓内 PyTorch/CPU reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，尾块按有效长度处理。
- 必须满足 `H_v % H_k == 0`；varlen 当前仅支持物理 `B=1`。
- `w`、`g_gamma` 当前为预留输入，必须传 `None`。
- `use_exp2` 与 `transpose_state_layout` 当前必须为 `false`。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_bwd_dqkwg python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_bwd_dqkwg/accuracy/test_chunk_bwd_dqkwg.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_bwd_dqkwg/routes/`，均使用同一份 JSON 规格。

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
| `H_k` | Query/Key head 数 |
| `H_v` | Value/Output/State head 数 |
| `K` | Query/Key 单 head 特征维 |
| `V` | Value/Output 单 head 特征维 |
| `N_c` | 当前调用中的 chunk 总数 |
