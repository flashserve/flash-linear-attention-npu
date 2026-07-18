# ChunkBwdDvLocal

## 1. 功能概述

Gated Delta Rule 反向过程的 Value 本地梯度算子。它在每个 chunk 内根据 `Q/K` score、门控差和 `dO` 生成 `dV_local`，不承担跨 chunk 状态梯度。

## 2. 数学定义

对 `h_v` 映射 `h_k=floor(h_v/(H_v/H_k))`，在每个 chunk 内：

```text
Ws       = K[h_k] @ Q[h_k]^T * scale
Ws_gated = triu(Ws * exp(g_col - g_row), diagonal=0)
dV_local = Ws_gated @ dO[h_v]
```

上三角包含对角线；partial chunk 的无效行列在乘法前清零。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key，与 q 同形 |
| `d_o` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 输出梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local 累积 gate |
| `g_gamma` | 预留 | `-` | FP32 | ND | 当前必须为 None |
| `A` | 预留 | `-` | FP16/BF16 | ND | 当前必须为 None |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平的 (seq_id,chunk_id) |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `d_v` | `[B,H_v,T,V]` | 与 d_o 一致 | Value 的 chunk-local 梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `scale` | double | `无` | - | 通常为 1/sqrt(K) |
| `chunk_size` | int | `无` | `{64, 128}` | chunk 长度 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | q/k/d_o 为 FP16/BF16，g 可为 FP16/BF16/FP32 |
| Format/Layout | BNSD；变长序列在 T 维拼接 |
| 模式 | 定长与变长序列；变长序列的两个索引必须同时提供 |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_bwd_dv_local` |
| aclnn | `aclnnChunkBwdDvLocalGetWorkspaceSize` / `aclnnChunkBwdDvLocal` |
| Ascend C `<<<>>>` | `chunk_bwd_dv_local<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_bwd_dv_local` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_bwd_dv_local/accuracy/test_chunk_bwd_dv_local.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_bwd_dv_local.json`；覆盖定长与变长序列；变长序列的两个索引必须同时提供。
- 参考实现：`仓内 PyTorch/CPU reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `g_gamma` 和 `A` 尚未实现，必须传 `None`。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_bwd_dv_local python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_bwd_dv_local/accuracy/test_chunk_bwd_dv_local.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_bwd_dv_local/routes/`，均使用同一份 JSON 规格。

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
| `V` | Value/Output 单 head 特征维 |
| `N_c` | 当前调用中的 chunk 总数 |
