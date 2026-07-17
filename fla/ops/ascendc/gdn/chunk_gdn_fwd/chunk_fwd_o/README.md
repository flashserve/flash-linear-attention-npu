# ChunkFwdO

## 1. 功能概述

Gated Delta Rule 的 chunk 输出阶段。算子把当前 chunk 的 Q/K/V、chunk 起始状态 H 和累积 gate 合并，生成注意力输出 O；状态推进由 `chunk_gated_delta_rule_fwd_h` 单独完成。

## 2. 数学定义

对 value head `h_v` 映射 key head `h_k=floor(h_v/(H_v/H_k))`：

```text
score = causal_mask((Q[h_k] @ K[h_k]^T) * scale * exp(g_row-g_col))
O[h_v] = Q[h_k] @ H_start[h_v] * scale + score @ V[h_v]
```

`H_start` 取当前 chunk 对应的状态切片；尾块只计算有效 token，输出保持 `[B,H_v,T,V]`。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `h` | 必选 | `[B,H_v,N_c,K,V]` | FP16/BF16 | ND | 每个 chunk 的起始状态 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 累积标量 gate |
| `g_gamma` | 预留 | `-` | - | - | 上层兼容参数，当前必须为 None |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `o` | `[B,H_v,T,V]` | 与 v 一致 | chunk 注意力输出 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `scale` | double | `无` | Query 缩放 |
| `chunk_size` | int | `64` | chunk 长度 |
| `transpose_state_layout` | bool | `false` | 预留参数，当前必须 false |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | Q/K/V/H 为 FP16/BF16；g 可为 FP32 |
| Format/Layout | BNSD；状态使用 `[B,H_v,N_c,K,V]` |
| 模式 | fixed/varlen、GVA、整块/尾块；g 为必选标量 gate |

varlen 模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。fixed 与 varlen、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_fwd_o` |
| aclnn | `aclnnChunkFwdOGetWorkspaceSize` / `aclnnChunkFwdO` |
| Ascend C `<<<>>>` | `chunk_fwd_o<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_fwd_o` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_fwd_o/accuracy/test_chunk_fwd_o.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_fwd_o.json`；覆盖 fixed/varlen、GVA、整块/尾块；g 为必选标量 gate。
- 参考实现：`torch_chunk_fwd_o_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`，h 的 chunk 数必须与索引推导一致。
- varlen 当前仅支持物理 `B=1`，两个索引必须同时提供。
- `g` 是 kernel 必选输入；`g_gamma` 必须为 None，`transpose_state_layout` 必须为 false。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_fwd_o/accuracy/test_chunk_fwd_o.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_fwd_o/routes/`，均使用同一份 JSON 规格。

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
| `C` | chunk_size |
| `N_c` | 当前调用中的 chunk 总数 |
