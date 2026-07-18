# PrepareWyReprBwdFull

## 1. 功能概述

WY 表示完整反向算子。它消费 `dA/dW/dU`，在 chunk 内反传到 K、V、beta 与 gate，是 `prepare_wy_repr_bwd_da` 之后的主梯度阶段。

## 2. 数学定义

对每个 chunk，令 `Kb=K*beta*exp(g)`、`Vb=V*beta`，则链式法则包含：

```text
dV     = A^T @ dU * beta
dK     = A^T @ dW * beta * exp(g) + dA-related terms
dBeta  = reduce_V(dVb * V) + reduce_K(dKb * K)
dG     = reverse_chunk_reduce(dKb * K)
```

`dA` 的因果区域和 GVA head 映射与前一阶段完全一致。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `beta` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | WY 权重 |
| `A` | 必选 | `[B,H_v,T,chunk_size]` | FP16/BF16 | BNSD | 前向局部矩阵 |
| `dA` | 必选 | `[B,H_v,T,chunk_size]` | FP16/BF16 | BNSD | A 梯度 |
| `dw` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | W 梯度 |
| `du` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | U 梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local gate |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dk` | `[B,H_k,T,K]` | 与 k 一致 | Key 梯度；GVA value head 已归约 |
| `dv` | `[B,H_v,T,V]` | 与 v 一致 | Value 梯度 |
| `dbeta` | `[B,H_v,T]` | 与 beta 一致 | beta 梯度 |
| `dg` | `[B,H_v,T]` | 与 g 一致 | gate 梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | `{64, 128}` | 必须等于 A/dA 最后一维 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 主张量 FP16/BF16；beta/g 可为 FP32 |
| Format/Layout | BNSD；A/dA 最后一维为 chunk_size |
| 模式 | 定长与变长序列，支持 GVA |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full` |
| aclnn | `aclnnPrepareWyReprBwdFullGetWorkspaceSize` / `aclnnPrepareWyReprBwdFull` |
| Ascend C `<<<>>>` | `prepare_wy_repr_bwd_full<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_prepare_wy_repr_bwd_full` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/prepare_wy_repr_bwd_full/accuracy/test_prepare_wy_repr_bwd_full.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/prepare_wy_repr_bwd_full.json`；覆盖定长与变长序列，支持 GVA。
- 参考实现：`仓内 PyTorch/CPU reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，并须等于 A/dA 最后一维。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=prepare_wy_repr_bwd_full python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/prepare_wy_repr_bwd_full/accuracy/test_prepare_wy_repr_bwd_full.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/prepare_wy_repr_bwd_full/routes/`，均使用同一份 JSON 规格。

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
| `chunk_size` | 每个 chunk 的 token 数，也是三角块宽度 |
| `N_c` | 当前调用中的 chunk 总数 |
