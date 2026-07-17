# PrepareWyReprBwdDa

## 1. 功能概述

WY 表示反向的 dA 子算子。它根据 K/V、beta、前向 A、dW/dU 与 gate，计算 chunk 局部矩阵 A 的梯度，供完整 WY 反向继续生成 dK/dV/dBeta/dG。

## 2. 数学定义

在每个 value head/chunk 内，先将对应 key head 按 GVA 关系广播，再组合两条矩阵链：

```text
dA = dU @ (V * beta)^T + dW @ (K * beta * exp(g))^T
dA = causal_mask(dA) + A-dependent triangular correction
```

`dA` 仅在 chunk 的有效因果区域有定义；尾块之外写零。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `beta` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | WY 权重 |
| `A` | 必选 | `[B,H_v,T,C]` | FP16/BF16 | BNSD | 前向 chunk 局部矩阵 |
| `dw` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | W 梯度 |
| `du` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | U 梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local gate |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dA` | `[B,H_v,T,C]` | 与 A 一致 | chunk 局部矩阵梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `无` | 必须等于 A/dA 最后一维 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | K/V/A/dW/dU 为 FP16/BF16；beta/g 可为 FP32 |
| Format/Layout | BNSD；A/dA 最后一维为 C |
| 模式 | fixed 与 varlen，支持 GVA |

varlen 模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。fixed 与 varlen、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_da` |
| aclnn | `aclnnPrepareWyReprBwdDaGetWorkspaceSize` / `aclnnPrepareWyReprBwdDa` |
| Ascend C `<<<>>>` | `prepare_wy_repr_bwd_da<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_prepare_wy_repr_bwd_da` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/prepare_wy_repr_bwd_da/accuracy/test_prepare_wy_repr_bwd_da.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/prepare_wy_repr_bwd_da.json`；覆盖 fixed 与 varlen，支持 GVA。
- 参考实现：`仓内 PyTorch/CPU reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，并须等于 A/dA 的最后一维。
- 必须满足 `H_v % H_k == 0`；varlen 当前仅支持物理 `B=1`。
- `cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=prepare_wy_repr_bwd_da python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/prepare_wy_repr_bwd_da/accuracy/test_prepare_wy_repr_bwd_da.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/prepare_wy_repr_bwd_da/routes/`，均使用同一份 JSON 规格。

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
