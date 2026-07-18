# RecomputeWUFwd

## 1. 功能概述

按需重计算 WY 中间张量 W/U，减少前向保存显存。输入 K/V/beta/A 与标量 gate g，在每个 chunk 内输出 W 和 U。

## 2. 数学定义

对 value head `h_v` 与映射后的 key head `h_k`：

```text
Vb = V[h_v] * beta[h_v]
Kb = K[h_k] * beta[h_v] * exp(g[h_v])
U  = A[h_v] @ Vb
W  = A[h_v] @ Kb
```

最后一个 chunk 使用实际有效行数，A 的其余列不参与结果。

## 3. 输入、输出和属性

本文使用的 Shape 符号统一引用[GDN 模型符号表](../../README.md#model-shape-symbols)，不在算子 README 中重复定义。

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `beta` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | WY 权重 |
| `A` | 必选 | `[B,H_v,T,chunk_size]` | FP16/BF16 | BNSD | 局部矩阵 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 标量 gate |
| `gk` | 预留 | `-` | - | - | 当前 kernel 不消费，必须为 None |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `w` | `[B,H_v,T,K]` | 与 k 一致 | WY 的 W |
| `u` | `[B,H_v,T,V]` | 与 v 一致 | WY 的 U |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | `{64, 128}` | chunk 长度并等于 A 最后一维 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 主张量 FP16/BF16；beta/g/gk 可为 FP32 |
| Format/Layout | BNSD |
| 模式 | 定长/变长序列、GVA、标量 gate g |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recompute_wu_fwd` |
| aclnn | `aclnnRecomputeWUFwdGetWorkspaceSize` / `aclnnRecomputeWUFwd` |
| Ascend C `<<<>>>` | `recompute_wu_fwd<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_recompute_w_u_fwd` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/recompute_wu_fwd/accuracy/test_recompute_wu_fwd.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/recompute_wu_fwd.json`；覆盖定长/变长序列、GVA、标量 gate g。
- 参考实现：`torch_recompute_wu_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- 当前实现仅支持 `K=128`、`V=128/256`、`chunk_size=64/128`。
- 必须满足 `H_v % H_k == 0`，A 最后一维等于 chunk_size。
- g 必须提供；gk 当前未实现且必须为 None；变长序列物理 B=1。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=recompute_wu_fwd python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/recompute_wu_fwd/accuracy/test_recompute_wu_fwd.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/recompute_wu_fwd/routes/`，均使用同一份 JSON 规格。
