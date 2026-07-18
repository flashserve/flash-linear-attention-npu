# ChunkGatedDeltaRuleBwdDhu

## 1. 功能概述

沿 chunk 反向计算隐藏状态梯度，并把状态分支对 Value 的贡献累加到 `dV2`。当前交付 kernel 只消费标量 gate `g`；逐 K gate、初末状态和 `dh0` 输出保留在 ABI 中但尚未实现。

## 2. 数学定义

对当前 chunk，合并输出分支和状态分支的梯度贡献：

```text
dV2_i = dV_i + state_value_contribution(dH_i, K_i, g_i)
dH_i  = Q_i^T @ dO_i + W_i^T @ dV2_i
```

GVA 下一个 key head 被 `H_v/H_k` 个 value head 复用；每个 value head 的状态和输出独立。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `w` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | WY 的 W |
| `d_o` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 输出梯度 |
| `dv` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 已有 Value 梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 标量 gate |
| `gK` | 预留 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | 当前必须为 None/nullptr |
| `h0` | 预留 | `[B,H_v,K,V]` | FP16/BF16 | ND | 当前必须为 None/nullptr |
| `dht` | 预留 | `[B,H_v,K,V]` | FP16/BF16 | ND | 当前必须为 None/nullptr |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dh` | `[B,H_v,N_c,K,V]` | 与 q 一致 | 每个 chunk 起点的状态梯度 |
| `dh0` | `-` | - | 预留输出；Python 固定返回 None，aclnn/直调必须传 nullptr |
| `dv2` | `[B,H_v,T,V]` | 与 dv 一致 | 合并状态贡献后的 Value 梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `scale` | double | `1.0` | - | Query 分支缩放 |
| `chunk_size` | int | `64` | `{64, 128}` | chunk 长度 |
| `use_exp2` | bool | `false` | `{false}` | 当前仅支持 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 当前仅支持 false |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | 主张量 FP16/BF16；g 可额外为 FP32 |
| Format/Layout | BNSD；状态为 `[B,H_v,K,V]` 或 `[B,H_v,N_c,K,V]` |
| 模式 | fixed/varlen、标量 g、FP16/BF16 主张量、FP16/BF16/FP32 gate |

varlen 模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。fixed 与 varlen、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu` |
| aclnn | `aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize` / `aclnnChunkGatedDeltaRuleBwdDhu` |
| Ascend C `<<<>>>` | `chunk_gated_delta_rule_bwd_dhu<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_gated_delta_rule_bwd_dhu/accuracy/test_chunk_gated_delta_rule_bwd_dhu.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_gated_delta_rule_bwd_dhu.json`；覆盖 fixed/varlen、标量 g、FP16/BF16 主张量、FP16/BF16/FP32 gate。
- 参考实现：`仓内 PyTorch/CPU reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K <= 128`、`V <= 256`；交付矩阵覆盖 K=128、V=128/256。
- `chunk_size` 仅支持 64/128；`H_v % H_k == 0`。
- `g` 必须提供；`gK`、`h0`、`dht` 和 `dh0` 当前为预留，必须为空。
- varlen 当前仅支持物理 `B=1`，两个索引必须同时提供。
- `use_exp2` 与 `transpose_state_layout` 当前必须为 false。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_gated_delta_rule_bwd_dhu python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_gated_delta_rule_bwd_dhu/accuracy/test_chunk_gated_delta_rule_bwd_dhu.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_gated_delta_rule_bwd_dhu/routes/`，均使用同一份 JSON 规格。

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
