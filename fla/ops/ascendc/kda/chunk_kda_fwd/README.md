# ChunkKdaFwd

## 1. 功能概述

Kimi Delta Attention 正向主算子。它消费已经按 chunk 累加的 key gate `gk`，分阶段生成 chunk 内矩阵项、递推状态和最终输出，并可返回完整中间量用于训练链路与精度定位。

## 2. 数学定义

对每个 value head 映射到对应 key head，在一个 chunk 内定义：

```text
Aqk[i,j] = tril(q_i @ k_j^T * exp2(gk_i-gk_j)) * scale
Akk      = inv(I + tril((k_i @ k_j^T) * exp2(gk_i-gk_j) * beta_i, -1))
w        = Akk @ (k * beta * exp2(gk))
u        = Akk @ (v * beta)
kg       = k * exp2(-gk)
v_new    = u - w @ h_prev
h_next   = exp2(gk_last) * h_prev + kg_state^T @ v_new
o        = (qg @ h_prev + Aqk @ v_new) * scale
```

`gk` 位于 log2 空间，因此 kernel 以 `exp(x*ln2)` 实现 `exp2(x)`。`final_state`
固定为 FP32；partial chunk 的补齐行使用中性值参与固定 tile，公开输出的无效区域写零。

## 3. 输入、输出和属性

本文使用的 Shape 符号统一引用[KDA 模型符号表](../README.md#model-shape-symbols)，不在算子 README 中重复定义。

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `按 layout 为 [B,T,H_k,K]/[B,H_k,T,K]/[T,H_k,K]/[H_k,T,K]` | FP16/BF16 | BSND/BNSD/TND/NTD | Query |
| `k` | 必选 | `与 q 相同` | 与 q 相同 | 与 q 相同 | Key |
| `v` | 必选 | `对应 [B,T,H_v,V]/[B,H_v,T,V]/[T,H_v,V]/[H_v,T,V]` | 与 q 相同 | 同 layout | Value |
| `gk` | 必选 | `与 k 的 token/head/K 维对应` | FP32/BF16 | 同 layout | chunk 内 log2 累积 key gate |
| `beta` | 必选 | `去掉 K 维的 gk shape` | FP32/BF16 | 同 layout | Delta 更新系数 |
| `initial_state` | 可选 | `[N,H_v,K,V]` | FP32 | ND | 每条逻辑序列的初始状态 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | sequence-major chunk 二元组 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `o` | `与 v 相同` | 与 v 相同 | KDA 输出 |
| `final_state` | `[N,H_v,K,V] 或空` | FP32 | output_final_state=false 时 Python 返回空 tensor |
| `g` | `与 gk 相同` | FP32 | Python 返回槽：gk 转 FP32 |
| `Aqk/Akk` | `按 layout 为 [...,T,chunk_size]` | 与 q 相同 | chunk 内因果矩阵；内部计算可使用 FP32 |
| `w/qg/kg` | `按 layout 为 [...,T,K]` | 与 q 相同 | K 维中间量 |
| `u/v_new` | `与 v 相同` | 与 v 相同 | V 维中间量 |
| `h` | `按 layout 为 [B,H_v,N_c,K,V] 或 [B,N_c,H_v,K,V]` | 与 q 相同 | 每个 chunk 的起始状态 |
| `initial_state_out` | `与 initial_state 相同或空` | FP32 | Python 预留透传槽 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | `{"BSND", "BNSD", "TND", "NTD"}` | 只接受大写 BSND/BNSD/TND/NTD |
| `scale` | double | `无` | - | 通常为 1/sqrt(K) |
| `chunk_size` | int | `无` | `{64, 128}` | 64 或 128 |
| `output_final_state` | bool | `false` | `{false, true}` | 是否返回有效 final_state |
| `return_intermediate` | bool | `false` | `{false, true}` | 是否物化八个中间张量 |
| `safe_gate` | bool | `false` | `{false}` | 预留，当前必须 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 预留，当前必须 false |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | q/k/v 为同一 FP16 或 BF16；gk/beta 为 FP32 或 BF16，由编译期模板直接读取；状态为 FP32 |
| Format/Layout | BSND/BNSD/TND/NTD；BNSD/NTD 为内部性能布局，BSND/TND 通过 KdaLayoutSwap12 转换 |
| 模式 | 定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量 |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

完整调用内部按 `PREPARE -> POST_WU -> STATE -> OUTPUT` 四个语义阶段执行。阶段仅在 L0 和
kernel 模板中可见，公共 Python/aclnn/直调 API 均不接收数值 `stage`。gk、beta 和阶段输出类型由
编译期模板选择，不在 L2 通过 Cast 拼出计算主路径。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_kda_fwd` |
| aclnn | `aclnnChunkKdaFwdGetWorkspaceSize` / `aclnnChunkKdaFwd` |
| Ascend C `<<<>>>` | `torch.ops.ascend_ops.chunk_kda_fwd_direct`（内部真实发射四阶段 kernel） |
| legacy（可选） | `torch.ops.npu.npu_chunk_kda_fwd` |

所有正式入口都表达一次完整算子语义，不要求调用者理解或传入内部阶段编号。直调示例当前覆盖连续
BNSD、K=128、V=128/256、chunk_size=64/128；稳定 Python/aclnn 入口覆盖本页完整能力矩阵。

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_kda_fwd/accuracy/test_chunk_kda_fwd.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_kda_fwd.json`；覆盖定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量。
- 参考实现：`tests/reference/chunk_kda_reference.py`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- chunk_size 仅支持 64/128；K/V 均须在 [16,256] 且为 16 的倍数；交付矩阵覆盖 K=128、V=128/256。
- H_k/H_v 必须在 [1,128] 且 H_v % H_k == 0；TND 仅支持 H_k=1，多 head rank3 使用 NTD。
- 变长序列的 cu_seqlens 至少含首尾、非递减且末项等于 T；单次最多 1024 条逻辑序列。
- 显式 chunk_indices 必须完整、合法并严格采用 sequence-major 规范顺序。
- safe_gate 与 transpose_state_layout 当前必须为 false；raw gate 应先调用 kda_gate_cumsum。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_fwd python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_kda_fwd/accuracy/test_chunk_kda_fwd.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 通路契约位于
`tests/operators/chunk_kda_fwd/routes/`；真实 `<<<>>>` 源码和运行测试位于
`examples/fast_kernel_launch_example/csrc/chunk_kda_fwd/` 与对应 `tests/chunk_kda_fwd/`，均使用同一份 JSON 规格。
