# KDA Forward 总体设计

## 接口边界

`ChunkKdaFwd` 是不包含 CP 切分的 FLA 顶层前向接口。它接收 q/k/v、raw 或已激活 gate、beta、
可选状态和变长元数据；`initial_state` 不是算子输出，Python 返回列表的第 12 项只做对象透传。

公共参数不包含 `total_chunks`、`gateScale`、`output_sequence_major` 或 stage。`N_c` 从
`cu_seqlens/chunk_indices/chunk_size` 推导；gate 累计固定乘 `1/ln(2)`。

## 算子拆分

| 算子 | 作用 | 是否有独立 L2 |
| --- | --- | --- |
| `KdaGateCumsum` | gate 激活和 chunk-local cumsum | 是，供 KDA/GDN2 复用 |
| `ChunkKdaFwdPrepare` | Aqk/Akk/qg/seed | 否，仅作为顶层阶段 |
| `ChunkKdaFwdPostWu` | w/u/kg/v_new seed | 否，仅作为顶层阶段 |
| `ChunkGatedDeltaRuleFwdH` | chunk 间状态递推 | 是，供 GDN/KDA 复用 |
| `ChunkKdaFwdFinalize` | 输出融合与 sequence-major 写回 | 否，仅作为顶层阶段 |

每个 L0 只声明本阶段实际读取的输入。阶段间张量通过显式 GM/workspace 传递，kernel 不跨算子 stage。

## Gate 与指数

```text
gate =
    g                                                       if !use_gate_in_kernel
    -exp(A_log) * softplus(g + dt_bias)                     if use_gate_in_kernel && !safe_gate
    lower_bound * sigmoid(exp(A_log) * (g + dt_bias))       if use_gate_in_kernel && safe_gate

gk = chunk_local_cumsum(gate) / ln(2)
```

后续 key-wise 衰减统一使用 `exp2(gk)`。因此 `gateScale` 不可配置，否则会破坏 gate 与指数基数的绑定。

## 布局

- `layout` 仅描述输入。
- 输入 BSND/TND 在 L2 通过 `l0op::Transpose` 转为连续 BNSD/NTD。
- `attn_out` 固定 BSND/TND。
- `final_state` 固定按序列排列。
- `Aqk/Akk/gk/w/u/qg/kg/v_new/h` 固定 head-major，供反向继续计算。
- `state_v_first` 控制 initial/final state 与 h 的末两维。

## FLA 输出策略

```text
attn_out: always
final_state: output_final_state
gk: !use_gate_in_kernel || disable_recompute
Aqk/Akk: always
w/u/qg/kg/v_new: disable_recompute
h: disable_recompute || return_intermediate_states
initial_state: Python passthrough only
```

aclnn 通过可空输出 descriptor 表达可选输出，OpDef 使用 `OPTIONAL_OUTPUT` 分组。

## 平台与性能

A2/A3/A5 使用相同 FP32 数学主路径。A5 的 VEC stage 使用 regbase 双发射，AIC/AIV 分别通过
L1/UB 双缓冲覆盖搬运和计算。性能门禁使用完整 L2 调用，包含 `KdaGateCumsum`。

详细阶段设计见 `chunk_kda_fwd/docs/design.md`，API 契约见 `chunk_kda_fwd/docs/api.md`。
