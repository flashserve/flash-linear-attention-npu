# KDA Forward 总体设计

## 接口边界

`ChunkKdaFwd` 是不包含 CP 切分的 FLA 顶层前向接口。它接收 q/k/v、raw 或已激活 gate、beta、
可选状态和变长元数据；`initial_state` 不是算子输出，Python 返回列表的第 12 项只做对象透传。

公共参数不包含 `total_chunks`、`gateScale`、`output_sequence_major` 或 stage。`N_c` 从
`cu_seqlens/chunk_indices/chunk_size` 推导；gate 累计固定乘 `1/ln(2)`。

## 实现拆分

| 组件 | 作用 | 是否有独立 L2 |
| --- | --- | --- |
| `KdaGateCumsum` | gate 激活和 chunk-local cumsum | 是，供 KDA/GDN2 复用 |
| Prepare | Aqk/Akk/qg/seed | 否，`ChunkKdaFwd` 内部阶段 |
| Post-WU | w/u/kg/v_new seed | 否，`ChunkKdaFwd` 内部阶段 |
| `ChunkGatedDeltaRuleFwdH` | chunk 间状态递推 | 是，供 GDN/KDA 复用 |
| Finalize | 输出融合与 sequence-major 写回 | 否，`ChunkKdaFwd` 内部阶段 |

不新增平台专用公开或私有原型。A2/A3/A5 均复用既有 `ChunkKdaFwd` 原型和同一个外层
`.cpp` 入口；上述内部阶段不再注册独立 L0。独立 `KdaGateCumsum` 和
`ChunkGatedDeltaRuleFwdH` 仍供其他模型单独调用。

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
- `layout` 和实际输入 shape 共同决定 kernel 内读取方式；TND 在 L2 物化为连续 head-major 视图。
- `attn_out` 固定 BSND/TND。
- `final_state` 固定按序列排列。
- `Aqk/Akk/gk/w/u/qg/kg/v_new` 固定 head-major，供反向继续计算；内部与公开 `h` 均为 NT-first，无 head/chunk 导出转置。
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

A2/A3/A5 使用相同 FP32 数学定义。`tiling key=1` 表示非 `chunk_size=64`、`K=V=128`
场景，`tiling key=2` 表示该场景并覆盖 dense、tail 和 varlen；tiling key 不区分平台。
同一 key 内由编译架构选择对应 kernel 模板，并在同一物理 kernel 内完成 gate cumsum、内部融合
与边界处理。独立 `KdaGateCumsum` 的 L2 接口保持不变。

详细阶段设计见 `chunk_kda_fwd/docs/design.md`，API 契约见 `chunk_kda_fwd/docs/api.md`。
