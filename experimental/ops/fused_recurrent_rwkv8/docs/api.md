# FusedRecurrentRwkv8 接口契约

## 功能

`FusedRecurrentRwkv8` 是 RWKV-v8（WKV7，z/b 参数化口径）的逐 token 递推前向算子，
面向推理路径（prefill/decode 共用一份 kernel）。语义对齐
BlinkDL/RWKV-LM 提交 `9521024`（`RWKV-v8/cuda/wkv7_cuda.cu` `forward_kernel` lines 10-52），
精度 peer 为 fla-org/flash-linear-attention 提交 `a4a2624b`
（`fla/ops/generalized_delta_rule/dplr` fused_recurrent 前向）。

递推公式（per head，state 数学朝向：行 = v/q 侧，列 = k/z 侧；接口 `initial_state` / s 快照以内核账本 `S^T` 朝向 `(B,H,K,V)` 传入/传出，零转置）：

```text
sa    = state @ z_t                          # 移除读出（z = -kk）
state = state * decay_t[None, :]             # 逐通道衰减，decay = exp(-exp(w))
      + sa[:, None] * b_t[None, :]           # 移除（rank-1，b = kk * a_inctx）
      + v_t[:, None] * k_t[None, :]          # delta-rule 写入
o_t   = state @ (q_t * scale)                # 读出（scale 作用在 q 上）
```

与官方 CUDA kernel 的差异：无 `T % 16` 约束（官方 kernel 要求 T 为 16 的倍数）。

> io 布局为 **BHTC = (B,H,T,C)**（H 在 T 前，2026-08-17 定档）：每个 (b,h) 的
> 数据在 GM 上连续，token 步长为 C。与 fla 的 (B,T,H,C) 口径不同，对接时需
> `transpose(1,2)`。

## 输入

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | `(B,H,T,N)`，FP16/BF16/FP32 | Query（读出向量） |
| `w` | 必选 | 同 q | log 域衰减参数，`decay = exp(-exp(w))` |
| `k` | 必选 | 同 q | Key（写入向量） |
| `v` | 必选 | `(B,H,T,V)`，同 q dtype | Value |
| `z` | 必选 | 同 q | 移除读出向量（`= -kk`，kk 为 L2 归一化） |
| `b` | 必选 | 同 q | 移除强度向量（`= kk * a_inctx`） |
| `initial_state` | 可选 | `(B,H,K,V)`，FP32 | 接口朝向（= 内核账本 `S^T`，与 s 快照一致），缺省为零态 |

## 输出

```text
(o, s, sa)
```

- `o`：`(B,H,T,V)`，dtype 跟随输入 q（FP16/BF16/FP32）。
- `s`：`(B,H,T//chunk_len,K,V)`，FP32；chunk state 快照，官方 CUDA 转置布局，`outputChunkState=true` 时必填。
- `sa`：`(B,H,T,V)`，FP32；每 token 移除读出（更新前的 `state@z`），`outputSa=true` 时必填。

## 属性

| 名称 | 默认值 | 支持范围 |
| --- | --- | --- |
| `scale` | `1.0` | FP32 标量，作用于 q 读出 |
| `reverse` | `false` | T 维倒序递推（对齐 fla reverse） |
| `output_chunk_state` | `false` | 产出 chunk 快照 `s`（训练预埋） |
| `output_sa` | `false` | 产出每 token 的 `sa`（训练预埋） |
| `chunk_len` | `16` | `s` 快照间隔（INT64，1 到 4294967295）；默认 16 对齐官方 CUDA backward 的 chunk 重建粒度，非 16 值与官方 backward 不兼容 |

`s`/`sa` 为训练预埋输出（对齐官方 CUDA 三输出 y_/s_/sa_），由 OpDef 属性门控；
缺省关闭时给零尺寸 shape、kernel 跳过写出。

## 支持范围

- A2 (`ascend910b`)：保留 PR #446 原实现与配置；原 PR 的历史测试不作为本迁移版本的复验结果。
- A5 (`ascend950`)：本轮新增构建目标，只验编译；未声明精度、性能或运行支持。
- `q/w/k/v/z/b` 支持 FP16/BF16/FP32（六者 dtype 必须一致）；递推与 state 全程 FP32，
  `o` 跟随输入 dtype，`initial_state`/`s`/`sa` 恒 FP32。
- `K <= 128`、`V <= 128`（均须为 8 的倍数且不切分，`K×V` 的 state 须整体放入单核 UB）。
- `B/H <= 65535` 且 `B*H <= 65535`（一个逻辑核负责一个 (b,h) 账本，blockDim 受 CANN
  约定的逻辑核数上限约束）；`T > 2^20`（1048576）由 tiling 拦截并返回 `GRAPH_FAILED`，
  该值仅为拦截上界（取自 kernel `sane_` 的 `T_` 上限）；实测最大 `T = 4096`，
  更大范围受时间限制未做测试。
- `B/T/H` 无对齐约束（含 `T=1` decode）。
- 仅前向；反向归 chunk 算子另行立项。

## 来源与本轮差异

算子源码和 CPU golden 来自 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446)
固定提交 `049eb0d`。本轮复用现有数学语义、参数顺序、属性默认值、dtype、layout、状态朝向
和 CPU golden；目录迁入 `experimental/ops/fused_recurrent_rwkv8`，新增 A5 编译配置。
Python 入口为独立的 `fla_npu_experimental.fused_recurrent_rwkv8`；不注册稳定 dispatcher。

修复来源实现的 `chunk_len` 窄化缺陷：aclnn/tiling/Python 明确拒绝不在
`[1,4294967295]` 内的值。aclnn/tiling 通过结构化错误上报给出 `chunk_len`、实际值和合法范围；
Python 错误包含 `chunk_len must be in [1, 4294967295]`。
原 int64 属性窄化到 uint32 时，2^32 会变成 0 并在 kernel 中触发除零；超界输入不再截断。

aclnn 参数顺序和两段式调用详见 [aclnn ABI](aclnnFusedRecurrentRwkv8.md)。
构建与验证范围见 [验证记录](../../../validation.md)。本轮从既有算子实现迁移路线进入方案适配；
A2 正式回归与 A5 运行资格仍待后续设备验证。
