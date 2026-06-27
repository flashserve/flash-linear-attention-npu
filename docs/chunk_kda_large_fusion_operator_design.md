# Chunk KDA AscendC 融合算子设计文档

## 1. 目标

为 Kimi Delta Attention (KDA) 提供一套原生 AscendC 正反向算子。实现方式不是在 Python 或 torch 接口层拼接已有算子，而是新增 AscendC L0 kernel，并在 aclnn L2 接口层编排 L0：

- forward L2：调用一个 `ChunkKdaFwd` L0 融合 kernel。
- backward L2：先调用 `ChunkKdaFwd` L0 重算中间量，再调用 `ChunkKdaBwd` L0 融合 kernel 完成反向。

该方案优先保证与 fla-org KDA 公式语义一致，并覆盖 KDA 训练/推理需要的状态递推、GVA、varlen、`chunk_size=128`、`vdim=256` 等典型场景。

## 2. 三方实现对标

对标 fla-org KDA chunk 语义：

- 输入语义：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- KDA 与 GDN 的核心差异：GDN 使用 head-wise scalar gate，KDA 使用 key-wise `gk`，状态衰减为 `exp2(gk_last)` 的 K 维对角衰减。
- GVA 约束：`HV % H == 0`。
- varlen 约束：输入 flatten 到 `B=1`，`cu_seqlens` 表示每个 sequence 边界。
- key head dim 约束：当前实现对齐三方语义并限制 `K <= 256`。

本仓 torch_custom 暴露 head-first NPU ABI：

```text
q/k:    [B, H,  T, K]
v:      [B, HV, T, V]
gk:     [B, HV, T, K]
beta:   [B, HV, T]
state:  [N, HV, K, V]
output: [B, HV, T, V]
```

其中 `gk` 是已经完成 KDA log2 gate 累计后的 key-wise gate。`safe_gate=True`、`transpose_state_layout=True` 当前显式报错，避免静默偏离三方语义；`use_gate_in_kernel`、QK L2Norm in-kernel、beta sigmoid in-kernel 可作为后续扩展。

## 3. Forward L0 设计

`ChunkKdaFwd` 输入 `q/k/v/gk/beta/initial_state/cu_seqlens/chunk_indices`，输出：

```text
o, final_state, Aqk, Akk, w, u, qg, kg, v_new, h
```

每个 chunk 内计算：

```text
Aqk[i,j] = tril(q_i * k_j * exp2(g_i - g_j)) * scale
Akk      = inv(I + tril(k_i * k_j * exp2(g_i - g_j) * beta_i, -1))
w        = Akk @ (k * beta * exp2(g))
u        = Akk @ (v * beta)
qg       = q * exp2(g)
kg       = k * exp2(g_last - g)
v_new    = u - w @ h_prev
o        = qg @ h_prev * scale + Aqk @ v_new
h_next   = exp2(g_last)[:, None] * h_prev + kg^T @ v_new
```

`h` 保存每个 chunk 的 `h_prev`，用于反向重算和推理中间状态返回。

### 3.1 exp2 实现

所有 `exp2(x)` 均对齐三方 Triton 的 fp32 exp2 语义：先在 UB 中以 fp32 写入 `x * ln2`，再调用 AscendC 向量 `Exp` 得到 `2^x`。K 维 gate 相关的 `exp2(g)`、`exp2(g_i-g_j)`、`exp2(g_last-g)`、`exp2(g_last)` 均按 K 向量成批计算，不再使用 scalar 多项式循环近似。

### 3.2 CATLASS score 路径

KDA 的 score 可按 gate 因子拆解：

```text
q_i * k_j * exp2(g_i - g_j) = (q_i * exp2(g_i)) @ (k_j * exp2(-g_j))^T
k_i * k_j * exp2(g_i - g_j) = (k_i * exp2(g_i)) @ (k_j * exp2(-g_j))^T
```

因此 forward 对 fp16/bf16 且 `K >= 16` 的 score 原始矩阵使用 CATLASS cube：

```text
q_pos = q * exp2(g)
k_pos = k * exp2(g)
k_neg = k * exp2(-g)
Aqk_raw = q_pos @ k_neg^T
Akk_raw = k_pos @ k_neg^T
```

实现细节：

- AIV 使用 `TQue<VECIN>` 搬入 `q/k/gk` 行，UB 中用 `Cast/Muls/Exp/Mul` 生成 `q_pos/k_pos/k_neg`，再通过 `TQue<VECOUT>` 与 `DataCopy/DataCopyPad` 写入 `qg/w/kg` 临时 GM。
- AIC 使用 CATLASS `BlockMmadTla` 执行两次 cube matmul。
- AIV/AIC 使用 `CrossCoreFlagWithReverse` 做 ready/done 同步。
- AIV 完成 mask、scale、`beta_i` 乘法和 lower-triangular inverse。
- `qg/w/kg` 最终会被覆盖为对外语义需要的 `qg/kg/w`。

fp32 和小 K fallback 走 AIV-only。AIV-only 与 CATLASS 前处理共用同一套 SIMD gate-product 管线：32B 对齐连续行走 `DataCopy`，小行宽或非 32B 对齐行走 `DataCopyPad`，不再用逐元素 `GetValue/SetValue` 生成 `qg/w/kg`。

### 3.3 tiling key

forward 当前有三个 tiling key：

```text
key 0: float32, AIV_ONLY
key 1: fp16/bf16 且 K >= 16, MIX_AIC_1_2
key 2: fp16/bf16 且 K < 16, AIV_ONLY
```

key 1 是 CATLASS score 主路径；key 0/key 2 是语义 fallback。这样既支持典型模型 `chunk_size=128`、`V=256`，也支持测试和小模型中常见的 `K=8`。

## 4. Backward L0 设计

`aclnnChunkKdaBwd` 是 backward L2 入口。它在 aclnn executor 内部完成：

```text
ChunkKdaFwd(...) -> recomputed o/final_state/Aqk/Akk/w/u/qg/kg/v_new/h
ChunkKdaBwd(...) -> dq/dk/dv/dbeta/dgk/dh0
ViewCopy(...)    -> user outputs
```

外部传入的 `Aqk/Akk` 保留为接口兼容参数和 shape 校验来源，但 L2 不再对其做无用 contiguous，也不把其数值传入反向 kernel；反向实际使用内部重算出的临时中间量。

`ChunkKdaBwd` 覆盖：

- `d_o + Aqk + v_new -> dAqk, d_v_new(local)`。
- `dht + kg + qg + w + d_o + d_v_new -> dh0`，即 KDA 版本的 `bwd_dhu`。
- `Akk/w/u` 反传到 `k/v/beta/gk`。
- `Aqk` 反传到 `q/k/gk`。
- GVA 下同一个 q-head 对应多个 value-head，kernel 以 `(sequence, q_head)` 为 task，内部顺序处理该 q-head 下的 value-head，避免 `dq/dk` 写冲突。

KDA 的反向状态递推需要先使用当前 chunk 的旧 `d_state` 计算完整 `d_h_prev`，然后再写回给前一个 chunk。实现中按 V 维列处理：

```text
for r in V:
    new_state[d] = f(old_d_state[:, r], qg, kg, w, d_o, d_v_new)
    dh0[:, r] = new_state
```

## 5. aclnn 接口

### 5.1 Forward

```text
npu_chunk_kda_fwd(
    q, k, v, gk, beta,
    scale: float,
    chunk_size: int,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False
) -> (o, final_state, Aqk, Akk, w, u, qg, kg, v_new, h)
```

### 5.2 Backward

```text
npu_chunk_kda_bwd(
    q, k, v, gk, beta, Aqk, Akk, d_o,
    scale: float,
    chunk_size: int,
    *,
    initial_state=None,
    dht=None,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    transpose_state_layout=False
) -> (dq, dk, dv, dbeta, dgk, dh0)
```

### 5.3 参数约束

- `chunk_size` 支持 `32/64/128`。
- `K <= 256`。
- `V` 已覆盖 `256`。
- `HV % H == 0`。
- `q/k/v/state/o` 支持 float32、float16、bfloat16；`gk/beta` 当前要求 float32。
- `safe_gate=True`、`transpose_state_layout=True` 当前显式报错。

## 6. 文件归档

新增 AscendC L0/L2：

- `fla/ops/ascendc/kda/chunk_kda_fwd/**`
- `fla/ops/ascendc/kda/chunk_kda_bwd/**`

新增 torch_custom ABI：

- `torch_custom/fla_npu/npu_custom.yaml`
- `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp`
- `torch_custom/fla_npu/fla_npu/kda.py`

新增验证：

- `torch_custom/fla_npu/test/test_npu_chunk_kda.py`
- `tests/reference/chunk_kda_reference.py`

## 7. 后续优化方向

当前实现已完成 KDA 正反向功能闭环和精度验证。面向吞吐优化时，建议继续：

- 将 `w/u`、`o`、`h_next` 的 V 维 tile 驻留 UB，减少 GM 往返。
- 将 backward 的 `d_v_new`、`dh`、`dq/dk` 分解为 cube 主路径和 vec 辅助路径。
- 为 BF16 gate 输入、`use_gate_in_kernel`、`safe_gate` 增加独立 L0/L2 扩展，进一步对齐 fla-org 高层接口。
