# Chunk KDA AscendC 融合算子设计文档

## 1. 目标

为 Kimi Delta Attention (KDA) 提供一套原生 AscendC 正反向算子。实现方式不是在 Python 或 torch 接口层拼接已有算子，而是新增 AscendC L0 kernel，并在 aclnn L2 接口层编排 L0：

- forward L2：调用一个 `ChunkKdaFwd` L0 融合 kernel。
- backward L2：先调用 `ChunkKdaFwd` L0 重算中间量，再调用 `ChunkKdaBwd` L0 融合 kernel 完成反向。

该方案优先保证与 fla-org KDA 公式语义一致，并覆盖 KDA 训练/推理需要的状态递推、GVA、varlen、`chunk_size=128`、`vdim=256` 等场景。

## 2. 三方实现对标

对标实现参考 fla-org `fla/ops/kda/chunk.py` 与公共 chunk 状态接口：

- 输入语义：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- KDA 与 GDN 的核心差异：GDN 使用 head-wise scalar gate，KDA 使用 key-wise gate，状态衰减为 `exp2(gk_last)` 的 K 维对角衰减。
- GVA 约束：`HV % H == 0`。
- varlen 约束：输入需要 flatten 到 `B=1`，`cu_seqlens` 表示每个 sequence 边界。
- key headdim 约束：对齐 fla-org 当前 KDA 约束，`K <= 256`。

本仓 torch_custom 暴露 head-first NPU ABI：

```text
q/k:    [B, H,  T, K]
v:      [B, HV, T, V]
gk:     [B, HV, T, K]
beta:   [B, HV, T]
state:  [N, HV, K, V]
output: [B, HV, T, V]
```

其中 `gk` 是已经完成 KDA log2 gate 累计后的 key-wise gate。`use_gate_in_kernel=True`、`safe_gate=True`、`state_v_first=True`、QK L2Norm in-kernel、beta sigmoid in-kernel 作为后续扩展点保留，不在本次 L0 ABI 中隐式拼接。

## 3. Forward L0 设计

### 3.1 L0 算子

`ChunkKdaFwd` 是一个 AscendC 融合 kernel，输入 `q/k/v/gk/beta/initial_state/cu_seqlens/chunk_indices`，输出：

```text
o, final_state, Aqk, Akk, w, u, qg, kg, v_new, h
```

### 3.2 计算语义

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

### 3.3 调度与状态

- task 维度按 `(sequence, value_head)` 切分。
- fixed-length 下每个 batch 独立递推；varlen 下通过 `cu_seqlens` 找到 sequence 边界。
- chunk 内存在严格数据依赖的 `Akk inverse`、`w/u`、`v_new`、`h_next` 在单个 L0 kernel 内完成，避免把状态递推拆回 torch 层。
- 当前版本为功能闭环实现，使用 GM scalar 读写保证语义正确；后续性能版本可将 `Aqk/Akk/w/u` 的局部矩阵计算迁移到 cube/UB tile，并将 `h_prev/v_new/h_next` 的 V 维 tile 常驻 UB。

## 4. Backward L0 设计

### 4.1 L2 编排

`aclnnChunkKdaBwd` 是 backward L2 入口。它在 aclnn executor 内部完成：

```text
ChunkKdaFwd(...) -> recomputed o/final_state/Aqk/Akk/w/u/qg/kg/v_new/h
ChunkKdaBwd(...) -> dq/dk/dv/dbeta/dgk/dh0
ViewCopy(...)    -> user outputs
```

因此 backward 不依赖 torch autograd 拼接，也不要求 Python 保存全部中间量。

### 4.2 `ChunkKdaBwd` 覆盖范围

`ChunkKdaBwd` 是一个反向融合 L0 kernel，覆盖原先拆分图中的以下逻辑：

- `d_o + Aqk + v_new -> dAqk, d_v_new(local)`。
- `dht + kg + qg + w + d_o + d_v_new -> dh0`，即 KDA 版本的 `bwd_dhu`。
- `Akk/w/u` 反传到 `k/v/beta/gk`。
- `Aqk` 反传到 `q/k/gk`。
- GVA 下同一个 q-head 对应多个 value-head；kernel 以 `(sequence, q_head)` 为 task，内部顺序处理该 q-head 下的 value-head，避免 `dq/dk` 写冲突。

### 4.3 `dh0` 数据依赖处理

KDA 的反向状态递推需要先使用当前 chunk 的旧 `d_state` 计算完整 `d_h_prev`，然后再写回给前一个 chunk。实现中按 V 维列处理：

```text
for r in V:
    new_state[d] = f(old_d_state[:, r], qg, kg, w, d_o, d_v_new)
    dh0[:, r] = new_state
```

这样避免 `dh0` 既作为工作区又作为输出时出现边读边覆盖。

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

- `chunk_size` 支持 `32/64/128`。其中 `32/64` 对齐三方 KDA，`128` 用于 NPU 侧大 chunk 场景。
- `K <= 256`。
- `V` 已覆盖 `256`。
- `HV % H == 0`。
- `gk/beta` 当前要求 float32。
- `safe_gate=True`、`transpose_state_layout=True` 当前显式报错，避免静默偏离三方语义。

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

当前实现已经完成 KDA 正反向功能闭环和精度验证。面向吞吐优化时，建议继续按以下方向演进：

- forward 将 `Aqk/Akk/w/u` 的块内矩阵计算迁移到 cube，并将 `h_prev/v_new/h_next` 的 V tile 驻留 UB。
- backward 将 `d_v_new`、`dh`、`dq/dk` 分解为 cube 主路径和 vec 辅助路径，通过 UB ping-pong 降低 GM 往返。
- 对 `gk/beta` 增加 L2 cast 或 raw-gate L0，使 BF16 gate 输入、`use_gate_in_kernel`、`safe_gate` 与 fla-org 高层接口进一步对齐。
