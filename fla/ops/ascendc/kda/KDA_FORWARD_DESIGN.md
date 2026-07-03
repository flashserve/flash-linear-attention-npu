# KDA 正向 AscendC 算子设计

## 1. 范围

本文档说明本 PR 中 KDA 正向算子的实现设计。目标是提供一套支持 `gk` 的 AscendC 算子栈，并在语义完全一致的地方复用已有 GDN 正向状态传播逻辑。

当前 PR 范围：

- 新增 KDA 正向 `ChunkKdaFwd` AscendC L0/L2 算子。
- 新增辅助算子 `KdaLayoutSwap12` 和 `KdaGateCumsum`。
- 为 `ChunkGatedDeltaRuleFwdH` 增加可选 `gk` 入参，因为 KDA 复用 GDN 的状态传播。
- 新增 PyTorch 自定义接口 `npu_chunk_kda_fwd` 和参考测试。
- 验证 `K=128, V=128, chunk_size=64` 下 dense BSND/TND 兼容输入，以及 BNSD/NTD 直通性能布局。

不在当前 PR 范围：

- KDA 反向算子。
- `V=256` 性能模板。`V=256` 应使用独立模板，因为 UB/L1 预算和 cube tiling 与 `V=128` 不同。
- 高吞吐的非 chunk 对齐 varlen partial chunk 路径。公开 API 保留 `cu_seqlens` 和 `chunk_indices`，但高 `K/V` 非 chunk 对齐 varlen 需要专门的 partial chunk 同步路径，验证完成前不应宣称已优化。

## 2. 对标语义

实现对齐三方对标实现 fla-org 的 KDA 正向分解：

```text
Aqk[i, j] = tril(q_i * k_j * exp2(g_i - g_j)) * scale
Akk       = inv(I + tril(k_i * k_j * exp2(g_i - g_j) * beta_i, -1))
w         = Akk @ (k * beta * exp2(g))
u         = Akk @ (v * beta)
v_new     = u - w @ h_prev
h_next    = exp2(g_last) * h_prev + kg @ v_new
o         = qg @ h_prev * scale + Aqk @ v_new
```

其中：

- `gk` 是 log2 空间下 key gate 的累积值。
- AscendC vector pipe 中使用 `exp(x * ln2)` 实现 `exp2(x)`。
- `final_state` 遵循三方对标实现和 GDN 语义，固定为 `float32`，即使 `q/k/v/o` 是 `fp16` 或 `bf16`。

## 3. 对外接口

PyTorch API：

```python
torch.ops.npu.npu_chunk_kda_fwd(
    q,
    k,
    v,
    gk,
    beta,
    scale,
    chunk_size,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False,
)
```

支持的 layout 语义：

- BSND：`q/k: [B, T, H, K]`，`v: [B, T, HV, V]`，`gk: [B, T, HV, K]`，`beta: [B, T, HV]`。
- BNSD：`q/k: [B, H, T, K]`，`v: [B, HV, T, V]`，`gk: [B, HV, T, K]`，`beta: [B, HV, T]`。
- TND：`q/k: [T, H, K]`，`v: [T, HV, V]`，`gk: [T, HV, K]`，`beta: [T, HV]`。
- NTD：`q/k: [H, T, K]`，`v: [HV, T, V]`，`gk: [HV, T, K]`，`beta: [HV, T]`。

BNSD 和 NTD 是性能布局，适用于上游 causal conv 已经完成数据排布转换的流水线。BSND 和 TND 是兼容布局，进入 kernel 前通过 `KdaLayoutSwap12` 转成内部布局。

支持的 dtype 语义：

- `q/k/v/o/Aqk/Akk/w/u/qg/kg/v_new/h`：根据张量角色跟随 `q` 或 `v` 的 dtype，算子注册覆盖 `fp16`、`bf16` 和 `fp32`。
- `gk/beta`：PyTorch 层接受 `fp32` 或 `bf16`，进入 `ChunkKdaFwd` 前统一 cast 到 `fp32`。
- `initial_state/final_state`：固定为 `fp32`。

预留或拦截：

- 当前 PR 中 PyTorch wrapper 拦截 `safe_gate=True`。
- 当前 PR 中 PyTorch wrapper 拦截 `transpose_state_layout=True`。

## 4. L2 组合设计

`aclnn_chunk_kda_fwd.cpp` 中的 L2 实现流程：

1. 将所有输入处理为 contiguous。
2. 根据 rank 和 shape 一致性推导 BSND/BNSD/TND/NTD。歧义场景优先选择 head 维更短的 layout。
3. 对 BNSD/NTD，NTD reshape 为 `[1, HV, T, D]` view 后直接进入 kernel，不触发布局转换。
4. 对 BSND/TND，TND reshape 为 `[1, T, H, D]` 后通过 `KdaLayoutSwap12` 转成 BNSD。
5. 如有需要，将 `gk/beta` cast 到 `fp32`。
6. 根据场景选择 dispatch：
   - 大 shape half/bfloat16 且 `chunk_size=64` 时走 split forward 路径；
   - 小 shape 或 `fp32` 场景走 monolithic `ChunkKdaFwd` 路径。
7. 对 BNSD/NTD，split 路径的临时输出 copy 回相同 layout 的用户输出。对 BSND/TND，将 BNSD 中间结果转回公开输出 layout。

split 路径包含三个 `ChunkKdaFwd` 阶段以及一次 GDN 状态传播：

```text
stage 1: 准备 qg/kg/w/u/Aqk/Akk 输入并求解 chunk 内部项
GDN fwd_h: 使用 kg、w、u、gk 更新 h/v_new/final_state
stage 2: 计算 output cube 主路径和最终 o 行
stage 3: 启用 post-WU cube 时后处理 w/kg/u
```

`ChunkGatedDeltaRuleFwdH` 扩展为可接收可选 `gk`。这是 KDA 复用状态传播的最小依赖；除该正向状态传播复用点外，不扩大修改 GDN 算子族。

BNSD/NTD split forward 中，raw `o/Aqk/Akk/w/u/qg/kg/v_new/h` 存放在 executor 管理的临时张量里。L2 最后一步使用同 layout 的 `ViewCopy` 写入用户输出。这样可以避免把用户输出张量作为 custom L0 和 elewise L0 之间的生产者-消费者中间张量，否则可能触发非法 tiling 或 workspace 推导。

## 5. L0 Kernel 设计

`ChunkKdaFwd` 内部包含 AIV 和 AIC 两类工作，通过跨核 flag 成对协作。

关键路径：

- Gate 乘积准备：
  - AIV 加载 `q/k/gk` 行。
  - AIV 计算 `qg = q * exp2(g)`、`w seed = k * exp2(g)`、`kg = k * exp2(-g)`。
  - 行输入和输出使用 double-buffer 队列。
- `Aqk/Akk` raw score：
  - 目标 `K>=16` 路径使用 Catlass cube GEMM。
  - scalar fallback 仅作为非目标 shape 的 correctness fallback，不能作为目标性能路径。
- `Akk` 求逆：
  - 完整 `BT=64` 使用 cube 辅助的 blocked matrix-chain iteration。
  - solve scratch 在状态传播消费 `h` 之前暂存在 `h` workspace slot 中。
- `w/u` 后处理：
  - 完整 `BT=64` 且 `K/V` 对齐时，使用 cube GEMM 计算 `Akk @ w` 和 `Akk @ v_new`。
  - AIV 准备 beta 缩放输入并完成 vector 后处理。
- Output：
  - AIC 计算 `qg @ h` 和 `Aqk @ v_new`。
  - AIV 合并 state contribution 和 local contribution，得到 `o`。

对于 half/bfloat16 且 `K>=16` 的目标路径，tiling 使用完整 AICore block 数启动，确保每个 AIC producer 都有成对的 AIV consumer，反之亦然。这是 cross-core flag 计数保持平衡的必要条件。

## 6. 内存与 Layout

L2 内部使用 BNSD，因为该 layout 在 kernel 使用的 head 和 token 维度上读写更连续：

```text
BSND/TND 公开输入 -> KdaLayoutSwap12 -> BNSD 内部张量
BNSD/NTD 直通输入 -> reshape/view     -> BNSD 内部张量
```

状态 layout：

```text
h:           kernel 内为 [B, HV, NT, K, V]
final_state: [seq_num, HV, K, V]，fp32
```

UB 使用原则：

- GM/UB 搬运使用 `DataCopy` 或 `DataCopyPad`。
- 目标路径避免使用 `GetValue` 和 `SetValue`。
- 复用大块 UB arena 做矩阵和向量 staging。
- `V=128` 和 `V=256` 模板保持独立，因为二者的 UB 驻留和 tile 复用计划不同。

## 7. 验证结果

构建验证：

- `chunk_kda_fwd`、`chunk_gated_delta_rule_fwd_h`、`kda_layout_swap12` 和 `kda_gate_cumsum` custom package 构建通过。

精度验证：

- 小 shape BSND/TND/BNSD/NTD 单测对齐 `tests/reference/chunk_kda_reference.py` 并通过。
- 目标 sampled BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` 通过：
  - `o`：`max_abs=4.26e-4`，`mean_abs=3.78e-5`。
  - `final_state`：`max_abs=1.09e-3`，`mean_abs=1.45e-5`。
- 目标 sampled BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64` 通过：
  - `o`：`max_abs=4.63e-4`，`mean_abs=4.01e-5`。
  - `final_state`：`max_abs=1.21e-4`，`mean_abs=9.68e-6`。
- 目标 sampled NTD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` 通过。

性能验证使用 `msopprof --aic-metrics=BasicInfo`：

- BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64`：相关 KDA+GDN 平均耗时 `3.05 ms`；`KdaLayoutSwap12` 次数为 `0`。
- BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64`：相关 KDA+GDN 平均耗时 `13.60 ms`；`KdaLayoutSwap12` 次数为 `0`。

已知验证边界：

- 高 `K/V` 的非 chunk 对齐 `cu_seqlens` 在当前 prototype 中可能触发 kernel timeout。在 dedicated partial-chunk 路径实现并验证前，不宣称已优化 varlen high `K/V` 支持。
- 当前 PR 有意不验证 `V=256`。

## 8. 后续扩展计划

建议后续工作：

1. 增加 dedicated partial-chunk varlen 路径，保证每个参与 subblock 的 AIC/AIV flag 计数平衡，并确保 cube 路径不会消费有效 chunk 外的脏行。
2. 增加 `V=256` 模板，明确 UB/L1 驻留计划，而不是拉伸 `V=128` 模板。
3. 将 KDA 反向算子放到独立 PR 中实现，并复用相同的 `gk` 和 state dtype 约定。
4. varlen partial 路径稳定后，补充 race、memory、init 和 sync sanitizer 验证。
