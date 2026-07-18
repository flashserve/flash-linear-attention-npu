# Kimi Delta Attention（KDA）正向 AscendC 算子设计

> **注意：本文第 4 节记录的是当前历史实现，不是推荐架构。** `stage=1/3/2` 复用同一 L0、直调通路暴露
> stage 语义、L2 拼接 Cast 均违反当前开发规范。整改应收敛为一个完整入口下的两个语义 phase，在 L0
> 内闭合必要的全核同步和 `TPipe::Reset()`，并把 cast 融入 kernel。阶段边界可参考
> [ops-nn PR #4803](https://gitcode.com/cann/ops-nn/pull/4803/diffs)，但必须按 KDA 自身参与核和状态依赖重新证明。

## 1. 范围

本文档说明本代码合入请求（Pull Request, PR）中 KDA 正向算子的实现设计。目标是提供一套支持 `gk` 的 AscendC 算子栈，并在语义完全一致的地方复用已有 Gated DeltaNet（GDN）正向状态传播逻辑。

当前 PR 范围：

- 新增 KDA 正向 `ChunkKdaFwd` AscendC 算子内核/算子接口层（L0/L2）算子。
- 新增辅助算子 `KdaLayoutSwap12` 和 `KdaGateCumsum`。
- 为 `ChunkGatedDeltaRuleFwdH` 增加可选 `gk` 入参，因为 KDA 复用 GDN 的状态传播。
- 新增 PyTorch 自定义接口 `npu_chunk_kda_fwd` 和参考测试。
- 验证 `K=128`、`V=128/256`、`chunk_size=64/128` 下 dense batch-sequence-head-dim/token-head-dim（BSND/TND）兼容输入，以及 batch-head-sequence-dim/head-token-dim（BNSD/NTD）直通性能布局。

不在当前 PR 范围：

- KDA 反向算子。
- 为非 chunk 对齐变长序列单独设计的高吞吐 partial-chunk 模板。当前实现已支持 partial chunk 的正确性，并采用完整 tile 补中性值计算、只回写有效区；其性能仍有进一步优化空间。

## 2. 对标语义

实现对齐三方对标实现 fla-org 的 KDA 正向分解：

```text
Aqk[i, j] = tril(q_i * k_j * exp2(g_i - g_j)) * scale
Akk       = inv(I + tril(k_i * k_j * exp2(g_i - g_j) * beta_i, -1))
w         = Akk @ (k * beta * exp2(g))
u         = Akk @ (v * beta)
v_new     = u - w @ h_prev
h_next    = exp2(g_last) * h_prev + kg_state @ v_new
o         = qg @ h_prev * scale + Aqk @ v_new
```

其中：

- `gk` 是 log2 空间下 key gate 的累积值。
- `qg` 是 `q * exp2(gk)` 的中间张量。
- `kg` 不是 `gk` 的笔误，而是 key-gated k 中间张量。kernel 中保存的 `kg = k * exp2(-gk)`；状态更新时结合当前 chunk 的 `g_last` 后，数学上等价于 `kg_state = k * exp2(g_last - gk)`。
- AscendC 向量流水（vector pipe）中使用 `exp(x * ln2)` 实现 `exp2(x)`。
- `final_state` 遵循三方对标实现和 GDN 语义，固定为 `float32`，即使 `q/k/v/o` 是 16 位浮点（fp16）或 Brain Floating Point 16（bf16）。

### 2.1 Gate Cumsum 语义

`KdaGateCumsum` 负责把模型侧 raw gate 转成 KDA 主算子消费的 `gk`。当 `use_gate_in_kernel=false` 时，输入 `g` 已经是 step gate，kernel 仅做 chunk 内 log2 空间累加：

```text
gk[b, t, hv, d] = cumsum(g[b, t, hv, d] * rcp_ln2, within_current_chunk)
```

当 `use_gate_in_kernel=true && safe_gate=true` 时，对齐三方对标实现的 safe gate raw path：

```text
x       = (g_raw + dt_bias[hv, d]) * exp(A_log[hv])
gate    = lower_bound * sigmoid(x)
gk      = cumsum(gate * rcp_ln2, within_current_chunk)
```

注意事项：

- `gk` 是 chunk 内累积，不跨 chunk 继续累加。`g_i - g_j` 的使用范围在 chunk 内，因此合法 `gk` 在同一 chunk 内应整体单调不增，且 causal 下 `g_i - g_j <= 0`。
- `safe_gate` 下每步 gate 位于 `[lower_bound, 0]`。默认 `lower_bound=-5` 时，一个 `chunk_size=64` 的 chunk 内 `gk` 可能到 `-300~-460`，这是合法值域。
- 如果某个 chunk 尾行被错误写成 0，会出现 `g_i - g_j` 正跳几百，随后 `exp2(g_i - g_j)` 溢出或产生极大中间值。这类问题的根因通常在 `KdaGateCumsum` 或同步/写回生命周期，不应先通过收紧输入 range、把 cube 改成 scalar 或调阈值规避。
- `safe_gate` 不应作为 AIV 热路径 runtime 分支，也不应依赖 tiling key 组合编码。`KdaGateCumsum` 参考 GDN 入口方式：host tiling 写入 `dataType/safeGate` 字段，kernel 入口选择 `<T, SAFE_GATE>` 模板实例，具体计算路径由编译期常量裁剪。

定位 gate cumsum 问题时，应先使用 gate-only 对比，而不是直接看 `o/final_state`：

```python
gk_npu = fla_npu.ops.ascendc.kda_gate_cumsum(
    g_raw, chunk_size,
    A_log=a_log,
    dt_bias=dt_bias,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
    layout="BSND",
)
gk_ref = reference_safe_gate_chunk_cumsum(g_raw, a_log, dt_bias, chunk_size)
torch.testing.assert_close(gk_npu.cpu(), gk_ref.cpu(), rtol=2e-3, atol=2e-3)
```

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
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False,
)
```

`layout` 是显式字符串参数，只接受全大写 `BSND`、`BNSD`、`TND`、`NTD`。默认值为 `BSND`，非 BSND 输入必须显式传入对应 layout；不再根据 shape 自动推导 layout。

支持的内存排布（layout）语义：

- BSND：`q/k: [B, T, H, K]`，`v: [B, T, HV, V]`，`gk: [B, T, HV, K]`，`beta: [B, T, HV]`。
- BNSD：`q/k: [B, H, T, K]`，`v: [B, HV, T, V]`，`gk: [B, HV, T, K]`，`beta: [B, HV, T]`。
- TND：`q/k: [T, H, K]`，`v: [T, HV, V]`，`gk: [T, HV, K]`，`beta: [T, HV]`。
- NTD：`q/k: [H, T, K]`，`v: [HV, T, V]`，`gk: [HV, T, K]`，`beta: [HV, T]`。

其中 `B/T/H/HV/K/V` 分别表示 batch、token 序列长度、query/key head 数、value head 数、key 维度和 value 维度。

BNSD 和 NTD 是性能布局，适用于上游 causal conv 已经完成数据排布转换的流水线。BSND 和 TND 是兼容布局，进入 kernel 前通过 `KdaLayoutSwap12` 转成内部布局。

当前 TND 兼容布局仅支持 `H=1` 的 rank3 输入。多 K head 的 rank3 输入必须使用 NTD 性能布局 `[H, T, D]`；host 侧会直接拦截 `layout=TND && H>1`，避免进入 `fwd_h` kernel 后触发非法访存。当前 `H/HV` 均要求不超过 128。

变长序列单次调用最多支持 1024 条序列，不限制单条序列或单次调用的 chunk 数。
kernel 的 tiling 仅保存每条序列的起止位置与累计 chunk 偏移，通过二分定位当前 chunk，避免把
逐 chunk metadata 塞入 tiling。显式传入的 `chunk_indices` 必须是按 sequence-major 排列的规范序列
`(seq_id, local_chunk_id)`；省略时由稳定 Python 入口和 L2 接口按同一规则生成。空序列仍按
`cu_seqlens` 非递减语义支持，不占用 chunk。

支持的数据类型（dtype）语义：

- `q/k/v/o/Aqk/Akk/w/u/qg/kg/v_new/h`：根据张量角色跟随 `q` 或 `v` 的 dtype，算子注册覆盖 `fp16`、`bf16` 和 32 位浮点（fp32）；其中 `kg` 表示 key-gated k 中间张量，不是 `gk` 输入。
- `g`：对齐三方对标实现的 gate 累积输出槽，当前由输入 `gk` 转为 `fp32` 后返回。
- `gk/beta`：PyTorch 层接受 `fp32` 或 `bf16`，进入 `ChunkKdaFwd` 前统一 cast 到 `fp32`。
- `initial_state/final_state`：固定为 `fp32`。

返回值顺序对齐三方对标实现：

```python
o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state = \
    torch.ops.npu.npu_chunk_kda_fwd(...)
```

其中 `initial_state` 输出为预留槽。未传入时返回空 tensor；传入时透传输入 `initial_state`，便于后续扩展到更完整的 state 语义。

预留或拦截：

- 当前 PR 中 PyTorch wrapper 拦截 `safe_gate=True`。
- 当前 PR 中 PyTorch wrapper 拦截 `transpose_state_layout=True`。

## 4. 现状 L2 组合设计（反面样例）

`aclnn_chunk_kda_fwd.cpp` 中的 L2 实现流程：

1. 将所有输入处理为 contiguous。
2. 校验显式 `layout` 参数和输入 rank/shape 是否一致，不做 layout 自动推导。
3. 对 BNSD/NTD，NTD reshape 为 `[1, HV, T, D]` view 后直接进入 kernel，不触发布局转换。
4. 对 BSND/TND，TND reshape 为 `[1, T, H, D]` 后通过 `KdaLayoutSwap12` 转成 BNSD。
5. 如有需要，将 `gk/beta` cast 到 `fp32`。
6. 校验并进入 split forward 路径。公开接口不再使用 monolithic `stage=0` scalar 路径兜底：
   - `q/k/v` 必须同为 `fp16` 或 `bf16`，`chunk_size` 必须为 `64` 或 `128`。
   - `K/V` 必须不小于 16 且按 16 对齐，当前 `V` 上限为 256；交付验证覆盖 `K=128, V=128/256`。
   - 求逆使用固定 16x16 对角块和 64x64 cube tile；`chunk_size=128` 继续用分块合并，不扩大单次 UB 驻留矩阵。
   - 不满足 cube 模板的 shape 由 host 明确拦截，不能回落到 scalar/逐元素计算。
7. 对 BNSD/NTD，split 路径的临时输出 copy 回相同 layout 的用户输出。对 BSND/TND，将 BNSD 中间结果转回公开输出 layout。

split 路径包含三个 `ChunkKdaFwd` 阶段以及一次 GDN 状态传播：

```text
stage 1: 准备 qg/kg_seed/w_seed/Aqk/Akk 并求解 chunk 内部项
stage 3: 消费 Akk/w_seed/v_new_seed，使用 cube 生成最终 w/u/kg
GDN fwd_h: 使用 kg、w、u、gk 更新 h/v_new/final_state
stage 2: 消费 h/v_new/Aqk/qg，计算 output cube 主路径和最终 o 行
```

这样设计的原因：

- KDA 正向天然分成“chunk 内矩阵项”和“chunk 间状态传播”两类依赖。`Aqk/Akk/w/u/qg/kg` 可以按 chunk 并行准备；`h/v_new/final_state` 依赖前一 chunk 状态，需要复用 GDN 的 `ChunkGatedDeltaRuleFwdH` 串起状态传播；最终 `o` 又依赖已经产出的 `h` 和 `v_new`。
- 如果把全部逻辑塞进一个大 kernel，需要在 `ChunkKdaFwd` 内重新实现 GDN 状态传播和输出后处理，既会重复已有可靠实现，也会把 cube 主路径、向量后处理、跨 chunk 依赖揉在一起，后续维护和定位都更困难。
- 对 `fp16/bf16, K=128, V=128/256, chunk_size=64/128` 场景，矩阵计算量足够大，拆分带来的 L0 调用和临时张量开销可以被 cube/向量主路径收益覆盖。不满足模板约束的 shape 明确报错，不能回落到 scalar/逐元素计算。

### 4.1 Stage 间数据依赖

KDA 正向的核心依赖关系如下：

```text
KdaGateCumsum(raw gate) -> gk

ChunkKdaFwd stage 1:
    输入 q/k/v/gk/beta
    产出 Aqk/Akk/qg/kg/w/u

ChunkKdaFwd stage 3:
    输入 stage 1 scratch
    产出 post-WU 后的 w/u/kg

ChunkGatedDeltaRuleFwdH:
    输入 kg/w/u/gk/initial_state
    产出 h/v_new/final_state

ChunkKdaFwd stage 2:
    输入 qg/Aqk/h/v_new
    产出 o
```

其中无 chunk 间依赖的部分可以按 `B/HV/chunk` 并行，存在 chunk 间依赖的是 `ChunkGatedDeltaRuleFwdH` 内部的状态传播。优化时不能把这两类依赖混在一起：

- `Aqk/Akk/qg/kg/w/u` 准备阶段允许多个 chunk 并行排布。
- `h/v_new/final_state` 状态传播必须尊重 chunk 顺序，除非引入明确的分段 prefix/scan 方案。
- `o` 只有在 `h/v_new` 已经可用后才能计算。

因此当前实现把 stage 拆开，是为了复用已有 GDN 状态传播，并让 cube 主路径和向量准备/后处理有清晰边界。后续如果进一步融合 stage，必须重新证明跨 stage 的生产者-消费者关系、workspace 生命周期和 cross-core flag 计数平衡。

目标 split forward 路径中，L2 对 L0 接口的拼接关系如下。公开目标 shape 统一进入 cube/vector 分阶段路径；不满足模板约束的 shape 由 host 明确拦截，不回落到逐元素计算。

```mermaid
flowchart LR
    In["输入: q/k/v/gk/beta<br/>initial_state/cu_seqlens/chunk_indices/layout"] --> Prep["L2: contiguous + layout 校验"]
    Prep -->|BSND/TND| SwapIn["L0: KdaLayoutSwap12<br/>转 BNSD 内部布局"]
    Prep -->|BNSD/NTD| ViewIn["L2: Reshape/View<br/>直通 BNSD 内部布局"]
    SwapIn --> CastGate["L0: Cast<br/>gk/beta -> fp32"]
    ViewIn --> CastGate

    CastGate --> S1["L0: ChunkKdaFwd(stage=1)<br/>产出 Aqk/Akk/w_pre/qg/kg_seed"]
    S1 --> AqkScale["L0: Muls<br/>Aqk * scale"]
    S1 --> S3["L0: ChunkKdaFwd(stage=3)<br/>post-WU 后处理"]
    S3 --> CastW["L0: Cast<br/>w -> fwd_h dtype"]
    S3 --> FwdH["L0: ChunkGatedDeltaRuleFwdH<br/>消费 kg/w/u/gk"]
    CastW --> FwdH
    FwdH --> S2["L0: ChunkKdaFwd(stage=2)<br/>消费 qg/Aqk/v_new/h"]
    S1 --> S2

    S2 --> OScale["L0: Muls<br/>o * scale"]
    OScale --> WriteO["L0: ViewCopy 或 KdaLayoutSwap12<br/>按公开 layout 回写"]
    WriteO --> OutO["输出: o"]
    AqkScale --> WriteAqk["L0: ViewCopy 或 KdaLayoutSwap12"]
    WriteAqk --> OutAqk["输出: Aqk"]
    S1 --> WriteAkk["L0: ViewCopy 或 KdaLayoutSwap12"]
    WriteAkk --> OutAkk["输出: Akk"]
    S3 --> WriteWu["L0: ViewCopy 或 KdaLayoutSwap12"]
    WriteWu --> OutWu["输出: w/u/qg/kg"]
    FwdH --> WriteState["L0: ViewCopy 或 KdaLayoutSwap12"]
    WriteState --> OutState["输出: h/v_new"]
    FwdH --> OutFinal["输出: final_state"]
    CastGate --> OutG["输出: g"]
    In --> OutInitial["输出: initial_state 预留槽"]
```

`ChunkGatedDeltaRuleFwdH` 扩展为可接收可选 `gk`。这是 KDA 复用状态传播的最小依赖；除该正向状态传播复用点外，不扩大修改 GDN 算子族。

BNSD/NTD split forward 中，未做最终后处理（raw）的 `o/Aqk/Akk/w/u/qg/kg/v_new/h` 存放在 executor 管理的临时张量里。L2 最后一步使用同 layout 的 `ViewCopy` 写入用户输出。这样可以避免把用户输出张量作为 custom L0 和逐元素 L0 算子（elementwise L0, elewise L0）之间的生产者-消费者中间张量，否则可能触发非法 tiling 或 workspace 推导。

### 4.2 L0 输入输出所有权

每个 stage 读取的中间张量必须出现在该 L0 算子的显式输入列表中，每个被生产的中间张量必须出现在显式输出列表中。不能把前一 stage 的输出仅作为后一 stage 的“输出参数”原地读取，也不能依赖 `return_intermediate=True` 恰好延长张量生命周期。

本实现为 stage 1 分配独立 `w_pre/kg_scratch`，stage 3 通过 `stage_qg/stage_aqk/stage_v_new` 显式消费 `w_pre/Akk/v_new_seed`，再写入独立的最终 `w/u/kg`。这样 executor 能构建真实的读后写依赖；`return_intermediate=False` 时也不会提前复用 workspace。

## 5. L0 Kernel 设计

`ChunkKdaFwd` 内部包含向量侧（AI Vector, AIV）和矩阵侧（AI Core/Cube, AIC）两类工作，通过跨核 flag 成对协作。

关键路径：

- Gate 乘积准备：
  - AIV 加载 `q/k/gk` 行。
  - AIV 计算 `qg = q * exp2(gk)`、`w seed = k * exp2(gk)`、`kg = k * exp2(-gk)`。
  - 行输入和输出使用 double-buffer 队列。
- `Aqk/Akk` raw score：
  - 目标 `K>=16` 路径使用 Catlass cube 通用矩阵乘（General Matrix Multiplication, GEMM）。
  - 不满足 cube 模板的 shape 在 host 拦截，kernel 不提供 scalar/vector 矩阵乘 fallback。
- `Akk` 求逆：
  - `BT=64/128` 均使用 16x16 对角块的矩阵链迭代（MCH）和 cube 分层合并；`BT=128` 不把整个 128x128 矩阵搬入 UB。
  - 求逆输入、迭代矩阵和合并结果全程为 IEEE FP32 cube 计算，编译期禁止 HF32。
  - 当前 PR 保留非 MXH 的 MCH + cube 求逆路径，优先保证输出语义和精度闭环。
  - MXH/L0C 驻留融合曾作为性能探索项实现，但在随机 `cu_seqlens` + 非零 `gk` 场景下仍存在尾块精度风险，本轮已下掉，不作为交付路径。
  - 非满 chunk 的尾块仍按完整 tile 计算，输入 pad 为中性值，回写只覆盖当前序列有效 token。
  - solve scratch 使用独立 user workspace，每个参与 core 配置 5 个 `BT x BT` FP32 slot，不再借用公开 `h` 输出。
- `w/u` 后处理：
  - `BT=64/128` 且 `K/V` 对齐时，使用 cube GEMM 计算 `Akk @ w` 和 `Akk @ v_new`。
  - AIV 准备 beta 缩放输入并完成 vector 后处理。
- Output：
  - AIC 计算 `qg @ h` 和 `Aqk @ v_new`。
  - 两项先分别写入 FP32 workspace；AIV 最后合并并在用户输出边界 cast 到 `fp16/bf16`，避免中间结果提前降精度。

对于 half/bfloat16 且 `K>=16` 的目标路径，tiling 使用完整 AICore block 数启动，确保每个 AIC producer 都有成对的 AIV consumer，反之亦然。这是 cross-core flag 计数保持平衡的必要条件。

### 5.1 Stage 1 双槽生产者消费者流水

Stage 1 的 gate factor 准备和 `Aqk/Akk` Catlass GEMM 没有 chunk 间数据依赖。实现按 score 行块建立深度为 2 的生产者消费者队列，使同一 chunk 内以及相邻 chunk 之间都能连续流水：

```text
AIV producer:
  等待 slot free
  -> 大块生成当前 score 行所需 qg/w 和 causal kg 前缀
  -> DataCopy 写入该 core 独立的 score scratch
  -> set slot ready

AIC consumer:
  wait slot ready
  -> Catlass cube 计算当前 Aqk/Akk score block
  -> set slot done/free

AIV producer:
  准备下一 score block；仅在轮转复用同一 slot 前 wait done
```

队列采用 `ready/free` 双向 flag，每个 slot 使用独立 flag id。score block 数向队列深度补齐，空 payload 也严格完成握手，避免 partial chunk 下生产者和消费者计数不平衡。队列排空后才允许后续 solve 协议复用相关 flag id。

score scratch 按参与 core 固定分配，而不是按序列长度或 chunk 数增长：

```text
score_scratch = core_num * 2 slots * 3 planes * BT * K * sizeof(T)
solve_scratch = core_num * 5 slots * BT * BT * sizeof(fp32)
```

在 `BT=128, K=128, bf16/fp16, core_num=20` 时，score scratch 约 3.75 MiB，solve scratch 约 6.25 MiB。超长序列只增加任务轮数，不增加这两部分峰值 workspace。

向量准备按连续 row range 分给两个 AIV subblock，并用大块 `DataCopy/DataCopyPad` 与 repeat vector 指令处理。不能用 `subBlockIdx >= curT` 提前退出，因为连续切分时 `curT=1` 的唯一有效行可能分给 AIV1；应先按 `subBlockNum` 校验参与者，再由 `[rowBegin, rowEnd)` 判断当前 subblock 是否有工作，同时保持 flag 协议闭环。

Stage 3 同样启用两个 AIV subblock，按连续有效行大块搬运和后处理，避免单 AIV 以及逐行小搬运成为长序列瓶颈。

### 5.2 AIV/MTE 生命周期

向量 kernel 中常见流水为：

```text
MTE2: GM -> UB
V:    UB vector compute
MTE3: UB -> GM
```

任一 UB buffer 被跨 pipe 复用前，必须表达真实依赖：

```cpp
// GM -> UB 后，V 读取前
DataCopy(ub, gm[offset], len);
SetFlag<HardEvent::MTE2_V>(event0);
WaitFlag<HardEvent::MTE2_V>(event0);

// V 计算后，MTE3 写回前
VectorCompute(ub, ...);
SetFlag<HardEvent::V_MTE3>(event1);
WaitFlag<HardEvent::V_MTE3>(event1);
DataCopy(gm[offset], ub, len);

// MTE3 仍在读 ub 时，V 或 MTE2 不能复用同一 ub
SetFlag<HardEvent::MTE3_V>(event2);
WaitFlag<HardEvent::MTE3_V>(event2);
SetFlag<HardEvent::MTE3_MTE2>(event3);
WaitFlag<HardEvent::MTE3_MTE2>(event3);
```

`KdaGateCumsum` 曾出现过一个典型同步缺口：每行 `acc` 写回 `gk` 后只等待了 `MTE3->MTE2`，没有等待 `MTE3->V`。当同一个 AIV core 写完一个 chunk 的最后一行后继续处理下一个 task，下一 task 开始时 `Duplicate(acc, 0)` 可能覆盖 MTE3 仍在读取的源 UB，导致上一 task 的最后一行被写成 0。

错误模式伪代码：

```cpp
for (task : tasks_on_this_core) {
    Duplicate(acc, 0);
    for (t in chunk) {
        Add(acc, acc, row);
        SetFlag<V_MTE3>(event);
        WaitFlag<V_MTE3>(event);
        DataCopy(gk[t], acc, k);
        SetFlag<MTE3_MTE2>(event);
        WaitFlag<MTE3_MTE2>(event);
        // 缺少 MTE3_V。下一 task 会复用 acc。
    }
}
```

修复模式：

```cpp
DataCopy(gk[t], acc, k);
SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
SetFlag<HardEvent::MTE3_V>(mte3ToVEvent);
WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent);
```

现象特征：

- gate-only 对比中，只坏固定 core 负责的 task 尾行，例如前若干 chunk 的 `t=chunk_end-1`。
- 坏值常为 0 或旧 buffer 值；前面行与 CPU reference 基本一致。
- `safe_gate` 大值域下，下游 `Aqk/Akk/w/kg` 出现 `1e23~1e35` 级极大值或 NaN/Inf。
- `o/final_state` 是下游症状，真正第一现场是 `gk` 的逐元素 diff。

回归覆盖要求：

- 至少包含一个 `safe_gate=True`、`K=128`、`chunk_size=64`、task 数大于 AIV core 数的 gate-only 用例。
- 该用例不需要非常长。`T=1536, HV=2, K=128` 已能稳定覆盖“一个 core 连续处理多个 task”的最后一行写回 hazard。
- 不能只跑 `T=40` 这类小 safe-gate 用例，也不能只跑外部已传入 `gk` 的 `chunk_kda_fwd` 主算子。

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

- 全局内存/统一缓冲（GM/UB）搬运使用 `DataCopy` 或 `DataCopyPad`。
- 目标路径避免使用 `GetValue` 和 `SetValue`。
- 复用大块 UB arena 做矩阵和向量 staging。
- Stage 1 每个 AIV tile 根据同时驻留的输入/输出张量数量计算最大连续行数，在 192 KiB UB 预算内尽量拉长单次搬运和 vector repeat；ping/pong slot 生命周期由事件闭合。
- Output cube 的 V 维按 128 列分块；`V=256` 连续执行两个 128 列 tile，避免扩大单次 L0/UB 驻留。
- `fwd_h` 的状态行 tile 按 `floor(32 KiB / (V * sizeof(ElementH)))` 计算，每个 ping/pong slot 最多 32 KiB；`V=256` 时每次搬运 32 行，两个 slot 分别位于独立 UB 区间，总 UB 不超过 192 KiB。
- `initial_state=None` 的变长序列首状态清零路径把整段 chunk offset metadata 一次 `DataCopyPad` 到 UB，再用于任务 offset 推导；不在逐 sequence/task 循环里下发单元素 GM 读取。

## 7. 验证结果

构建验证：

- `chunk_kda_fwd`、`chunk_gated_delta_rule_fwd_h`、`kda_layout_swap12` 和 `kda_gate_cumsum` custom package 构建通过。

精度验证：

以下验证 shape 中，`H_K/H_V` 表示 key/value head 数。
完整 C128/V256 用例、相对误差和性能原始汇总结论见
[`KDA_FORWARD_VALIDATION_REPORT.md`](KDA_FORWARD_VALIDATION_REPORT.md)。

- 小 shape BSND/TND/BNSD/NTD 单测对齐 `tests/reference/chunk_kda_reference.py` 并通过。
- 目标 sampled BNSD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` 通过：
  - `o`：`max_abs=4.26e-4`，`mean_abs=3.78e-5`。
  - `final_state`：`max_abs=1.09e-3`，`mean_abs=1.45e-5`。
- 目标 sampled BNSD `B=1, H_K=32, H_V=64, T=4096, K=128, V=128, chunk_size=64` 通过：
  - `o`：`max_abs=4.63e-4`，`mean_abs=4.01e-5`。
  - `final_state`：`max_abs=1.21e-4`，`mean_abs=9.68e-6`。
- 目标 sampled NTD `B=1, H_K=1, H_V=2, T=16384, K=128, V=128, chunk_size=64` 通过。
- `KdaGateCumsum` safe-gate 多 task 回归 `B=1, T=1536, H_V=2, K=128, chunk_size=64` 通过：
  - gate-only CPU reference 对比 `bad_count=0`。
  - 最大绝对误差约 `1.5e-4`，处于 `fp32` 累加和向量指数误差可接受范围。
- 极端 safe-gate 模型复现 `B=1, H_K=2, H_V=2, T=131072, K=128, V=128, chunk_size=64, bf16` 的 NPU 路径通过有限值检查：
  - `gk/o/final_state` 全量 finite。
  - 原先由 chunk 尾行错误写 0 引发的 `exp2(g_i - g_j)` 放大链路已消除。
- `chunk_size=128` 的 C9-C12、C26、C30 规格，以及 `V=256` 的 C21-C24、C27-C29、C31-C33 规格完成 sampled CPU FP32 reference 对比和 `ct viz -wl 1 -sc 100000` 可视化：
  - 16 个重点用例 `o` 的 cosine 范围为 `0.9999845~0.9999936`。
  - `max_abs` 范围为 `5.21e-6~1.55e-5`，未观察到块状、条纹状或序列边界相关的结构性误差。
  - 覆盖 BF16、BSND/NTD、定长/变长序列、随机非对齐尾块、`chunk_size=64/128` 和 `V=128/256`。
- `return_intermediate=False` 与 `return_intermediate=True` 在相同输入下逐位一致；两者均与 CPU FP32 reference 对齐。该用例专门看护 L0 中间量生命周期和 workspace 复用依赖。

性能验证使用 `msopprof --aic-metrics=BasicInfo`：

下表时间均为设备侧 kernel duration，不使用 Python wall time。`总计` 包含表中三段 `ChunkKdaFwd`、
`ChunkGatedDeltaRuleFwdH`，BSND C9 还包含 6 次必要的 layout swap。

| 用例 | stage 1 | stage 3 | fwd_h | stage 2 | layout | 优化前总计 | 优化后总计 | 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BNSD `B=1,H_K=1,H_V=2,T=16384,K=V=128,chunk_size=64` | 1.512 ms | 0.190 ms | 1.385 ms | 0.444 ms | 0 | 4.751 ms | 3.531 ms | 25.7% |
| NTD `B=1,H_K=H_V=32,T=65536,K=V=128,chunk_size=64` | 93.123 ms | 7.029 ms | 5.505 ms | 21.991 ms | 0 | 206.142 ms | 127.648 ms | 38.1% |
| BSND C9 `B=64,H_K=H_V=8,T=2048,K=V=128,chunk_size=128` | 42.374 ms | 2.785 ms | 1.754 ms | 11.111 ms | 10.011 ms | 114.572 ms | 68.035 ms | 40.6% |
| NTD C33 `B=1,H_K=16,H_V=48,T=8999,K=128,V=256,chunk_size=128` | 19.064 ms | 1.413 ms | 1.619 ms | 5.114 ms | 0 | 48.621 ms | 27.210 ms | 44.0% |

Stage 3 双 AIV 大块搬运/后处理相对优化前降低 `84.8%~92.2%`；Stage 1 两槽 score
生产者消费者流水降低 `10.0%~31.7%`。在三个长序列/大 shape 中，Stage 1 的 AIC
`wait_id4` 平均等待分别由 `29.94/20.43/12.26 ms` 降到 `14.92/7.20/3.40 ms`，说明
AIV 准备和 Catlass 消费已经形成有效重叠。

`fwd_h` 的变长序列 chunk offset 元数据由逐任务 GM 标量读取改为单次 `DataCopyPad` 搬入 UB，
随后在 UB 中索引；目标 NTD 长序列的 `fwd_h` 从 `6.545 ms` 降至 `5.505 ms`。

当前剩余主瓶颈是 Stage 1，占优化后长序列链路的约 `62%~72%`；下一轮应优先减少 score
scratch 往返、score block 控制开销和 solve 串行段。BSND 大 head 场景还需要继续降低 layout
转换成本。不能把矩阵数值计算迁到 scalar/vector 路径来换取局部计时改善。

已知验证边界：

- 高 `K/V` 的非 chunk 对齐 `cu_seqlens` 已改为由 L2 一次性规范化 chunk 元数据，并通过 tiling 下发紧凑索引；kernel 热点循环不再逐项从 GM 搬运 `int64` 元数据。`T=131072, H_K=H_V=2, K=V=128, chunk_size=64` 的 BF16 模型形状已覆盖非对齐尾块、非空 `initial_state` 和 `initial_state=None`，未再出现 AIV timeout。
- 该紧凑 tiling 元数据路径单次调用最多支持 1024 条序列，不限制单条序列或单次调用的
  chunk 数；序列长度不会因为 tiling 容量被截断。
- `return_intermediate=True` 已覆盖 BSND、BNSD、TND、NTD、FP16/BF16、尾块和模型长序列；
  `Aqk/Akk/w/u/qg/kg/v_new/h` 的无效上三角区域按接口语义清零。

## 8. 开发与验证闭环

KDA forward 的开发应按“语义 -> 结构 -> 单算子 -> 组合 -> 精度 -> 性能 -> 回归”推进。推荐流程：

1. 先对齐三方对标实现的数学语义，确认 `Aqk/Akk/w/u/qg/kg/v_new/h/o/final_state` 的公式、dtype 和返回顺序。
2. 再对齐本仓可复用 NPU 模块：
   - GDN `ChunkGatedDeltaRuleFwdH` 用于跨 chunk 状态传播。
   - GDN `solve_tri`/MCH 设计用于三角求逆思路。
   - Catlass cube GEMM 用于矩阵主路径。
   - causal conv layout 转换思路用于 BSND/TND 与 BNSD/NTD 的边界设计。
3. 每增加一个 L0/L2 拼接点，先做小 shape reference 对比，再做目标 shape sampled 对比。
4. 出现精度异常时，必须先定位第一处偏差：
   - `gk` 偏：先看 `KdaGateCumsum`。
   - `Aqk/Akk` 偏：先看 gate factorization、mask、solve。
   - `v_new/h/final_state` 偏：先看 GDN 状态传播、initial_state、chunk 顺序。
   - `o` 偏：先看 `qg @ h` 和 `Aqk @ v_new` 两项是否分别对齐。
5. 出现 NaN/Inf 时，不要先改阈值或输入 range。先检查是否存在：
   - causal mask 前已经 `inf * 0 -> nan`。
   - `g_i - g_j` 非法正跳。
   - UB/GM 写回生命周期未闭合。
   - partial chunk 脏行进入 cube/solve。
6. 性能优化只能在精度第一现场明确后进行。目标形状中矩阵类计算必须保持 cube 主路径，不能为了修精度退回 scalar/逐元素循环。
7. 每个已修复问题必须补“能稳定触发该问题”的回归用例，而不是只补更大的随机全量用例。

## 9. 后续扩展计划

建议后续工作：

1. 继续降低长序列 stage 1 的 GM score scratch 往返，并基于 profiling 评估 Catlass score/solve tile；当前两槽深度已能形成稳定流水，不能在没有流水证据时盲目扩大队列和 workspace。
2. 将 KDA 反向算子放到独立 PR 中实现，并复用相同的 `gk` 和 state dtype 约定。
3. 补充 race、memory、init 和 sync 内存/同步检查工具（sanitizer）验证。
