# ChunkGatedDeltaRuleFwdH 支持 gk 设计文档

## 1. 背景与目标

`ChunkGatedDeltaRuleFwdH` 当前负责计算 chunk 级隐藏状态 `h`、经 WY 修正后的 `v_new`，以及可选 `final_state`。现有实现只支持标量 gate `g`，`gk` 在 Python/ACLNN 接口中保留但被显式限制为 `None/nullptr`。

本设计目标是在不破坏现有 `g` 路径的前提下，为 `chunk_gated_delta_rule_fwd_h` 增加 `gk` 支持，使其与三方仓 `fla/ops/common/chunk_h.py` 中 `USE_GK` 分支的语义对齐：

- `g`：按 token/head 的标量衰减，作用在整个状态矩阵和 `v_new` token 维。
- `gk`：按 token/head/K 维的逐 Key 维衰减，作用在状态矩阵 K 维和参与状态更新的 `k`。
- `gk=None` 时保持当前行为和性能路径。
- `gk=0` 时结果应与未启用 `gk` 完全等价。

本设计只覆盖 `chunk_gated_delta_rule_fwd_h` 算子本身。若要在完整 Gated Delta Rule 前后向链路中启用 `gate_source=gk` 或 `g+gk`，还需要同步检查 `recompute_wu_fwd`、`chunk_fwd_o`、反向算子和示例脚本的 `gk` 约束。

## 2. 当前实现现状

相关文件：

- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_def.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/op_api/chunk_gated_delta_rule_fwd_h.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.*`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h.cpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/epilogue/block/block_epilogue_gdn_fwdh_update.hpp`

当前主要限制：

- Python API 已有 `gk` 参数，但 `FLANpuOpApi.cpp` 中要求 `gk` 必须为 `None`。
- ACLNN API 已有 `gkOptional` 参数，但 `CheckReservedOptions` 要求 `gkOptional == nullptr`。
- OpDef 输入列表没有 `gk`，当前输入顺序为 `k, w, u, g, inital_state, cu_seqlens, chunk_indices`。
- L0 `ChunkGatedDeltaRuleFwdH` 不接收 `gk`，AICore `OP_INPUT` 未下发 `gk`。
- kernel 入口形参未包含 `gk`，tiling data 中也没有 `hasGk/useGk` 或 `gk` workspace 信息。
- kernel 当前计算分两段：
  - C1/V1：`v_new = u - w @ h[i]`，并生成 `v_new_decay = v_new * exp(g_last - g_t)`。
  - C2/V2：`h[i+1] = exp(g_last) * h[i] + k^T @ v_new_decay`，末块可写 `final_state`。

## 3. 目标语义

当前实现 `useExp2=false`，kernel 内使用 `exp`。为避免同时改动 `useExp2` 语义，本文用 `E(x)` 表示当前算子采用的指数函数：

```text
E(x) = exp(x)
```

若后续支持 `useExp2=true`，可将 `E(x)` 切换为 `exp2(x)`，但不纳入本次范围。

对第 `c` 个 chunk，记：

- `S_c`: 当前 chunk 起始状态，形状 `[K, V]`
- `k_c`: 当前 chunk 的 key，形状 `[BT, K]`
- `w_c`: 当前 chunk 的 W，形状 `[BT, K]`
- `u_c`: 当前 chunk 的 U，形状 `[BT, V]`
- `g_c`: 当前 chunk 的标量 gate，形状 `[BT]`
- `gk_c`: 当前 chunk 的 key-wise gate，形状 `[BT, K]`
- `L`: 当前 chunk 实际 token 数，最后一个 chunk 可小于 `BT`

现有路径：

```text
v_new = u_c - w_c @ S_c
v_decay = v_new * E(g_c[L - 1] - g_c)[:, None]
S_{c+1} = E(g_c[L - 1]) * S_c + k_c^T @ v_decay
```

启用 `gk` 后，参考三方 `chunk_h.py` 的 `USE_GK` 分支，扩展为：

```text
v_new = u_c - w_c @ S_c

v_decay = v_new * E(g_c[L - 1] - g_c)[:, None]
k_decay = k_c * E(gk_c[L - 1, :][None, :] - gk_c)

S_decay = E(g_c[L - 1]) * E(gk_c[L - 1, :])[:, None] * S_c
S_{c+1} = S_decay + k_decay^T @ v_decay
```

关键约束：

- `hOut` 仍存储每个 chunk 起始处的 `S_c`，不存储衰减后的 `S_decay`。
- `vNewOut` 仍输出 `v_new = u - w @ S_c`，不包含 `g` 或 `gk` 衰减。
- `gk` 只影响状态跨 chunk 更新和 `final_state`，不改变 `v_new` 的数学定义。
- `gk=None` 时不创建 `k_decay` workspace，C2 继续读取原始 `k`。

## 4. 接口与数据布局设计

### 4.1 Python API

保持现有 `npu_custom.yaml` 签名不变：

```text
npu_chunk_gated_delta_rule_fwd_h(
    Tensor k,
    Tensor w,
    Tensor u,
    Tensor? g=None,
    *,
    Tensor? gk=None,
    Tensor? initial_state=None,
    bool? output_final_state=False,
    int? chunk_size=None,
    bool? save_new_value=True,
    int[]? cu_seqlens=None,
    int[]? chunk_indices=None,
    bool? use_exp2=False,
    bool? transpose_state_layout=False
) -> (Tensor, Tensor, Tensor)
```

新增行为：

- 删除 `gk is reserved and only None is supported` 检查。
- 当 `gk` 有值时，向 ACLNN 正常传入 `gk_`。
- `gk` 推荐形状为 `[B, HV, T, K]`，与 `w` 的 head 维对齐。
- 变长场景仍沿用当前 `B=1`、`cu_seqlens`、`chunk_indices` 约束。

### 4.2 ACLNN / L0 API

ACLNN GetWorkspaceSize 函数签名已有 `gkOptional`，无需改 ABI；只需放开参数检查并做连续化：

```cpp
if (params.gkOptional != nullptr) {
    DataContiguous(params.gkOptional, executorPtr);
}
```

L0 层函数需要新增 `gkOptional` 参数：

```cpp
const std::array<const aclTensor *, 3> ChunkGatedDeltaRuleFwdH(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *gkOptional,
    const aclTensor *initalStateOptional,
    ...
);
```

AICore 下发输入顺序调整为：

```text
k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices
```

注意：这是本次改造最容易错位的地方。OpDef、tiling 输入下标、L0 `OP_INPUT` 和 kernel 入口形参必须保持完全一致。

### 4.3 OpDef

在 `g` 后新增 optional 输入 `gk`：

```cpp
this->Input("gk")
    .ParamType(OPTIONAL)
    .DataType({...})
    .Format({...})
    .UnknownShapeFormat({...})
    .AutoContiguous();
```

首期建议约束：

- `gk.dtype` 与 `g.dtype` 保持一致。
- 支持 dtype 组合沿用 `g` 当前组合：`FLOAT`，或与主输入一致的 `FLOAT16/BFLOAT16`。
- 若后续需要支持 `g` 与 `gk` 不同 dtype，再新增独立 `GK_TYPE` 模板参数和 `gkDataType` tiling 字段。

### 4.4 Shape 检查

当 `gkOptional != nullptr` 时增加检查：

```text
rank(gk) == 4
gk.shape == [shapeBatch, HV, T, K]
```

其中：

- 定长：`shapeBatch = B`
- 变长：`shapeBatch = 1`
- `HV = u.shape[1] = w.shape[1] = hOut.shape[1]`
- `T = k.shape[2] = u.shape[2]`
- `K = k.shape[3] = w.shape[3]`

保留当前 GQA 约束：

```text
HV % HK == 0
headGroups = HV / HK
kHeadIdx = vHeadIdx / headGroups
```

也就是说，`k` 可按 key head 共享，但 `gk` 按 value head 生效，使同一 `kHeadIdx` 下不同 `vHeadIdx` 可以使用不同 key-wise 衰减。

## 5. Tiling 设计

### 5.1 输入下标调整

新增 `INPUT_GK_IDX` 后，下标建议调整如下：

```cpp
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_W_IDX = 1;
static constexpr size_t INPUT_U_IDX = 2;
static constexpr size_t INPUT_G_IDX = 3;
static constexpr size_t INPUT_GK_IDX = 4;
static constexpr size_t INPUT_INITIAL_STATE_IDX = 5;
static constexpr size_t INPUT_SEQLENS_IDX = 6;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
```

### 5.2 TilingData 新增字段

建议新增：

```cpp
TILING_DATA_FIELD_DEF(bool, useGk);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
```

若首期强制 `gk.dtype == g.dtype`，无需新增 `gkDataType`。若放开 dtype，则新增：

```cpp
TILING_DATA_FIELD_DEF(int64_t, gkDataType);
```

### 5.3 Workspace

新增 `k_decay` workspace，仅 `useGk=true` 时使用：

```text
kDecayWorkspace:
  shape per stream: [chunkSize, kHeadDim]
  logical layout for C2: column-major view [K, BT]
  element type: ElementK
  allocation size: aicCoreNum * chunkSize * kHeadDim * sizeof(float) * PING_PONG_STAGES
```

分配建议放在 `vUpdateWorkspaceOffset` 之后、`hWorkspaceOffset` 之前：

```text
vWorkspaceOffset
vUpdateWorkspaceOffset
kDecayWorkspaceOffset   // new, only meaningful when useGk=true
hWorkspaceOffset
numSeqWorkspaceOffset
numChunksWorkspaceOffset
```

沿用 `sizeof(float)` 计算 workspace 可以保持当前风格并预留对齐空间；实际 `GlobalTensor<ElementK>` 访问时只写入 `ElementK` 数据。

## 6. Kernel 设计

### 6.1 kernel 入口与 Init

kernel 入口改为：

```cpp
void chunk_gated_delta_rule_fwd_h(
    GM_ADDR k,
    GM_ADDR w,
    GM_ADDR u,
    GM_ADDR g,
    GM_ADDR gk,
    GM_ADDR inital_state,
    GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices,
    GM_ADDR h,
    GM_ADDR v_new,
    GM_ADDR final_state,
    GM_ADDR workspace,
    GM_ADDR tiling)
```

`GDNFwdHKernel::Init` 同步新增 `gk`，并初始化：

```cpp
bool useGk;
AscendC::GlobalTensor<ElementG> gmGk;
AscendC::GlobalTensor<ElementK> gmKDecayWorkspace;
```

### 6.2 Scheduler offset

`GDNFwdHOffsets` 新增：

```cpp
uint32_t gkOffset;
uint32_t kDecayWorkOffset;
```

offset 计算：

```text
gkOffset =
  (shapeBatchIdx * vNumHead * totalTokens
   + vHeadIdx * totalTokens
   + tokenOffset
   + chunkIdx * chunkSize) * kHeadDim

kDecayWorkOffset =
  (cubeCoreIdx * PING_PONG_STAGES + streamId) * chunkSize * kHeadDim
```

最后一个有效 token 的 `gk_last` 位置：

```text
gkLastOffset = gkOffset + (blockTokens - 1) * kHeadDim
```

### 6.3 V1 阶段生成 k_decay

当前 V1 做：

```text
v_new = u - w @ h
v_decay = v_new * E(g_last - g_t)
write v_decay to vUpdateWorkspace
signal vec1Done
```

启用 `gk` 后，V1 增加：

```text
k_decay = k * E(gk_last[None, :] - gk)
write k_decay to kDecayWorkspace
signal vec1Done after both v_decay and k_decay are ready
```

实现建议：

- 将 `gmK`、`gmGk`、`gmKDecayWorkspace` 传入 `BlockEpilogueGDNFwdHVnew`。
- `useGk=false` 时保持现有逻辑，不读取 `gk`，不写 `kDecayWorkspace`。
- `useGk=true` 时在 V1 内复用现有 UB 缓冲区，先完成 `v_decay` 和 `vNewOut` 写回，再复用 `calcUbTensor` 等 buffer 生成 `k_decay`。
- `vec1Done` 必须在 `k_decay` 写回后再置位，否则 C2 可能读到未完成的 workspace。

首期可利用当前约束 `K=128`、`chunkSize in {64, 128}` 控制 UB 压力。若未来放宽 K，需要将 `k_decay` 计算再按 K 维分块。

### 6.4 C2 阶段选择 K 输入

C2 当前读取原始 `gmK`：

```text
h_work = k^T @ v_decay
```

改为：

```text
if useGk:
    h_work = k_decay^T @ v_decay
else:
    h_work = k^T @ v_decay
```

布局建议：

- 原始 `k`：按 `[K, totalTokens]` column-major 视图读取。
- `k_decay` workspace：按 `[K, chunkSize]` column-major 视图读取。
- 两者都保持 `ElementK`，尽量复用现有 `BlockMmadKV` 模板。

### 6.5 V2 阶段状态衰减

当前 V2 做：

```text
S_decay = E(g_last) * S_c
S_{c+1} = S_decay + h_work
```

启用 `gk` 后改为：

```text
S_decay = E(g_last) * E(gk_last)[:, None] * S_c
S_{c+1} = S_decay + h_work
```

实现建议：

- 将 `gmGk` 和 `gkOffset` 传入 `BlockEpilogueGDNFwdHUpdate`。
- V2 已按 K 维拆给 AIV sub block，可以只读取本 sub block 对应的 `gk_last[mOffset : mOffset + mActualThisSubBlock]`。
- 将 `E(gk_last)` 广播到 `[mActualThisSubBlock, vHeadDim]` 后乘到 `calcUbTensor`。
- `useGk=false` 时保持当前只乘 `E(g_last)` 的路径。

### 6.6 final_state

`final_state` 应写入完成 `g` 与 `gk` 衰减、再加上 `k_decay^T @ v_decay` 后的状态：

```text
final_state = S_{last+1}
```

这与当前 `storeFinalState` 分支位置一致，只需确保 V2 的 `S_decay` 已包含 `gk_last`。

## 7. 兼容性与错误处理

保持不变：

- `save_new_value` 仍只支持 `true`。
- `use_exp2` 仍只支持 `false`。
- `transpose_state_layout` 仍只支持 `false`。
- `g` 仍要求非空，直到单独支持 `g=None`。
- `chunkSize` 仍只支持当前 kernel 覆盖的值。
- 变长模式仍沿用当前 `B=1` 约束。

新增错误处理：

- `gk` 非空但 rank 不是 4：返回参数错误。
- `gk.shape != [shapeBatch, HV, T, K]`：返回参数错误。
- `gk.dtype` 不在支持列表：返回参数错误。
- 首期若 `gk.dtype != g.dtype`：返回参数错误，并在错误信息中说明当前要求同 dtype。

## 8. 修改清单

建议按以下顺序实施：

1. Python wrapper
   - 删除 `FLANpuOpApi.cpp` 中 `gk` 必须为 `None` 的 `TORCH_CHECK`。
   - 保留 `gk_ = value_or_else(...)`，继续传给 ACLNN。

2. ACLNN/L0 host
   - 放开 `CheckReservedOptions` 中 `gkOptional == nullptr` 限制。
   - `ParamsDataContiguous` 中对非空 `gkOptional` 做 `Contiguous`。
   - L0 `ChunkGatedDeltaRuleFwdH` 函数声明与实现新增 `gkOptional`。
   - `OP_INPUT` 插入 `gkOptional`。

3. OpDef/tiling
   - OpDef 在 `g` 后增加 optional `gk`。
   - tiling 输入下标整体调整。
   - tiling data 增加 `useGk` 和 `kDecayWorkspaceOffset`。
   - workspace 增加 `k_decay` 区域。

4. kernel
   - kernel 入口和 `GDNFwdHKernel::Init` 增加 `gk`。
   - scheduler 增加 `gkOffset`、`kDecayWorkOffset`。
   - V1 增加 `k_decay` 生成逻辑。
   - C2 根据 `useGk` 切换 `k` 来源。
   - V2 增加 `E(gk_last)` 状态 K 维衰减。
   - `arch35` 目录下同名 kernel、scheduler、epilogue 文件同步修改。

5. 文档和测试
   - README 更新 `gkOptional` 参数说明。
   - PTA/torch 测试增加 `gk` case。
   - 示例脚本解除仅 `gate_source=g` 的限制前，先确认完整链路其它算子均支持 `gk`。

## 9. 测试设计

### 9.1 CPU reference

在 `forward_h_trans_cpu` 中新增 `gk` 逻辑：

```python
if gk is None:
    k_decay = k_sel
    state_decay = S * exp(g_last)
else:
    gk_last = gk_sel[actual_len - 1]          # [K]
    k_decay = k_sel * exp(gk_last[None, :] - gk_sel)
    state_decay = S * exp(g_last) * exp(gk_last)[:, None]

v_new = u_sel - w_sel @ S
v_decay = v_new * exp(g_last - g_sel)[:, None]
next_state = state_decay + k_decay.T @ v_decay
```

### 9.2 功能用例

必测：

- `gk=None`：对比旧版本，确保回归通过。
- `gk=zeros`：结果应与 `gk=None` 一致。
- `gk=random negative`：对比 CPU reference。
- `output_final_state=false/true`。
- `initial_state=None` 与 `initial_state` 非空。
- 定长与变长。
- `chunkSize=64/128`。
- `dtype=fp16/bf16`。
- `g.dtype=float32` 与 `g.dtype=input dtype`。
- `V=128/256`。
- `HV=HK` 与 `HV > HK` 的 grouped value heads。

建议补充：

- 最后一个 chunk 不满 `chunkSize`。
- 非连续输入触发 host contiguous。
- `gk` shape/dtype 不合法的负例。

### 9.3 精度验收

建议沿用现有 `data_compare_h.py` 阈值策略，并额外关注：

- `gk` 取值过大可能导致指数溢出，测试数据应主要使用非正值或接近现有 gate 生成范围。
- `gk=zeros` case 应作为严格回归基线。
- `final_state` 使用 fp32 时，与 `hOut`/`vNewOut` 的 dtype 差异要单独校验。

### 9.4 内存与同步验证

新增 `kDecayWorkspace` 后，需要重点验证：

- V1 写 `k_decay` 与 C2 读 `k_decay` 之间的 cross-core flag 顺序。
- ping/pong stage 的 `kDecayWorkOffset` 是否互不覆盖。
- `chunkSize=128, K=128` 下 UB 复用是否越界。
- 变长末块 `blockTokens < chunkSize` 时，`gk_last` 与 `k_decay` 写入长度是否正确。

如排查流水 hazard、读写冲突或疑似未初始化读取，优先使用 MindStudio Sanitizer：

- race 类问题使用 `mssanitizer --tool=racecheck`。
- 越界/泄漏类问题使用 `memcheck`。
- 未初始化类问题使用 `initcheck`。
- 同步类问题使用 `synccheck`。

验证结论只记录测试项和结果，不记录机器、用户、绝对路径、临时目录或日志路径。

## 10. 风险与备选方案

### 风险 1：插入 optional 输入导致下标错位

这是最高风险。必须保证以下位置输入顺序一致：

- OpDef 输入定义
- tiling input index
- ACLNN/L0 参数
- `OP_INPUT`
- kernel 入口形参
- kernel `Init`

建议在首个 host UT 中直接检查 `gk` 非空时 tiling 能读取正确 shape，并检查 `initial_state/cu_seqlens/chunk_indices` 没有错位。

### 风险 2：UB 空间紧张

V1 新增 `k_decay` 生成会额外读取 `k/gk` 并做广播。首期建议：

- 只覆盖当前已支持的 `K=128`。
- 使用现有 ping/pong buffer 复用，避免新增大 UB 常驻区。
- 若后续支持更大 K，再将 `k_decay` 计算按 K 维分块。

### 风险 3：性能回退

`gk=None` 必须走原路径，不增加 GM 读写。

`gk!=None` 时新增一次 `gk` 读、一次 `k_decay` workspace 写、C2 读 workspace 代替读原始 `k`。这是支持逐 K 维 gate 的必要成本。若后续性能不满足，可评估在 Catlass tile copy 中融合 `k * exp(gk_last - gk)`，但首期复杂度和风险更高。

### 风险 4：完整链路未全部支持 gk

本设计只让 fwd_h 能接收和消费 `gk`。完整前向若选择 `gate_source=gk` 或 `g+gk`，还需要确认：

- `chunk_scaled_dot_kkt_fwd` 是否按目标 gate 生成 A。
- `recompute_wu_fwd` 是否需要同步支持 `gk`。
- `chunk_fwd_o` 是否需要同步支持 `gk`。
- 反向相关算子的 `gk` 梯度路径是否完整。

因此建议先以单算子 PTA/torch 测试验收，再推进完整模型链路。

## 11. 推荐实施顺序

1. 先完成 host 接口、OpDef、tiling 的 `gk` 下发，写 shape/dtype 负例测试。
2. kernel 先实现 `gk=zeros` 路径，确认与 `gk=None` 完全一致。
3. 再实现随机 `gk` 的 V1 `k_decay` 和 V2 state decay，使用 CPU reference 对比。
4. 覆盖 `arch35` 同步实现。
5. 最后更新 README 与示例限制说明。
