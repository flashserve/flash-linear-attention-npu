# ChunkKdaFwdIntraSubChunk 设计方案

> 状态：**已实现并对齐当前已验证最优路径**（板端模型 case Task Dur med **2.075 ms**，目标 ≤1.5 ms 未结案）。  
> GPU 对标：`flash-linear-attention/fla/ops/kda/chunk_intra.py` `chunk_kda_fwd_kernel_intra_sub_chunk`  
> 流水参考：`chunk_bwd_dqkwg`（raw credit）、`prepare_wy_repr_bwd_full`（SetFree 预置）、`a2_a3_common_optimization_notes.md`  
> 精度测试：`torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py`  
> 迭代档案：[`ITERATION_LOG.md`](ITERATION_LOG.md) · 采集：[`MSPROF_GUIDE.md`](MSPROF_GUIDE.md) · Cube 理论：[`CUBE_OPTIMAL_PIPELINE.md`](CUBE_OPTIMAL_PIPELINE.md)

---

## 0. 已拍板决策

| 项 | 锁定 |
|----|------|
| 定位 | **独立算子、长期并行 API**；不改、不 drop-in 替换 `chunk_kda_fwd` Stage1 |
| SOC | A2/A3 **精度+可跑**；A5 **可编译**（`arch35/` 骨架，精度后补） |
| Layout | 大写 **`BNSD` / `TND`**（可兼容小写解析）；dense 原生 BNSD，不做 BSND/NTD Swap12 |
| Tile | `K=128`，`BC=16`，**`BT∈{32,64,128}`**（`NC=2/4/8`） |
| 求逆 | **Forward Substitution**（对齐 Triton）；**不**嵌 MCH |
| 写回 | 对角 BC×BC 正确；**非对角块 / 无效行算子写 0** |
| 变长 | TND 需 `cu_seqlens`；`chunk_indices` 由 wrapper 按仓内惯例生成 |
| 流水 | 先发 **2 window / 4 slot**；`SetFree` 预置 + S0 实灌 + raw ready；**禁** `WithReverse` 充当深度 |
| Kernel | `KERNEL_TYPE_MIX_AIC_1_2` |

命名：

| 项 | 值 |
|----|-----|
| 目录 | `fla/ops/ascendc/kda/chunk_kda_fwd_intra_sub_chunk/` |
| OpDef / ACLNN | `ChunkKdaFwdIntraSubChunk` / `aclnnChunkKdaFwdIntraSubChunk` |
| Python | `fla_npu.ops.ascendc.npu_chunk_kda_fwd_intra_sub_chunk` |

---

## 1. 问题定义

对每个 `(chunk, sub_chunk i_i, hv)`，子块起点 `i_ti = chunkLocal*BT + i_i*BC`：

1. 门控归一化后的对角分块 `Aqk`（下三角含对角 × scale）
2. 对角 `Akk` 严格下三角 × β 后的 **`Akkd = (I+L)^{-1}`**（fp32，Forward Substitution）

对应 Triton grid：`(NT, NC, B*HV)`，`NC=BT/BC`，`BC=16`。

**不含** `inter_solve_fused`。

---

## 2. 接口契约

### 2.1 Layout 与形状

| layout | 含义 | metadata |
|--------|------|----------|
| **BNSD** | 定长 4D | 不传 cu_seqlens |
| **TND** | 变长 packed 3D | API：`cu_seqlens`；kernel：+`chunk_indices`（wrapper 生成） |

**BNSD**

```text
q/k:  [B, H,  T, K]
g:    [B, HV, T, K]
beta: [B, HV, T]
Aqk:  [B, HV, T, BT]     # qk dtype
Akkd: [B, HV, T, BC]     # fp32
```

**TND**

```text
q/k:  [T, H,  K]
g:    [T, HV, K]
beta: [T, HV]
Aqk:  [T, HV, BT]
Akkd: [T, HV, BC]
cu_seqlens: [S+1] int64
chunk_indices: flat [seq_id, local_chunk_id, ...]  # wrapper 生成
```

约束：`HV % H == 0`，`H,HV ≤ 128`；TND 支持多 H/HV（不沿用整网 `chunk_kda_fwd` 的 TND∧H>1 限制）。

ATTR：`scale`（float）、`chunk_size∈{32,64,128}`、`layout`（大写字符串）。

Dtype：`q/k/Aqk` = fp16/bf16；`g/beta` = fp32；`Akkd` = fp32。

### 2.2 寻址

**BNSD**：`timeBos = batch*T + chunkLocal*BT`；`q/k` 按 H，`g/Aqk/Akkd/beta` 按 HV；GVA：`hk = hv / (HV/H)`。

**TND**（每 AIC task 只读一次 metadata）：

```text
(seqIdx, chunkLocal) = chunk_indices[task]
timeBos  = cu_seqlens[seqIdx] + chunkLocal * BT
validLen = min(BT, cu_seqlens[seqIdx+1] - timeBos)
```

写回：

- `Aqk`：行 `i_ti:i_ti+BC`，列 `i_i*BC : (i_i+1)*BC`
- `Akkd`：行 `i_ti:i_ti+BC`，列 `0:BC`（整块 BC×BC 按行铺在 `[T,BC]`）
- **非对角列块、越界/无效行：算子写 0**

`g_ref`：`mid = min(BC//2, validRows-1)`（对齐 Triton）。

---

## 3. 数学语义

### 3.1 计算链（每对角子块）

```text
(1) gate:  gq = exp2(g - g_ref);  gk = exp2(-(g - g_ref))   # 无效行 0
           Qg = q*gq;  Kg = k*gk;  Kgq = k*gq
(2) gemm:  Aqk_raw = Qg @ Kg.T
           Akk_raw = Kgq @ Kg.T
(3) mask:  Aqk = tril_incl(Aqk_raw) * scale
           L   = strict_tril(Akk_raw) * beta[:, None]
(4) inv:   Akkd = ForwardSub(I + L)     # 即 Triton 路径；注释里亦称 (I-L)^{-1} 当 L 取 -Akk
```

### 3.2 为何不用 MCH

| | Forward-sub（锁定） | 嵌 MCH |
|--|---------------------|--------|
| 落点 | Vector，接在 tril 后 | 再进 Cube |
| 额外 CrossCore | **0** | +1～2 |
| A2/A3 | 仅 UB | GM scratch bounce |
| 对标 | Triton 同构 | 另证数值差 |

Forward-sub 的「乘」是 **1×BC @ BC×BC 行更新**（Mul + 按列归约），**禁止**为此调 Cube。

```text
Ai = -L                                    # UB fp32 16×16
for i in range(2, validRows):
    a = -L[i, :];  a = where(o < i, a, 0)
    a = a + (a as row) @ Ai               # Vector only
    Ai[i, :] = a
Akkd = Ai + I
```

`validRows < 2` → `Akkd = I`（仅对角有效）。

### 3.3 已验证最优：FwdSub Vector 落法（实现锁定）

数学不变：\(a \leftarrow a + \sum_p a_p\,(A_i)_{p,:}\)（对齐 Triton `tl.sum(b_a[:, None] * b_Ai, 0)`）。

| 步骤 | 实现 | 同步 |
|------|------|------|
| 取行 | `DataCopy(tmp, akk[i,:])` | `PipeBarrier<PIPE_V>` |
| 广播 | `Brcb(aBrcb, tmp)` → 每行 8 路重复 | BAR |
| 物化 | **行广播 `Mul`**：`prod[p,c]=akk[p,c]*a[p]`，`src1BlkStride=0`；按列 tile（步长 8）两趟盖满 BC=16 | **两趟 Mul 连发，无 mid-tile BAR** |
| 归约 | Add-fold 对行维折半（`BC=16=2^n`） | **每层 fold 一次 BAR**（真 RAW） |
| `+I` | 对角 `GetValue`/`SetValue` | `V_S` / `S_V` |

**明确不做（已否决）**

- 铺满 `brcd` 矩阵再大 `Mul`（UB2UB 热点）  
- 每 col-tile 后 `PipeBarrier`（BAR cyc/call 暴涨，板端不稳）  
- `Pattern::RA` / 库 `ReduceSum` 做 16×16 列和（~7.5× 更慢）  
- 按 `next_pow2(i)` 稀疏 Mul/fold（精度偶过但不稳，sim tick 回退）

### 3.4 Cube 路径数值语义（golden 必须对齐）

NPU：Vector fp32 gate → **Cast 到 qk dtype** → Cube MMAD → Post fp32 FwdSub。

Golden 必须用 `score_dtype=qk.dtype` 在两笔 score GEMM 前 cast `Qg/Kg/Kgq`（参考仓内 `chunk_kda_fwd_intra_sub_chunk_ref(..., score_dtype=dtype)`），FwdSub 可在 fp32/fp64 中做。  
**禁止**用「全程 fp64 GEMM」单标杆冒充 Cube 精度，否则 Aqk 系统性偏差会被误判为 kernel bug。

---

## 4. Stage 与流水深度

### 4.1 依赖图

```text
stage_0 Vector:  Qg, Kg, Kgq  → WS
                 -- SetS0Ready (PIPE_MTE3) -->
stage_1 Cube:    Aqk_raw, Akk_raw (Kg L1 复用) → WS
                 -- SetCubeReady (PIPE_FIX) -->
stage_1 Vector:  tril + FwdSub + store Aqk/Akkd
                 -- SetFree(slot) -->
```

同 stage 内允许 Cube→Vector 边（对齐 PR190 `stage_1` 的 tril）；**不**拆 outer stage_2。  
stage_1 Vector **无**可先于本窗 Cube 的独立活；并行靠 **先发多窗**。

### 4.2 流水深度协议（最终锁定）

合成：`prepare_wy_full` 的 **SetFree 预置** + 本算子 **S0 实灌** + dqkwg 的 **raw 计数 / 0x2 / 禁 reverse**。

| 量 | 值 |
|----|-----|
| 握手单位 | **1 window**（2 head；`headCnt==1` 时一实一空） |
| 先发深度 | **2 window** |
| GM slot | **4**（奇偶窗 × 2 head） |
| slot | `slot0=(windowId%2)*2+0`，`slot1=...+1` |

```text
启动（每 AIV，0x2）:
  for s in 0..3: SetFree(s)
  for w in 0 .. min(2,W)-1:
      WaitFree(slot0); WaitFree(slot1)
      AIV0 S0(h0); AIV1 S0(h1)
      PipeBarrier<MTE3>; SetS0Ready()     // 每窗 1 次逻辑 ready

Cube 每窗:
  WaitS0Ready()
  GEMM h0; GEMM h1                        // 各 head：装数+两笔 MMAD+Fixpipe
  PipeBarrier<FIX>; SetCubeReady()

Vector 稳态每窗:
  WaitCubeReady()
  Post h0 || Post h1                      // tril+FwdSub+写 0
  SetFree(slot0); SetFree(slot1)
  if w+2 < W:
      WaitFree → S0(w+2) → SetS0Ready     // 补液，保持深度

Cube 结束:
  WaitFree(0..3)                          // 收回启动预置
```

时序：

```text
V: [S0_w0 S0_w1] [Post_w0 S0_w2] [Post_w1 S0_w3] ...
C:               [GEMM_w0][GEMM_w1][GEMM_w2] ...
```

**禁止**

- `CrossCoreFlagWithReverse` 充当深度（会 lockstep）
- MMAD 完未 Fixpipe 就 Set / VEC 完未 MTE3 就 Set
- 只写完 `Aqk_raw` 就 Set（Akk 也要用）
- `0x2` 下单侧漏 Set（尾窗空 AIV 必须空转握手）

### 4.3 Flag 与次数

```cpp
FLAG_S0_READY, FLAG_CUBE_READY          // 每窗 1 次逻辑握手
FLAG_SLOT_FREE[4]                       // 每 slot
// SetS0Ready:  PIPE_MTE3 + Barrier MTE3, mode 0x2
// SetCubeReady: PIPE_FIX  + Barrier FIX,  mode 0x2
// SetFree:      PIPE_MTE3（或 Post 末 GM 读后 MTE2）, 0x2
```

| 动作 | 每 AIV | Cube |
|------|--------|------|
| 启动 SetFree | 4 | — |
| SetS0Ready / WaitS0Ready | W | W |
| SetCubeReady / WaitCubeReady | W | W |
| Post 后 SetFree | 每窗本侧 slot | 尾 WaitFree×4 |

`W` = 本核 window 数。改循环先改次数表。

### 4.4 对照仓内修订摘要

| 问题 | 修正 |
|------|------|
| 空 credit 不适用于本算子 | S0 **实灌** 2 窗 |
| 隐式 free + 双 AIV 不安全 | 显式 `FLAG_SLOT_FREE` |
| head 粒度单 AIV Set + 0x2 死等 | **窗粒度握手** |
| 缺 Barrier | Set 前强制 PipeBarrier |
| 与 Stage1 reverse 混用 | 禁 reverse 做深度 |

---

## 5. 分核 / 调度

### 5.1 AIC 分核

| | |
|--|--|
| 主路径 | task = **chunk**（`chunkNum >= aicCoreNum`） |
| 短序回退 | task = **(chunk, i_i)** |
| **不做** | 按 hv 分 AIC；按 window 分 AIC task |

核内：`GetTaskInfo` **每 task 一次** → `for i_i` → `for hvBase += 2` 走 §4.2 流水。

### 5.2 双 AIV

```text
AIV0 ↔ hvBase     ↔ slot 偶
AIV1 ↔ hvBase + 1 ↔ slot 奇
```

GVA：`hk = hv/(HV/H)`；同 chunk 全部 hv 留在本 AIC。

### 5.3 Tile 常量

| BC | BT | NC | K | Window | Slots |
|----|----|----|---|--------|-------|
| 16 | 32/64/128 | 2/4/8 | 128 | 2 head | 4 |

---

## 6. 缓冲与 Workspace

### 6.1 Cube L1 / L0（路径 A · 已验证默认）

与 [`CUBE_OPTIMAL_PIPELINE.md`](CUBE_OPTIMAL_PIPELINE.md) **路径 A** 一致：

| 宏 / 行为 | 默认 | 作用 |
|-----------|------|------|
| `USE_SCORE_L1A_DBUF` | **1** | `l1A[0]=Qg`，`l1A[1]=W`；`MTE2(W)‖MMAD1`，Wait W 再 Fix Aqk |
| `USE_SCORE_FIX_MTE2_DBUF` | **1** | Akk Fix ‖ 下一 tile MTE2；SetCubeDone 前 Drain |
| `USE_SCORE_WIN_L1_RESIDENT` | **0** | 双头 L1 Prefetch；精度失败，禁止 default on |
| `USE_SCORE_MMAD1_LOAD_W` | **0** | 单槽 W‖MMAD（由 L1A_DBUF 取代） |

- 单 head 内 **Kg → L1B 驻留**，两笔 GEMM 复用  
- L0A/L0B 双缓冲；L0C 单槽（A2/A3）

```text
GM→L1A[0]: Qg ; GM→L1B: Kg ; GM→L1A[1]: W (‖ MMAD1)
L1→L0 → MMAD Aqk → Fixpipe
复用 L1(Kg) → MMAD Akk → Fixpipe (‖ 下 tile MTE2)
```

### 6.2 Vector UB（已验证默认）

```text
vecBuf_     : Prep 六段 fp32 / FwdSub prod|aBrcb|acc / tril mask 雕刻
midBuf_     : g_ref 一行
betaBuf_    : BC
aqkBuf_/akkBuf_ : 16×16 fp32 分块
tmpBuf_     : FwdSub 行向量
inBuf_      : MTE2 入口（含 beta）
aqkTBuf_    : store 前 cast
zeroBuf_    : 无效行清零
```

MTE2 合并（无宏，主路径写死）：

- Prepare：`mid ‖ q/k/g` → 一次 `Wait(MTE2_V)`  
- Post：`cmat(Aqk/Akk) ‖ beta` → 一次 `Wait(MTE2_V)`  

`MTE3_MTE2` 不能替代 `MTE3_V`。Post 存完后立刻 Wait MTE3（不做 Post‖S0 defer）。

### 6.3 Workspace

```text
slotBytes = align32(BC*K*s*3 + BC*BC*s*2)   # Qg,W,Kg + Aqk_raw,Akk_raw
total     = aicCoreNum * 4 * slotBytes
```

---

## 7. Host / 工程

### 7.1 TilingData（建议）

```text
B, T, H, HV, K, BT, BC, NC, scale
chunkNum, aicCoreNum, taskGrain
layoutMode (0=BNSD, 1=TND)
workspacePerCore, workspaceSlotSize, workspaceBufferCount(=4)
```

TilingKey：`D_T_QK ∈ {fp16,bf16}` × `CHUNK_SIZE ∈ {32,64,128}`。

### 7.2 校验

- `K==128`, `BT∈{32,64,128}`, `BC==16`, `HV%H==0`
- TND：`B==1` + `cu_seqlens`；BNSD：无变长 metadata
- 非法 layout / dtype / BT → host 明确报错

### 7.3 目录

```text
chunk_kda_fwd_intra_sub_chunk/
  DESIGN.md              ← 本文（方案 + 已验证最优落法）
  ITERATION_LOG.md       ← 性能迭代 / 否决档案
  PERF_ITER_LOG.md       ← 当前基线快照（指向 ITERATION_LOG）
  MSPROF_GUIDE.md
  CUBE_OPTIMAL_PIPELINE.md
  README.md
  op_host/               def, tiling, aclnn
  op_kernel/
    *_common.h
    *_cube.h
    *_vector.h
    arch35/              A5 可编译骨架
  test/                  golden ref
```

Python：`fla_npu.ops.ascendc`（ctypes→aclnn）。

### 7.4 A2/A3 vs A5

| | A2/A3 | A5 |
|--|-------|-----|
| 交付 | 完整实现 + 精度 | 注册 + 编译通过 |
| 数学/stage/slot | 同语义 | 同文档；精度后补 |
| 禁止 | — | 另写一套 FwdSub / mid-gate |

---

## 8. 精度验证（标杆必须对齐）

参考实现与 case 矩阵对齐：

- Golden：`.../chunk_kda_fwd_intra_sub_chunk/test/test_chunk_kda_fwd_intra_sub_chunk.py`
- NPU 单测：`.../torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py`

### 8.1 标杆对齐原则（强制）

1. **单标杆时必须 Cube 同构**  
   - 使用 `score_dtype = qk.dtype`（bf16/fp16）cast 后再做两笔 score GEMM。  
   - FwdSub 可用 fp32/fp64；**不要**用「全程 fp64 GEMM」当唯一 golden 判 NPU 失败。

2. **强 gate 下 Akkd 用相对误差**  
   - `(I-L)^{-1}` 在 `lin_strong` 下幅值可极大；`akkd_rel = max(|err| / max(|ref|, 1))`。  
   - Aqk 仍可用绝对误差（如 `5e-2`）。

3. **大 T 不可只靠全量 CPU golden**  
   - `T=8192` 全量 ref 过慢 → 用 **Cube-faithful 子块采样**（`_bf16_sim_sub`）对若干 `(h, ti)` 比对。  
   - 采样须覆盖多 head、多 chunk 列偏移 `col0 = ti % BT`。

4. **单标杆必须泛化多 case**（不要只跑一个默认 shape）  
   覆盖维度至少包括：

| 维度 | case 示例 |
|------|-----------|
| BT | 32 / 64 / 128 |
| dense | B>1、多 H、尾块 `T%BT!=0`、短 T |
| GVA | `HV=2H/4H`，含 varlen GVA |
| varlen | 不等长、多 seq、短 seq（&lt;BC）、BT 变体 |
| gate | `lin_strong` / `lin_mild` / `rand` |
| dtype | bf16 / fp16 |
| 模型尺度 | `T=2048/4096/8192, H=16/32` 采样路径 |
| 写 0 | 断言非对角块 / 越界区为 0（或与 ref 一致的零） |
| 反向 | 非法 BT/layout/缺 cu_seqlens → host 拒绝 |

5. **禁止**  
   - 收窄 gate range、删失败 case、无依据放宽阈值来制造通过。  
   - 未跑项须在报告中明示（缺 NPU / 缺 SOC 等）。

### 8.2 建议阈值（与参考测试同量级，可按门控微调）

| 输出 | 典型 | 强门控 / 长序列 / 大 GVA |
|------|------|---------------------------|
| Aqk max abs | `< 5e-2` | 可达 `1e-1`（须注明） |
| Akkd max rel | `< 1e-3` | `5e-3`～`5e-2`（须注明） |

### 8.3 结构定位序（精度失败时）

```text
gq/gk → Aqk_raw/Akk_raw → mask 后 Aqk/L → Akkd → 最终 GM
```

结构性误差（块错位、符号、固定头/尾条纹）优先查：layout、bos、`i_i` 列偏移、GVA `hk`、写 0、flag 失衡。

### 8.4 性能验收（A2/A3）

- 口径与流程：[`MSPROF_GUIDE.md`](MSPROF_GUIDE.md)。  
- 主 case：`T=8192,H=32,K=128,BT=64,bf16` 的 **Task Duration 中位**；不用 host wall 作结论。  
- 先发生效：prefill 后出现 **GEMM ∥ Post**；大段互等 → 查 Set/Wait 与 free。  
- **当前基线（已验证最优）**：med **2.075 ms**；硬目标 **≤ 1.5 ms**（未结案）。历程见 [`ITERATION_LOG.md`](ITERATION_LOG.md)。

### 8.5 SOC 门禁

| SOC | 状态 |
|-----|------|
| A2 / A3 | 精度矩阵 + 主 case 性能（当前最优已上板） |
| A5 | **编译通过**；精度可选后补 |

---

## 9. 风险

| 风险 | 应对 |
|------|------|
| 仍 AIV-bound（BAR/fold） | 接受 Cube 路径 A 已尽；下一刀须结构级，见 `ITERATION_LOG` §7 |
| FwdSub 再换库 reduce | **禁止**（RA 已证伪） |
| 与 Stage1 MCH 不一致 | 只对齐 Triton/Cube-faithful golden |
| flag 失衡 | 窗级次数表 + 空 AIV 空转 |
| golden 不对齐 | §8.1：score_dtype + 相对误差 + 多样本 |
| aicore timeout 楔卡 | 避让长任务后全卡 reset；见 `MSPROF_GUIDE` |

---

## 10. 实现阶段（已完成 / 遗留）

| 阶段 | 状态 |
|------|------|
| Host def / tiling / aclnn | 完成 |
| Golden + 单测 + 模型采样 | 完成 |
| A2/A3：§4.2 流水 + Cube 路径 A + FwdSub P1 | **完成（当前最优）** |
| A5 arch35 编译门禁 | 完成（精度后补） |
| Task Dur ≤ 1.5 ms | **未结案**（基线 2.075 ms） |

---

## 11. 文档维护

| 文档 | 职责 |
|------|------|
| `DESIGN.md`（本文） | 数学 / stage / 缓冲 / 已验证最优落法 / 验收 |
| `ITERATION_LOG.md` | 性能刀时间线、否决清单、sim 画像 |
| `CUBE_OPTIMAL_PIPELINE.md` | Cube 路径 A/B 理论与宏 |
| `MSPROF_GUIDE.md` | 采集命令与门禁口径 |
| `README.md` | 接口、构建、测试入口 |

改握手/slot/BT/默认宏或 FwdSub 算法时：同步 §3.3、§6、§8.4，并在 `ITERATION_LOG` 追加条目。
