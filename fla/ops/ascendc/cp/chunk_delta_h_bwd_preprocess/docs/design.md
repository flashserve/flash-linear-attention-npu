# ChunkDeltaHBwdPreprocess 设计

## 1. 定位

本算子实现 CP 反向状态预处理的本地部分：把本 rank 边界序列的反向状态递推压缩成仿射摘要
`dH_start = P_r @ dH_end + E_r`，输出一个 FP32 张量 `dhm`。跨 rank 的 all-gather 与按逻辑时间逆序的
合并由框架侧与后续的 `ChunkDeltaHBwdMerge` 完成，不在本算子内。

上游对应实现是 Triton 的 `pre_process_bwd_kernel_merged`（GDN/KDA/GDN2 共用），一次 launch 内同时计算
`E_r` 与 `P_r`。

## 2. 数学目标

按 chunk 从最后一个（`i_t = NT-1`）向第一个扫描，`M` 为当前 chunk 的有效 token 数：

```text
Q̄_c  = q·2^{g}                      (USE_G)        Q̄s_c = scale · Q̄_c
     = qg                            (USE_GK)
     = q                             (无门控)
K̄_c  = k·2^{g_last - g}              (USE_G)
     = kg                            (USE_GK)
     = k                             (无门控)
decayK_c = 2^{g_last}                (USE_G，沿 K 广播)
         = 2^{gk_last}               (USE_GK，逐 K)
         = 1                        (无门控)
```

E 分支（写 `dhm[..., 0:V]`）：

```text
dV_pre = K̄_c @ dH_old                 [M,K] @ [K,BS]      → [M,BS]
dV̂'    = -(dV_pre + dv_local)         取负号使 C3 只做累加
inc    = Q̄s_c^T @ do_c + W_c^T @ dV̂'   [K,M]@[M,BS] ×2     → [K,BS]（同一 L0C）
dH_new = decayK_c ⊙ dH_old + inc       初值 0 ⇒ 末值即 E_r
```

P 分支（写 `dhm[..., V:V+K]`）：

```text
T1    = W_c^T @ K̄_c                    [K,M] @ [M,K]       → [K,K]
P_c   = diag(decayK_c) - T1
P_new = P_c @ P_old                    初值 I ⇒ 末值即 P_r = P_0 P_1 … P_{NT-1}
```

关键等价关系：上游把 `2^{g_last-g}` 乘在 `K̄ @ dH` 的乘积上；由于该衰减只沿 token 轴逐行作用，把它折进
`K̄` 与乘在乘积上等价。本设计统一折进 `K̄`（P 分支本来就需要带衰减的 `K̄`），因此 `USE_G` 与 `USE_GK`
共用同一套 Cube 公式，V2 没有分支。代价是 V0 必须在 `K̄` 上把 `[M, BT)` 的无效行写零。

## 3. 六 Stage 划分

原八 Stage 版本把 `C1/C5`、`V4/V6` 分开；合并后为：

```text
V0 ──▶ C1 ──▶ V2 ──▶ C3 ──▶ V4 ──▶ C5
        └── 路 B（T1）──┘
```

| 阶段 | 引擎 | 内容 | 入口 ready | 出口 ready |
| --- | --- | --- | --- | --- |
| `V0` | Vector | 门控/衰减准备：`Q̄s`、`K̄`、`decayK`；`[M,BT)` 写零 | 本 chunk 的 `q/k/g`（或 `qg/kg/gk`） | `V0ExportReady`（多读者） |
| `C1` | Cube | 路 A `dV_pre`；路 B `T1` | `V0ExportReady` + `dH` 旧值 + `W_c` | `C1DvPreReady`（先）、`C1T1Ready`（后） |
| `V2` | Vector | `dV̂' = -(dV_pre + dv_local)` | `C1DvPreReady` + `dv_local` | `V2DvHatReady`、`C1DvPrePayloadFree` |
| `C3` | Cube | `inc = Q̄sᵀ@do + Wᵀ@dV̂'`（同 L0C） | `V2DvHatReady` + `Q̄s` 在 L1 + `do` | `C3IncReady`、`V2SourceFree` |
| `V4` | Vector | 路 A `dH_new`；路 B `P_c = diag(decayK) - T1` | `C3IncReady` + `C1T1Ready` + `dH` 旧值 + `decayK` | `V4DhtReady`（下一轮 C1）、`V4PcReady`（本 chunk C5） |
| `C5` | Cube | `P_new = P_c @ P_old`（K 归约分块） | `V4PcReady` + `P` 旧值 | `C5PReady`（下一轮 C5）、`V4PcPayloadFree` |

约束：

1. 每个 Stage 只含一种计算引擎；Cube 不消费本 Stage 新输出；
2. `C1` 必须先发布路 A 再发布路 B，不得把 `dV_pre` 拖到 `T1` 之后；
3. 合并 Stage 的两路 ready/free **各自独立**，不能并成一条；
4. `V4` 的两路共用同一份 `decayK` 装载；`C1` 的两路共用同一次 `K̄` 的 L1 装载。

### 3.1 为什么合并后是 6 个 Stage

两组可合并的 Stage 都满足"同一计算引擎 + 入口操作数同一时刻 ready"，合并只共享入口与操作数装载，
不改变数学语义：

1. 原 `C1` + 原 `C5`（都是 Cube，都吃 `K̄_c`）：`K̄_c`/`dH_old`/`W_c` 都在 `V0` 之后同一时刻 ready，
   两路只是两个 MMAD，共用同一次 `K̄_c` 的 L1 装载；`dV_pre` 先于 `T1` 发布，因此 `V2` 不被 `T1` 阻塞。
2. 原 `V4` + 原 `V6`（都是 Vector，都吃 `decayK`）：路 B 的 `T1` 由合并后的 `C1` 提前发布，
   本 Stage 的实际入口条件仍是 `C3IncReady`；`decayK` 一次 UB 装载同时服务两路输出。

不能继续合并的部分：

- `V0` 不能并入 Cube Stage：它是唯一做门控/指数/截断/掩码的 Stage，两个分支都依赖它的输出，
  数值规则（截断区间、掩码零行）集中在一处才能一处保证；
- `C1` 与 `C3` 之间必须有 `V2`：`C3` 的右操作数是 `V2` 的乘积结果，Cube 不消费同 Stage 新产生的数据；
- `C3` 与下一轮 `C1` 之间必须有 `V4` 路 A：`dH` 的衰减 + 累加是逐 K 行的 Vector 语义，
  且必须回写 workspace 才能被下一轮 `C1` 当 L1 操作数读取；
- `V4` 路 B 与 `C5` 之间必须有对角注入：`diag(decayK)` 必须在链乘之前按 `row == col` 的位置语义
  落入 `P_c`，留到 `C5` 的 L0C 初始化就无法注入。

收益与代价：Stage 数由 8 降到 6，每 chunk 少两次 Stage 入口 handshake，`K̄_c` 每 chunk 只做一次 L1 装载、
`decayK` 只做一次 UB 装载；代价是 P 链的 `P_c` 与本 chunk 的 E 链绑定（E 链更长时关键路径不变；
若实测 `tileK` 远多于 `tileV` 导致 P 链成为关键路径，回退到八 Stage 版本即可解耦）。

## 4. 分核规则

```text
tileV = cdiv(V, BS)，tileK = cdiv(K, BS)，tileNum = tileV + tileK
BS = 32 if K <= 64 else 64
```

1. **禁止按 chunk 分核**：`dH` 与 `P` 都是跨 chunk 状态，同一 head 的全部 chunk 必须留在同一工作组内按
   `i_t = NT-1 → 0` 逆序执行。
2. **默认（`Hv >= 核数`）**：仅按 head 连续分核，`groupHeads = ceil(Hv/核数)`，工作组 `i` 负责
   `[i*groupHeads, min((i+1)*groupHeads, Hv))`，内部遍历该 head 的全部 `tileNum` 个列 tile。
3. **补充（`Hv < 核数`）**：把 `(hv, 列 tile)` 展平为 `Hv * tileNum` 个 task，按 balanced half-open range
   分配。依据是列 tile 相互独立：V 方向列 tile 只用 `dH`/`do`/`dv` 的同一列区间，K 方向列 tile 只用 `P`
   的同一列区间。展平后每个 task 只执行合并 Stage 中属于自己分支的那一路。
4. **多 segment**：一次 launch 只处理一个 `[bos, eos)`；varlen 时 `bos/eos` 由 kernel 从 GM 读取
   `cu_seqlens[0:2]`，host 只校验 shape。

`Hv` 与 `Hk` 的映射：`hk = hv // (Hv / Hk)`；host 校验 `Hv % Hk == 0`，不要求物理 repeat `q/k`。

## 5. workspace 布局

所有子区位于 user workspace，偏移由 host tiling 规划、按 512 B 对齐：

| 子区 | 大小 | 用途 |
| --- | --- | --- |
| `meta` | 512 B | segment / chunk 元数据（预留） |
| `slot` | `slotNum × slotBytes` | V0 的 chunk slot：`Q̄s[M,K]` + `K̄[M,K]`（模型 dtype）+ `decayK[K]`（FP32） |
| `t1` | `K*K*4` | `T1` 平面（与输出列 tile 无关，每 chunk 一次） |
| `pc` | `K*K*4` | `P_c` 平面 |
| `dh` | `2*K*V*4` | `dH` ping-pong（跨 chunk 状态，FP32） |
| `p` | `2*K*K*4` | `P` ping-pong（跨 chunk 状态，FP32） |

以 `K = 128`（本版唯一支持取值）的账本为例；若将来放宽到 `K = 256`，`T1`/`P_c` 会各占 256 KiB、
`P` 两个面共 512 KiB、`dH` 两个面共 512 KiB，这些平面就必须驻留
workspace，由 Cube 侧按块 MTE2，不能假设能整体进 L1。`W`/`do`/`dv_local` 直接从 GM 取，不进入 slot。

## 6. 同步合同

```text
data-ready：V0 → C1(两路) ; C1路A → V2 → C3 → V4路A ; C1路B → V4路B → C5 ; V4路A → 下一轮 C1 ; C5 → 下一轮 C5
storage-free：V2 → C1DvPrePayloadFree ; V4 → C3PayloadFree ; C5 → V4PcPayloadFree ;
              slot 最后一个读者 → V0SlotFree ; 下一轮 C1+V4 → DhtPrevFree ; 下一轮 C5 → PPrevFree
```

- data-ready 不能替代 storage-free；上游 profiling 中偶然先完成的边不能省略或合并。
- `V0` 的 slot 是多读者资源（C1 两路、C3、V4），必须用"最后一个读者归还"的计数语义。
- `dH` 旧值在同一 chunk 内有 C1（经 L1）与 V4（经 UB）两个读者；`W` 有 C1 路 B 与 C3 两个读者。
- ping-pong 复用必须等下一轮消费者读完；`dH`/`P` 的 free 边由生产者与消费者共同确认。

## 7. 与 Triton 参考实现的差异

| 项 | Triton | 本算子 |
| --- | --- | --- |
| 并行度 | `grid = (cdiv(V,BS)+cdiv(K,BS), Hv)`，每个 program 独立 | 单工作组持有整头（或 `(head, 列 tile)` 任务），chunk 在核内逆序 |
| `T1`（`b_kw`） | 每个 program 各算一遍 | 每 chunk 一次（列 tile 复用） |
| 衰减位置 | `USE_G` 下乘在 `K̄@dH` 的乘积上 | 折进 `K̄`（V0 内），两模式共用 Cube 公式 |
| Stage 数 | 一次 launch 两个 program 组 | 六 Stage（原八 Stage 合并 `C1+C5`、`V4+V6`） |
| 精度 | `dhm`/链 FP32；`AFFINE_CHAIN_PRECISION` 可选 | `dhm`/链固定 FP32，不提供链精度开关 |

## 8. 平台与限制

- 平台：`ascend910b`（A2）、`ascend910_93`（A3）、`ascend950`（A5）。
- 本版**只支持 `K = V = 128`、`chunk_size = 64`**（host 拦截其它取值）：状态行按 64 行分组、
  Cube 列 tile 按 16 个元素（64B）成组、Vector 逐行的寄存器读写也依赖该行宽，改尺寸需要同步改
  tiling、workspace 账本与 tile 布局；`K=64/96/256`、`V=96` 等取值均已实测会出现设备报错或结果错误。
- shape/取值范围的唯一维护处是 [README](../README.md) 的「支持的场景」「不支持（本版显式拦截）」「已知限制」三节，
  本文只讨论设计取舍，不重复列举取值。
- layout：`[B,H,T,D]`（BSND）。TND/NTD 需由调用方或 L2 侧 layout sweep 后进入本算子。
- `state_v_first` 与本算子无关：本算子只读 token-major 的 `q/k/w/do/dv`，只写固定 `[Hv, K, V+K]` 的 `dhm`。
- 不支持 `USE_BG`（DPLR）与 `AFFINE_CHAIN_PRECISION`；`g`/`gk` 互斥由 host 拦截。
- Vector 侧实现按平台拆分：`op_kernel/arch22/`（A2/A3）用 AscendC 向量 API + 手写事件对；
  `op_kernel/arch35/`（A5，dav-3510）把 V0 门控行缩放、V2 `dV̂'`、V4 状态更新与 `P_c` 对角注入
  下沉到 RegBase `__simd_vf__`（一个 fp32 寄存器 256B / 64 lane）一趟融合，去掉逐行 `ExpScalar`
  的 V→S 同步与逐元素 `SetValue`/`GetValue` 的标量往返；Cube 侧两平台共用同一份实现。

## 9. 接入步骤

1. **kernel 落地（已完成）**：`op_kernel` 六个 Stage 的真实实现已落地（Catlass `BlockMmadTla` +
   Vector 侧手写事件对），`PROPOSED` 标记已随实现删除；Vector 侧按平台拆成 `arch22`（A2/A3）与
   `arch35`（A5，RegBase `__simd_vf__` 融合）两份实现，Cube 侧共用。
2. **构建接入（已完成）**：`chunk_delta_h_bwd_preprocess/CMakeLists.txt`（glob 风格）与
   `op_host/CMakeLists.txt`（`add_op_to_compiled_list()` + `target_sources(op_host_aclnnExc ...)` +
   `add_modules_sources(OPTYPE chunk_delta_h_bwd_preprocess ACLNNTYPE aclnn_exclude)` +
   `add_ops_compile_options(OP_NAME ChunkDeltaHBwdPreprocess OPTIONS --cce-auto-sync=off
   -Wno-deprecated-declarations)`）已提交；aclnn 走手写 exc 通路，不走自动生成。
3. **Python 入口（已完成）**：`fla_npu.ops.ascendc.chunk_delta_h_bwd_preprocess` 已接入，走 ctypes
   直调手写 aclnn（默认调用路径不依赖 `torch_npu` dispatcher 与 `torch.ops.npu` 注册），A2/A5 双平台
   设备侧验证通过。
4. **测试（设备侧精度矩阵已完成，见 README）**：以 `tests/atk/chunk_delta_h_bwd_preprocess/cases.json`
   为唯一用例来源，用 `tests/atk/chunk_delta_h_bwd_preprocess/reference.py` 作为 CPU 标杆，覆盖
   A2/A5 两平台与 `USE_G`/`USE_GK`/无门控、dense/varlen、尾块、GVA（`K = V = 128`，本版唯一支持取值）。
5. **精度与内存（部分完成）**：设备侧精度矩阵已通过；sanitizer（UB/L1 复用、跨核 slot、ping-pong）
   尚未执行，需要按仓库规范补做并确认运行命中的是 sanitizer 版本对象。

## 10. 验证方案

1. **数学基线**：`reference.py` 逐 chunk 复算 `E_r`、`P_r`，并用非零 `dht` 校验
   `P_r @ dht + E_r` 与整段 backward 的 `dh0` 一致（只测 `dht = 0` 无法验证 `P`）。
2. **Stage 边界**：在 `C1`/`V2`/`C3`/`V4`/`C5` 出口抽 `dV_pre`、`T1`、`dV̂'`、`inc`、`dH`、`P_c`、`P`
   与参考实现逐块比对，区分"公式错误"与"ready/free 缺边导致的旧值/新值混用"。
3. **链方向**：`NT >= 3` 的用例单独验证 `P_new = P_c @ P_old` 与 `dH_new = decayK ⊙ dH_old + inc` 的 chunk
   顺序，避免 `NT = 1` 掩盖方向错误。
4. **两路独立性**：构造"路 A 已就绪、路 B 未就绪"与反过来的用例，确认 `C1` 的 `dV_pre`/`T1`、`V4` 的
   `dH`/`P_c` 各自独立发布与释放。
5. **分核**：覆盖仅按 head 分核与 `(hv, 列 tile)` 展平两条路径，结果需与单 task 全量执行逐位一致。
6. **跨 rank 一致性**：以单 rank 全序列为基线，只改 CP 切分，对比最终 `dq/dk/dv/dg`（或 `dgk/dbeta`）。

### 10.1 已执行情况

- 第 1 项：纯 Python 自检已通过（三种 gate 模式，仿射恒等式最大绝对误差 ~1e-16）。
- 第 2、3 项：按 `tests/atk/chunk_delta_h_bwd_preprocess/`（ATK 精度矩阵 + `scripts/` 逐平面核对）执行，
  A2/A5 均 12/12 PASS（全部 `K = V = 128`），含 32 chunk 长链（`pos_13`）与尾块（`pos_06`）。
- 第 4 项：`C1`/`C3` 的两路输出、`V4` 的两路输出在实现上分别发布/释放（各自独立 flag 边界），
  逐平面核对未发现两路混用；受控实验（`scripts/ctrl_case.py`）用于定位过 A 路。
- 第 5 项：本版只实现"仅按 head 连续分核"（`BY_HEAD`），`(hv, 列 tile)` 展平（`BY_TILE`）尚未实现；
  原 `pos_16_tile_split_partition`（K=V=256）已随 `K = V = 128` 的收敛移出正向矩阵，改为不支持拦截用例。
- 第 6 项：跨 rank 一致性验证需要上层 CP 切分链路，本版未覆盖。
- 另需补做 sanitizer（`racecheck`/`memcheck`/`initcheck`/`synccheck`）与官方 ATK/CI 接入。

## 11. v2 流水排布（4 head / 双 window / 1:2 核型）

### 11.1 现状为什么慢（设备实测）

| 平台 / 用例 | AIC | AIV |
| --- | --- | --- |
| A2 gva_cp16（T=8192, 128 chunk, Hv=32） | `aic_time` 10.714 ms；cube 2.3%、fixpipe 9.8%、mte1 2.5%、mte2 3.9%；**等 AIV flag 95.7%**（wait_id0 42.6% + id2 11.5% + id4 41.6%） | 10.719 ms；vec 26.6%、mte2 28.3%（**34.2 GB/s**）、mte3 18.8%（**28.8 GB/s**）、等 AIC flag 27.5% |
| A5 gva（T=2048, 32 chunk, Hv=32） | 2.143 ms；cube 2.8%、mte1 2.4%、mte2 5.8%、fixpipe 99.9%（见下方 unit flag 说明，**该值不可直接采信**） | 2.148 ms；mte2 42.5%（**29.2 GB/s**）、vec 21.0%、mte3 17.0% |

结论：瓶颈不是算力，而是 **①AIC/AIV 严格交替（每 chunk 6 次阻塞握手，无跨 chunk 重叠）；
②中间量全部落 GM 再搬回，且每次只有 `[64,128]` 级小包（有效带宽 29–34 GB/s）；
③段内 `SetFlag`+紧邻 `WaitFlag` 全阻塞**。

### 11.1.1 `aic_fixpipe_ratio` 的 unit flag 陷阱（A5 实测 A/B）

`Catlass::Gemm::MmadPingpong<ArchTag, ENABLE_UNIT_FLAG, ...>` 的第二个模板参数是 unit flag。
置 `true` 时 Catlass 内部 **L0C 只开 1 个 stage**，M↔FIX 的同步由 MMAD 指令的 unit flag 承担
（`common/kernel_utils/block/block_mmad_pingpong_tla.hpp` 里 `static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1))`
与 `if constexpr(!ENABLE_UNIT_FLAG) { … FIX_M … }` 可以印证）。于是 profiler 会把整条 MMAD→Fixpipe
链的时间都记到 fixpipe 上，出现"99.87% 占用"的假象。同用例 A/B（A5 / gva / T=2048）：

| 变体 | `aic_fixpipe_ratio` | `aic_fixpipe_active_bw` | Task Duration |
| --- | --- | --- | --- |
| unit flag = `true`（L0C 单 stage） | 99.87%（2035 µs） | 8.6 GB/s | 2052.1 µs |
| unit flag = `false`（L0C 2 stage + 显式 FIX_M） | **6.49%（133 µs）** | **132.5 GB/s** | 2058.7 µs |

结论：**99.87% 是 unit flag 造成的统计口径，不是 fixpipe 瓶颈**——真实运行时间几乎不变
（2052 vs 2059 µs，噪声内）。A5 上 AIC 各 pipe 占用都只有 3–7%，且 `aic_time ≈ aiv_time`
（2044 vs 2049 µs）⇒ 该 kernel 是**依赖/同步受限**，不是任何单个 pipe 打满。

### 11.2 参考实现 `chunk_gated_delta_rule_bwd_finalize` 的可用机制

该算子（`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_finalize`）在同一套 Cube/Vector 分工下
把流水排起来了，关键机制如下（源码位置见括号）：

1. **1 AIC : 2 AIV**：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`；AIV 侧用
   `AscendC::GetSubBlockIdx()` 把两个 AIV 拆开承包（`..._vector.h`）。
2. **每核 4 head 常驻 L1**：`L1_RESIDENT_HEAD_COUNT_4`，`h/du/vb/kbg/dw0/a/dA0` 每个都有 4 份 L1 物理槽，
   AIC 对 4 个 head 连续发 MMAD（`..._cube.h` 的 `Process()` 开头）。注意 L1 只能由 AIC 写，
   AIV 的结果先落 GM、AIC 在对应 Stage 内搬入，**不存在 UB→L1 通路**。
3. **双 bank + per-slot 事件数组**：`BANK_COUNT_2`、`streamSlot_` 轮转，`mte2ToV_[slot]`、`vToMte3_[slot]`、
   `mte3ToMte2_[slot]`、`stateVToMte2_[slot]`、`mte3ToV_[slot]` 都是 2 元素数组 → 搬入/计算/搬出跨 slot 重叠。
4. **workspace 双 window × 4 head**：`WORKSPACE_HEADS_PER_GROUP_4 = 4`、`WORKSPACE_WINDOW_COUNT_2 = 2`、
   `WORKSPACE_BUFFER_COUNT_8 = 8`，`GetWorkspaceHeadOffset(coreIdx, groupRound, headOffset)` 用
   `(coreIdx*8 + (groupRound&1)*4 + headOffset) * 64*128` 定位（`..._common.h`）。
5. **每个方向只有一条 flag 链**：`VEC_TO_CUBE_READY_FLAG = 1`、`CUBE_TO_VEC_READY_FLAG = 3`，
   靠"固定顺序 set/wait"保证次序，从而允许生产者领先消费者一整个 window（`..._common.h` + 两侧 `Process()`）。
6. **任务按 `(chunk, head-group)` 展平**：`taskNum = totalChunkNum * headGroupNum`，
   `for (taskIdx = coreIdx_; taskIdx < taskNum; taskIdx += coreNum_)`（`..._tiling.cpp`）。

### 11.3 v2 排布（本算子）

约束：chunk 维不可并行（`dH`/`P` 链式），因此并行度只能来自 head；v2 在**核内跨 chunk 排流水**。

**缓冲**（把 v1 的 1 份 slot 改成 2 window × 4 head，沿用 finalize 的 `WORKSPACE_BUFFER_COUNT_8`）：

| 缓冲 | v1 | v2 |
| --- | --- | --- |
| slot（Q̄s/K̄/W/do + decayK） | `blockDim` × 1 | `blockDim` × **8**（2 window × 4 head），索引 `(coreIdx*8 + (chunkIdx&1)*4 + headInGroup)` |
| dH ping-pong（FP32 + dH_bf） | 2 × `blockDim` | 2 × `blockDim` × **4 head**（window 索引与 slot 对齐） |
| P ping-pong（FP32 + P_bf） | 2 × `blockDim` | 2 × `blockDim` × **4 head** |
| T1 / P_c / dV_pre / dV̂ / qterm / wterm | 1 份/核 | 2 window × 4 head |

**核间协议**：6 个 flag（`V0/C1/V2/C3/V4/C5_DONE` 严格交替）→ **每方向 1 条 ready 链**
（`VEC_TO_CUBE_READY = 1`、`CUBE_TO_VEC_READY = 3`），配合 window 索引形成"生产者领先 win=1"的握手：

```text
win = chunkIdx & 1
AIV:  … V0(c) → SET(vec→cube)   [等 cube→vec 的 win 空出] V0(c+1) …
AIC:  WAIT(vec→cube) → C1(c) → C3(c) → C5(c) → SET(cube→vec)
```

**波前（wavefront）排布**（`c` 为 chunk 序号，逆序；`w = c & 1`）：

| 时间片 | AIV（VEC→CUBE=1） | AIC（CUBE→VEC=3） |
| --- | --- | --- |
| t0 | V0(c0) → set#1 | 等 #1 |
| t1 | V0(c1) **（w=1，与下片并行）** | C1(c0)+C3(c0)+C5(c0) → set#3 |
| t2 | V2(c0) → V4(c0) → set#2 | 等 #2 → C1(c1)+C3(c1)+C5(c1) |
| t3 | V2(c1) → V4(c1) → set#3 | 等 #3 → C1(c2)… |

即 v1 的「V0→等→V2→等→V4→等」变成了 v2 的「AIV 一路向前做操作数准备与后处理，AIC 只在自己的
window 就绪后连续发 4 head × 5 GEMM」，AIC 不再空等。

**每核任务**：`blockDim = min(aicCoreNum, ceil(Hv/4) * tileNum)`，每核一组 4 个 head（GVA 下按 `headRatio`
折算），列 tile（K/V 方向）仍在核内串行；若 `Hv/4 < 核数` 则启用 `SPLIT_BY_TILE` 把 `(head-group, tile)` 展平。

**段内**：`SetFlag`+紧邻 `WaitFlag` → per-slot 事件数组（`mte2ToV_[2]`、`vToMte3_[2]`、`mte3ToMte2_[2]`），
只在真正消费点 wait，搬入/计算/搬出跨 slot 重叠。

### 11.4 代码改动清单

1. `op_kernel/<op>_policy.h`：flag id 改为每方向一条 ready 链；新增 `Window/HeadsPerGroup` 常量与
   `SlotIndex(coreIdx, window, headInGroup)` 寻址函数。
2. `op_host/op_tiling/*`：`blockDim = min(aicCoreNum, ceil(Hv/4)*tileNum)`；workspace 按 `blockDim*8` 规划；
   `KERNEL_TASK_TYPE` 改 `MIX_AIC_1_2`。
3. `op_kernel/<op>.cpp`（入口）：`MIX_AIC_1_1` → `MIX_AIC_1_2`；AIV 用 `GetSubBlockIdx()` 分包。
4. `op_kernel/arch3x/<op>_vector.h`：`Process()` 改为 `for (win …)` 的窗口循环 + per-slot 事件数组；
   V0/V2/V4 按 4 head 合并搬运（整 chunk 平面一次搬，替代逐 `[64,128]` 小包）。
5. `op_kernel/arch3x/<op>_cube.h`：`Process()` 改为窗口循环；一次搬入 4 head 的 K̄/W/Q̄s/do 到 L1
   （`L1_RESIDENT_HEAD_COUNT_4` 布局），再对 4 head 连续发 GEMM；`RunGemm` 前的 3 个全阻塞同步改为 per-slot 事件。

### 11.5 预期收益与验收

### 11.6 v2.x 实测进展（A5 / 246，gva，Hv=32，T=2048，32 chunk，28 核）

| 版本 | 改动 | Task Duration | 相对 v1 | 关键 profile |
| --- | --- | --- | --- | --- |
| v1 | 严格交替 + 单窗口 | 2169.3 µs | 1.00× | AIC 等 flag 95.7%；AIV vec 21%/mte2 42.5%(29 GB/s)/mte3 17% |
| v2.1 | 双 window + per-window ready 链（V0 领先 2 个 chunk） | 2052.1 µs | 1.06× | 同上，`C1→V2→C3→V4` 链长不变 |
| v2.1b | + `MmadPingpong<ArchTag,false>`（L0C 2 stage，关闭 unit flag） | 2058.7 µs | 1.05× | fixpipe 99.9% → **6.5%**（8.6 → 132 GB/s），证明该指标是 unit flag 口径 |
| v2.1c | + `CDHP_VEC_TILE` 16 → 32（搬运笔数减半） | 1465.0 µs | 1.48× | AIV mte2 39.4%（57 GB/s）、vec 26.7%、scalar 8.5% |
| **v2.1d** | + 搬入/搬出各用**独立双槽暂存区**（去掉全排空） | **1223.1 µs** | **1.77×** | AIV mte2 38.7%（57 GB/s）、vec 31.7%、mte3 17.0%、scalar 10.5%；AIV 各 pipe 之和 ≈ 总时间，仍未重叠 |
| **v3** | 代数改写：删掉 V2 级 / `dV_pre` / `dV̂ᵀ` / `P_c`；`AB = Q̄sᵀ@do + Wᵀ@(-dv)` 用一次合并 GEMM；状态链 `dH_new = decay⊙dH + AB + (-T1)@dH`（2 步）；`P_new = decay⊙P + (-T1)@P` | **1171.0 µs** | **1.85×** | AIC cube 5.9%、fixpipe 10.7%(124 GB/s)、mte1 4.4%、mte2 11.8%；AIV mte2 44.0%（55 GB/s）、vec 29.5%、mte3 21.2%、scalar 10.1%（仍近乎串行） |
| **v3b** | + 状态链的 3 笔搬入改为"背靠背下发、统一等"（`IssueLoadPlaneF32`/`WaitPlaneLoad`） | **1145.6 µs** | **1.89×** | AIV mte2 41.2%（463 µs）、vec 31.0%、scalar 10.2%；各 pipe 之和仍 ≈ 总时间 |

**搬出（MTE3）延后这一半会挂死**：把 `StorePlaneF32` 换成 `IssueStorePlaneF32` + 下一次复用前 `WaitPlaneStore`
的写法（信用在 InitBuffer/循环前置一次）会让 kernel 卡在首个 launch（`run_case` 10 分钟 0 进展）。
当时把搬入延后保留、只回退搬出延后即恢复正常，说明问题在 MTE3 侧信用/排空语义（`PipeBarrier<PIPE_MTE3>` 可能同时
在承担跨核 flag 前的排空职责）。后续若要再动 MTE3，需要单独隔离验证。

### 11.7.1 A5 逐档实测（v3b，msopprof `Task Duration`）与 H20 对照

| 用例 | Hk/Hv | T_rank | A5 | H20 | 比值 |
| --- | --- | --- | --- | --- | --- |
| gva_cp2 | 16/32 | 65536 | 35.8 ms | 10.35 ms | 3.46× |
| gva_cp8 | 16/32 | 16384 | 9.0 ms | 2.59 ms | 3.46× |
| gva_cp64 | 16/32 | 2048 | 1.1 ms | 0.42 ms | 2.73× |
| h8_cp2 | 8/8 | 65536 | 17.7 ms | 5.13 ms | 3.45× |
| kda_cp2 | 64/64 | 65536 | 52.9 ms | 14.32 ms | 3.70× |
| long_cp2 | 16/32 | 131072 | 71.5 ms | 20.66 ms | 3.46× |

稳态每 chunk 每核 ≈ 35 µs（35.8 ms / 1024 chunk），与 T=2048 档一致；AIV 每 chunk ≈ 52 条搬运/向量指令
（每条约 0.7 µs）是当前主成本，因此**减少指令条数与把 AIV 从链上摘下来**是下一步的两条主线：

1. 合并 slot 落盘（`[Q̄s|W]`、`[do;-dv]` 各一次连续搬出）、去掉可省的中间平面；
2. 用 `MIX_AIC_1_2` 让两个 AIV 分担 V0 与 dtype 转换（finalize 已验证的核型），预计 AIV 侧时间近似减半；
3. 之后把 arch35 的 v2.1/v3/v3b 一次性镜像到 `arch22/vector.h` + `arch22/cube.h`（A2 侧目前仍是 v1 结构）。

## 12. v5 排布：参考 `ChunkFwdH`（`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h`）

v3 之后 A5 仍差 3.46×，瓶颈是"状态每 chunk 经 GM workspace 回写再读回"与"两个核逐 chunk 交替"。
`ChunkFwdH` 用同一套 Cube/Vector 分工解决了同类问题（它也是跨 chunk 的状态递推 `R_next = decay*R + D`），
可直接照搬的写法如下（原文位置：`chunk_fwd_h/docs/design.md` §3~§6、`op_kernel/chunk_fwd_h_policy.h`）。

### 12.1 可照搬的六条

| # | ChunkFwdH 写法 | 对本算子的映射 |
| --- | --- | --- |
| 1 | `MIX_AIC_1_2`：1 AIC + 2 AIV（`FWD_H_AIV_COUNT=2`）；AIC 每 round 4 个 head 的 L1 槽；`AIV0` 处理 round head 0/2、`AIV1` 处理 1/3；每个 AIV 两个 local slot（ping-pong） | 我们的 head 维并行度是唯一来源（Hv=32），现在 1 AIC:1 AIV、每核 1 head、每 chunk 6 次交替；改成 1:2 后 AIV 侧工作量近似减半，且一个 AIC 能同时喂两个 AIV |
| 2 | **rolling state 常驻 UB**："仅首块读取 initial state、末块写 final state，不再逐 chunk 经 GM workspace 回写和恢复" | 我们现在每 chunk 把 `dH`（fp32+bf16）和 `P`（fp32+bf16）各读写一遍 ≈ 384 KiB/chunk/head；常驻后只剩链上必需的输入（AB/Z/ZP） |
| 3 | A5 上 **L0C 经 Fixpipe 直接写入配对 AIV 的 local UB slot** | 我们的 `AB`(链外) / `Z` / `ZP` 三个矩阵现在都落 GM 再被 AIV 读回；直送后省掉 3 个平面的 GM 往返（这解释了我此前"让 fixpipe 直接写 bf16"的方向是对的，但要走 L0C→UB 通路而不是 GM） |
| 4 | ready/free 双向 credit：每个 local slot × 每类数据用**同一个 ID 双向计数**，ready 由真实生产 pipe 发布、free 由最后消费者发布，复用前必须完成上一代 wait；A5 的 AIV 本地 ID 0..10，AIC 侧用 16-ID 步长区分 AIV0/AIV1 | 替换我现在 5 条单向 ready 链（`V0/T1/T1BF/Z/STATE`）；双向 credit 才能让"生产者领先消费者"真正成立 |
| 5 | 跨 chunk lookahead：AIC 把 W/K 放 L1 slot 0/1、2/3 按 chunk 奇偶轮转，当前搬运下发后立即预取下一 chunk；AIV 用两个独立 input bank 轮转 U/g | 我们的 V0（门控 + 4 个操作数平面）由 AIV 做，按"当前 bank 消费前先下发下一 bank"轮转；AIC 侧把它的两个矩阵操作数（`W` 与 `-K̄`）按 chunk 奇偶放 L1 槽 |
| 6 | 尾块用 MTE2 `InitConstValue` 清零 L1 NZ 槽再覆盖有效行；流式数据（W/U/g/gk）走 L2 bypass，递推数据（state/right）保留默认策略 | 我们目前靠 V0 的 `Duplicate(0)` + 部分行覆盖，等价；L2 策略直接照搬 |

### 12.2 存储预算（决定哪些状态能常驻）

每个 head 的 rolling state：`dH` = `K*V*4` = 64 KiB，`P` = `K*K*4` = 64 KiB，合计 128 KiB。
A5 每 AIV 的 UB 是 256 KiB（ChunkFwdH 用到 226 KiB）。四 head/round 下每个 AIV 要管两个 head：

| 方案 | 布局 | 结论 |
| --- | --- | --- |
| A（推荐） | `dH` 常驻（2 head × 64 KiB = 128 KiB）+ P/D 物理数据槽 64 KiB + bf16 work bank 32 KiB + gate bank 2 KiB ≈ 226 KiB | 与 ChunkFwdH 的布局同量级；**`P` 链仍走 GM scratch**（每 chunk 读 64 KiB + 写 64 KiB），但它是"次链"，被 `dH` 链与 Cube 的 MMA 掩盖 |
| B | 两个 AIV 分工：AIV0 管 `dH` 链、AIV1 管 `P` 链（各 2 head × 64 KiB = 128 KiB） | 省 GM，但两条链的 ready/free 要跨 AIV 协调，复杂度高于收益 |

### 12.3 预期收益与验收

| 阶段 | 预期 | 依据 |
| --- | --- | --- |
| v5.1 核型 1:2 + round 4 head + L1 槽轮转 | A5 3.46× → ≈ 2.0× | AIV 工作量近似减半（对应 §11.7.1 的 AIV 52 条指令/chunk） |
| v5.2 state 常驻 UB + L0C→UB 直送 + 双向 credit | A5 → ≈ 1.2–1.4×（目标 1.25×） | 去掉每 chunk ~384 KiB 的状态 GM 往返与 3 个中间平面往返 |
| v5.3 跨 chunk lookahead（AIV input bank + AIC W/-K̄ 槽轮转） | 收尾到 ≤1.25× | ChunkFwdH 实测该条贡献显著（当前 chunk 的 MTE2 与上一 chunk 的 MTE1/MMAD 重叠） |

同时这套改写会**顺带消除当前的一个正确性缺口**：短链（NT=1/2/尾块）用例的 `P` 面初值（单位阵）
依赖"跨核 GM 平面初始化 + 跨核 flag 可见性"，实测该初值没有落到 workspace（两种注入方式都试过：
RegBase VF 对角注入、标量 `SetValue` 注入），长链用例因 `P` 面判据退化而掩盖。改成"state 在 UB 内初始化、
仅末块写回"后，这个跨核初始化路径整体消失。

### 12.4 A2（arch22）现状：结构已对齐，向量实现是新的瓶颈

arch22 的 v3 移植已完成，A2 与 A5 的精度**逐位一致**（gva 1.760e-03 / kda 1.919e-03）。但 A2 的逐档性能
仍差 6~9×（同一份 v3 结构）：

| 用例 | T_rank | A2(v3) | A5(v3b) | H20 | A2/H20 | A5/H20 |
| --- | --- | --- | --- | --- | --- | --- |
| gva_cp2 | 65536 | 82.2 ms | 35.8 ms | 10.35 ms | 7.94× | 3.46× |
| gva_cp8 | 16384 | 20.6 ms | 9.0 ms | 2.59 ms | 7.95× | 3.46× |
| gva_cp64 | 2048 | 2.6 ms | 1.1 ms | 0.42 ms | 6.08× | 2.73× |
| h8_cp2 | 65536 | 40.9 ms | 17.7 ms | 5.13 ms | 7.96× | 3.45× |
| kda_cp2 | 65536 | 131.7 ms | 52.9 ms | 14.32 ms | 9.20× | 3.70× |
| long_cp2 | 131072 | 163.3 ms | 71.5 ms | 20.66 ms | 7.90× | 3.46× |

折算到每 head-chunk：A2 ≈ 40 µs、A5 ≈ 35 µs、H20 ≈ 5 µs。两者结构相同，差距来自 **arch22 的向量实现**：
它按行做 `ExpScalar` + 逐行 `Muls`（每 chunk 每 head 64 次标量 exp + 逐行向量调用），而 arch35 用 RegBase
`__simd_vf__` 融合。因此 A2 要达标（0.4× H20 ⇒ 2.5× H20），需要两条并行工作：

1. **arch22 向量化**：把门控指数（`exp2(g_last-g)`、`exp2(g)`）按行一次性算出（`Exp` 一条指令 + 广播乘），
   去掉 64 次 `ExpScalar` 与逐行 `Muls`；
2. v5 结构（同上 12.1~12.3）：`MIX_AIC_1_2` + round 4 head + state 常驻 + L0C→UB 直送 + 双向 credit。

### 12.5 已执行：arch22 门控向量化（A2 第一刀）

`StageV0` 里原来每行调 2 次 `ExpScalar`（每 chunk/head 128 次 V↔S 往返）；现在按 chunk 一次算好
`qFac_r = exp2(g_r)·scale`、`kFac_r = -exp2(g_last-g_r)`（6 条向量指令 + 1 次 V→S），tile 循环只做逐行 `Muls`。
精度与改造前**逐位一致**（A5/A2 均 gva 1.760e-03、kda 1.919e-03）。A2 逐档：

| 用例 | 改造前 | 改造后 | A2/H20（前→后） |
| --- | --- | --- | --- |
| gva_cp2 | 82.2 ms | **68.0 ms** | 7.94× → **6.57×** |
| gva_cp8 | 20.6 ms | **16.9 ms** | 7.95× → **6.51×** |
| gva_cp64 | 2.6 ms | **2.1 ms** | 6.08× → **5.04×** |
| h8_cp2 | 40.9 ms | **33.5 ms** | 7.96× → **6.53×** |
| kda_cp2 | 131.7 ms | 132.2 ms（gk 路径本就没有逐行 exp） | 9.20× → 9.23× |
| long_cp2 | 163.3 ms | **136.9 ms** | 7.90× → **6.63×** |

### 12.6 A2 新剖面指出的下一步：T1 转换仍在关键链上

改造后 A2（gva T=2048，blockDim=20）的 AIC 剖面：cube 3.3%、mte1 3.2%、mte2 4.5%、fixpipe 12.2%，
而 **`scalar_wait_id4 = 901 µs` + `id5 = 895 µs`（合计 88%）** —— `CdhpFlag(CDHP_FLAG_T1BF_READY, win)`，
即 AIC 在等"T1 的 bf16 转换"。这与 §12.1 第 5 条（fwd_h 的 lookahead）对应：把 Cube 的链外工作
（`T1`、`AB`）**提前一个 chunk** 计算，让下个 chunk 的 T1 转换落在当前 chunk 的 `Z/ZP` MMA 窗口内，
就把这一等从关键链上摘掉。AIV 侧当前是 mte2 37.6%（35.7 GB/s）+ mte3 27.9%（32 GB/s）+ vec 21.4%
+ scalar 30.7%，说明**搬运条数与有效带宽**仍是墙，需要 §12.1 第 2/3 条（state 常驻 + L0C→UB 直送）
一起解决。

### 12.7 已执行：Cube 链外 lookahead（A5/A2 都无性能变化 —— 但说明了真正的墙）

按 §12.1 第 5 条把 Cube 的链外工作提前一个 chunk：序言先算第一个 chunk 的 `T1/AB`，之后每轮
先补下一个 chunk 的 `T1/AB`，再做本 chunk 的 `Z/ZP`。这样 `T1BF_READY` 的等待在时序上被前移的
转换消掉（改造前 A2 上该项占 AIC 时间 88%）。

实测（两平台都逐位一致的精度：gva 1.760e-03 / kda 1.919e-03）：

| 用例 | A5（改造前→后） | A2（改造前→后） |
| --- | --- | --- |
| gva_cp2 | 35.8 → 35.8 ms（3.46×） | 68.0 → 68.6 ms（6.63×） |
| gva_cp64 | 1.1 → 1.1 ms（2.73×） | 2.1 → 2.1 ms（5.09×） |
| kda_cp2 | 52.9 → 52.9 ms（3.70×） | 132.2 → 131.6 ms（9.19×） |
| long_cp2 | 71.5 → 71.5 ms（3.46×） | 136.9 → 136.5 ms（6.61×） |

**结论（重要）**：lookahead 不改变吞吐，因为两平台都是 `aic_time ≈ aiv_time`（A5 2044 vs 2049；
A2 2033 vs 2052），流水吞吐由**较慢的一侧——AIV 的每 chunk 工作量**决定，AIC 的等待并不在墙
上。所以后续必须直接砍 AIV 的每 chunk 工作量：①state 常驻 UB（省 384 KiB/chunk/head 的状态往返）；
②`AB/Z/ZP` 改 L0C→Fixpipe 直送 AIV 的 UB（省 3 个中间平面的 GM 往返）；③`MIX_AIC_1_2` 让两个 AIV
分担 V0/转换（把每 AIV 的每 chunk 工作量近似减半）。lookahead 保留：一旦 AIV 被加速，AIC 侧那 88%
的等待就会变成新墙，这一条正是为它准备的（且是 ChunkFwdH 的原写法）。

精度（A5/246）：`gva_cp64 E=1.760e-03`、`kda_cp64 E=1.919e-03`（与 v1 的 1.837e-03/1.900e-03 同量级，改写带来的舍入路径差异），均 PASS。

v3 落地过程中修掉的两个协议级问题（很重要，写下来避免重犯）：

1. **首 chunk 的 `T1BF_READY` 无人置位** → Cube 第 0 轮永久等待。原因是把 T1 转换排成了"下一 chunk"；
   正确做法是同轮内"先转换、再等 Z/ZP"（转换与 Cube 的 AB 计算并行，仍不进关键链）。
2. v3 的 flag 由 6 条严格交替改成 5 条 ready 链（`V0/T1/T1BF/Z/STATE`），每条按 window 分 2 个 id；
   `STATE_READY` 在 `InitState` 之后要多置一次（初值 dH=0、P=I 也算"状态就绪"）。

精度：每一步在 A5 与 A2 上都是 **逐位一致**（gva_cp64 `E=1.837e-03`、kda_cp64 `E=1.900e-03`，无 NaN/Inf）。

踩过的坑（写入文档避免重复）：

1. `CDHP_VEC_TILE = 64`（一次吃满 chunk）会让 kernel **挂死**——arch35 的 RegBase VF 单次处理 64x128 元素超限；32 行可用。
2. 双槽暂存区若让 **搬入路径等 `MTE3_MTE2` 信用**（该信用只由搬出路径置位），长期计数不对等 → 永久阻塞；必须让搬入/搬出各自独立成对（本版做法）。
3. 门控（g/gk 行）不能再借用搬运暂存槽，必须单独一块 `gDT_`；否则 gk 路径数值崩（`E` 6.5e-01）。

### 11.7 v3 计划：把状态链从 4 步压到 2 步

代数等价改写（与 `reference.py` 逐项对应）：

```text
inc      = Q̄sᵀ@do + Wᵀ@dV̂ᵀ            （现状 C3）
dV̂ᵀ     = -(K̄@dH + dv)              （现状 C1/V2）
⇒ inc    = AB + (-T1)@dH,  AB = Q̄sᵀ@do + Wᵀ@(-dv),  T1 = Wᵀ@K̄
⇒ dH_new = decay⊙dH_prev + AB + (-T1)@dH_prev
   P_new = decay⊙P_prev + (-T1)@P_prev
```

即：`V2` 整级可以删掉，`AB` 与 `T1` 都**不依赖状态**（属链外工作），链上只剩
`Z = (-T1)@dH_prev` 一次 MMAD + 一次向量累加 = **2 步**；`P` 链同理。

落地要点（本轮已完成前半部分）：

1. ✅ slot 布局改成 `Q̄s|W|K̄|do|negdv|decay`：`Q̄s` 与 `W` 相邻、`do` 与 `negdv` 相邻，于是
   `AB` 可用**一次** GEMM 完成（`A=[Q̄s|W]ᵀ`、`B=[do;-dv]`，k=2M），省掉一次 MMAD 与一次累加。
2. ✅ 平面集合改成 `T1`(FP32 [K,K]) / `T1Bf`(模型 dtype [K,K]，Cube 当矩阵操作数) / `AB` / `Z` / `ZP`，
   删除 `dV_pre`/`dV̂ᵀ`/`P_c`；`negdv` 放进 slot。
3. ⏳ Vector：`V0` 增写 `-K̄` 与 `-dv`；新增 `StageT1Convert`（T1 FP32→模型 dtype，链外）、
   `StageStateDH`（`dH = decay⊙dH + AB + Z`）、`StageStateP`（`P = decay⊙P + ZP`）。
4. ⏳ Cube：链外 `T1`+`AB` → 等 `T1Bf` → 链上 `Z`/`ZP` → 通知 Vector。
5. ⏳ flag 改成 5 条 × 2 window：`V0_READY / T1_READY / T1BF_READY / Z_READY / STATE_READY`。

**跨 chunk 关键路径决定了上界**：`C1 → V2 → C3 → V4 → C1(next)` 是不可消除的依赖链（`C1(c+1)` 需要 `V4(c)`
刚写出的 `dH` 状态），因此双 window 只能把 `V0` 以及 `C5` 藏到别的 chunk 下面。可回收的部分是实测的
「AIC 等 V0 42.6%」与「AIV 等 C5 8.6%/AIC 等 V4 41.6% 中可重叠的那一段」。

| 阶段 | 预期 | 依据 |
| --- | --- | --- |
| v2.1 双 window + ready 链（本版） | A2 10.71 ms → ≈ 6.5–7.5 ms；A5 2.14 ms → ≈ 1.4–1.6 ms | 消掉 AIC 等 V0 的 42.6%；`C1→V2→C3→V4` 关键链长度不变 |
| v2.2a 去掉 `RunGemm` 里每次 MMAD 前的三个全排空 + unit flag = `false`（L0C 2 stage） | 先测 AIC 侧串行化程度 | 现在 5 个 GEMM/chunk × 3 个全排空把 M/MTE1/MTE2/FIX 全串起来；AIC 各 pipe 只占 3–7% 说明时间花在等待上 |
| v2.2b 融合（把 V2 折进 C1 的写回路径、V4 折进 C3 的累加）+ 4 head 批处理 | A2 → ≈ 3–3.5 ms；A5 → ≈ 0.7–0.9 ms | 关键链从 4 段降到 2 段；MTE 有效带宽 29–34 GB/s → ≥150 GB/s |
| v2.3 1:2 核型 + 列 tile 并行（`SPLIT_BY_TILE`）+ 把 dH/P 常驻 L1 | 目标 A5 ≈ 0.4–0.6 ms（对齐 H20 同档 0.42 ms） | AIV 工作量是 AIC 的 ~3 倍，需要两个 AIV 分担；并行度 Hv → Hv×tileNum |

验收：`tests/atk/chunk_delta_h_bwd_preprocess/` 的 12 条正向用例 A2/A5 必须仍 12/12 PASS（数值与 v1 逐项一致），
再用 `cp_pprof_pipe.sh`（A5/A2 同口径）对比 `aic_time`、cube 占用率与 AIV 的 MTE 带宽。

## 13. v5 实施结果：rolling state 常驻 UB（参考 `ChunkFwdH`）

### 13.1 落地内容（arch35 / arch22 同步）

照搬 `ChunkFwdH` 的机制 2（rolling state 常驻 UB），两个架构的 Vector 侧都改成：

1. 新增 `dhStateF32_`（`K*V*4` = 64 KiB）常驻 UB；`InitState` 只做 `Duplicate(dhState, 0)`；
2. 新增 `StageStateStore(window, isFirstChunk)`：把"本 chunk 之前的"状态写成 Cube 需要的模型 dtype 操作数
   ——`dH` 直接由常驻状态转 bf16 落 `DhBfAt(prevWin)`；`P` 首 chunk 内联生成单位阵、之后从
   `PAt(prevWin)` 读回再转 bf16；末尾置 `STATE_READY(prevWin)`；
3. `StageState` 的 `dH` 更新改为**就地**读写 `dhStateF32_`（arch35 传给 `CDHP_V4StateUpdateVF` 的
   `state` 指针按 `r0*V` 偏移；arch22 直接对 `dhState[(r0+r)*V]` 切片做 `Muls/Add`），不再读写
   `DhAt(parity)`，也不再写 `DhBfAt(window)`；
4. `WriteOutput` 的 E 面直接从 `dhStateF32_` 搬出（去掉一次 `LoadPlaneF32`，改成 V→MTE3 顺序）；
5. `arch22` 的 `Process()` 与 `arch35` 对齐：删掉循环前的 `STATE_READY(chunkNum&1)` 与循环内的
   `STATE_READY(win)`，统一由 `StageStateStore` 在放行 Cube 之前置位。

### 13.2 精度：短链 `P` 面初值的缺口被消掉

模型规模用例与改造前**逐位一致**（A5/A2 均 gva_cp64 `E=1.760e-03`、kda_cp64 `E=1.919e-03`）。
更重要的是此前一直 FAIL 的短链用例现在通过（A5 与 A2 数值完全相同）：

| 用例 | 改造前 `P_rel_norm` | 改造后 |
| --- | --- | --- |
| sh_single_chunk（T=64） | ≈ 9.9e-01 FAIL | **6.2e-05 PASS** |
| sh_two_chunk（T=128） | ≈ 9.9e-01 FAIL | **1.3e-04 PASS** |
| sh_tail（T=200，尾块） | ≈ 9.9e-01 FAIL | **4.9e-04 PASS** |

根因：`P` 的单位阵初值原来靠 `InitState` 早期单独落 GM 平面 + `STATE_READY` 跨核可见性，实测没有落到
workspace；现在改成"首 chunk 的 `StageStateStore` 里内联生成、紧跟 flag 之后"（arch35 用 RegBase
`CDHP_InitIdentityVF`，arch22 用 `Duplicate` + `SetValue` 对角注入），跨核初始化路径整体消失。

### 13.3 性能：**no-op**（数据量不是瓶颈）

| 用例 | A5 v3b | A5 v5 | A2 前 | A2 v5 |
| --- | --- | --- | --- | --- |
| gva_cp2 (T=65536) | 35.8 ms | 36.5 ms | 68.0 ms | 68.9 ms |
| gva_cp8 (T=16384) | 9.0 ms | 9.1 ms | 16.9 ms | 17.2 ms |
| gva_cp64 (T=2048) | 1.1 ms | 1.2 ms | 2.1 ms | 2.1 ms |
| h8_cp2 (T=65536) | 17.7 ms | 17.9 ms | 33.5 ms | 34.1 ms |
| kda_cp2 (T=65536) | 52.9 ms | 54.0 ms | 132.2 ms | 133.3 ms |
| long_cp2 (T=131072) | 71.5 ms | 73.0 ms | 136.9 ms | 137.8 ms |

省掉每 chunk 每 head 的 128 KiB FP32 状态 GM 往返（读写各 64 KiB）后**时间零变化**，
说明 AIV 不是被搬入搬出的**数据量**卡住。

### 13.4 新剖面：两侧都被"指令下发 + 屏障停顿"占满（A5 / gva / T=2048 / 28 AIC + 28 AIV）

`msopprof --aic-metrics=PipeUtilization` 汇总（均值）：

| AIC | cube 3.6% | scalar **43.4%** | mte1 2.8% | mte2 6.9% | fixpipe 6.2% |
| --- | --- | --- | --- | --- | --- |
| AIV | vec 18.6% | scalar **41.3%** | mte2 30.9%（有效 46 GB/s） | mte3 11.6% | icache miss 1.3% |

两条结论：

1. **AIV 各 pipe 之和 ≈ 102%**（vec+scalar+mte2+mte3）⇒ 搬入 / 计算 / 搬出之间**完全没有重叠**，
   全部被串行化在标量线程的"下发 → 等屏障 → 再下发"节奏上；
2. 两侧**最忙的都是 scalar pipe**（41~43%），而真正干活的 pipe（cube 3.6%、mte2 7~31%）都远未打满
   ⇒ 当前成本 = 指令条数 × 每条指令前后的 `SetFlag`/`WaitFlag` 停顿，不是带宽也不是算力。
   （折算：每 head-chunk 35 µs ≈ 50 条指令 × ~0.7 µs。）

### 13.5 下一步优先级（按"每改动一处的收益/风险"排序）

| 优先 | 改动 | 依据 |
| --- | --- | --- |
| 1 | **去掉逐笔搬运前后的全排空**：`LoadTileF32`/`StoreTileModel`/`LoadPlaneF32`/`StorePlaneF32` 现在每笔都做 `SetFlag+紧邻 WaitFlag`（等于把 V/MTE2/MTE3 排空一次，StageV0 每 chunk 约 40 次）；改成 `ChunkFwdH` 的 **per-slot credit**（`IoFree/WorkFree/GateFree` 各 slot 预置一次、消费者用完再置位）+ 5 个平面**批量下发、统一等** | §13.4：pipe 串行化的直接来源 |
| 2 | **减少 `RunGemm` 调用数与每次调用的 3 个全排空**：可先把 `Z` 与 `ZP` 合并成一次 GEMM（A 都是 `T1Bf`，B 取 `[dH_bf | P_bf]` 的 `[K, V+K]` 拼接，C 为 `[K, V+K]`） | AIC scalar 43%，4 次 GEMM/chunk × 3 排空 + 每次 `BlockMmadTla` 构造的事件初始化 |
| 3 | **状态链代数合并**：`decay⊙dH + (-T1)@dH = P_c @ dH`（`P_c = diag(decay) - T1`，Vector 侧的 `CDHP_V4PcDiagVF` 已实现同一表达式），于是 `dH` 与 `P` 可堆叠成 `[K, V+K]` 一条状态、每 chunk 只发 1 次链上 GEMM；代价是 `P_c` 需要按模型 dtype 再舍入一次（精度需回归验证） | 与 2 合并后 AIC 每 chunk 只剩 3 次 GEMM |
| 4 | `MIX_AIC_1_2` + round 4 head + L1 槽轮转 + L0C→UB 直送（§12.1 机制 1/3/4） | 前 3 项落实后再评估；1:2 只对"AIV 工作量占主导"有效，而 §13.4 显示两侧近似对称，单独上 1:2 收益有限且改动最大 |

已排除/勿重犯：`CDHP_VEC_TILE=64` 挂死；搬出延后（`IssueStorePlaneF32` + 复用前 `WaitPlaneStore`）挂死；
fixpipe 直接落 bf16（`TileCopyColRowBf16`）结果为空；状态常驻 UB 本身对性能是 no-op（本节的负结果）。

## 14. v8：1 AIC : 2 AIV 核型（fwd_h 机制 1）——实测 1.69×，但被 P 面时序问题挡住

### 14.1 已落地的框架（当前**默认关闭**）

`op_host` 的 `aivPerBlock` + `op_kernel/*_common.h` 的 subblock 寻址 + `policy.h` 的
`CdhpPeerFlag`（A5 用 16 的 id 步长选择配对 AIV）+ cube 侧"按 chunk 步交错服务两个 head"的
`rounds` 循环，整套框架都在树里，开关只有两处：

1. `op_host/op_tiling/*_tiling.cpp`：`aivPerBlock = 2`（当前写死 1）；
2. `op_kernel/*.cpp`：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（当前 1_1）。

**mode 必须和核型一起切**：1:1 用 `0x2`；1:2 在 A5 上必须 `0x4`（否则 launch 后直接
`synchronize failed 507015`，实测过一次，不是精度问题是运行期错误）。

### 14.2 实测收益（A5，msopprof `Task Duration`）

| 用例 | 1:1 | **1:2** | H20 | 1:2 / H20 |
| --- | --- | --- | --- | --- |
| gva_cp2 (T=65536) | 34.6 ms | **17.5 ms** | 10.35 | 1.69× |
| gva_cp8 | 8.7 | **4.4** | 2.59 | 1.70× |
| gva_cp64 (T=2048) | 1.1 | **0.6** | 0.42 | 1.35× |
| h8_cp2 (Hv=8) | 17.0 | 15.8 | 5.13 | 3.08× |
| kda_cp2 (Hv=64) | 51.3 | **34.2** | 14.32 | 2.39× |
| long_cp2 (T=131072) | 69.4 | **35.5** | 20.66 | 1.72× |

归因（剖面）：1:1 时 AIV 各 pipe 之和 ≈102%（完全串行），是唯一瓶颈；1:2 后两个 AIV 各承包
一个 head，AIC 被喂满（pipe 之和 98.6%），每个 head-chunk 的墙从 ~35 µs 降到 ~17.5 µs。

### 14.3 关于"P 面出错"的更正：那是**A5 机器状态被搞坏**，不是算子问题

本节原先记过"P 面（PAt/PBf/ZP）存在时序可见性缺口"。**该结论是错的，已更正**：

现象：A5 上短链用例 `sh_two_chunk` 的 P 面稳定报错（`P_rel_norm` ≈ 1.3~1.7）、`sh_tail` 挂死，
换任何分核/flag/落盘写法都一样，且同一份**早先验证通过的代码**（`/tmp/cp_v6_verified` 快照）
重新编译后也复现。把同一份代码放到 **A2 机器**上跑，结果与历史**逐位一致**：

```text
smoke      : gva_cp64 E=1.760e-03 PASS  kda_cp64 E=1.919e-03 PASS
short      : sh_single_chunk P=6.202e-05 PASS  sh_two_chunk P=1.334e-04 PASS  sh_tail P=4.872e-04 PASS
```

真因：A5 那台机器上被 `kill -9` 掉过卡住的测试进程，**device 0 至今停在 100% 占用**（`npu-smi`
里 NPU 0 的 Util=100、温度 99℃），`npu-smi set -t reset` 被拒（`User aborts reset chip by outband`，
即芯片仍在被占用）。此后所有落在该机的用例（换到 device 1/2 也一样，驱动侧被拖住）都会随机
报 `synchronize failed 507015`／精度错乱／挂死。**结论：A5 上的短链/精度结论在机器恢复前都不可信，
必须以 A2 或恢复后的 A5 为准。**

顺带确认的两条硬约束（有效，勿踩）：

- `CDHP_CROSS_CORE_MODE` 必须与核型配对：1:1 ⇒ `0x2`；1:2 在 A5 ⇒ `0x4`。用错会**直接**
  `synchronize failed 507015`；
- 在 flag 前加 `SetFlag/WaitFlag<HardEvent::FIX_M>` 不是"等 fixpipe 落盘"的手段（它阻塞的是
  **M pipe**，不是标量线程），真正让 flag 晚于 fixpipe 写完成的是 `CrossCoreSetFlag<.., PIPE_FIX>`
  本身。

### 14.4 v8：P 状态常驻 UB（在 A2 上验证通过；A5 数值待机器恢复后复测）

V0 里 `w` 只做搬运（不改数值），因此改成"只在模型 dtype 上补零搬运"，省下一块 16 KiB FP32 tile，
正好给 P 状态腾出 64 KiB 常驻 UB：

- `InitState`：`pState = I`（RegBase 对角注入）；`StageStateStore` 的 P 操作数直接由 `pState` 转模型
  dtype；`StageState` 就地累加 `pState = decayK ⊙ pState + ZP`；`WriteOutput` 的 P 面直接读 `pState`；
  同时删掉 `StageState` 里那次与下一轮完全重复的 `PBf` 落盘。
- A2（arch22，未启用该改动）验证：smoke 与短链全部逐位一致（见 14.3 的表格）。
- A5 侧在被搞坏的机器上测到 `gva_cp2 34.6 → 28.9 ms`（3.35× → 2.79×）、`kda 51.3 → 42.6`
  （2.97×）、`long 69.4 → 57.5`（2.78×）——**方向明确（P 的 GM 往返是实打实的开销），但数值需在
  恢复后的 A5 上复测确认。**

### 14.5 下一步

1. **先恢复 A5 机器**（需要重启那台测试机，或由有权限的人 reset 芯片）——否则 A5 的性能/精度
   都无法作为结论；恢复后用 §14.2 的 1:2 + 本节 v8 一起复测；
2. `aivPerBlock=2` + `mode=0x4` 打开后（预期 1.69×），再做 **`AB/Z/ZP` 的 L0C→UB 直送**
   （fwd_h 机制 3，每 head-chunk 省 256 KiB 的 AIV 搬运与同量 cube fixpipe 落盘），继续往
   1.25×（A5）推进；
3. A2 侧：把 1:2 框架接到 arch22（mode `0x2` 集合同步，需要每个 block 恰好 2 个 head 的
   "无 dummy 同步"配置），再削 AIV 工作量，目标 2.5×。

### 14.6 A2（arch22）1:2 尝试失败：0x2 集合同步需要"三方 rendezvous"协议

本轮试过把 1:2 直接接到 arch22（host `aivPerBlock=2` + `KERNEL_TYPE_MIX_AIC_1_2` + cube 的
rounds 交错 + 每 block head 数补齐到 2 的倍数）。结果：**smoke 用例 10 分钟不返回（死锁）**。

原因分析：910B 的 `mode=0x2` 是"AIC:2*AIV 集合同步"——每个 flag id 的语义是 **AIC + 两个 AIV
三方 rendezvous**，而我方协议是按 chunk 递进的 5 条 ready 链（每轮 AIC 与两个 AIV 各 set/wait 一次）。
把这套协议直接套上去后，计数器对不齐就会永久阻塞。要启用必须按 `ChunkFwdH` 的写法逐条核对：

- 每条 ready 的 set 侧必须由**真实生产 pipe** 发布、wait 侧绑定**消费者 pipe**，且两端 AIV 与 AIC
  的 set/wait 次数严格一一对应；
- 每轮参与的 head 数必须是 `aivPerBlock` 的倍数（不足时由缺 head 的 subblock 做 dummy set/wait），
  这一条我已用"补齐 + 重复最后一个 head（幂等）"实现，但显然还缺别的对齐；
- 建议先做一个只保留 1 条 flag 的最小 kernel 验证 0x2 的语义，再回来改本算子。

**当前状态**：arch22 保持 1:1（已实测通过：smoke + 短链三例逐位一致，A2 逐档 6.65×~9.27×）。
`arch22_cube.h` 里的 rounds 交错结构保留（`aivPerBlock=1` 时与改造前语义等价、A2 已验证通过），
`arch22_vector.h` 已回退到验证过的版本。

### 14.7 运维教训（重要）：不要在设备侧内核卡死时 `kill -9`

本算子调优过程中出现过用 `kill -9` 杀掉"卡在设备侧等待"的 `run_case` 进程，导致 **A5(246) 与
A2(221) 的 device 0 至今 100% 占用**、`npu-smi set -t reset` 被拒（`chip in use`），并且同机其他
device 也开始报 `setDevice failed 507033` / `rtGetFaultEvent ... context is a null pointer`
——两台机器的驱动上下文都被拖坏，**任何落在这两台机上的算子精度/性能结论都不可信**。

正确做法：

1. 测试进程一律用 `timeout` 包住，让它自然退出；不要 `kill -9`；
2. 一旦某个 device 出现 100% 占用且进程已不在用户态，就**只**用 `npu-smi set -t reset -i <id>`
   （若被拒，说明有内核占着，需要重启那台机器）；
3. 恢复后先跑一个十几行的 torch 小算子确认 `npu-smi` Util 归零、`torch_npu` 能正常出结果，
   再跑本算子的冒烟用例；
4. 同一台机器上**串行**跑用例，不要并发跑多个 `run_case`（会互相拖慢甚至互相卡住）。

## 15. v9：P 面"时好时坏"的两个真实根因（已定位并修复）

把设备恢复到干净状态后，用 Hv=32、T=128/200 的短链用例做二分，终于把 P 面长期"时好时坏"的问题
钉死成两个具体 bug。**这两个都是真实缺陷，不是机器问题**（14.3 那次的机器问题掩盖了它们）。

### 15.1 `CDHP_InitIdentityVF` 少了"先把整行寄存器清零"这一步

```cpp
// 错（历史写法）：
Duplicate(zeroReg, 0.0f, maskAll);
Add(diagReg, oneReg, zeroReg, diagMask);   // 带 mask 的 Add 是 merge 语义
StoreAlign(dst + off, diagReg, mask);      // mask 是整行 → 非对角 lane 写的是 diagReg 的旧值
```

带 mask 的 `Add(dst, a, b, mask)` 在**未命中 lane** 上保留的是 `dst` 这个**寄存器**里的旧值，
不是内存里的旧值。因此 `StoreAlign` 整行写回时，非对角 lane 落的是上一轮残留在 `diagReg` 里的
脏数据。`CDHP_V4PcDiagVF` 里写法是对的（先 `Duplicate(diagReg, 0.0f, maskAll)` 再带 mask 加），
`CDHP_InitIdentityVF` 漏了这一步。

症状与历史记录完全吻合：**只有 P 面错、E 面永远正常**（I 只进 P 链），且**随负载/时序时好时坏**
（寄存器残留内容变了就换个 head 出错）。修复：VF 内部 `Duplicate(diagReg, 0.0f, maskAll)` 之后再
`Add(diagReg, oneReg, diagReg, diagMask)`；调用方不需要再整块清零。

### 15.2 A5（dav-3510）上暂存区必须用 per-slot credit，不能用自排空

同一份短链用例做 A/B：

| V0 搬入/搬出写法 | A5 短链 P 面 | 说明 |
| --- | --- | --- |
| 5 槽批量搬入 + per-slot credit（v6，本算子里自己发明的那套） | `P_rel_norm≈1.5` FAIL | 只有 P 面错（K̄ 经 T1 只喂 P 链） |
| 逐平面 `LoadTileF32` + 全排空（arch22 在 A2 上验证通过的那套） | E 也错（`E_rel_norm≈12`） | 在 A5 上连基本语义都不成立 |
| **双槽 + per-slot credit（`WaitFlag<V_MTE2>(slot)` / `SetFlag<V_MTE2>(slot)`）** | **PASS（1.334e-04）** | 当前采用（v9） |

结论：在 A5 上复用 UB 暂存区必须让生产/消费两侧用**同一槽的 credit 闭环**，
`PipeBarrier` /「SetFlag 紧跟 WaitFlag」这种自排空不足以保证顺序（A2 上够用，不能跨架构照抄）。

### 15.3 v9 实测（A5，1:1 + 上述两个修复）

```text
smoke : gva_cp64 E=1.760e-03 PASS   kda_cp64 E=1.919e-03 PASS
short : sh_single P=6.202e-05 PASS  sh_two_chunk P=1.334e-04 PASS  sh_tail P=4.872e-04 PASS
perf  : gva_cp2 36.6ms(3.54×) gva_cp8 9.2(3.55×) gva_cp64 1.2(2.76×) h8 18.0(3.51×)
        kda_cp2 54.2(3.78×) long_cp2 73.3(3.55×)
```

（比 v6 的 3.35× 略慢 ~5%，因为 credit 版搬入比 5 槽批量版保守；这是把 P 面修正过来的代价。）

### 15.4 最终方案：1:2 + per-tile P credit（A5 实测 1.71×，全绿）

把 §15.1/15.2 的修复与下面这条合起来，1:2 才算真正落地：

**P 链只由 `StageState` 生产、并且它的 fp32 平面用 per-tile credit 交接**

1. `StageStateStore` 不再碰 P：非首 chunk 只写 `dH` 的模型 dtype 操作数；首 chunk 额外写一次
   `PBf = I`（Cube 的 ZP 操作数），**不写 `PAt`**；
2. `StageState` 是 `PAt`（fp32）与 `PBf`（模型 dtype）的唯一生产者：`P_new = decayK ⊙ P_prev + ZP`
   之后写 `PAt(window)` 和 `PBf(window)`；首 chunk 的 `P_prev = I` 直接内联生成（不读 GM）；
3. `PAt` 的"本核 MTE3 写 → 下一轮本核 MTE2 读"用 **per-tile credit**：4 个 tile 用 4 个独立
   event id（`CDHP_EV_P_TILE0 + tile`），写侧 `SetFlag`、下一轮读侧 `WaitFlag`，同一 id 严格
   set/wait 交替（单比特 event 不允许连续 set 同一 id）；末 chunk 多出的 4 份在 `WriteOutput` 配平。
   这样 P 链每 chunk 每 tile 只有一次跨 GM 交接，且与 §15.2 里验证过的 credit 形态一致。

同时修掉的两处（见 15.1/15.2）：`CDHP_InitIdentityVF` 先清寄存器整行；A5 上暂存区用 per-slot credit。

### 15.5 A5 实测（1:2 + 上述修复，干净设备串行）

```text
smoke  : gva_cp64 E=1.760e-03 PASS   kda_cp64 E=1.919e-03 PASS
short  : sh_single P=6.202e-05 PASS  sh_two_chunk P=1.334e-04 PASS  sh_tail P=4.872e-04 PASS
sweep  : c3(T=192) P=4.35e-04 PASS   c4(T=256) P=1.08e-03 PASS
         c8(T=512) P=5.30e-03 PASS   c16(T=1024) P=3.07e-17 PASS
perf   : gva_cp2 17.7ms(1.71×) gva_cp8 4.5(1.72×) gva_cp64 0.6(1.36×)
         h8_cp2 17.4(3.40×) kda_cp2 37.9(2.65×) long_cp2 35.3(1.71×)
```

对照（同一轮内测的同口径数据，供选型参考）：

| 配置 | 短链/尾块 | gva_cp2 | kda_cp2 |
| --- | --- | --- | --- |
| **1:2 + per-tile P credit（当前）** | ✅ 全绿 | **17.7 ms（1.71×）** | 37.9（2.65×） |
| 1:1 + credit（无 per-tile） | ✅ 全绿 | 36.6（3.53×） | 54.3（3.79×） |
| 1:2 + P 走 GM 自排空（旧） | ❌ 尾块 P 偏 | 18.4（1.78×） | 39.2（2.74×） |
| 1:1/1:2 + P 常驻 UB | ✅ 全绿 | 31.1 / 32.0（3.01× / 3.09×） | 45.9 / 98.8 |

即：**P 常驻 UB 不是必需**（per-tile credit 已经解决时序问题），而且它在当前实现下反而更慢，
所以不采用；A5 现在取"1:2 + per-tile P credit"，**1.71×H20**，与 0.8×H20 的验收线（即 1.25×）
还差最后一步。

### 15.6 当前瓶颈剖面（1:2 + per-tile P credit，gva / T=2048 / 16 blocks × 2 AIV）

| 引擎 | 时间 | 组成（占用率之和） |
| --- | --- | --- |
| AIC（关键块） | 566 µs | mte2 26.6% + **fixpipe 23.3%** + scalar 21.7% + cube 12.3% + mte1 9.2% ⇒ **93.1%** |
| AIV v0 | 566 µs | mte2 39.9%（54 GB/s）+ vec 31.0% + mte3 18.5% + scalar 10.9% ⇒ **100.3%** |
| AIV v1 | 574 µs | mte2 35.9% + vec 30.7% + mte3 18.6% + scalar 10.6% ⇒ **95.7%** |

`aic_time ≈ aiv_time`（566/574）、各 pipe 之和接近 100%、max/mean=1.01 ⇒ **两个引擎都已打满且负载均衡**，
剩下只能靠"减少总工作量"，不能靠调分配。

### 15.7 下一步：收敛最后 ~1.37×（含 UB 预算约束）

1. **`AB/Z/ZP` 走 L0C→UB 直送**（fwd_h 机制 3）——收益最大：每 head-chunk 省 192 KiB 的 AIV
   MTE2（AIV mte2 40% → ~20%）**以及** Cube fixpipe 的 192 KiB GM 写（AIC fixpipe 23% → 接近 0）。
   **但卡在 UB 预算**：arch35 的 AIV 现在已用 178.5 KiB / 256 KiB，而三个 fp32 平面要 3×64=192 KiB。
   两条可行路线：
   - 先把 V0 的 `w`/`do`/`dv` 改成"只在模型 dtype 上搬运/取负"（省 3 块 FP32 tile = 48 KiB），
     再把 AB/Z/ZP 以 **bf16** 落到 AIV 的 UB（3×32=96 KiB，总量 ~226 KiB，能放下）；代价是这三个
     中间量多一次 bf16 舍入（E 面 rel_norm 预计从 1.7e-3 涨到 ~4e-3，仍在 2e-2 之内）；
   - 或者只把其中 1 个平面（64 KiB fp32）直送，收益按比例缩小。
2. `Z` 与 `ZP` 合并成一次 GEMM（A 都是 `T1Bf`，B 拼 `[dH_bf | P_bf]` 的 `[K,V+K]` 行交错布局）：
   每 chunk 4 次 GEMM → 3 次，AIC 侧 ~25% 的 GEMM 工作量。需要改 workspace 平面布局 + 一个带行距的
   `LoadPlaneF32` 变体，改动中等。
3. `PBf/dH_bf → Cube` 这条跨核链做成 fwd_h 的双向 credit（ready/free 成对）——它是 15.6.1 那个
   "改 head 分布就翻车"的根因，也是 h8（3.40×）/kda（2.66×）进一步优化的前提。
4. 之后再回头看 AIV 的 vec 31%（V0 的 10 次 Cast + 门控 VF）与 mte2 带宽（54 GB/s，credit 双槽
   只有 2 深流水）：把搬入槽加到 4~5 个（每槽独立 id、交替使用）能提带宽。

### 15.6.1 已试过并回退：去掉"补齐"、把 blockDim 开到 Hv（小 Hv 想每 head 一个 AIC）

想法：A5 是 0x4 peer 模式，某个 AIV 不参与是允许的，于是 `blockDim = min(Hv, 核数)`、每个 block
只挂 1 个 head（Hv=8 用 8 个 AIC 而不是 4 个）。结果：

```text
smoke PASS   short: sh_single PASS  sh_two_chunk FAIL(P=5.65)  sh_tail PASS
perf : gva_cp2 18.5(1.79×, 反而更慢)  kda 35.3(2.47×)  h8 16.4(3.21×, 只快了 6% 而不是预期的 2×)
```

即：Hv=8 并没有拿到"每 head 一个 AIC"的收益（说明那里不是 AIC 串行而是别的东西在限速），
而 2-chunk 用例的 P 面又被打回失败——**说明还差一条跨核 handoff 的时序保证**：现在 `PBf` 完全由
`StageState` 生产、Cube 在下一轮读，这条跨核链仍靠 `STATE_READY` + `PipeBarrier<MTE3>`（§15.2 已证明
A5 上不够）。要再动 head 分配，先把 `PBf/dH_bf → Cube` 这条也做成 `ChunkFwdH` 那种**双向 credit**
（ready/free 成对），再回头试分布方案。

### 15.7 附：旧的 1:2 尾块记录（已被 15.4 覆盖）

在 §15.1/15.2 两个修复的基础上，把 1:2、P 常驻 UB 两个开关都试过，结果如下（都在干净设备上串行跑）：

| 配置 | 短链正确性 | gva_cp2 | kda_cp2 | 说明 |
| --- | --- | --- | --- | --- |
| 1:1 + credit 搬入，P 走 GM（**当前采用**） | ✅ 三例全过 | **36.6 ms（3.53×）** | 54.3（3.79×） | 最快且全绿 |
| 1:1 + credit，P 常驻 UB | ✅ 三例全过 | 31.1 ms（3.01×） | 45.9（3.20×） | 数值全对但**更慢** |
| 1:2 + credit，P 走 GM | ⚠️ 满块对，**尾块 P 错**（P_max_abs≈1e-4，判据退化放大成 rel≈0.5） | 18.4 ms（1.78×） | 39.2（2.74×） | 最快，但尾块不达标 |
| 1:2 + credit，P 常驻 UB | ✅ 三例全过 | 32.0 ms（3.09×） | 98.8（6.90×） | 修正了尾块，速度反而不如 1:1 |

两个结论：

1. **P 走 GM 时，`PAt`（fp32 P 平面）那条"同核 MTE3 写 → MTE2 读"是尾块/1:2 下 P 面偏差的来源**
   （E 面永远 bit-identical，只有 P 面差；P 常驻 UB 后尾块立刻变绿）；
2. **但 P 常驻 UB 在当前实现下会明显变慢**（1:1 慢 15%、1:2 慢 42%），原因待查
   （怀疑 UB 占用从 178.5 KiB 涨到 242.5 KiB 后编译期调度变差 / AIC 成为瓶颈），
   所以当前先取"1:1 + credit + P 走 GM"这个**最快且全绿**的组合。

### 15.5 下一步（按收益排序）

1. **把 `PAt` 往返换成 per-tile 显式 credit**（4 个 tile × 2 个读者 ⇒ 8 个 MTE3_MTE2 id，需同时去掉
   `IssueLoadPlaneF32` 里那个自排空 id 6 腾出 id 空间；每轮 set/wait 必须严格交替，最后在 `WriteOutput`
   里把末 chunk 多出的 4+4 份 credit 配平）。这条做完就能在 **不** 开 P 常驻的前提下把 1:2 修绿，
   直接拿到 **1.78×（gva）/2.74×（kda）**。
2. 查清 P 常驻 UB 变慢的原因（UB 预算/调度），若能把它的开销压掉，则 1:2 + 常驻 ≈ 更快且更稳。
3. `AB/Z/ZP` 的 L0C→UB 直送（fwd_h 机制 3）：省 256 KiB/head-chunk 的 AIV 搬运 + 同量 cube fixpipe
   落盘，是冲 1.25×（0.8×H20）的主线。
4. A2(221) 恢复后：先按 §14.6 做 0x2 的 rendezvous 协议，再套上面这些。

### 15.6 附：1:2 尾块的旧记录（已被 §15.4 的结论覆盖）

在 v9 基础上打开 `aivPerBlock=2` + `mode=0x4` + MIX_AIC_1_2（A5）：

```text
smoke : PASS            short : sh_single PASS  sh_two_chunk PASS  sh_tail FAIL(P=0.46)
perf  : gva_cp2 18.4ms(1.78×) gva_cp8 4.6(1.78×) gva_cp64 0.6(1.42×) h8 18.0(3.50×)
        kda_cp2 39.2(2.74×)  long_cp2 36.6(1.77×)
```

即 15.1/15.2 修好之后，1:2 的**满 chunk** 路径已经正确（E/P 都对），只剩
**尾块（`T % chunkSize != 0`，例如 T=200）的 P 面**不对（E 面正常）——这是下一步的唯一拦路项；
修掉之后 A5 就能落到 1.78×（再叠加 §14.4 的 L0C→UB 等改动即可冲 1.25×）。

### 14.5 （历史）5 槽批量 V0 在"整块无效行"上的 credit 漂移 —— 该实现已在 §15.2 被 credit 版替换

`StageV0` 的 tile 循环里，当 `validRows == 0`（尾块里 `[rows, chunkSize)` 的那半块 tile）时原来
**跳过 5 笔搬入但照常 5 笔搬出**：搬入路径里的 `WaitFlag<MTE3_MTE2>(slot)` 被跳过、搬出路径里的
`SetFlag<MTE3_MTE2>(slot)` 照旧 → per-slot credit 计数一路漂移，之后会覆盖"正在被 MTE3 读"的
暂存槽。修法：`validRows == 0` 时把这 5 个 wait 补上（保持 set/wait 配对）。

### 13.6 第二个负结果：删掉落盘前后的 `PipeBarrier<PIPE_V>` 也是 no-op

依据 `ChunkFwdH` 的 `GateFreeEvent` 写法（`SetFlag<V_MTE2>` 本身按 V pipe 顺序生效，代码里紧接 VF 之后、
不夹 `PipeBarrier<PIPE_V>`），把三处冗余排空删掉：

1. `LoadTileF32` 里 `Cast` 之后、`SetFlag<V_MTE2>(slot)` 之前的 `PipeBarrier<PIPE_V>`（每 chunk 10 次）；
2. `StoreTileModel` / `StoreModelNoPad` 里 `Cast` 之后、`SetFlag<V_MTE3>(slot)` 之前的同款屏障（每 chunk 10 次）；
3. `StageV0` 里 5 笔 `StoreTileModel` 之前的 `PipeBarrier<PIPE_V>`（同 pipe 顺序执行）。

结果：精度**逐位一致**（gva 1.760e-03 / kda 1.919e-03），A5 逐档 36.5/9.1/1.2/18.0/54.0/72.9 ms
（与改前 36.5/9.1/1.2/17.9/54.0/73.0 相同）。改动本身保留（少 22 条冗余指令、语义等价），
但它证明了**单个屏障不是成本**——成本在"每笔搬运前后的 `SetFlag`+`WaitFlag` 等待"的**次数**上：

按当前结构折算，每个 chunk 每 head 的等待次数约为

| Stage | 搬运笔数 | 每笔等待 | 合计 |
| --- | --- | --- | --- |
| `V0` | 10 载入 + 10 落盘 | 2 + 2 | ≈ 40 |
| `T1Convert` | 8 | 3 + 2 | ≈ 20 |
| `StateStore` | 12 | 2~3 | ≈ 28 |
| `State`（dH + P） | 4×4 | 3~4 | ≈ 70 |
| 合计 | | | **≈ 160 次/chunk** |

即"少而大"的搬运（整平面、批量下发、per-slot credit）比"多而小"的搬运（32 行 tile + 每笔全排空）
在当前指令间隙下要快数倍；这也解释了为什么单纯减数据量（v5）没有收益——数据量没变慢，**笔数**才慢。

### 13.7 下一步的 UB 预算约束（决定先做哪一项）

"整平面 + 批量下发"的前提是让一个 FP32 平面（`[128,128]` = 64 KiB）或半平面（64 行 = 32 KiB）
同时驻留。当前 UB 占用（arch35）已到 178.5 KiB / 192 KiB：

| 缓冲 | 大小 |
| --- | --- |
| `s0F32_~s4F32_`（5 × 32 行 tile） | 80 KiB |
| `s0DT_`/`s1DT_`（搬入/搬出各 2 槽） | 32 KiB |
| `dhStateF32_`（v5 常驻状态） | 64 KiB |
| g/decay/gate/dh 等 | ≈ 2.5 KiB |

因此下一步需要先腾空间，两个可选项：

1. `StageV0` 改成两趟（先 `q/k/w` 再 `do/dv`，3 个 FP32 缓冲即可），`s0F32_~s4F32_` 从 80 KiB 降到 48 KiB，
   腾出 32 KiB 做半平面批量搬运；`P` 链的 `P/ZP` 与 `dH` 链的 `AB/Z` 用同一组半平面缓冲（各自成对下发）；
2. 或者回退 `dhStateF32_`（v5 已证性能 no-op，但保留它能把短链 `P` 初值问题按 §13.2 的方式修掉——
   该修复只依赖 `StageStateStore` 的内联单位阵，不依赖状态常驻），把 64 KiB 让给"整平面 + 批量下发"。

## 16. v10 本轮：T1 直接落模型 dtype + P 状态常驻 UB（A5 17.7 → 15.7 ms，1.71× → 1.51×）

### 16.1 判据与入手点

§15 之后的剖面（`gva` / T=2048 / 16 blocks × 2 AIV）显示 AIC 与 AIV **同时打满**：

| 引擎 | 时间 | pipe 占用率之和 |
| --- | --- | --- |
| AIC（关键块） | 566 µs | mte2 26.6% + fixpipe 23.3% + scalar 21.7% + cube 12.3% + mte1 9.2% ≈ 93% |
| AIV v0 | 566 µs | mte2 39.9%（54 GB/s）+ vec 31.0% + mte3 18.5% + scalar 10.9% ≈ 100% |

`max/mean ≈ 1.01`、两侧之和都接近 100% ⇒ 单边优化无收益，只能砍**两侧共享的搬运**。
本轮因此挑了两条"减少 GM 往返"的改动，都不改数学式子。

### 16.2 改动一：T1 由 Cube 直接落模型 dtype（删掉 Vector 的 `StageT1Convert`）

旧结构是三步握手：`AIC` 写 FP32 的 T1 平面 → `AIV` 读回、转模型 dtype、写另一份平面 →
`AIV` 通知 `AIC` 可以用（`T1_READY` / `T1BF_READY` 两条跨核 flag）。
但 T1 只有"模型 dtype 操作数"这一种用途，所以让 Cube 的 fixpipe 直接把 L0C 的 FP32 结果转成模型
dtype 落盘即可：

* cube：`T1` 走 `TileCopyColRowBf16`（`PackedTileCopyTla<..., ElementC = DT>`），输出写到 `T1BfAt`；
* "写落盘 → 本核 MTE2 读回"改成本核 `HardEvent::FIX_MTE2`（每个 window 一个 id，set/wait 严格
  交替，1:2 下两个 head 共用一个 id ⇒ set/wait 都放在 head 循环外）；
* vector：删掉 `StageT1Convert` 与两条 flag；`policy.h` 的 flag 由 5 条降到 3 条
  （`V0_READY` / `Z_READY` / `STATE_READY`，id 最大 5，仍在 A5 允许范围内）。

收益：AIV 每 chunk 少 64 KiB 回读 + 32 KiB 落盘、AIC 少写 32 KiB（fixpipe 减半），少一次跨核往返。
实测 A5 逐档 **17.7 → 17.1 ms（−3.5%）**，精度与改前**逐位一致**（`gva_cp64` 1.760e-03、
`kda_cp64` 1.919e-03）——因为 fixpipe 的 `F322BF16` 与原来 Vector 侧的 `Cast(CAST_RINT)` 是同一种舍入。

### 16.3 改动二：P 状态常驻 UB —— 同时修掉 2 chunk 用例的 P 面错误

`T = 128`（恰好 2 个 chunk）的短链用例此前一直失败，且**失败形态每次不同**（8 次历史运行
`P_r rel_norm` 在 0.99 ~ 1.70 之间跳，`E` 面则每次逐位相同）⇒ 典型的时序问题，不是算错。

本轮把 workspace dump 出来做定点定位（`PAt` 是 per-tile credit 保护的 fp32 往返平面）：

* `PAt(1)`（第一个 chunk 写出的 P）第 32~36 行**整行被写成 0**，只有 4 个 `1.0` 落在
  `(32,64) (33,65) (34,66) (35,67)`，即"**只写了一部分的单位阵**、且单位值落到 +32 列"；
* 同一轮、同一块 UB 紧接着落盘的模型 dtype 操作数 `PBf(1)`（含第 32 行对角 0.11377）**完全正确**；
* 于是 `P0 = decay ⊙ P1_读回 + ZP0` 在第 32~36 行丢掉 `decay ⊙ P_prev` 项，
  并在 `+32 列` 位置留下 `decay`（0.3812）——与实测偏差位置完全对上。

结论：**A5 上"本核 MTE3 写 → 本核下一轮 MTE2 读"的 fp32 状态往返不可靠**（per-tile credit 也没兜住），
而"同核 VF 算完 → 立刻落模型 dtype"是可靠的。

修法（`v11`）：P 的 fp32 状态与 `dH` 一样**常驻 UB**（新增 `pStateF32_` 64 KiB）：

* 首 chunk 用 `CDHP_InitIdentityVF` 就地写单位阵，之后每 chunk 就地 `CDHP_V3AccumVF` 累加
  （与 `dH` 共用同一套"传 tile 基址 + `rowBase`"的写法）；
* 模型 dtype 操作数 `PBf` 仍照旧落盘（Cube 的 ZP 要读）；
* `WriteOutput` 的 P 末值直接从常驻 UB 写 `dhm`（与 E 面同一条路径）；
* 删掉 `PAt` 的 per-tile credit 与 `WriteOutput` 的 credit 回收。

收益有两头：**正确性**（`sh_two_chunk` `P_rel_norm` 6.248e+00 → 1.334e-04，连跑 3 次逐位一致）
与**性能**（AIV 每 chunk 少 64 KiB 回读 + 64 KiB 状态落盘）。

### 16.4 本轮实测（A5 = Ascend950；msopprof `Task Duration`，逐档）

| 用例 | T_rank | 本轮前 | 改动一后 | 本轮后 | H20 | 本轮后 / H20 |
| --- | --- | --- | --- | --- | --- | --- |
| gva_cp2 | 65536 | 17.7 | 17.1 | **15.7** | 10.35 | **1.51×** |
| gva_cp8 | 16384 | 4.5 | 4.3 | **3.9** | 2.59 | **1.52×** |
| gva_cp64 | 2048 | 0.6 | 0.6 | **0.5** | 0.42 | **1.21×** |
| h8_cp2 | 65536 | 17.4 | 16.8 | **15.5** | 5.13 | **3.03×** |
| kda_cp2 | 65536 | 38.0 | 36.4 | **32.1** | 14.32 | **2.24×** |
| long_cp2 | 131072 | 35.3 | 34.2 | **31.4** | 20.66 | **1.52×** |

（优化起点为 36.6 ms / 3.53×，累计已 2.33×。）

精度（A5，本轮后全部 PASS，且 `E` 面与改动前逐位相同）：

```
smoke  : gva_cp64 E=1.760e-03  kda_cp64 E=1.919e-03
short  : sh_single P=6.202e-05  sh_two_chunk P=1.334e-04  sh_tail P=4.872e-04
sweep  : c3 P=4.35e-04  c4 P=1.08e-03  c8 P=5.30e-03  c16 P=3.07e-17
model  : gva/h8/kda/long × cp16/cp64 共 7 条全 PASS
```

### 16.5 本轮后的剖面（`gva` / T=2048 / 16 blocks × 2 AIV）

| 引擎 | 时间 | pipe 占用 |
| --- | --- | --- |
| AIC | 498 µs | mte2 27.1% + scalar 23.8% + **fixpipe 21.8%** + cube 13.9% + mte1 10.6% |
| AIV v0 | 499 µs | **mte2 33.7%（49 GB/s）** + **vec 33.3%** + mte3 12.8% + scalar 10.8% |

结构没有变：两侧仍然同时打满、依旧只能继续砍共享搬运。

### 16.6 下一步（按收益排序）

1. **`AB` 与 `Z` 合并成一次 GEMM**：两者同为 `[K,V]` FP32，且 A 操作数可以拼成
   `[Q̄s|W|(-T1)]ᵀ [K, 2M+K]`、B 拼成 `[do; -dv; dH_bf] [2M+K, V]`，一次 MMAD 得到 `AB + Z`。
   每 chunk 省 64 KiB 回读、64 KiB fixpipe，并少一次 `RunGemm` 的标量开销
   （AIC 的 scalar 已占 23.8%）。需要把 slot 布局改成"`A1 = [Q̄s; W; T1bf]`、`B1 = [do; -dv; dH_bf]`
   各自连续"，同时 `T1` 输出要转置存放（把 `A/B` 操作数对调即可）。
2. **链上平面走 L0C→UB 直送**（参考 `ChunkFwdH` 的 `copy_l0c_to_ub.hpp`）：把 `AB+Z`、`ZP`
   直接放进配对 AIV 的 UB，彻底去掉这两笔 GM 往返（AIV 少 128 KiB 回读、AIC 少 128 KiB fixpipe）。
   前置条件是先腾 UB：常驻 `dH` + `P` 已占 128 KiB，`s0~s4F32_` 80 KiB 里可省掉 2~3 块
   （`W`/`do` 不需要实数运算，可以走模型 dtype 纯拷贝，代价是尾块无效行要单独清）。
3. **`V0` 改成批量下发**：`mte2` 只有 49 GB/s，双槽流水偏浅（§13.6 已证"笔数"是成本主因）。
4. **`h8` / `kda` 的 head 分布**：`h8`（`Hv=8`）1:2 下只有 4 个 block 在跑，
   `kda`（`Hv=64`）则是 32 个 block 各跑 1 轮；前者需要按 chunk 或 KV 列 tile 分核
   （`tiling` 里的 `splitMode` / `tileV` / `tileK` 目前只规划未启用），后者需要检查
   轮次间的流水是否被 `rounds` 循环切断。

### 16.7 平台一致性

* 改动一（T1 直落模型 dtype + flag 重编号）**已在 arch22 同步**（`StageT1Convert` 同样删除，
  `RunGemm` 模板增加输出元素类型参数），A2/A3 的 1:1 路径保持原语义；
* 改动二（P 常驻）目前只在 arch35 落地：A2 上该 fp32 往返此前实测稳定，暂不动，避免在
  A5 尚未收敛时引入 A2 的未验证改动；A2 复测（精度 + 逐档性能）放到下一轮。

## 17. A2/A3（arch22）现状、1:2 尝试与"设备卡死"记录

### 17.1 arch22 的 T1 改动必须回退

§16.2 的"T1 直落模型 dtype + 核内 `HardEvent::FIX_MTE2` 自排空"在 **A5 上可用，在 A2 上会挂死**：
A2 smoke（`gva_cp64`）跑 9 分钟不返回、AICore 100%、`npu-smi` 显示 device 100% 且进程消失。
因此 arch22 保留原来的三步握手（`AIC` 写 FP32 T1 → `AIV` 读回转模型 dtype → `AIV` 通知 `AIC`），
`policy.h` 恢复 `T1_READY` / `T1BF_READY` 两条 flag；arch35 继续用 §16.2 的写法。

### 17.2 A2 基线与剖面（1:1，逐档实测）

| 用例 | T_rank | a2_ms | H20 | 比值 |
| --- | --- | --- | --- | --- |
| gva_cp2 | 65536 | 68.9 | 10.35 | 6.66× |
| gva_cp8 | 16384 | 17.2 | 2.59 | 6.65× |
| gva_cp64 | 2048 | 2.1 | 0.42 | 5.04× |
| h8_cp2 | 65536 | 33.6 | 5.13 | 6.55× |
| kda_cp2 | 65536 | 132.7 | 14.32 | 9.26× |
| long_cp2 | 131072 | 138.3 | 20.66 | 6.69× |

精度：smoke、短链（`sh_single/two/tail`）、逐档全部 PASS，与 A5 的数值逐位一致。

剖面（`gva` / T=2048 / Hv=32 / 1:1，每核平均）：

| 引擎 | 时间 | pipe 占用 |
| --- | --- | --- |
| AIC | 1654 µs | cube 2.6% + fixpipe 9.5% + scalar 9.6% + mte1 2.6% + mte2 3.8% ⇒ **≈28%** |
| AIV | 1665 µs | **mte2 30.5%（28 GB/s）** + **scalar 28.6%** + mte3 21.6% + vec 17.1% ⇒ **≈98%** |

两点结论：

1. 瓶颈完全在 AIV：`scalar` 占 28.6%（arch22 没有 RegBase VF，门控/状态更新是逐行
   `ExpScalar` + `SetValue/GetValue` 的标量往返），`mte2` 只有 28 GB/s（双槽浅流水、每笔都等）。
2. 拓扑上也有浪费：Hv=32、1:1 时 `blockDim = min(ceil(32/1), 20) = 20`、`groupHeads = 2`，
   实际只有 16 个 block 有活（每个串行做 2 个 head），4 个 AIC 空转。

### 17.3 1:2 的两次尝试（均挂死，设备被打到 100%）

目标是把 AIV 侧墙时间近似减半（A5 上 1:2 的实测收益就在这个量级）。两次都在 smoke 用例挂死：

* **尝试 1（沿用 A5 的 id 口径）**：AIC/AIV 两侧共用同一批 id。910B 的 `mode 0x2` 是
  "AIC:2*AIV 集合同步"——AIC 的一次 set 会同时放行两个 AIV，而两个 AIV 又各自 set 同一 id，
  "一 set 一 wait"的计数纪律被破坏 ⇒ 死锁。
* **尝试 2（按 `fwd_h` 的写法给每个 AIV 一段独立 id）**：先确认 910B 的 flag 上限只有 8 个
  （catlass `FFTS_MAX_FLAG = 7`），因此把 `T1_READY`/`T1BF_READY` 合并成一个 base（方向相反、
  落在不同核的寄存器上，可共用 id），并**去掉 window 维度**，得到 AIV0 = 0..3、AIV1 = 4..7：

  | base | 方向 | AIV0 id | AIV1 id |
  | --- | --- | --- | --- |
  | V0_READY | AIV→AIC | 0 | 4 |
  | T1_READY / T1BF_READY | AIC→AIV / AIV→AIC | 1 | 5 |
  | Z_READY | AIC→AIV | 2 | 6 |
  | STATE_READY | AIV→AIC | 3 | 7 |

  结果：`gva` 档（Hv=32，1 轮）**通过**；`kda` 档（Hv=64，2 轮）仍挂住，原因未定。
  （`window` 维度去掉是安全的：flag 是计数器、允许 setter 领先最多 15 次，只要两侧按同一
  chunk 顺序推进，一个 base 一个 id 就够；双 window 只省一次跨核往返，不是正确性前提。）

**运维注意**：两次挂死后 `run_case` 进程被 `kill -9`，device 停在 AICore 100% 且无进程占用，
`npu-smi set -t reset -i 0 -c 0` 被拒（AMP + HCCS 模式下会要求重启全部设备且执行失败），
**必须重启该主机才能恢复**；期间同一台机器上的任何用例都会同样挂住（实测：原本通过的
`gva_cp64` 在设备卡住后也变成 10 分钟不返回）。因此排查 A2 的 1:2 之前要先确认设备已恢复。

### 17.4 下一步（A2）

1. 设备重启后先复跑 1:1 基线（确认设备健康），再继续查 `kda`（两轮 head）在 1:2 下的挂点：
   优先核对"每轮首 chunk 的 `STATE_READY`/`V0_READY` 配对"与"轮次之间 id 复用"是否仍然严格交替。
2. 若 1:2 短期无法稳定，改走"同 head 内按列拆分"的 1:2 分工（两个 AIV 各算 dH 的 V 半平面 /
   P 的 K 半平面），那样两侧共享同一批 flag、天然是集合同步语义，不需要 per-AIV 段。
3. 与 1:2 正交的两项 AIV 优化（A5 同样受益，可先做）：
   * 把 AIV 的 `mte2` 改成批量下发（多槽 + per-slot credit），目标把 28 GB/s（A5 49 GB/s）拉起来；
   * 降低 arch22 的标量占比（门控/状态更新的逐行/逐元素标量往返改成整行向量化）。

## 18. v12：E/P 两条链按 tile 合并成一趟（A5 15.7 → 15.2 ms，1.51× → 1.47×）

### 18.1 改动

`arch35` 的 `StageState` 原来分两个循环：先按 32 行 tile 取 `AB`/`Z` 做 dH 的 VF，再另一个循环取 `ZP`
做 P 的 VF。这样 P 链的 `ZP` 搬运**完全排在 dH 的 VF 之后**，MTE2 队列深度只有 2。

改成**每个 tile 一趟**：先把三条链上平面（`AB`/`Z`/`ZP`）一起排进 MTE2，统一等到齐后再做两个 VF，
最后落 `PBf`。第三个 FP32 缓冲复用 `V0` 的 `do` 平面（`s3F32_`）：`StageState` 与 `StageV0` 在同一轮里
串行、且 `StageV0` 排在后面，因此这块 UB 在 `StageState` 期间是空闲的，**不新增 UB**。

实测（`gva` / T=2048 / 16 blocks × 2 AIV）：

| 指标 | 改前 | 改后 |
| --- | --- | --- |
| AIV `mte2` | 169 µs | **147 µs** |
| AIV 总时间 | 499 µs | 488 µs |
| AIC 总时间 | 498 µs | 485 µs |

逐档（A5）：gva_cp2 15.7 → **15.2 ms（1.47×）**、gva_cp8 3.9 → 3.8、h8_cp2 15.5 → 15.1、
kda_cp2 32.1 → 31.1、long_cp2 31.4 → 30.5；精度 smoke/短链/扫描**逐位不变**。

### 18.2 同期做的一个负结果（已回退）

把 `StageV0` 的 5 笔搬入拆成"显式槽号 + 2 深流水"（先下发两笔、之后"等一笔 / cast 一笔 / 补一笔"），
用 `IssueLoadTile` / `CastLoadedTile` 两个半程函数实现。逐档实测与改前**完全一致**（15.2 vs 15.2，
h8 15.1 vs 15.1），说明 V0 的搬入并不是当前停顿来源（现有的双槽交替已经做到 2 深），
因此该改动（含两个半程函数）已回退，避免无收益的复杂度。

### 18.3 v12 之后的剖面与下一步

| 引擎 | 时间 | pipe 占用 |
| --- | --- | --- |
| AIC | 485 µs | mte2 28.4%（99 GB/s）+ scalar 24.5% + fixpipe 22.3% + cube 14.3% + mte1 11.0% |
| AIV | 488 µs | vec 33.7% + **mte2 30.2%（57 GB/s）** + mte3 13.2% + scalar 10.5% |

仍是"两侧同时打满"，继续砍共享搬运。按收益排的前两项已经明确到可实现的程度：

1. **`Z` 与 `ZP` 合成一次 GEMM**：两者 A 操作数相同（`T1Bf`）、B 操作数分别是 `dH_bf`（[K,V]）与
   `P_bf`（[K,K]），把两份操作数拼成一份 `[K, V+K]` 的模型 dtype 平面（`dH_bf` 放前 128 列、
   `P_bf` 放后 128 列），一次 MMAD 得到 `[K, V+K]` 的 FP32 输出（`Z` 与 `ZP` 各占 128 列），
   AIV 侧用"行距 `V+K`、列偏移 0 / V"的两笔 `DataCopyPad` 分别读入两个 32 行 tile，
   VF 本身不需要改 stride（UB 内仍是紧凑 tile）。收益：GEMM 4→3（AIC scalar 24.5% 里对应的一次
   `RunGemm` 开销）、`T1Bf` 只读一次、平面笔数减少；tiling 侧只需新增"每个 (head, window) 一份
   `[K, V+K]` 的模型 dtype 操作数平面 + 一份 `[K, V+K]` FP32 输出平面"（arch22 的旧平面保留不动，
   避免动到当前无法验证的 A2 路径）。
2. **`AB` 与 `Z` 合成一次 GEMM**：A 拼 `[Q̄s|W|(-T1)]ᵀ [K, 2M+K]`、B 拼 `[do; -dv; dH_bf] [2M+K, V]`，
   一次 MMAD 得到 `AB + Z`。需要把 slot 布局改成"`Q̄s | W | TT | do | -dv | dH_bf`"并把 `T1` 输出
   转置存放（A/B 操作数对调即可），改动比第 1 项大。

## 19. v13：W/do 同 dtype 纯拷贝 + AB/Z 合并 GEMM（A5 15.2 → 12.2 ms，1.47× → 1.18×）

两处改动都在 `arch35`（A5）；arch22（A2/A3）保持原路径不动（当前无法验证 A2 的改动）。

### 19.1 W / do 改同 dtype 纯拷贝

`StageV0` 里 5 个平面原本一律走"载入 → Cast 到 FP32 → （门控/取负）→ Cast 回模型 dtype → 落盘"。
但 `W` 与 `do` 在 Vector 侧**没有任何实数运算**（既不门控也不取负），于是新增一块专用单槽暂存
`s2DT_` 与 `CopyTileModel`：只保留 MTE2→MTE3 顺序（一对 per-slot credit），完全不进 V pipe。
尾块（`T_rank` 不是 chunkSize 整数倍）的无效行必须落 0，因此尾块仍走原来的 FP32 通路
（先 `Duplicate` 清零），满 tile 才走拷贝。

每 tile 的 Cast 数从 10 降到 6（q/k 的载入 + q/k/W/do/dv 的落出里，W/do 各占 2 次）。

### 19.2 AB 与 Z 合并成一次 GEMM

新增两份拼接操作数平面（arch35 专用，`aOperWs` / `bOperWs`；arch22 不用，但一起分配以保持
tiling 只有一套布局）：

```
aOper = [ Q̄s(M,K) | W(M,K) | (-T1)ᵀ(K,K) ]   → Cube 的 A 操作数，按列主序 [K, 2M+K] 读
bOper = [ do(M,V) | -dv(M,V) | dH_bf(K,V) ]  → Cube 的 B 操作数，按行主序 [2M+K, V] 读
```

* `T1 = Wᵀ@(-K̄)` ⇒ `(-T1)ᵀ = (-K̄)ᵀ@W`，所以 T1 的 A/B 操作数对调、结果直接**转置存放**，
  正好等于合并 GEMM 里 k 段（行）与列主序 A 需要的排布；`ZP` 的 A 也从同一段按列主序读回来。
* 链上一步变成"等 `STATE_READY` → 合并 GEMM(AB+Z) → ZP"；`dH_bf` 由 AIV 写进**同 window** 的
  `bOper` 第三段（Cube 从同 window 读 B，靠 `STATE_READY` 保证可见性），`P_bf` 仍保留原 window 约定。
* Vector 侧因此从"读 AB、Z 两个 FP32 平面"变成"读一份 `inc = AB+Z`"，`dH_new = decay⊙dH + inc`
  直接复用 P 链的单项累加 VF（`V3Accum`），少一次 GEMM 调用。

### 19.3 实测（A5，msopprof `Task Duration`）

| 用例 | §18 后 | 19.1 后 | **19.2 后** | H20 | 19.2 后 / H20 |
| --- | --- | --- | --- | --- | --- |
| gva_cp2 | 15.2 | 13.8 | **12.2** | 10.35 | **1.18×** |
| gva_cp8 | 3.8 | 3.4 | **3.1** | 2.59 | **1.19×** |
| gva_cp64 | 0.5 | 0.5 | **0.4** | 0.42 | **0.96×** |
| h8_cp2 | 15.1 | 13.5 | **12.0** | 5.13 | 2.34× |
| kda_cp2 | 31.1 | 28.7 | **25.5** | 14.32 | 1.78× |
| long_cp2 | 30.5 | 27.7 | **24.6** | 20.66 | **1.19×** |

精度：smoke / 短链 / 扫描 / 7 条模型用例全部 PASS（`gva_cp64` E = 1.760e-03，`max_abs` 由
1.072e-05 变 1.073e-05，属累加顺序变化的 1 ulp 量级；其余用例逐位相同）。

剖面（`gva` / T=2048）：AIC 439 → **392 µs**（mte2 32.7% + scalar 26.8% + fixpipe 20.5% +
cube 17.7% + mte1 14.5%），AIV 442 → **395 µs**（vec 35.8% + mte2 29.2% + mte3 16.5% + scalar 11.5%）。

### 19.4 结论与下一步

**主场景（`gva` 家族与 `long`）已经达到 0.8+ × H20 的目标**（1.18~1.19× H20，`gva_cp64` 已快于 H20）。
拉低整体均值的是两个 head 分布极端的用例：

* `h8`（Hv=8）：1:2 下 `blockDim = ceil(8/2) = 4`，**只有 4 个 AIC / 8 个 AIV 在跑**（芯片有 20/40），
  时间完全由"每核要串完整个 head 的链"决定；要提速必须做 **head 内并行**（按 V 列拆 dH、
  按行拆 P，两个 AIV 共享同一批 flag 的集合同步语义天然可用），这是下一个结构性改动。
* `kda`（Hv=64，走逐 K 门控 `gk` 路径）：核已经用满（20 block × 2 AIV），差距来自每 chunk 的成本
  本身（`gk` 路径的 gate/decay 是 128 lane 的向量 Exp，且 q/k 不共享）；对应优化是把 gate 的
  逐行计算改成"先向量算一遍 64 行因子、再逐行广播"（省掉每行 3 Muls + 2 Exp），以及评估
  ZP 与 Z 的进一步合并（只减 AIC 侧，需先确认 AIC 是否为该档的瓶颈）。
