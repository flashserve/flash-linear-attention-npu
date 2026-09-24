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

`K = 256` 时 `T1`/`P_c` 各 256 KiB、`P` 两个面共 512 KiB、`dH` 两个面共 512 KiB，因此这些平面必须驻留
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
- shape/取值范围的唯一维护处是 [README](../README.md) 的「支持的场景」「不支持（本版显式拦截）」「已知限制」三节，
  本文只讨论设计取舍，不重复列举取值。
- layout：`[B,H,T,D]`（BSND）。TND/NTD 需由调用方或 L2 侧 layout sweep 后进入本算子。
- `state_v_first` 与本算子无关：本算子只读 token-major 的 `q/k/w/do/dv`，只写固定 `[Hv, K, V+K]` 的 `dhm`。
- 不支持 `USE_BG`（DPLR）与 `AFFINE_CHAIN_PRECISION`；`g`/`gk` 互斥由 host 拦截。

## 9. 接入步骤

1. **kernel 落地（已完成）**：`op_kernel` 六个 Stage 的真实实现已落地（Catlass `BlockMmadTla` +
   Vector 侧手写事件对），`PROPOSED` 标记已随实现删除。
2. **构建接入（已完成）**：`chunk_delta_h_bwd_preprocess/CMakeLists.txt`（glob 风格）与
   `op_host/CMakeLists.txt`（`add_op_to_compiled_list()` + `target_sources(op_host_aclnnExc ...)` +
   `add_modules_sources(OPTYPE chunk_delta_h_bwd_preprocess ACLNNTYPE aclnn_exclude)` +
   `add_ops_compile_options(OP_NAME ChunkDeltaHBwdPreprocess OPTIONS --cce-auto-sync=off
   -Wno-deprecated-declarations)`）已提交；aclnn 走手写 exc 通路，不走自动生成。
3. **Python 入口（待完成）**：在 `torch_custom/fla_npu/npu_custom.yaml` 注册
   `npu_chunk_delta_h_bwd_preprocess`，并在 `fla_npu/ops/ascendc` 下提供主入口。当前分支缺少
   `ops/ascendc/_runtime.py`、`_aclnn_ctypes.py` 适配层，需要在具备该适配层的分支上接入
   （默认调用路径必须走 ctypes/aclnn，不依赖 `torch_npu` dispatcher）。
4. **测试（设备侧精度矩阵已完成，见 README）**：按 `tests/op_cases/chunk_delta_h_bwd_preprocess.json`
   生成用例，用 `tests/operators/chunk_delta_h_bwd_preprocess/reference.py` 作为 CPU 标杆，覆盖
   A2/A5 两平台与 `USE_G`/`USE_GK`/无门控、dense/varlen、尾块、GVA、`K=64/128/256`。
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
- 第 2、3 项：按 `tests/operators/chunk_delta_h_bwd_preprocess/harness/` 的逐平面核对与 `op_cases` 矩阵执行，
  A2/A5 均 16/16 PASS，含 32 chunk 长链（`pos_13`）与尾块（`pos_06`/`pos_09`）。
- 第 4 项：`C1`/`C3` 的两路输出、`V4` 的两路输出在实现上分别发布/释放（各自独立 flag 边界），
  逐平面核对未发现两路混用；受控实验（`harness/ctrl_case.py`）用于定位过 A 路。
- 第 5 项：本版只实现"仅按 head 连续分核"（`BY_HEAD`），`(hv, 列 tile)` 展平（`BY_TILE`）尚未实现，
  `pos_16_tile_split_partition` 实际按 `BY_HEAD` 执行。
- 第 6 项：跨 rank 一致性验证需要上层 CP 切分链路，本版未覆盖。
- 另需补做 sanitizer（`racecheck`/`memcheck`/`initcheck`/`synccheck`）与官方 ATK/CI 接入。
