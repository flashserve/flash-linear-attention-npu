# pre_process_fwd_kernel_merged 设计

`workflow_id`: `catlass-linear-attention-v1`
`design_rule_version`: `V1`
`sync_pattern`: `SYNC_L1_HANDOFF`
`pipeline_pattern`: `PIPE_SERIAL`
`status`: `design_ready_for_review`

> **范围变更（2026-09-30）：DPLR 不支持。** 本文档第 1、3、4 章里所有 DPLR 条目
> （`USE_BG`、`bg`、`v` 与 `u` 分离、`M_c` 取 `+`）都是**该分支的历史设计记录**，不构成本
> 版本的接口承诺：接口已把 `bg` / `v` 收敛成"必须为空、非空直接拒绝"（`docs/api.md`
> §0 / §2 / §6），TilingKey 可达 **4 个**，`USE_BG` 只保留槽位、host 永不产生。
> 实现与验收范围只有 **GDN（`g`）与 KDA（`gk`）**。

> 03 方案设计分两步完成。**第一步（Stage 划分）与第二步（具体详设）均已完成**：
> 第 1、2、3、6 章已填实，包含完整依赖图与 Stage 表、逐 Stage 的公式/地址/生命周期/同步、
> L1/UB/L0 容量、`TilingKey` 与 workspace、以及 R01–R21 逐条结论。
> 第 4 章的性能目标与第 5 章的未决问题里仍有需要用户输入或 04 实测的项，逐条列在第 5 章。

> **修订记录**
>
> | 日期 | 修订 | 影响位置 |
> | --- | --- | --- |
> | 2026-09-21 | `K` 收敛为 `<=128`、`V` 收敛为 `<=128`、`chunk_size` 固定 64（`K=256`/`V=256`/`128` 移出本轮）；GVA 保留，`HK` 与 `HV` 成倍数 | 1.1、1.2、1.3、2.1.4、3.1、3.3、3.4、4.3、5.2 |
> | 2026-09-21 | 分档由容量模型算出；收敛范围后全档位只落在一组配置 | 3.1、3.4、4.3 |
> | 2026-09-21 | 输入输出布局定为 BNSD，与仓内其他 AscendC 算子一致 | 1.4、4.3、5.2 |
> | 2026-09-21 | 第二步详设完成：2.7 同步（CrossCore/HardEvent 分配 + 完整伪代码）、2.8 组件与 TileShape、3.1 L0 占用、3.2 TilingKey 与 workspace、3.3 模板字段划分 | 2.7、2.8、3.1、3.2、3.3 |
> | 2026-09-21 | `K`/`V` 由"`<=128` 的档位"改为**写死 128**；`TilingKey` 收敛到 6 个；3.4 改写为写死依据 | 1.1、1.2、2.1.4、2.2、2.4、2.6、2.7、2.8、3.1、3.2、3.3、3.4、4.3、5 |

---

## 1. 目标与数学语义

### 1.1 目标与范围

- **目标 SoC**：`Ascend950PR_9579`，`NpuArch=3510`（`CATLASS_ARCH=3510`），`AIC_version=AIC-C-310`。
  平台参数取自 `${ASCEND_HOME_PATH}/../latest/acllib/data/platform_config/Ascend950PR_9579.ini`：
  Cube 核 **28**、Vector 核 **56**（`cube_vector_combine=split`）、L1 **512 KiB**、
  单 AIV UB **248 KiB**（253952 B）、L0A/L0B **64 KiB**、L0C **256 KiB**、L2 128 MiB、
  约 1.5 TB/s HBM、`cube_freq=1650 MHz`、BF16 峰值约 378 TFLOPS（28 核档）。
- **输入输出 dtype/layout/关键维度**：`k/w/u` BF16，`g/gk` FP32 或 BF16，`hm` FP32；
  布局 `[B,H,T,D]`（BNSD）；`BT=64`、**`B ≡ 1`**（CP 契约，见下）、
  `K = V = 128`、`chunk_size` 固定 64（编译期规格，host 拦截其它值）。`HK` 与 `HV` 可以不一致但
  **必须成倍数**（`HV >= HK` 且 `HV % HK == 0`），这就是 GVA：`k` 在 `HK` 维，
  `w/v/u/g/gk` 在 `HV` 维，输出 `hm` 与两个状态都在 `HV` 维，算子内部按
  `hk = hv // (HV/HK)` 取对应的 key head。
- **`hm` 的前导维是"链条数"**：`[Nseq, HV, K, V+K]`，`Nseq = len(cu_seqlens)-1`。语义等价说法：**竞品一次调用
  产出一份 `hm`，我们一次调用产出 `Nseq` 份，第 i 份与竞品针对第 i 段单独调用一次逐位相同**
  （kernel 侧 `MULTI_SEQS` 与逐段调用逐位相等，见 2.1.4 第 4 条）。CP 场景下本 rank 的窗口
  往往是单段（`Nseq=1`），此时去掉 size-1 维与竞品 `hm` 逐字节相同。
- **支持场景（2026-09-22 定稿，与竞品 CP 契约完全一致）**：**序列表达只有一种——varlen
  打包窗口**：`B ≡ 1`、`cu_seqlens` 必给、`Nseq = len(cu_seqlens)-1 >= 1`。单段窗口写
  `cu_seqlens=[0, T_win]`（= 竞品跨卡调用的形态）；多条等长序列由**调用方打包**成一个窗口
  （`B` 条 `T` 长序列 ⇔ `cu_seqlens=[0,T,2T,…]`、`T_win = B·T`）。尾块（不足 `BT` 的最后一
  chunk 与前缀补齐）照旧支持。
  **实现与验收范围是 GDN（`g`）+ KDA（`gk`）两条路径**；
  **DPLR（`gk`+`bg`）不支持**（2026-09-30 定稿）：`bg` / `v` 必须为空，非空在 host tiling /
  aclnn / ctypes / stable 四个入口直接被拒；`TilingKey` 的 `USE_BG` 槽位与本文档里
  `bg^T @ V_c`、`M_c` 取 `+`、L1 的 `bg`/`V_c` 槽等条目**只是历史设计记录**，不实现、不验收。
  变长与尾块复用同一条主路径，只改变有效长度与 mask，不改变 Stage 结构。
- **性能目标**：对标 **1.0x H20** 上同一 `pre_process_fwd_kernel_merged`，判定指标为
  `msprof` `op_summary` 的 `Task Duration(us)`。模型 case 的 `CP world_size`、`T_win`、
  `precision` 档位、采样口径与基线数值由用户提供后填入第 4 章（见第 5 章未决问题）。
  **不处理范围**：CP 多卡编排（`all_gather_into_tensor`、`merge_fwd_bwd_kernel`、
  zigzag part 选择）、`MULTI_SEQS`、`state_v_first`、`AFFINE_CHAIN_PRECISION=tf32x3`。

### 1.2 输入输出和调用方式

与 `docs/api.md` 一致，此处只列设计直接依赖的部分：

| 名称 | 必需 | shape | dtype | 设计语义 |
| --- | --- | --- | --- | --- |
| `k` | 是 | g-only `[B,HK,T,K]`；gk 路径 `[B,HV,T,K]`（要求 `HK==HV`） | BF16 | g-only 为 raw key，按 `hk` 复用；gk 路径为已 gate 好的 `kg`，本算子不再乘 gate |
| `w` | 是 | GDN/KDA `[B,HV,T,K]`；DPLR `[B,HK,T,K]` | BF16 | erase/WY 输出；S0 的左操作数、S2 的右操作数 |
| `u` | 是 | `[1,HV,T,V]` | BF16 | 仅 DPLR 读取 |
| `v` | 是 | `[1,HV,T,V]` | BF16 | DPLR 下独立于 `u`；GDN/KDA 下 `v` 与 `u` 为同一张量 |
| `g` | 二选一 | `[1,HV,T]` | FP32/BF16 | 标量 gate，base-2 chunk 内累积对数衰减 |
| `gk` | 二选一 | `[1,HV,T,K]` | FP32/BF16 | 逐 K gate，同为 base-2 chunk 内累积量 |
| `bg` | 仅 DPLR | `[1,HK,T,K]` | BF16 | DPLR 的 K 侧项 |
| `cu_seqlens` | **是** | `[N+1]` | **host `list[int]`**（非张量） | 本窗口内的段边界（T 轴打包多段），`N = Nseq >= 1`；首元素 0、严格递增、末元素 = `T_win`（唯一形态，与 `api.md` 3.2 一致） |
| `hm` | 输出 | `[Nseq,HV,K,V+K]` | FP32 | 左 `[0,V)` 为 `h`，右 `[V,V+K)` 为 `m`；`Nseq = len(cu_seqlens)-1` |

属性、边界与异常行为以 `docs/api.md` 第 3、6 节为准；host 负责全部 shape/dtype/属性校验并拦截
非法组合（`K!=128`、`V!=128`、`HK>HV` 或 `HV%HK!=0`、`g`/`gk` 同缺或同给、
`bg`/`v` 非空（DPLR 不支持）、空窗口、`chunk_size!=64`）。标杆与算子接口的差异（NPU 固定
`ieee` 口径、`hm` 单 part 布局）已在 `docs/api.md` 第 2、4 节记录。

### 1.3 完整数学语义

符号：

```text
B 窗口条数、T 窗口长度、BT=64、NT=ceil(T/BT)、c in [0,NT) chunk 序号
HK key head 数、HV value head 数、G=HV/HK（成倍数）、hk(hv)=floor(hv/G)
K key 维（=128）、V value 维（=128）          固定规格，见 3.4
last(c) = min((c+1)*BT, T) - 1            当前 chunk 最后一个有效 token 的全局下标
E(x) = exp2(x)                            全部衰减量均为 base-2 对数域，不做底数换算
FP32(x) 表示"该项以 FP32 累加/存放"，不是把操作数升到 FP32；
        右侧显式的 BF16(x) 才表示先量化到 BF16 再入 Cube
h_c [K,V] FP32   窗口内 rolling 状态，初值全 0
m_c [K,K] FP32   仿射链矩阵，初值单位阵 I
```

对每个 value head `i_h` 与第 `c` 个 chunk（`s,t` 为 chunk 内 token 下标，累加一律 FP32）：

```text
# S0  Cube
P[BT,V]        = FP32(W_c[BT,K]) @ FP32(BF16(h_c)[K,V])            # FP32 累加

# S1  Vector
GDN/KDA: v_new[BT,V] = FP32(V_c[BT,V]) - P
DPLR   : v_new[BT,V] = P + FP32(U_c[BT,V])
USE_G  : v_new[t,:] *= E(g[last]-g[t]) ;  v_new -> BF16
USE_G  : L_c        = BF16(E(g[last]-g[t]) * FP32(K_c))             # 逐行缩放后再量化
USE_GK : L_c        = K_c，不缩放
DPLR   : 左侧要用两个因子：K_c（`dH` 主项）与 bg_c（`dH` 追加项与 `Kw`）

# S2  Cube
dH[K,V]        = FP32(K_c^T[K,BT]) @ FP32(BF16(v_new)[BT,V])        # FP32 累加
DPLR   : dH    += FP32(bg_c^T[K,BT]) @ FP32(BF16(V_c)[BT,V])
Kw[K,K]        = FP32(L_c^T[K,BT]) @ FP32(W_c[BT,K])                # FP32 累加

# S3  Vector
USE_G  : decay = E(g[last])                    标量
USE_GK : decay[k] = E(gk[last,k])              [K] 逐行
h_c            = decay (*) h_c + dH                                 # (*) 为标量或逐行乘
M_c[K,K]       = diag(decay) - Kw              （DPLR 为 “+”）
h_c -> BF16 影子 H_c 写 L1（供下一 chunk 的 S0）

# S4  Cube
m_c            = FP32(M_c[K,K]) @ FP32(m_c[K,K])

# 窗口结束（c = NT-1）
hm[i_h, 0:K, 0:V]   = FP32(h_{NT})
hm[i_h, 0:K, V:V+K] = FP32(m_{NT})
```

与 CPU 标杆（`tests/atk/pre_process_fwd_kernel_merged/scripts/pre_process_fwd_kernel_merged_cpu.py`）
的对应关系与**必须保持的舍入点**（见 `docs/api.md` 第 4 节）：

1. S0 的右操作数 `h` 先降 BF16 再入 Cube；S1 产出的 `v_new` 在进入 S2 前量化 BF16；
2. `h` 的累加 `decay*h + dH` 在 FP32 上完成，chunk 之间不降精度；
3. `m` 的整条链（`Kw` -> `M_c` -> `M_c@m`）在 FP32 内完成，chunk 之间不降精度；
4. `hm` 为 FP32 输出，不做 BF16 舍入。

`USE_G` 下 `Kw = (E(dg)*K_c)^T @ W_c` 与代数等价的 `K_c^T @ (E(dg)*W_c)` 在 BF16 舍入上
不等价。本设计**采用上游写法：缩放 `K_c` 后再量化**，以保持与标杆相同的舍入位置。

### 1.4 Python 入口与对标接口的差异

**本算子当前只有一个窗口一个 part 的粒度**，因此公开入口是"直接产出 `hm`"的函数，
而不是竞品的 CP 编排函数。对标关系如下（详见 `docs/api.md` 第 3 节）：

| 层级 | 竞品（fla-org @ e52dbc0e） | 本算子 |
| --- | --- | --- |
| Triton kernel | `pre_process_fwd_kernel_merged[grid](k, v, w, g, gk, bg, u, hm, cu_seqlens, T, H, HV, K, V, BT, BK1, BLOCK_SIZE, MULTI_SEQS, AFFINE_CHAIN_PRECISION)`，`hm` 是输出指针 | 不暴露 kernel；由 host tiling 推导 |
| Python 包装 | `chunk_gated_delta_rule_fwd_h_pre_process(k, w, u, g=None, gk=None, bg=None, v=None, chunk_size=64, state_v_first=False, cu_seqlens=None, initial_state=None, context=None, use_graph=False) -> initial_state`，内部做 zigzag part 循环 + `all_gather` + `merge_fwd_bwd_kernel` | 不做 CP 编排；只承担其中对 `pre_process_fwd_kernel_merged` 的单次调用 |
| 调用入口 | 无独立算子名 | `from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged`（同时导出 `npu_pre_process_fwd_kernel_merged`） |
| 输入布局 | token-major `[B, T, H, D]` | **BNSD `[B, H, T, D]`**，与仓内其他 AscendC 算子一致（见 5.2） |
| `u` / `v` | 位置参数 `u`，关键字 `v`；GDN/KDA 下 `v=u if v is None else v` | 同左（保持别名语义与 host 校验） |
| `state_v_first` | 有，但只影响包装内部 `initial_state` 的布局 | 不适用（本算子不接触 `initial_state`） |
| `AFFINE_CHAIN_PRECISION` | 编译期常量，NVIDIA 上可为 `tf32x3` | 固定 `ieee`（NPU），不作为参数暴露 |
| `MULTI_SEQS` | 编译期常量，上游包装恒传 `False` | 不适用 |
| `hm` | 包装内部 buffer，不返回 | 作为函数返回值，`[B, HV, K, V+K]` FP32 |

> 布局差异已按用户要求定稿为 BNSD（理由见 5.2），竞品的 token-major 由调用方吸收。
> 仍需确认的一处是 `hm` 的返回方式；它属于 `operator_contract` 的范围，若在 04 之前定稿只需
> 同步更新 `docs/api.md`；一旦进入 04 再改动，按主 Skill 的恢复矩阵回到 01。

**`hm` 的 `B` 维与上游的"part 维"是同一个东西**：

| 场景 | 上游 `hm` | 本算子 `hm` |
| --- | --- | --- |
| 单 part（contiguous CP 或非 CP） | `[HV, K, V+K]` | `B = 1` → `[1, HV, K, V+K]` |
| zigzag CP（每 rank 两个 part：front/back） | `[2, HV, K, V+K]` | `B = 2` → `[2, HV, K, V+K]` |

所以本设计**不是偏离上游**：`B = 1` 时去掉 size-1 维后内存布局逐字节相同，zigzag 下形状完全一致。
带上 `B` 维的三条理由（`api.md` 第 3.4 节有完整表述）：输入输出对称；与仓内
`npu_chunk_fwd_h`（输出 `h` 是 `[B,HV,NT,K,V]`）的约定一致；`Nseq > 1` 正是并行度
`Nseq × HV` 的来源（2.1.4）。上游"没有 `B` 维"是它一次只算一段窗口的调用粒度造成的，
它一用 zigzag 就不得不加一个 `2` 维。

**CP layout 与算子的感知边界**：`layout` 由建 CP context 的一层决定
（`build_cp_context(..., layout=...)`，默认 `'contiguous'`；`FLACPContext.layout` 只是记录，
kernel 侧唯一的消费者是 `chunk_delta_h.py` 的 `if context.layout == 'zigzag'`）。
**本算子不感知 `layout` 这个参数、也不应该加**——它是编排概念。算子需要感知的是它的**后果**：
本 rank 的窗口要切成**几个 part**、每个 part 覆盖哪些 token。表达手段就是 `B`（part 数）
与 `cu_seqlens`（段边界）。

| 场景 | part 结构 | 我们的表达 | 是否够 |
| --- | --- | --- | --- |
| contiguous（默认） | 1 个 part = 本 rank 的整段 token | `B` 个独立窗口 + 一维 `cu_seqlens` | ✅ 够 |
| zigzag | 2 个 part = 本地 buffer `[front; back]` 的前后两半，**切点是 `part_len`** | `B = 2` 能表达"两个 part"，但**切点不在 `cu_seqlens` 里** | ⚠️ 缺一条 part 边界 |

上游是用 `context.front_num_seqs` 补上这条边界的（`cu_seqlens[fns-1:fns+1]` 取 front 的末段、
`cu_seqlens[-2:]` 取 back 的末段，两次调用分别写 `hm[0]`/`hm[1]`）。我们要在一次调用里产出
`[B=2, HV, K, V+K]` 的话，有三种做法：

| 方案 | 接口改动 | 代价 |
| --- | --- | --- |
| **(c) 不感知（本轮建议）** | 无。调用方每个 part 调一次（`B=1`），算子只算一段 | 与上游现状**完全一致**；zigzag 下每个 part 各自 `Nseq_part × HV` 个工作项，并行度和固定开销摊销都减半 |
| **(d) 整窗一次算（新增，推荐）** | **无新增参数**：把整个本地 buffer（`[front; back]` 拼在一根 T 轴上）当一个窗口，`cu_seqlens` 覆盖全部段——part 边界天然在其中（`front_cu[-1] == part_len`），算子产出"每段一条链" `[Nseq, HV, K, V+K]`，调用方按 `hm[fns-1]`（front 末段）与 `hm[Nseq-1]`（back 末段）取两条 | 与 (a)/(b) 相比零接口改动、零搬运；代价是多算 `Nseq-2` 条链（填核，不取用），且要求"张量 T 轴 = 窗口"（`cu_seqlens[0]=0`、`cu_seqlens[-1]=T`）——整窗传参天然满足，所以**不需要** §5.2 第 12 项的子区间支持 |
| (a) `cu_seqlens` 升成 `[B, N+1]` | 01 接口：`cu_seqlens` 由一维变二维 | 最干净，每 part 一组边界；但要改 host 校验与标杆 |
| (b) 一维 `cu_seqlens` + `part_offsets[B+1]` | 01 接口：新增一个辅助数组（各 part 在段列表中的起止索引） | 与上游 `front_num_seqs` 同构，改动小；比 (a) 少一次语义澄清 |

**本轮取 (d) 为主、(c) 为兜底**：(d) 零接口改动、零搬运，整窗一次调用即可覆盖两个 part
（前提"part 边界在 `cu_seqlens` 里"对 zigzag 天然成立）；(c) 保留为"最保守对标竞品"的调用方式
（每 part 调一次），但它要求窗口是张量 T 轴的子区间（§5.2 第 12 项）。目标场景是 contiguous
（默认）时两者等价——contiguous 只有一个 part，`cu_seqlens` 一维就够。

#### 1.4.1 "窗口能否是子区间"到底差在哪（四种做法）

竞品 kernel 的寻址是 **`bos` 偏移 + 张量自身 stride**：
`k += ((bos*H + i_h//R) * K)`、`w += ((bos*HV + i_h) * K)`，且 varlen 分支里
`T = eos - bos`。所以它天然支持"窗口只是张量 T 轴的一段"——zigzag 的两次调用正是如此
（张量 `T_local = 2·part_len`，front 调 `[0, 256]`（`eos < T`）、back 调 `[444, 512]`
（`bos > 0`），这就是"子区间"）。

我们当前契约要求 **张量 T 轴 = 窗口**（`cu[0]=0`、`cu[-1]=T`），因为 host tiling 与 L1 搬运
都按"窗口是连续的、stride 由 shape 决定"来算。四种做法与代价：

| 方案 | 局部数据形态 | 调用方式 | 额外搬运 | 多算的链 | 需要改内核？ |
| --- | --- | --- | --- | --- | --- |
| **A 整窗一次算（推荐）** | `[front; back]` 拼在一根 T 轴上 | 1 次调用，传整条本地 `cu_seqlens` | 0 | `Nseq-2` 条（填核，不取用） | 否 |
| B 两段各自成张量 | front / back 各自一个 `[1,H,part_len,D]` | 2 次调用，各传该 part 的完整段边界 | 0（gather 阶段就分开写） | 每个 part 内非末段 | 否 |
| C 逐 part 调 + 子区间 | `[front; back]` 一根轴 | 2 次调用，各传 2 元素（末段） | 0 | 0 | **见下**：分 C1/C2 两档，C1 成本很低 |
| D 逐 part 调 + 切片 | 同上 | 2 次调用，但先把该段 `.contiguous()` | ≈ `2 × 40.96 KB/token × L` | 0 | 否 |

> **注意"2 个元素"本身不是关键，"张量只含那一段"才是。** 我们的契约要求窗口 = 张量 T 轴
> （`cu[0]=0`、`cu[-1]=T`）。所以：
> * 若上游把数据按段组织（或按段写进独立 buffer）→ 传 `[1,H,L,D]` + `cu_seqlens=[0,L]` 即可
>   （2 个元素，但是 `[0,L]`），**内核零改动**——这就是方案 B；
> * 若上游只有一根拼好的大轴，只切 `cu_seqlens` 成 2 个元素**不行**：那是 `[bos,eos]`（`bos>0`
>   或 `eos<T`），属于子区间（方案 C）；
> * 也别指望"传个 T 轴 slice 的 view"能绕过：BNSD 下沿 T 切片后，H 维 stride 仍是**整根长度**
>   `T_full·D`，而我们的内核按 `shape` 推 stride（会按 `L·D` 算）→ 直接算错。要么按真实 stride
>   寻址（C），要么先 `.contiguous()`（D）。

**方案 C 要分两档，成本差很多**（我此前把它整体算成"要改地址层"，过重了）：

| 档 | 输入形态 | 要改什么 | 成本 |
| --- | --- | --- | --- |
| **C1（推荐）** | **整根张量** + `cu_seqlens=[bos, eos]`（允许 `bos>0`、`eos ≤ T`）——**这正是竞品的用法** | ① host 校验放宽成 `cu[-1] ≤ T`（仍能挡住误传全局数组，因为全局末值 > `T_local`）；② host tiling 里段的基址按 `bos` 相对**张量起点**算、`T_win = eos-bos`、h 维 stride 用 `shape[2]`（整根长）；③ tiling 多带一个"张量 T 全长/stride"字段。**内核的地址表达式不变形**（本来就 `base + h*stride_h + t*stride_t`），L1 常驻搬运也照旧（同一段的行，只是起点偏移） | **低**：改动集中在 host + 一个 tiling 字段；标杆侧零改动（`reference.py` 本来就是按 `(bos,eos)` 起算） |
| C2（不必做） | **任意非连续 view**（stride 与 shape 不一致） | 真正的 stride 参数化：tiling 带全 stride、地址全换、L1 搬运行跨度重核 | 高 |

所以"要不要支持子区间"的正确问法是"**要不要做 C1**"：它零拷贝零浪费、与竞品调用形态 1:1
对齐，成本主要在 host 侧；只有框架侧连"整根张量"都给不了（只能给非连续 view）时才涉及 C2。

搬运量的量级（模型 case：`HK=HV=32, K=V=128`，`k/w/u/gk` 合计约 460 MB / 11264 token
≈ **40.96 KB/token**；1.5 TB/s）：若某段 `L = 1408`（CP=8 的一个 part），方案 D 的搬运
≈ `2 × 40.96KB × 1408 ≈ 115 MB ≈ 77 µs`，而该 rank 上算子自身只有 ~35 µs 量级
（模型 case 下界 281 µs ÷ 8）——**搬运比算子本体还慢，所以 D 不可取**（段很短时另说）。
这也是 §5.2 里"全量 `[B,T,H,D]↔[B,H,T,D]` 转置 613 µs > 算子下界 281 µs"的同一笔账。

> **要框架侧回答的问题**：gather 局部 buffer 时，front/back 是**拼在一根 T 轴上**（→ 走 A），
> 还是**各自独立成张量**（→ 走 B，完全不需要子区间支持）？两条都不需要动内核；只有当框架侧
> 坚持"一根轴 + 每个 part 单独调"时才需要 C（内核加 stride 支持）或 D（接受搬运）。

**方案 A 的"多算"什么时候真的亏**（不能只看"填核"）：每个工作项 = 一条完整链（链内串行
`NT` 个 chunk），所以运行时 ≈ `ceil(Nwork / AIC_NUM) × 单链时长`，其中 `Nwork = Nseq × HV`。

| 情形 | 多算的段 | 是否真的亏 |
| --- | --- | --- |
| `Nseq × HV ≤ AIC_NUM`（有空闲核） | 落在空闲核上 | **不亏**（还改善带宽/延迟隐藏） |
| `Nseq × HV > AIC_NUM`（要排队） | 占用波次 | **亏**：波数按 `Nwork/AIC_NUM` 线性涨，多算 `k` 个段就多花 `k×HV/AIC_NUM` 波 |

所以"只算末段"这件事在大 `Nseq` 场景下是有实际时间的——此时应优先方案 B（上游把要导出的
那一段作为**独立连续张量**交给算子，零浪费零拷贝），C（内核支持子区间）作为 04 备选。
这也正是 §5.2 第 11 项要框架侧给"每 rank 待用链数 / `Nseq` 分布"的原因：`Nseq` 长期为 1 时
本问题不存在，长期很大时就得选 B 或 C。

#### 1.4.2 定稿（2026-09-23）：**支持子区间窗口**（= 竞品调用形态）

**决定**：本算子支持"窗口是张量 T 轴的子区间"——`cu_seqlens=[bos, eos]`，`bos > 0` 与 `eos < T`
都合法。这样**调用形态与竞品 1:1 一致**（竞品每次就是"整根张量 + `cu_seqlens[-2:]` /
`cu_seqlens[fns-1:fns+1]`"）。内核尚未实现，此时纳入成本最低。

**契约影响**：`api.md` 3.2 / 6 已同步——校验从"`cu[0]=0` 且 `cu[-1]=T`"放宽为
`0 ≤ cu[0] < cu[-1] ≤ T`；`B ≡ 1`、零长段、非递增、越界仍拒绝。

**实现影响**（集中在 host + 一个 tiling 字段；内核地址表达式不变形）：

1. **host tiling**：每段基址按 `bos` 相对**张量起点**算（不再 shift 到窗口坐标）；`T_win = eos - bos`；
   h 维 stride 用 `shape[2]`（**整根** T）；
2. **tiling 多带一个字段**：张量 T 全长（或等价 stride），供内核算 `h * T_full * K`；
3. **内核地址**：`base + bos*stride_t + t*stride_t + h*T_full*K`——与原来的
   `base + t*stride_t + h*T_win*K` 同形，只是 `T_win → T_full`、基址多一个 `bos` 偏移；
4. **L1 常驻搬运**：行范围 = 该段的行（起点 `bos`），容量与流量不变；
5. **标杆与精度**：`reference.py` 本就按 `(bos,eos)` 起算 → 零改动；同一段、同一分块口径 →
   精度策略与容差不变；
6. **用例表**：新增"子区间窗口"用例覆盖竞品形态（见 4.3）。

**随之作废/降级**：(d) 整窗多段（方案 A）保留为**可选扩展**（吃并行度，`Nseq > 1`）；
B / D 不再需要；§5.2 第 12 项关闭；第 9 项（zigzag 一次吃两个 part）也关闭——zigzag 与竞品一样
**每个 part 调一次**（各传该 part 末段的子区间）即可。

**最小例子（8 个 token）**：张量 `[t0 t1 t2 t3 t4 t5 t6 t7]`，只要算 `t4..t7` 这条链。

```
竞品：     传 cu_seqlens=[4,8]        → 它从第 4 格开始读，只读 4 格（子区间）
我们(现在)：得先把 [t4..t7] 单独拷出来 → 再传 cu_seqlens=[0,4]（张量 T 轴 = 窗口）
我们(方案A)：不拷，整根传 + cu_seqlens=[0,4,8] → 产出 2 条链，取第 2 条
```

即："**竞品的算子会说'从第 x 格看到第 y 格'；我们的算子只会说'从头看到尾'**"。想让我们的算子
只看中间一段，就得先把那段"剪下来"（拷贝）；不然就整根传进来多算几条。这不是语义差异，是
**输入形态**差异——两种都能算对，区别是"谁来切、要不要拷贝"。

> 触发条件、计算结果与结果去向的完整链路（含源码行号、`all_gather`/`merge`/`compress_h0`
> 的消费路径、卡内切段形态、框架侧调用伪代码）属于**编排层**，不在本算子范围内。

---

## 2. Stage 总览与完整详设

### 2.1 依赖图与 Stage 总览

#### 2.1.1 完整计算依赖图

```text
GM k/w/v/u/bg/g/gk/cu_seqlens
        |
        +--> [S0 Cube]  P = W_c @ bf16(h_c)               AIC
        |         |  (L0C FP32 -> Fixpipe -> AIV UB)
        |         v
        +--> [S1 Vector] v_new = V_c - P                    两个 AIV
        |         |      +gate(USE_G) / +U_c(DPLR)；量化 BF16
        |         |  (UB -> L1，B 操作数 NZ)
        |         v
        +--> [S2 Cube]  dH = K_c^T @ bf16(v_new)            AIC
        |         |      Kw = L_c^T @ W_c (+ DPLR: bg_c^T @ bf16(V_c))
        |         |  (L0C FP32 -> Fixpipe -> AIV UB)
        |         v
        +--> [S3 Vector] h_c = decay*h_c + dH                两个 AIV
        |         |      M_c = diag(decay) - Kw；h_c -> BF16 影子 H_c
        |         |  (M_c: UB -> L1 A 操作数 NZ；H_c: UB -> L1 B 操作数 NZ)
        |         v
        +--> [S4 Cube]  m_c = M_c @ m_c                      AIC
                  |  (L0C FP32 -> L1 NZ 回写 m；尾 chunk 直接 Fixpipe -> GM hm)
                  +--> 下一 chunk 回到 S0

窗口结束：S3 写 h 到 GM hm[.., 0:V]；S4 的尾 chunk 结果写 GM hm[.., V:V+K]
```

每个节点恰好属于一个 Stage，Cube/Vector 不混用；数据边见 2.1.2 与 2.1.3。

#### 2.1.2 Stage 依赖表

| Stage | 类型 / 执行单元 | 公式 | 前驱 | 可并行 Stage | 输入来源 | 输出落点 | 任务映射 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | Cube / AIC | `P = W_c @ bf16(h_c)` | 同 chunk 无前驱；`h_c` 来自上一 chunk 的 S3 | 无 | `w` GM->L1；`H_c` L1 驻留 | L0C -> Fixpipe -> AIV UB | 每工作项每 chunk 1 个逻辑 MMAD（`K=128` 一次算完，`w` 16 KiB / `H_c` 32 KiB 都容得下） |
| S1 | Vector / 两个 AIV | `v_new = V_c - P`（DPLR `+U_c`）、`*= E(dg)`、量化 BF16；USE_G 另产生 `L_c` | S0 | 无 | `P` 驻 UB；`v/u` GM->UB；`g`、`K_c` GM->UB | `v_new` UB -> L1（B NZ）；`L_c` UB -> L1（A NZ） | 按 `hm` 列分片，本 AIV 的列段一次装齐、一次 VF |
| S2 | Cube / AIC | `dH = L_c^T @ bf16(v_new)`；`Kw = L_c^T @ W_c`（DPLR 追加 `bg_c^T @ bf16(V_c)`） | S1 | 无 | `L_c`、`w`、`v_new`、（DPLR `bg`、`V_c`）均 L1 驻留 | L0C -> Fixpipe -> AIV UB | `L_c^T` 只从 L1 取一次进 L0A，服务 `dH` 与 `Kw` 两次 MMAD |
| S3 | Vector / 两个 AIV | `h_c = decay*h_c + dH`；`M_c = diag(decay) - Kw`；`H_c = BF16(h_c)` | S2 与 S3 自身上一 chunk 的 `h` | 无 | `dH`、`Kw` 驻 UB；`h_c` 驻 UB；`decay` 由 `g`/`gk` 现算 | `h_c` 留 UB；`H_c` -> L1；`M_c` -> L1（A NZ） | 同 head 分片，两半各一次 VF；`M_c` 按 K 行分片（每个 AIV 负责 `K/2` 行） |
| S4 | Cube / AIC | `m_c = M_c @ m_c` | S3 | 无 | `M_c` L1；`m_c` L1 驻留 | 中间 chunk：L0C -> L1 NZ 回写 `m`；尾 chunk：L0C -> Fixpipe -> GM `hm` | `M_c`、`m` 各一次进 L0A/L0B（各 64 KiB 占满），`K=128` 单次 MMAD |

**Stage 顺序**：`S0 -> S1 -> S2 -> S3 -> S4`，同一工作项内严格串行；窗口内 `NT` 个 chunk
依次执行同一序列。不同工作项（不同 value head / 不同列段）之间完全独立，可并行。

#### 2.1.3 跨 Stage 数据表

| 数据 | shape | dtype | 生产 | 消费 | 最后消费者 | 存放 | 份数依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `H_c`（`bf16(h_c)`） | `[K,V]`（按列段） | BF16 | S3 | S0 | S0 的 MTE1 最后一次读 L0B | L1，B NZ | 1 份：跨 chunk 常驻，下一 chunk 的 S0 读完后 S3 才可覆盖 |
| `h_c` | `[K,V]`（按列段） | FP32 | S3 | S3 | 本 chunk S3 末尾（窗口结束时写 GM） | AIV UB | 1 份：跨 chunk 常驻的 FP32 主副本，向量就地更新 |
| `P` | `[BT,V]`（按列段） | FP32 -> BF16 | S0 | S1 | S1 的 VF | UB（Fixpipe 落点） | 1 份：S1 消费后即可被下一 chunk 覆盖 |
| `v_new` | `[BT,V]`（按列段） | BF16 | S1 | S2 | S2 的 MTE1 | L1，B NZ | 1 份：S2 消费后即可被下一 chunk 覆盖 |
| `L_c` | `[BT,K]` | BF16 | 算子输入 `k`；USE_G 时由 S1 产生缩放副本 | S2 | S2 的 MTE1 | L1，A NZ | 输入 `k` 1 份常驻；USE_G 追加 1 份缩放副本 |
| `dH` | `[K,V]` | FP32 | S2 | S3 | S3 的 VF | L0C -> Fixpipe -> UB | 1 份：`L0C[0,64)`，阶段内独占；Fixpipe 按列分片落两个 AIV 的 UB |
| `Kw` | `[K,K]` | FP32 | S2 | S3 | S3 的 VF | L0C -> Fixpipe -> UB | 1 份：`L0C[64,128)`，阶段内与 `dH` 同时存活（峰值 128 KiB） |
| `M_c` | `[K,K]` | FP32 | S3 | S4 | S4 的 MTE1 | L1，A NZ | 1 份：S4 读完即被下一 chunk 覆盖 |
| `m_c` | `[K,K]`（按列段） | FP32 | S4 | S4 | S4 的 MTE1；尾 chunk 由 Fixpipe 写出 | L1，B NZ | 1 份：`M_c@m` 的 L0B 读完之后才写回，无需 ping/pong |
| `hm` | `[K,V+K]` | FP32 | S3（`h`）、S4（`m`） | 调用方 | 调用方 | GM | 最终输出 |
| `decay` | 标量 / `[K]` | FP32 | S3 | S3 | S3 的 VF | UB | 由 `g`/`gk` 现算，两个分片各算自身 K 行范围 |

#### 2.1.4 调度一致性、AIC/AIV 映射与 Stage 数量依据

**分核与任务映射（R20）**：

```text
Nseq      = len(cu_seqlens) - 1              varlen 打包窗口（B ≡ 1，唯一形态）
T_full    = 张量 T 轴全长（= shape[2]）      子区间窗口时 T_win < T_full；段基址按 bos 相对张量起点算
T_win(n)  = cu_seqlens[n+1] - cu_seqlens[n]  第 n 段长度（chunk 从该段起点对齐）
Nbase     = Nseq                              chunk 间有依赖 → Nbase 取序列总数
Nwork     = Nseq * ceil(HV/CG) = Nseq * HV    CG = 1，一个工作项 = (sequence, value head)
AIC_NUM   = 平台可用 Cube 核数（host 读，950PR 上为 28）   来源见 3.5
blockDim  = min(AIC_NUM, Nwork)
work_id   = block_idx + i * blockDim           grid-stride
n         = work_id / HV ; hv = work_id % HV   先跳 sequence，再跳 head
CG        = 1                                 一个工作项处理一个完整的 value head
```

序列表达只有一种（varlen 打包窗口，`Nseq = len(cu_seqlens)-1`），与竞品 CP 路径的
`N = len(cu_seqlens)-1` 分支完全一致（竞品 `cu_seqlens is None` 那一支在 CP 下不可达）。
`cu_seqlens` 按仓内约定是 **host 侧的 `list[int]`**（`int[]?` → `at::OptionalIntArrayRef` →
aclIntArray），**不是 device 张量**：边界校验与每段 `bos/T_win/NT` 的展开都在 host tiling 里做，
不占 GM 带宽，也省掉一次 `cu_seqlens` 的 device 读取（竞品把它当 device tensor 传入 kernel）。
每个工作项带自己的 `bos`、`T_win`、`NT` 与 `M(c)`——host 按仓内 chunk 依赖参考的做法
"把 fixed/varlen 输入统一转换成可执行的 chunk 任务"后写进 tiling。

**并行度只有 `Nseq × HV` 两维**，`chunk` 维被依赖砍掉。R20 原文是"chunk 间有依赖时 `Nbase` 为
sequence 总数"；仓内的 chunk 依赖参考给出同一口径（`docs/agents/reference/04-operator-development/
chunk-dependent-development.md` 第 2 条："同一序列的 chunk 按依赖顺序执行；**不同序列或互不依赖的
head 可以并行**"，第 4~5 条："每核按 head round 推进一个 chunk……当前 chunk 的全部 head round
完成状态更新后，再进入下一 chunk，避免下一 chunk 读到不完整状态"）。参考算子
`ChunkGatedDeltaRuleFwdH` 也是只按 sequence × head 分核，**没有任何一层按 chunk 或按列分核**。

**竞品怎么处理多序列：分三层，本 kernel 那一层不吃序列维。**

| 层 | 谁 | 并行度 / 承载的维度 | 证据 |
| --- | --- | --- | --- |
| CP 编排层 | `fla/ops/cp/context.py` + 包装 | **按 token 切**：`part_len = total_tokens / num_parts`（`total_tokens = cu_seqlens[-1]`），`num_parts = W`（contiguous）或 `2W`（zigzag）。一个 rank 的窗口**可以跨多个 sequence** —— `_interval_cp_meta` 返回的 `local` 是区间内序列边界 clamp+shift 后的结果（`unique_consecutive` 后可能多段） | `context.py` 的 `get_cp_cu_seqlens` |
| CP 编排层 | 同上 | `initial_state = k.new_zeros(N, HV, K, V)`，**`N = len(cu_seqlens)-1` 就是本 rank 的 sequence 数**；`merge` 只给边界行写值（`initial_state[0]` 与 `initial_state[fns]`），其余 sequence 本来就从零状态开始 | `chunk_delta_h.py` 804、873、846、1142 行 |
| **本 kernel** | `pre_process_fwd_kernel_merged` | **grid 恒为 `(列块数, HV)`，`hm` 恒无 sequence 维**（`[HV,K,V+K]` 或 `[2,HV,K,V+K]`，`2` 是 part）。**序列再多也不会多开一个 program** | 804、873、810、889 行 |
| 下游算子 | `chunk_gated_delta_rule_fwd_h` | 拿 `initial_state[N,HV,K,V]` 处理本 rank 的**全部 N 个 sequence**；sequence 并行度落在这里 | GDN `chunk.py` 的调用链 |

所以准确的说法是：**竞品在系统层完全支持多序列（甚至一个窗口跨多 sequence），
但 `pre_process` 这个 kernel 只负责"窗口摘要"，一次只喂一段窗口**（上游注释：
*each part exports the affine chain of its last segment*），**序列数不会增加它的 grid**。

`MULTI_SEQS` 就是为"一段窗口里含多个 sequence、一次算完"准备的（`i_n = tl.program_id(2)`
用 grid dim 2 承载 sequence 号、`hm += i_n * HV * K * (K+V)` 按它偏移），
但这条 CP 路径没启用它——`hm` 的布局装不下 sequence。多 sequence 只出现在
`merge_fwd_bwd_kernel` 的 `INTRACARD_MODE`（`ag_hm` 注释里的 `[S_split, HV, K, K+V]`）。

**对标的含义**：

1. **多序列时我们和竞品的这一层完全同构**——都是 `列块数 × HV` 个 program，
   序列维不进 grid。所以 1.0x 是可比口径。
2. **"用不满核"这个问题在竞品里同样存在**，只是它把序列的并行度交给了下游算子。
3. 我们要不要吃这块并行度是**独立选择**：吃就是"sequence 进 grid 第三维"（上游 `MULTI_SEQS`
   的思路），不吃就和竞品同口径、同短板。若吃，**1.0x 口径必须按"一个 rank 的整个 pre_process
   总时长"定**（竞品多次调用 vs 我们 1 次），见 5.2 第 3 项。
4. **实测依据（2026-09-22，H20，`benchmarks/cp/check_cp_alignment.py`）**：kernel 的
   `MULTI_SEQS` 与逐段调用**逐位相等**（A3 `max_abs=0.000e+00`，三个 case 含非 64 倍数段
   与 GVA），所以"sequence 进 grid 第三维"是竞品**已有且已验证**的能力，不是我们新造的语义；
   CP 级（part 级 wrap 的等价性）待 H20 重跑。

**`Nseq = 1` 且 `HV` 小时会严重用不满核**：`Nseq=1, HV=8` 只有 8 个工作项，28 个 AIC 里 20 个空转
（`GDN泛化用例表` 里 V1/V2 就是单序列配 `HV=16/8`；但 C1~C16 是 `B=8~128` 配 `HV=4~32`，
`Nseq` 充足）。三条出路：

| 方案 | 做法 | 代价 |
| --- | --- | --- |
| **吃满多序列（推荐）** | 一次调用处理**一个打包窗口里的多条序列**（`cu_seqlens` 多段，`Nwork = Nseq × HV`）。kernel 侧把段号放进 grid 第三维（上游 `MULTI_SEQS` 就是这个思路：`i_n = tl.program_id(2)` + `hm` 的段前导维；卡内 CP 的 `intracard_pre_scan` 正是这么用的） | 只需把用例从 `Nseq=1` 扩到 `Nseq>1` 并定 1.0x 口径；**接口形状不用改**。零额外搬运 |
| 按 `hm` 列拆工作项 | 上游 `i_col` 的做法，`Nwork = Nseq × HV × SPLIT` | `k`/`w` 按 `SPLIT` 倍数重复读，HBM 流量上升；3.1 的地址图要重算（L1 余量 192 KiB 够，但流量代价要实测） |
| 把 m 半边的预算拆成独立并行 phase | `M_c = diag(decay) ∓ k^T w` 与 `m` 无关，NT 个 `M_c` 可提前并行算 | 需要额外的缓冲与跨 phase 同步，收益要权衡 |

**优先走第一条**：唯一零额外搬运、且与上游既有能力一致。`CG=1/2/4` 只是同一个 (n,hv) 分片策略下的
head 分组参数，不改变并行度上限；**不拆列段**的容量证明见 3.1.4 末尾（最紧档位 L1 只用 256 KiB / 上限 448 KiB）。
**`AIC_NUM` 由 host 从设备读取后经 tiling 传入，kernel 不写死**（见 3.5）。

**AIV 分工（R01）**：采用**同 head 分片**，两个 AIV 把该工作项的列段对半切。三条理由：

1. **容量**：单 AIV 若承接整个列段，UB 峰值按 §3.1.2 的分项翻倍估算约 **264 KiB**，
   **超过单 AIV UB 的 248 KiB 硬上限**（折半后每 AIV 峰值 132 KiB）；该翻倍估算让 04 用
   目标 CANN 版本下组件的实际占用核对。
2. **关键路径**：S1/S3 两个 Vector Stage 是 5 段串行链上的瓶颈段，两个 AIV 并行承担能把
   该段耗时切半。
3. **列天然可分**：`h` 的 V 列、`m` 的 K 列、`M_c` 的 K 行在数学上互相独立，切分不需要
   跨片归约。

```text
part     = 0 或 1
W        = 列段宽 / 2（按 64 对齐；K=V=128 时 W=64）
h 分片   : AIV part 负责 h[:, part*W : (part+1)*W]
m 分片   : 同一列规则作用在 m 的 K 列上
M_c 分片 : AIV part 负责 M_c 的 K 行 [part*(K/2), (part+1)*(K/2))，一次构造
```

沿列切分不需要跨片归约：`h` 的 V 列、`m` 的 K 列、`M_c` 的 K 行在数学上互相独立。
唯一共享量是 `Kw`（`M_c = diag(decay) - Kw`），它由 AIC 一次算出，两个 AIV 各取自己需要的
K 行分片，不做跨片合并。有效宽为 0 的分片跳过计算与搬运，但仍按协议发布/消费同步通知。

**Stage 与执行单元的映射**：AIC 承载 S0、S2、S4（三个 Cube Stage），两个 AIV 承载 S1、S3
（两个 Vector Stage），按程序顺序依次执行。同一执行单元承载多个 Stage 时，各 Stage 的逻辑
边界、数据交接（`P` / `v_new` / `M_c` / `H_c`）和观察点均独立保留。

**为什么 Stage 数不能继续减少**：

相邻 Stage 两两类型不同（S0/S1、S1/S2、S2/S3、S3/S4），按 R01 天然不能合并；
"同类 Stage 能否合并"只需检查三对：

1. **S0 与 S2（Cube|Cube）不能合并**：`dH` 与 `Kw` 的右操作数 `v_new`，是由 S0 的 Cube
   输出 `P` 经 S1 的 Vector 变换（减/加、逐行缩放、量化 BF16）得到的。合并会让 Cube Stage
   读取本 Stage 的输出（违反 R02），把该 Vector 变换并入又违反 R01。
2. **S1 与 S3（Vector|Vector）不能合并**：`h` 的更新依赖 `dH`、`M_c` 的构造依赖 `Kw`，
   两者都是 S2 的 Cube 输出；理由同上。
3. **S2 与 S4（Cube|Cube）不能合并**：S4 的 A 操作数是 `M_c`，它派生自 S2 的 Cube 输出
   `Kw`（`M_c = diag(decay) - Kw`，DPLR 为 `+`）。`Kw -> M_c` 是逐元素相减加对角广播，
   属于 Vector 操作，因此 S2 与 S4 之间必须夹一个 Vector Stage（即 S3）；若把两者并成
   一个 Cube Stage，该 Stage 的第二个 MMAD 就要读本 Stage 第一个 MMAD 的输出，同样触发
   R02/R01。**注意 S4 并不直接读 `Kw`**：它从 L1 读的是 S3 写出的 `M_c`。

因此 5 个 Stage 中没有任何一对同类 Stage 可以合并。S3 是把两个 Vector 操作（`h` 更新、
`M_c` 构造）按 R15 合并后的**唯一 Vector Stage**：两者都只依赖 S2 的输出，同类型且相邻；
拆成两个 Vector Stage 反而抬高 UB 峰值（`dH` 与 `Kw` 必须同时驻留）。
所以 5 个 Stage 是当前依赖、容量与精度观察点下的最小值。

#### 2.1.5 第一步评审清单

| 评审项 | 结论 | 证据 |
| --- | --- | --- |
| 所有计算节点恰好分配到一个 Stage，Cube/Vector 不混用 | 通过 | 2.1.1、2.1.2 |
| 所有依赖边由合法前序或同 Stage Vector 结果提供 | 通过 | 2.1.2 前驱列、2.1.3 |
| L1/UB 峰值、地址连续性、驻留份数、GM 回写可复核 | 通过 | 3.1、3.2、3.3 |
| Stage 划分依据与 AIC/AIV 映射分别可复核 | 通过 | 2.1.4 |
| 已解释 Stage 数为何不能继续减少 | 通过 | 2.1.4 同类 Stage 合并检查（3 对） |
| 无法满足的规则已标记为阻塞 | 通过 | 5.2 未决问题；R14 兜底路径已定义 |

### 2.2 Stage 0：Cube，`P = W_c @ bf16(h_c)`

- **公式、shape 与 dtype**：`P[BT,V] = W_c[BT,K] @ bf16(h_c)[K,V]`，左操作数 dtype BF16、
  右操作数 dtype BF16、累加 dtype FP32；有效区 `[M,V]`，其中 `M = min(BT, T - c*BT)`，补齐行补零。
- **执行单元与任务映射**：AIC。每工作项每 chunk 一个逻辑 MMAD，`K=128` 一次算完（`w` 16 KiB、
  `H_c` 32 KiB，L0A/L0B 各 64 KiB 足够，不再按 `K` 行切块）；`P` 按列段分别落到两个 AIV 的
  `UB[96,112)`。
- **地址、大小、份数**：左操作数 `w` 取 `L1[128,144)`（A 操作数 NZ，A/B 双用同一份）；
  右操作数 `H_c` 取 `L1[64,96)`（B 操作数 NZ，S3 产物，1 份）。结果 `[BT,V]` FP32 落在
  `L0C[0,32)`，Fixpipe 按列段写到两个 AIV 的 `UB[96,112)`（各 16 KiB，互不重叠）。
  `c==0` 时本 Stage 跳过，`L0C`/`UB` 目标区不写但仍发布 `P_ready`。
- **生命周期**：`H_c` 跨 chunk 常驻，本 chunk S0 的 MTE1 最后一次读完后，下一 chunk 的 S3
  才可覆盖；`P` 在 UB 中只存活到本 chunk S1 的 VF 结束。
- **tail / 空任务**：`M<BT` 时只搬 `M` 行，其余行物理补零；`w` 的越界列补零，
  使 `P` 的无效行自然为 0。
- **同步**：`c > 0` 时进入前等待 S3 发布的 `H_ready`（跨核，`PIPE_MTE3` 语义）；**`c == 0`
  不需要也不得等待 `H_ready`**（`H_0 ≡ 0`，且首 chunk 之前只有 `InitWindow` 发布的
  `init_ready`），本 Stage 整体跳过。两个分支都必须在 `PIPE_FIX` 完成后向两个 AIV 发布
  `P_ready`（`c == 0` 时为"空发布"，保证通知计数配平，见 2.7.3 的边界行为）。核内按
  `MTE2_MTE1 / MTE1_M / M_FIX` 组织 L1->L0->L0C 的顺序。
- **Cube 细节**：`w` 以 A 操作数 NZ 读入 L0A，`H_c` 以 B 操作数 NZ 读入 L0B；MMAD 首次
  累加清零，其余块沿 `K` 方向累加；累加器 FP32，Fixpipe 输出 FP32。
- **执行顺序**：`c > 0`：等 `H_ready` -> 装载 L0A(`w`)/L0B(`H_c`) 并 MMAD -> Fixpipe 写两个
  AIV 的 UB -> 发布 `P_ready`；`c == 0`：直接发布 `P_ready`（不写 L0C/UB，AIV 侧不得读 `P` 区）。

```text
Stage0AIC(item, c):
  if c > 0:
    wait H_ready(item)                     # 上一 chunk 的 S3 产物
    MTE1: L1 -> L0A(w) ; MTE1: L1 -> L0B(H_c)
    MMAD: L0C = L0A @ L0B                  # init_flag 清零；K=128 单次 MMAD
    Fixpipe: L0C -> AIV[0].UB(P_part0), AIV[1].UB(P_part1)
  publish P_ready(item) after PIPE_FIX     # c==0 也发布（空发布），保证 AIV 侧等得到
```

### 2.3 Stage 1：Vector，`v_new` 与 S2 的操作数准备

- **公式、shape 与 dtype**：`v_new[BT,V] = V_c[BT,V] - P`（DPLR：`= P + U_c`）；
  `USE_G` 时 `v_new[t,:] *= E(g[last]-g[t])`。输入 dtype 为 BF16/FP32 混合（`V_c` BF16、
  `P` FP32），计算 dtype 为 FP32，输出 dtype 量化为 BF16。同一 VF 另行产生
  `L_c = BF16(E(dg) * FP32(K_c))`（仅 `USE_G`）供 S2 作 A 操作数。
- **执行单元与任务映射**：两个 AIV，按 2.1.4 的列分片。每个 AIV 一次性装入本分片
  `[BT,W]` 的 `v/u`、`P`、`g` 与 `K_c` 行数据，用**一次 VF**完成
  `减（加）-> 逐行缩放 -> 量化`。
- **地址、大小、份数**：`P` 取本 AIV 的 `UB[96,112)`（S0 的 Fixpipe 落点）；`v/u` 经 MTE2
  搬入 `UB[112,128)` 的计算缓冲；结果 `v_new` 由 UB 写到 `L1[144,160)`（B 操作数 NZ，1 份），
  `L_c`（`USE_G`/`USE_BG`）写到 `L1[112,128)`（A 操作数 NZ，1 份）。
- **GM 搬运**：`v`（或 `u`）每 chunk 每分片一次；`g` 为 `[BT]` 小张量。输入 `k`、`w`
  各只从 GM 搬入一次并常驻 L1（见 R10）。
- **拼 NZ 的方式**：UB->L1 无随路 ND2NZ，按仓内 `chunk_gdn_fwd_prepare` 的做法用 `DataCopy`
  以分型宽（BF16 为 16 列）为单位手工构造 L1 的 NZ 分型；该项在 04 用目标 CANN 版本核对。
- **tail / 空任务**：`M<BT` 时无效行写 0；有效宽为 0 的分片跳过 VF 与搬运，但仍参与通知。
- **同步**：等待 `P_ready`；`v_new` / `L_c` 的 MTE3 完成后发布 `vec_ready`（跨核，
  `PIPE_MTE3` 语义），并在覆盖 L1 目标区前等待上一 chunk 的 `vec_free`。
- **生命周期**：`P` 从 S0 的 Fixpipe 起存活到本 Stage 的 VF 结束；`v_new`、`L_c` 从本
  Stage 的 MTE3 起存活到本 chunk S2 的 MTE1（`vec_free` 由 AIC 发布）。
- **执行顺序**：搬入 `v`/`u` -> 等 `P_ready` -> 一次 VF 完成减（加）、逐行缩放与量化 ->
  等 `vec_free` -> 写 L1 -> 发布 `vec_ready`。

```text
Stage1AIV(item, c, part):
  if validWidth(part) > 0:
    MTE2: GM v/u -> UB ; wait P_ready(item) ; MTE2 -> V
    OneVF: v_new = v - P ; if USE_G: v_new *= E(g_last - g)
           if USE_G: L_c = BF16(E(g_last - g) * FP32(K_c))
    wait vec_free(item)                      # 上一 chunk 的 S2 已读完 L1 区
    V -> MTE3 ; MTE3: UB -> L1 (v_new: B NZ ; L_c: A NZ)
  publish vec_ready(item)                    # 空分片同样贡献通知
```

### 2.4 Stage 2：Cube，`dH` 与 `Kw`

- **公式、shape 与 dtype**：`dH[K,V] = L_c^T[K,BT] @ bf16(v_new)[BT,V]`；`Kw[K,K] = L_c^T @ W_c`；
  DPLR 追加 `dH += bg_c^T @ bf16(V_c)` 并把 `Kw` 的左因子换成 `bg_c^T`。
  两个 MMAD 的左/右操作数 dtype 均为 BF16、累加 dtype 为 FP32。
- **执行单元与任务映射**：AIC。**`L_c^T` 只从 L1 取一次进 L0A**，同时服务 `dH` 与 `Kw`
  两次 MMAD。这是本设计相对上游的核心优化：上游把 `k`/`w` 在每个 h 列块和每个 m 列块各读
  一遍（`K=V=128` 时每 chunk 4 遍），本设计降到 1 遍；同时消除上游
  `i_col=2/3` 两个 m program 对 `Kw` 的重复计算。**DPLR 预留分支例外**：它的左因子有两个
  （`K_c^T` 服务 `dH` 主项、`bg_c^T` 服务 `dH` 追加项与 `Kw`），需要两次 L0A 装载。
- **地址、大小、份数**：`L_c` 取 `L1[112,128)`、`v_new` 取 `L1[144,160)`、`w` 取
  `L1[128,144)`、`k` 取 `L1[96,112)`（DPLR 另加 `bg` `L1[160,176)` 与 `V_c`
  `L1[176,192)`），各 1 份。`dH` 的 `[K,V]` FP32 落 `L0C[0,64)`，Fixpipe 按列分片写到两个
  AIV 的 `UB[32,64)`；`Kw` 的 `[K,K]` FP32 落 `L0C[64,128)`，Fixpipe 按行分片写到两个 AIV
  的 `UB[64,96)`。两者在 `L0C` 上同时存活（阶段内峰值 128 KiB）。
- **容量驱动的分块**：`K=V=128` 时 `dH` `[128,128]` 与 `Kw` `[128,128]` 的 FP32 累加器各 64 KiB，
  在 `L0C` 上可以同时存活（阶段内峰值 128 KiB），因此**不需要按 K 行再分块**，
  每个 MMAD 一次算完。
- **tail / 空任务**：`v_new` / `V_c` 的补齐行为 0，使 `dH` 只由有效行贡献；
  `w` 的 `t >= M` 行补零使 `Kw` 不含无效 token。
- **同步**：等待两个 AIV 的 `vec_ready` 聚合；L1 读完后发布 `vec_free` 供下一 chunk 的 S1
  覆盖；Fixpipe 完成后按行块发布 `dh_ready` / `kw_ready`。
- **生命周期**：`L_c`、`v_new`、`w`（DPLR 另加 `bg`、`V_c`）从写入 L1 起存活到本 chunk S2
  的 MTE1 结束（`vec_free`）；`dH`、`Kw` 的行块从 Fixpipe 起存活到本 chunk S3 的 VF 结束。
- **执行顺序**：等 `vec_ready` -> `L_c^T` 取一次进 L0A -> 按 K 行块算 `dH` 并逐块搬出 ->
  复用同一 L0A 按 K 行块算 `Kw` 并逐块搬出 -> 发布 `vec_free`。

```text
Stage2AIC(item, c):
  wait vec_ready(item)                      # 两个 AIV 聚合
  MTE1: L1 -> L0A(L_c^T)                    # 只取一次
  MTE1: L1 -> L0B(v_new) ; MMAD: L0C_dh = L0A @ L0B        # K=128 单次 MMAD
  if USE_BG:                                                # 预留分支：需第二次装载左因子
    MTE1: L1 -> L0A(bg_c^T) ; MTE1: L1 -> L0B(V_c) ; MMAD: L0C_dh += L0A @ L0B
    MTE1: L1 -> L0B(w) ; MMAD: L0C_kw = L0A @ L0B          # DPLR 的 Kw 左因子是 bg_c^T
  else:
    MTE1: L1 -> L0B(w) ; MMAD: L0C_kw = L0A @ L0B          # L0A 复用同一份 L_c^T
  Fixpipe: L0C[0,64)  -> AIV[0/1].UB(dH) ; publish dh_ready
  Fixpipe: L0C[64,128) -> AIV[0/1].UB(Kw) ; publish kw_ready
  publish vec_free(item) after PIPE_MTE2     # L1 的 v_new / L_c 可被下一 chunk 覆盖
```

### 2.5 Stage 3：Vector，`h` 与 `M_c` 更新

- **公式、shape 与 dtype**：`h_c[K,V] = decay * h_c + dH`（`decay` 为标量或 `[K]` 逐行）；
  `M_c[K,K] = diag(decay) - Kw`（DPLR 为 `+ Kw`）；同时产生 `H_c = BF16(h_c)`。
  `h_c`、`dH`、`Kw`、`M_c` 的计算与存放 dtype 均为 FP32，仅 `H_c` 量化为 BF16。
- **执行单元与任务映射**：两个 AIV，同 head 分片。`h` 分片按列范围就地更新；`M_c` 按
  K 行分片构造（每个 AIV 负责 `K/2=64` 行，一次构造完）；`decay` 由 `g` / `gk` 现算。
- **地址、大小、份数**：`h_c` 常驻本 AIV 的 `UB[0,32)`（FP32 主副本，跨 chunk，1 份）；
  `dH` 取 `UB[32,64)`、`Kw` 取 `UB[64,96)`，都是本 AIV 的行/列分片。输出 `H_c` 写到
  `L1[64,96)`（B 操作数 NZ，1 份）、`M_c` 写到 `L1[192,256)`（A 操作数 NZ，1 份）；
  `decay`/mask 用 `UB[128,132)`。
- **生命周期**：`h_c` 跨 chunk 常驻至窗口末（其后写 GM）；`H_c` 存活到下一 chunk S0 的
  MTE1；`M_c` 存活到本 chunk S4 的 MTE1。
- **tail / 空任务**：`h_c` 的补齐列保持 0；`decay` 对无效行取 1（中性值）避免污染。
- **精度观察点**：本 Stage 是 `h` 的 FP32 累加点，也是 `M_c` 的 FP32 构造点，
  两点均在第 4 章列为观察点。
- **同步**：等待 `dh_ready` / `kw_ready`；`H_c`、`M_c` 的 MTE3 完成后发布
  `H_ready` / `M_c_ready`；窗口结束时按 `hm` 的 `h` 段地址写 GM。
- **执行顺序**：等 `dh_ready`/`kw_ready` 行块 -> 一次 VF 完成 `h` 更新、`M_c` 构造与
  `H_c` 量化 -> 写 L1 -> 发布 `H_ready`/`M_c_ready` -> 尾 chunk 追加写 GM `hm`。

```text
Stage3AIV(item, c, part):
  wait dh_ready(item), kw_ready(item)       # K=128 单次 MMAD，无行块参数
  OneVF: decay = E(g_last) 或 E(gk_last[本分片 K 行范围])
         h[:, part]     = decay * h[:, part] + dH_part
         M_c[K 行范围]  = diag(decay) - Kw[K 行范围]
         H_c_part       = BF16(h[:, part])
  V -> MTE3 ; MTE3: H_c -> L1(B NZ) ; M_c -> L1(A NZ)
  publish H_ready(item), M_c_ready(item) after PIPE_MTE3
  if c == NT-1: MTE3: h -> GM hm[.., h 列段]
```

### 2.6 Stage 4：Cube，`m_c = M_c @ m_c`

- **公式、shape 与 dtype**：`m_c[K,K] = FP32(M_c[K,K]) @ FP32(m_c[K,K])`。**两侧操作数 dtype
  都是 FP32**，累加 dtype 也是 FP32；这是全算子唯一需要 **FP32 原生** MMAD 的运算
  （`AFFINE_CHAIN_PRECISION = ieee`，口径已定稿，见 2.8）。
- **执行单元与任务映射**：AIC。`M_c` 的 `[K,K]` 一次进 L0A、`m` 的 `[K,W]` 一次进 L0B；
  `K=V=128` 时**每个 chunk 只需 1 次 MMAD**。
- **地址、大小、份数**：`M_c` 从 `L1[192,256)` 取 A 操作数 `[K,K]` FP32 = 64 KiB，
  **占满 L0A[0,64)**；`m_c` 从 `L1[0,64)` 取 B 操作数 `[K,K]` FP32 = 64 KiB，
  **占满 L0B[0,64)**。结果 `m_new` 落 `L0C[0,64)`。
- **回写**：中间 chunk 的 L0C 结果经 Fixpipe 以 NZ 格式写回 L1 的同一 `m` 区（该 N 列块的
  L0B 读取在 MMAD 之前已完成，因此**不需要 ping/pong**）；**尾 chunk（`c = NT-1`）直接由
  Fixpipe 以 NZ2ND 写入 GM `hm[.., V:V+K]`**，省掉一次 L1 往返。
- **tail / 空任务**：`m` 的补齐行/列不参与输出；`hm` 的越界区域不写。
- **同步**：等待 `M_c_ready`（AIV S3 -> AIC S4，跨核）；`m` 的 L0B 读取与随后 Fixpipe 的
  同区回写是**核内**依赖，用 `MTE1 -> FIX` HardEvent 保护即可（pipe 对按目标 CANN 版本核对）；
  尾 chunk 在 kernel 退出前排空 Fixpipe。
- **生命周期**：`M_c` 从本 chunk S3 的 MTE3 起存活到本 Stage 的 MTE1 结束——该"释放"就是
  `M_c_ready` 的单槽 ping-pong：S4 读完 → 下一 chunk 的 S3 再次发布，**不存在独立的 free 边**；
  `m` 跨 chunk 常驻 L1，本 Stage 的 L0B 读取完成后才允许 Fixpipe 覆盖同一区。
- **执行顺序**：等 `M_c_ready` -> 装载 L0A(`M_c`)/L0B(`m`) 并单次 MMAD -> 中间 chunk 回写
  L1 / 尾 chunk Fixpipe 直写 GM（两者都在同区内完成，由核内 HardEvent 保序）。

```text
Stage4AIC(item, c):
  wait M_c_ready(item)
  MTE1: L1 -> L0A(M_c)                       # [K,K] FP32 = 64 KiB，占满 L0A
  MTE1: L1 -> L0B(m)                         # [K,K] FP32 = 64 KiB，占满 L0B
  MMAD: L0C = L0A @ L0B                      # FP32 x FP32，init_flag 清零；K=128 单次
  wait HardEvent(MTE1 -> FIX)                # L0B 读完才能覆盖同一 L1 区
  if c == NT-1: Fixpipe(L0C -> GM hm[.., V:V+K], NZ2ND)
  else:         Fixpipe(L0C -> L1 m, NZ2NZ)
```

### 2.7 Stage 间同步方案

本设计选用 `SYNC_L1_HANDOFF`：AIV 直接写 L1，AIC 用跨核 flag 等待，**不使用 GM 中转**
（与仓内先例 `chunk_gdn_fwd_prepare` 的 AIV -> L1 写法一致）。GM 中转按 R14 保留为容量或
路径受限时的兜底（见 5.1、5.3）。

| 通知 | 发布方与条件 | 消费方与地址保护 | 粒度 |
| --- | --- | --- | --- |
| `H_ready` | AIV：`H_c` 的 MTE3 写入完成 | AIC：下一 chunk 的 S0 读 L0B | 每工作项每 chunk 1 次（两个 AIV 聚合） |
| `M_c_ready` | AIV：`M_c` 的 MTE3 写入完成 | AIC：本 chunk 的 S4 读 L0A | 每工作项每 chunk 1 次（两个 AIV 聚合） |
| `P_ready` | AIC：`P` 的 Fixpipe 完成 | AIV：本 chunk 的 S1 读 UB | 每分片 1 次 |
| `dh_ready` / `kw_ready` | AIC：对应行块的 Fixpipe 完成 | AIV：本 chunk 的 S3 读 UB | 每行块 1 次 |
| `vec_ready` | AIV：`v_new` / `L_c` 的 MTE3 完成 | AIC：本 chunk 的 S2 读 L1 | 每工作项每 chunk 1 次（两个 AIV 聚合） |
| `vec_free` | AIC：S2 的 MTE1 读完 L1 | AIV：下一 chunk 的 S1 覆盖该区 | 每工作项每 chunk 1 次 |

> `m` 的"读完后才能被 Fixpipe 覆盖"是 **AIC 核内**依赖（`MTE1 -> FIX` HardEvent），
> 不是跨核通知，因此不进本表。

#### 2.7.1 CrossCore 通知分配

`PIPE_SERIAL` 基线下同一工作项的 chunk 串行推进，每条逻辑边只需要 **1 个物理 flag**
（不按 chunk 轮转、不积压）；`PIPE_PACK_OVERLAP` 候选再按流水深度扩展成多份。

| 逻辑边 | 方向 | 聚合/广播 | 发布 pipe | flag 数 | 复用条件 |
| --- | --- | --- | --- | ---: | --- |
| `init_ready` | AIV ×2 -> AIC | 聚合（2 份） | `PIPE_MTE3` | 1 | 每工作项只发布一次；AIC 进入 chunk 循环前等待一次 |
| `H_ready` | AIV ×2 -> AIC | 聚合（2 份） | `PIPE_MTE3` | 1 | 下一 chunk 的 S0 MTE1 读完 `H_c` 后，由 S3 再次发布 |
| `M_c_ready` | AIV ×2 -> AIC | 聚合（2 份） | `PIPE_MTE3` | 1 | 本 chunk S4 读完 `M_c` 后，由下一 chunk 的 S3 再次发布 |
| `vec_ready` | AIV ×2 -> AIC | 聚合（2 份） | `PIPE_MTE3` | 1 | 本 chunk S2 聚合完成后，下一 chunk S1 的 MTE3 再次发布 |
| `P_ready` | AIC -> AIV ×2 | 广播 | `PIPE_FIX` | 1 | 本 chunk S1 的 VF 读完 `P` 后，下一 chunk S0 再次发布 |
| `dh_ready` | AIC -> AIV ×2 | 广播 | `PIPE_FIX` | 1 | 本 chunk S3 的 VF 读完 `dH` 后释放 |
| `kw_ready` | AIC -> AIV ×2 | 广播 | `PIPE_FIX` | 1 | 本 chunk S3 的 VF 读完 `Kw` 后释放 |
| `vec_free` | AIC -> AIV ×2 | 广播 | `PIPE_MTE2` | 1 | 下一 chunk S1 覆盖 L1 该区前等待 |

合计 **8 个 CrossCore flag**（7 条 chunk 内边 + 1 条窗口初始化边），需在 04 对照目标 CANN
版本的 CrossCore flag 上限核验。`m` 的 L0B 读 / Fixpipe 同区回写是核内 `MTE1 -> FIX`
HardEvent（pipe 对按目标 CANN 版本核对），不占跨核 flag。

所有跨核边都按"**设置（set）-> 等待（wait）**"成对使用，`ready` 与 `free` 两类语义分开：
`ready` 由生产方在对应 pipe 完成后**设置**，消费方访问数据前**等待**；`free` 由最后一个
消费者**设置**，下一次写入该地址的核**等待**。核内则用成对 `HardEvent`：写方在搬运或计算
完成后**设置**，读方在使用前**等待**。同一执行管线的连续操作用程序顺序表达，不额外插事件。

```text
Producer(slot):
  wait free[slot]                     # slot 初始为空闲，首轮直接过
  <compute / store>
  set ready[slot] after <对应 pipe>    # 跨核用 CrossCoreSetFlag，核内用 SetFlag<HardEvent::..>

Consumer(slot):
  wait ready[slot]                    # 跨核用 CrossCoreWaitFlag，核内用 WaitFlag<HardEvent::..>
  <load / compute>
  set free[slot] after <对应 pipe>
```

#### 2.7.2 核内 HardEvent 分配

| 核 | 事件 | 生产 -> 消费 | 保护的对象 | 首次使用 | 末次排空 |
| --- | --- | --- | --- | --- | --- |
| AIC | `MTE2_MTE1` | MTE2 -> MTE1 | L1 中的 `w` / `H_c` / `L_c` / `v_new` 装载 | S0 | kernel 尾部逐项 Wait |
| AIC | `MTE1_M` | MTE1 -> Cube | L0A / L0B 槽 | S0 | 同上 |
| AIC | `M_FIX` | Cube -> Fixpipe | L0C 槽 | S0 | 同上（含尾部 Fixpipe） |
| AIC | `FIX_M` | Fixpipe -> Cube | L0C 槽复用 | S2 | 同上 |
| AIC | `MTE1_MTE2` | MTE1 -> MTE2 | L1 区复用（`H_c` / `v_new` / `L_c` / `M_c`） | S0 | 同上 |
| AIV | `MTE2_V` | MTE2 -> Vector | UB 中的 `V_c`/`U_c`/`P`/`g` | S1 | kernel 尾部逐项 Wait |
| AIV | `V_MTE3` | Vector -> MTE3 | UB 结果区 -> L1 | S1 | 同上（含尾部 MTE3） |
| AIV | `MTE3_MTE2` | MTE3 -> MTE2 | UB 区复用 | S1 | 同上 |
| AIV | `MTE3_V` | MTE3 -> Vector | UB 区复用 | S1 | 同上 |

两核各自维护事件数组，`Process()` 末尾逐项 `WaitFlag` 后 `ReleaseEventID`，避免未消费事件
影响下一次调用。CATLASS 组件内部的事件也计入同一套上限。

#### 2.7.3 完整调度伪代码

`Schedule(blockIdx, Nwork, blockDim)` 是 R20 的 grid-stride 分发；本设计
`Nwork = B × HV`、`work_id -> (n, hv)`。所有 slot 初始为空闲，**首轮不等待 `free`**。

```text
Process(blockIdx):
  allocate flags, hard events; mark every slot free
  if ASCEND_IS_AIC: AICLoop(blockIdx)
  else:             AIVLoop(blockIdx, subBlockIdx)

AICLoop(blockIdx):
  for hv in Schedule(blockIdx, HV, blockDim):
    wait init_ready(hv)             # 窗口初始化完成（m=I 已写入 L1）
    for c in 0 .. NT(hv)-1:            # NT(hv) = ceil(M_win(hv)/BT)
      Stage0AIC(hv, c); Stage2AIC(hv, c); Stage4AIC(hv, c)
  wait remaining CrossCore flags(H_ready, M_c_ready, vec_ready)
  drain HardEvents; release flags

Stage0AIC(hv, c):
  if c == 0:
    skip                            # H_0 ≡ 0 ⇒ v_decay = 0 ⇒ v_new = V_c（与上游逐位等价）
  else:
    wait H_ready(hv)                # 上一 chunk 的 S3 产物
    TileCopy: L1 -> L0A(w) ; L1 -> L0B(H_c)
    LocalMmad -> L0C(P); Fixpipe -> AIV part0/part1 的 UB(P 分片)
  publish P_ready(hv) after PIPE_FIX     # c==0 也发布，保证通知平衡

Stage2AIC(hv, c):
  wait vec_ready(hv)                # 两个 AIV 聚合
  TileCopy: L1 -> L0A(Lc^T)          # 只取一次，服务 dH 与 Kw
  LocalMmad(dH = Lc^T @ v_new) ; Fixpipe -> AIV part0/part1 的 UB(dH)
  if USE_BG:
    TileCopy: L1 -> L0B(V_c) ; LocalMmad(dH += bg^T @ V_c) ; Fixpipe -> UB(dH)
  publish dh_ready(hv) after PIPE_FIX
  LocalMmad(Kw = Lc^T @ w) ; Fixpipe -> AIV part0/part1 的 UB(Kw)
  publish kw_ready(hv) after PIPE_FIX
  publish vec_free(hv) after PIPE_MTE2   # L1 的 v_new / L_c 可被下一 chunk 覆盖

Stage4AIC(hv, c):
  wait M_c_ready(hv)
  TileCopy: L1 -> L0A(M_c) ; L1 -> L0B(m)              # 各 64 KiB，K=128 单次 MMAD
  LocalMmad -> L0C(m_new)
  wait HardEvent(MTE1 -> FIX)                          # L0B 读完才能覆盖同一 L1 区
  if c == NT-1: Fixpipe NZ2ND -> GM hm[.., V:V+K]       # 尾 chunk 直写，省一次 L1 往返
  else:         Fixpipe NZ2NZ -> L1 m

AIVLoop(blockIdx, part):
  for hv in Schedule(blockIdx, HV, blockDim):
    width = 本 AIV 承担的列数 = (V + K) 段的一半（按 64 对齐）
    InitWindow(hv)                   # 不属于 chunk 循环，不占 Stage 编号
    for c in 0 .. NT(hv)-1:
      Stage1AIV(hv, c, part); Stage3AIV(hv, c, part)
  wait remaining CrossCore flags(P_ready, dh_ready, kw_ready, vec_free)
  drain HardEvents; release flags

InitWindow(hv):
  OneVF: 生成 I_K 的单位阵分片 -> UB 临时区
  MTE3: UB -> L1[0,64) 的 m 区          # 只写一次，每工作项一次
  V -> MTE3 ; MTE3: 清零 UB[0,32) 的 h 分片
  publish init_ready(hv) after PIPE_MTE3

Stage1AIV(hv, c, part):
  if width == 0:                      # 空分片
    publish vec_ready(hv)             # 只贡献通知，不做搬运与计算
    return
  MTE2: GM v/u -> UB ; MTE2 -> V
  wait P_ready(hv)  # c == 0 同样要等（S0 的空发布），但**不得读 P 区**，直接取 v_new = V_c
  OneVF: v_new = V_c - P (或 P + U_c) ; if USE_G: v_new *= E(g_last - g)
         if USE_G: L_c = BF16(E(g_last - g) * FP32(K_c))
  wait vec_free(hv)  if c > 0         # L1 目标区上次已被 S2 读完
  V -> MTE3 ; MTE3: UB -> L1 (v_new: B NZ ; L_c: A NZ)
  publish vec_ready(hv) after PIPE_MTE3

Stage3AIV(hv, c, part):
  wait dh_ready(hv), kw_ready(hv)
  OneVF: decay = E(g_last) 或 E(gk_last[K 行范围])
         h[:, part] = decay * h[:, part] + dH_part
         M_c[K 行范围] = diag(decay) - Kw[K 行范围]   (USE_BG 取 +)
         H_c_part = BF16(h[:, part])
  V -> MTE3 ; MTE3: H_c -> L1(B NZ) ; M_c -> L1(A NZ)
  publish H_ready(hv), M_c_ready(hv) after PIPE_MTE3
  if c == NT-1: MTE3: h[:, part] -> GM hm[.., h 列段]
```

边界行为：

- **`NT=1`**：循环只跑一次，`c>0` 的等待全部跳过；S0 因 `c==0` 跳过，`P_ready` 仍然发布。
- **`c == 0`**：S0 整体跳过（`H_0 ≡ 0`，既不 wait `H_ready` 也不写 L0C/UB），但**仍发布
  `P_ready`**；S1 侧同样 wait `P_ready` 配平，且**不得读 `P` 区**（该区此刻未初始化），
  直接取 `v_new = V_c`（DPLR 为 `U_c`）。这两条是协议要求，不是可选优化。
- **tail（`M(c) < BT`）**：所有搬运只搬 `M(c)` 行，物理 tile 的其余行补零；
  `hm` 只写有效行，不跨 chunk 覆写。
- **varlen**：`M_win` 与 `NT` 由 `cu_seqlens` 推出，其他路径完全相同。
- **空分片**：只发 `vec_ready`，不搬运不计算；`H_ready`/`M_c_ready` 仍按本 AIV 的空行区间
  参与协议，保证通知计数平衡。
- **退出排空**：两核在 `Process()` 末尾等待所有未消费的 CrossCore flag 与 HardEvent，
  再释放事件资源；尾 chunk 的 Fixpipe 必须在返回前完成。

一处可选的进一步优化（**不进基线**）：`c==0` 时 `m_0 = M_0 @ I = M_0` 在 FP32 下逐位精确，
可以用一次 L0C `Fixpipe` 直通替代 S4 的 MMAD；代价是引入一条只在首 chunk 生效的分支，
留给 04 按实测收益决定。

### 2.8 Kernel 组件与调用

- **入口**：MIX kernel（`KERNEL_TYPE_MIX_AIC_1_2`）。AIC 与 AIV 各自 `Process()` 按
  `block_idx` / `blockDim` 以 grid-stride 领取工作项（R20），并按 2.1.4 的列分片取自己的部分。
- **CATLASS 组件粒度**：使用 **Tile 级**（`Gemm::Tile::TileMmadTla` + `TileCopy`）手写，
  chunk 循环留在核内，L1/L0 槽与 `HardEvent` 自管；**不使用**
  `BlockMmad` / `BlockScheduler` / `Kernel` / `DeviceGemm`。依据：
  chunk 循环共 `NT` 次（模型 case 176 次），单次 `DeviceGemm` 调用在本算子形状上 99% 是
  固定开销（实测 `128x64x128` FP32 为 8.092 us，纯 Cube 仅 0.089 us），外推比核内循环
  差约 17 倍；仓内先例 `chunk_fwd_h_cube.h` 正是 `TileMmadTla` + 自管 L0 槽 +
  `SetFlag<HardEvent::MTE1_MTE2>` 的写法。
- **ArchTag**：`Arch::Ascend950`（`CATLASS_ARCH=3510`）。
- **逐 Stage 组件与 TileShape**（`(M, N, Kdim)`；`K=V=128`、`BT=64`）：

| Stage | 运算 | TileShape | 左操作数（L0A） | 右操作数（L0B） | 累加 | Fixpipe |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | `P = W_c @ bf16(h)` | `(BT=64, V, K)` | `w` L1 A-NZ BF16 | `H_c` L1 B-NZ BF16 | FP32 | NZ -> AIV UB |
| S2 | `dH = L_c^T @ bf16(v_new)` | `(K, V, BT=64)` | `L_c^T` L1 A-NZ BF16 | `v_new` L1 B-NZ BF16 | FP32 | NZ -> AIV UB |
| S2 | `Kw = L_c^T @ W_c` | `(K, K, BT=64)` | **复用同一份 L0A** | `w` L1 B-NZ BF16 | FP32 | NZ -> AIV UB |
| S2（DPLR） | `dH += bg_c^T @ bf16(V_c)` | `(K, V, BT=64)` | `bg^T` L1 A-NZ BF16 | `V_c` L1 B-NZ BF16 | FP32 | 累加到同一 L0C |
| S4 | `m = M_c @ m` | `(K, W, K)` | `M_c` L1 A-NZ FP32 | `m` L1 B-NZ FP32 | FP32 | NZ2NZ -> L1 / NZ2ND -> GM |

  `K=V=128` 时 S0 的 `N=V=128`、S2 的 `N=V/K`、S4 的 `Kdim=K=128` 都在一次 MMAD 内完成，
  不需要再切 tile。`w` 在 S0 作 A、在 S2 作 B，本形状下两者在 NZ 下的分型索引一致（3.1 脚注），
  只占一份 L1。
- **S4 的精度路径（已定稿，2026-09-22）**：这是全算子唯一 FP32×FP32 的 MMAD，**基线取
  FP32 原生**（两侧操作数 FP32、累加 FP32），即 `AFFINE_CHAIN_PRECISION = ieee` 在 NPU 侧的
  落法；**不做 BF16 三分拆**——`m` 链长约 176、有效尾数需求约 20 位，三分拆虽能补尾数，但会把
  `M_c` 的 L0A 占用从 64 KiB 涨到 96 KiB（**超过 L0A 的 64 KiB 上限**），必须把 `M_c` 按 K 行
  再切一刀，属于退回第一步的结构性改动。
  H20 的 `default`（NVIDIA TF32）只是**性能对标**口径（同 case 实测 `default` 1620.7 us vs
  `ieee` 18321.6 us，差 11×）；**精度验收**按 ATK 双标杆（`tests/atk/pre_process_fwd_kernel_merged/`）对 CPU 契约标杆
（CPU 标杆：FP32 累加 + 四个舍入点）。FP32 原生比 H20 `default` 更接近真值，
  因此这条口径在精度与性能两侧都不吃亏。若目标 CANN 版本的 Cube 不支持 FP32 原生或吞吐不达
  预期，回退候选顺序为**单遍 HF32 → BF16 三分拆**，04 用核内循环实测单步成本后决定。
- **合法分支组合**：`K = V = 128`、`BT = 64`、`layout = BNSD`、
  `gate ∈ {USE_G, USE_GK}`、`gate dtype ∈ {BF16, FP32}`，即 3.2.1 的 4 个 `TilingKey`
  （`USE_BG` 槽位保留但 host 永不产生，见文档开头的范围变更）。
- **Vector API**：`exp2`、逐行广播乘、cast、mask、`DataCopy`（含手工构造 NZ 分型）；
  全部走批量路径，热路径不做逐元素标量访问。
- **输出写回**：`h` 由 S3 的 MTE3 写 GM；`m` 由 S4 的 Fixpipe（NZ2ND）写 GM。

---

## 3. 全局资源、Tiling 和 Workspace

### 3.1 L1 / UB / L0 地址与容量

预算上限取 §1.1 的平台参数：L1 512 KiB、单 AIV UB 248 KiB、
L0A/L0B 64 KiB、L0C 256 KiB。地址单位是 KiB 半开区间，全部按 512 B 对齐；
下表按固定规格 `K=V=128` 给出具体数字。`h_c` 的 FP32 主副本放 UB，`m_c` 放 L1，
两者不共用存储。

**为什么一个 head 就够一个工作项**：把 `hm` 的列再拆成多个工作项，只在单片 L1 装不下
一个 head 的常驻量时才需要。DPLR 档位 L1 峰值 256 KiB，离 448 KiB 的设计上限还有 192 KiB
余量，所以**不拆列段**，一个工作项 = 一个完整 head，
`Nwork = B × HV`，`k`/`w` 每 chunk 只从 GM 读一遍。

#### 3.1.1 L1 固定地址图（全部 6 个 TilingKey 共用同一张图）

每种模式使用同一组 offset，未使用的槽保留但不访问；这样 L1 布局与 `GATE_MODE` 无关，
省掉一整类分支。

| L1 区间 | 内容、shape、dtype、份数 | 大小 | 首次写入 → 最后消费 | 复用保护 |
| --- | --- | ---: | --- | --- |
| L1[0,64) | `m` `[128,128]` FP32 NZ(B)，1 份 | 64 | 窗口初始化（写 `I`）与每 chunk S4 的 Fixpipe → 下一 chunk S4 的 MTE1 | `MTE1_MTE2` |
| L1[64,96) | `H_c` `[128,128]` BF16 NZ(B)，1 份 | 32 | S3 MTE3 → 下一 chunk S0 的 MTE1 | `H_ready` / `MTE1_MTE2` |
| L1[96,112) | `k` `[BT,K]=[64,128]` BF16 NZ(A)，1 份 | 16 | MTE2 → S2 的 MTE1（`dH`） | `MTE1_MTE2` |
| L1[112,128) | `L_c` `[64,128]` BF16 NZ(A)，1 份 | 16 | S1 MTE3 → S2 的 MTE1（`Kw`） | 仅 `USE_G`/`USE_BG`；`vec_free` |
| L1[128,144) | `w` `[64,128]` BF16 NZ，1 份（A/B 双用） | 16 | MTE2 → S0 的 L0A 与 S2 的 L0B | `MTE1_MTE2` |
| L1[144,160) | `v_new` `[64,128]` BF16 NZ(B)，1 份 | 16 | S1 MTE3 → S2 的 MTE1 | `vec_free` |
| L1[160,176) | `bg` `[64,128]` BF16 NZ(A)，1 份 | 16 | MTE2 → S2 的 MTE1 | 仅 `USE_BG` |
| L1[176,192) | `V_c` `[64,128]` BF16 NZ(B)，1 份 | 16 | MTE2 → S2 的 MTE1 | 仅 `USE_BG` |
| L1[192,256) | `M_c` `[128,128]` FP32 NZ(A)，1 份 | 64 | S3 MTE3 → S4 的 MTE1 | `M_c_ready` / `MTE1_MTE2` |
| **小计** | | **256** | 硬上限 512 KiB；设计预算 448 KiB（512 − 64 预留） | 最大连续空闲 `L1[256,512)` = **256 KiB** |

> `w` 的 A/B 双用依据：`[BT,K]` 矩阵在 NZ 下的分型索引对 A 操作数（外维 `K`、内维 `M`）
> 与 B 操作数（外维 `N`、内维 `Kdim`）在本形状上一致，因此 S0 的 A 与 S2 的 B 可共用同一份
> L1 数据，不需要第二份副本。该结论在 04 用目标版本 `TileCopy` 的实际 layout 支持复核；
> 不成立时在 `L1[256,272)` 增加一份副本（L1 仍只用 272 KiB）。

#### 3.1.2 UB 固定地址图（每个 AIV）

| UB 区间 | 内容、shape、dtype、份数 | 大小 | 首次写入 → 最后消费 | 复用保护 |
| --- | --- | ---: | --- | --- |
| UB[0,32) | `h` 分片 `[K,V/2]=[128,64]` FP32，1 份，跨 chunk 常驻 | 32 | 窗口初始化清零 → 每 chunk S3 就地更新 → 窗口末 MTE3 写 GM | `MTE3_V` |
| UB[32,64) | `dH` 分片 `[128,64]` FP32（S2 Fixpipe 落点） | 32 | S2 Fixpipe → 本 chunk S3 的 VF | `FIX_M` / `MTE3_V` |
| UB[64,96) | `Kw` 行分片 `[K/2,K]=[64,128]` FP32（S2 Fixpipe 落点） | 32 | S2 Fixpipe → 本 chunk S3 的 VF | `FIX_M` / `MTE3_V` |
| UB[96,112) | `P` 分片 `[BT,V/2]=[64,64]` FP32（S0 Fixpipe 落点） | 16 | S0 Fixpipe → 本 chunk S1 的 VF | `MTE2_V` |
| UB[112,128) | `v_new` / `L_c` 计算缓冲 `[64,64]` FP32，1 份 | 16 | 本 chunk S1 的 VF 内部 | `MTE2_V` |
| UB[128,132) | `decay`、mask、标量小量 | 4 | 本 chunk S3 的 VF | —— |
| **小计** | | **132** | 硬上限 248 KiB；设计预算 224 KiB（248 − 24 预留） | 最大连续空闲 `UB[132,248)` = **116 KiB** |

`h` 分片 32 KiB、`dH`/`Kw` staging 各 32 KiB，合计 132 KiB，是 UB 的主要占用；
余量 116 KiB 留给将来的双缓冲或精度分支。

#### 3.1.3 L0 槽位与复用

L0 的布局取决于阶段而非固定 offset：**S4 的两份 FP32 操作数各占满一个 64 KiB 的 L0A/L0B**，
所以 S4 期间独占两者，S0/S2 的操作数在进入 S4 前必须已经消费完（由核内事件保证）。

| L0 | 槽区间 | 内容 | 大小 | 使用者 | 复用关系 |
| --- | --- | --- | ---: | --- | --- |
| L0A | [0,16) | `w` `[64,128]` BF16 A-NZ | 16 | S0 | 与 S2 的 `L_c^T` 分时复用 |
| L0A | [0,16) | `L_c^T` `[128,64]` BF16 A-NZ | 16 | S2（`dH` 与 `Kw` 共用同一份） | 同上 |
| L0A | [16,32) | `bg^T` `[128,64]` BF16 A-NZ | 16 | S2（仅 DPLR） | —— |
| L0A | [0,64) | `M_c` `[128,128]` **FP32** A-NZ | 64（占满） | S4 | S4 独占 |
| L0B | [0,32) | `H_c` `[128,128]` BF16 B-NZ | 32 | S0 | 与 S2 的 `v_new` 分时复用 |
| L0B | [0,16) | `v_new` `[64,128]` BF16 B-NZ | 16 | S2（`dH`） | 同上 |
| L0B | [16,32) | `w` `[64,128]` BF16 B-NZ | 16 | S2（`Kw`） | —— |
| L0B | [32,48) | `V_c` `[64,128]` BF16 B-NZ | 16 | S2（仅 DPLR） | —— |
| L0B | [0,64) | `m` `[128,128]` **FP32** B-NZ | 64（占满） | S4 | S4 独占 |
| L0C | [0,32) | `P` `[64,128]` FP32 | 32 | S0 | 与 S2/S4 分时复用 |
| L0C | [0,64) | `dH` `[128,128]` FP32 | 64 | S2 | 与 `Kw` **同时存活** |
| L0C | [64,128) | `Kw` `[128,128]` FP32 | 64 | S2 | 同上 |
| L0C | [0,64) | `m_new` `[128,128]` FP32 | 64 | S4 | 与 S2 分时复用 |

**L0C 阶段内峰值 128 KiB**（S2 的 `dH` + `Kw`）≤ 256 KiB，因此 S2 的两个 MMAD 可以连续发射，
不需要串行化；S4 的 `m_new` 单独一份。**L0A/L0B 的阶段内峰值各 64 KiB**，正好等于容量上限。

事件分配见 2.7.2；ping/pong 只发生在跨 chunk 常驻的 `m` 与 `H_c` 上，两者都由对应 `free`
通知或核内事件保护，不需要额外的 L0 物理槽。若将来开 `PIPE_PACK_OVERLAP`，需要给 `dH`/`Kw`
各加一份 staging（UB +64 KiB，仍有 52 KiB 余量）并给 L0 槽加代际标记。

#### 3.1.4 生命周期表

| 数据 | 位置 | 生产 | 各次消费 | 释放 / 复用点 | 份数依据 |
| --- | --- | --- | --- | --- | --- |
| `m` | L1[0,64) | 窗口初始化写 `I`；每 chunk S4 的 Fixpipe | 每 chunk S4 的 MTE1 | 本 chunk S4 读完即被下一 chunk 的 Fixpipe 覆盖 | 1 份：跨 chunk 常驻，读写都在 AIC，串行覆盖 |
| `H_c` | L1[64,96) | 每 chunk S3 的 MTE3 | 下一 chunk S0 的 MTE1 | `H_ready` 发布、下一 chunk S0 读完后 `vec_free` 之外由 `MTE1_MTE2` 保护 | 1 份：跨 chunk 常驻 |
| `h` | UB[0,32) | 窗口初始化清零 | 每 chunk S3 的 VF（读写） | 窗口末 MTE3 写 GM 后释放 | 1 份：跨 chunk 常驻 |
| `k` | L1[96,112) | GM -> MTE2 | 每 chunk S2 的 MTE1 | 本 chunk S2 读完即被下一 chunk 的 MTE2 覆盖 | 1 份：每 chunk 一次 |
| `L_c` | L1[112,128) | 每 chunk S1 的 MTE3 | 本 chunk S2 的 MTE1 | `vec_free` | 1 份 |
| `w` | L1[128,144) | GM -> MTE2 | S0 的 L0A、S2 的 L0B | 本 chunk S2 读完即被覆盖 | 1 份（A/B 双用） |
| `v_new` | L1[144,160) | 每 chunk S1 的 MTE3 | 本 chunk S2 的 MTE1 | `vec_free` | 1 份 |
| `bg` / `V_c` | L1[160,192) | GM -> MTE2 | 本 chunk S2 的 MTE1 | 本 chunk S2 读完即被覆盖 | 各 1 份（仅 DPLR） |
| `M_c` | L1[192,256) | 每 chunk S3 的 MTE3 | 本 chunk S4 的 MTE1 | `M_c_ready` / `MTE1_MTE2` | 1 份 |
| `P` | UB[96,112) | 每 chunk S0 的 Fixpipe | 本 chunk S1 的 VF | 本 chunk S1 的 VF 结束时释放 | 1 份 |
| `dH` / `Kw` | UB[32,96) | 每 chunk S2 的 Fixpipe | 本 chunk S3 的 VF | 本 chunk S3 的 VF 结束时释放 | 各 1 份（串行基线不需要 ping/pong） |
| `v_new`/`L_c` 缓冲 | UB[112,128) | 本 chunk S1 的 VF | 本 chunk S1 的 VF | VF 内部 | 1 份 |
| `hm` | GM | S3（`h` 段）与 S4（`m` 段，尾 chunk） | 调用方 | —— | 最终输出 |

窗口初始化（不属于 chunk 循环，也不占 Stage 编号）：AIV 把单位阵写进 `L1[0,64)` 的 `m` 区、
把 `UB[0,32)` 的 `h` 分片清零，完成后发布 `init_ready`；AIC 进入 chunk 循环前等待一次。
这与参考设计里"每个核在进入 chunk 循环前先造常数阵"的做法一致，代价是每工作项一次
64 KiB 的 L1 写入与一次 UB 清零。

**片内余量总结**：DPLR 用到 L1 256 / UB 132 / L0C 128 KiB，对照**设计预算** 448 / 224 / 256 KiB
（硬上限 512 / 248 / 256 KiB），
余量分别是 192 / 92 / 128 KiB。这个余量分布是"不拆列段、不切 L0 tile"的前提，
数值由 `scripts/capacity_check.py` 复算，事件分配见 2.7.2。

### 3.2 GM / Workspace

本设计采用 `SYNC_L1_HANDOFF`，**所有跨 Stage 交接都在片内完成，不需要 workspace**
（`workspace_bytes = 0`），因此没有 GM 中转的重复搬运。GM 流量只有一次性输入读与最终输出写：

| 项 | 每 chunk 每 head 的字节数 | 说明 |
| --- | --- | --- |
| 读 `k` | `2*BT*K` | 每 chunk 只读一次（不再是上游的 4 次或 8 次） |
| 读 `w` | `2*BT*K` | 同上 |
| 读 `v`（或 `u`） | `2*BT*V` | 每分片读自身列段 |
| 读 `g` / `gk` | `4*BT` 或 `4*BT*K` | `gk` 为 FP32 时是本算子的主要带宽项之一 |
| 写 `hm` | `4*K*(V+K)` | 窗口结束时一次 |

模型 case（`K=V=128, BT=64, T=11264, HV=32`）：输入读约 203 MB，输出写 `32*128*256*4`
= 4 MB，合计约 207 MB；按 1.5 TB/s 约 **138 us**，低于 Cube 吞吐下界，说明本设计已回到
**计算受限**一侧（上游逐列块重复读 `k`/`w` 的方案约 553.6 us，属带宽受限）。

#### 3.2.1 `TilingKey` 枚举

`TilingKey` 只由编译期维度组成，规模有界：

```text
TilingKey = (GATE_MODE, GATE_T)

GATE_MODE ∈ {USE_G, USE_GK}           # 核心路径：是否需要 L_c（USE_BG 槽位保留但不产生）
GATE_T    ∈ {BF16, FP32}              # g/gk 的存储 dtype
```

`K`/`V`/`BT` 是固定规格（128 / 128 / 64），不进 `TilingKey`。可达组合
**2 × 2 = 4 个 `TilingKey`**（`USE_G` / `USE_GK` × `GATE_T ∈ {BF16, FP32}`）。
`USE_BG`（DPLR）**不产生**：`bg` 非空被 host 直接拒绝（`docs/api.md` §6），
模板槽位与相关布局条目只作历史记录，不进精度/性能用例。选择条件全部来自 host 校验后的属性，kernel 内只有一个
`switch(TilingKey)` 的模板分派，没有运行时分支。不满足固定规格的输入在 host 侧被拒绝
（`K!=128`、`V!=128`、`chunk_size!=64`、`HK>HV`、`HV%HK!=0`）。

#### 3.2.2 Workspace

`workspace_bytes = 0`。理由是 `SYNC_L1_HANDOFF` 让所有跨 Stage 交接都在片内完成：
`P`、`dH`、`Kw` 走 L0C -> Fixpipe -> UB，`v_new`、`L_c`、`H_c`、`M_c` 走 UB -> MTE3 -> L1，
`m` 留在 L1。host 仍需按 ACLNN 两段式接口返回 `workspaceSize`，本算子返回 0 即可
（`hm` 是显式输出张量，不走 workspace）。

若 R14 兜底被触发（5.1 中的 AIV -> L1 受限场景），GM 中转区按
`AlignUp(blockDim × (BT×K×2 + BT×V×2 + K×K×4) × 2, 512)` 预留，即每核一份
`v_new`/`L_c`/`M_c` 的双缓冲；该路径在 04 触发时再按实测量化收益后决定是否保留。

### 3.3 Tiling 与边界

```text
host 校验：B/HK/HV/T=dtype/连续性、K==128、V==128、chunk_size==64、
           HV>=HK 且 HV%HK==0（两条路径都支持 GVA：k 在 HK 维、gate 在 HV 维）、
           g 与 gk 恰提供一个、bg/v 必须为空（DPLR 不支持）、
           cu_seqlens 严格递增、B==1
kernel 入参：Nwork=HV, blockDim, NT, M(c), 各张量 stride
分支：完整 chunk / tail；USE_G / USE_GK
```

`NT`、`M(c)` 在 device 侧由窗口长度推出；进入 kernel 的 chunk 保证 `1 <= M <= 64`。
`T=0` 由 host 拦截。`hm` 的越界行/列不写、不参与比较。
**编译期模板 vs 运行时 tiling 的字段划分**：

| 承载方式 | 字段 | 说明 |
| --- | --- | --- |
| 编译期常量（固定规格） | `K = V = 128`、`BT = 64` | 决定 L1 / UB / L0 的地址布局与 `TileShape`；host 拦截其它值 |
| 编译期模板 | `GATE_MODE` / `GATE_T` | 核心路径选项，即 3.2.1 的 6 个 `TilingKey` |
| 运行时 tiling | `NT`、`M(c)` | 由窗口长度与 `cu_seqlens` 推出 |
| 运行时 tiling | `B`、`HV`、`Nwork = B × HV`、`blockDim` | R20 的分核参数（并行度只有 `B × HV`，见 2.1.4） |
| 运行时 tiling | 窗口基址、各张量 stride 与每段的 `bos/T_win/NT` | 段边界（`cu_seqlens`）、BNSD 寻址 |

所有可达 `TilingKey` 的列表见 3.2.1（6 个），选择条件全部由 host 在进入 kernel 前确定。

### 3.4 `K`/`V` 写死为 128 的依据

`K`/`V` 是**编译期固定规格**，不设档位、不做 fallback，host 直接拦截其它值。三条依据：

1. **仓内通行做法**：`npu_chunk_fwd_h`（本算子的直接兄弟）在 1539 行硬校验
   `"K and V must both be 128."`；`npu_chunk_gated_delta_rule_bwd`、`npu_chunk_gdn_bwd_intra`、
   `npu_chunk_kda_bwd`、`npu_chunk_kda_bwd_recompute`、`npu_chunk_kda_fwd_finalize` 等同样是
   精确 128；只有 `npu_chunk_gated_delta_rule_fwd_prepare`、`npu_recurrent_kda` 把 `V` 放宽到
   `{128, 256}`。**没有一个 AscendC 算子的 `K` 不是 128。**
2. **参考设计文档的写法**：`gdn-fwd-h`、`gdn-backward-finalize`、`chunk_fwd_o`、
   `chunk_gdn_fwd_prepare` 都把 `BT/K/V` 当固定规格，host 必须显式拦截，
   "不能静默 fallback"。
3. **模型 case 的实际取值**：`GDN泛化用例表.xlsx` 的 34 行里 `Kdim` 全部是 128，
   `Vdim` 只有 128 与 256，没有任何 case 用到别的值。

写死之后的容量一次算完，`usage` 见 3.1.1～3.1.3：

| 模式 | L1 峰值 | UB 峰值（每 AIV） | L0C 阶段内峰值 |
| --- | ---: | ---: | ---: |
| `USE_GK` | 208 KiB | 132 KiB | 128 KiB |
| `USE_G` | 224 KiB | 132 KiB | 128 KiB |

对照**设计预算** 448 / 224 / 256 KiB（硬上限 512 / 248 / 256 KiB），余量 192 / 92 / 128 KiB。数值由
`scripts/capacity_check.py` 复算。

> 历史备注：`K`/`V` 曾经按"档位"处理（`K_TILE`/`V_TILE` 取 `{64,128}`，还讨论过
> `K=256` 的 `SPLIT=2` 拆分），那些档位来自上游 Triton 的 `assert K <= 256` 与
> `BLOCK_SIZE = 32 if K <= 64 else 64`，**不是这个仓库的做法**，现已全部删除。

### 3.5 平台常量与可移植性

本文的容量、地址图与分核规则都建立在一组**平台常量**上。把它们和"换硬件时怎么处理"一次列清，
避免把某一档位的数字当成通用结论：

| 常量 | 本设计取值 | 来源 | 换硬件时 |
| --- | --- | --- | --- |
| `AIC_NUM`（可用 Cube 核数） | 28 | 平台 ini 的 `cube_core_cnt`（§1.1）；**host 侧改为运行时从设备读**（`aclrtGetDeviceInfo` 一类接口）后经 tiling 传入 | **不写死在 kernel**：`blockDim = min(AIC_NUM, Nwork)`，换硬件自动跟随 |
| `AIV_NUM` | 56（每 AIC 配 2 AIV） | 同上 `vector_core_cnt` | 核配比变化要换 kernel 类型（当前是 `KERNEL_TYPE_MIX_AIC_1_2`） |
| `L1` | 512 KiB | 同上 `l1_size` | 编译期常量；换 SoC 必须重算 3.1.1 的地址图 |
| `UB`（每 AIV） | 248 KiB（253952 B） | 同上 `ub_size` | 同上，重算 3.1.2 |
| `L0A` / `L0B` | 64 KiB / 64 KiB | 同上 | 同上，重算 3.1.3；`M_c` 与 `m` 各占满 L0A/L0B 是**本规格下**的结论 |
| `L0C` | 256 KiB | 同上 | 同上 |
| 设计预留 | L1 按 448 KiB、UB 按 224 KiB 计 | **本设计自定**（512−64 / 248−24），留给对齐空隙与组件临时区 | ⚠️ **两个名词分开**：**硬上限**取平台 ini 的实测值（L1 512 / UB 248 KiB，§1.1）；**设计预算**（448 / 224）是本设计拍的余量、没有硬件依据。**实际占用一律按目标芯片 + 目标 CANN 版本的组件实际用量核对**（04 落地时用真机实测重算 3.1.1/3.1.2 的地址图），不能按比例缩放或跨芯片套用 |
| 对齐 | 512 B | 搬运 API 与 L1 分型要求 | 按目标 CANN 版本核对 |
| NZ 分型 | BF16 16 列 / FP32 16×8 | `NpuArch=3510` 的 L0 分型 | 与 `CATLASS_ARCH` 绑定；换架构要重核 S1 的手工 NZ 构造（2.3） |
| CrossCore flag | 8 个 | 逻辑边数量（2.7.1） | 需对照目标版本的 flag 上限，换硬件先看上限 |
| 片上事件 | 9 个 | 2.7.2 的表 | 同上 |
| `ArchTag` | `Arch::Ascend950`（`CATLASS_ARCH=3510`） | §1.1 平台参数 | 换 SoC 换模板参数 |
| 性能用参数 | `cube_freq=1650 MHz`、BF16 约 378 TFLOPS、HBM 约 1.5 TB/s、H20 脊点对比值 | 同上 | **只用于估算**（4.2 的 281 us 下界、3.2 的 138 us），不参与 kernel 逻辑 |

不在这张表里的数字都不是硬件绑定：`K = V = 128`、`BT = 64`、`layout = BNSD` 是算子接口规格；
`B`、`HV`、`HK:HV`、`Nseq`、`NT`、`M(c)`、`Nwork` 是 shape 派生量；
`PIPE_SERIAL` / `SYNC_L1_HANDOFF` 是工作流术语。

复算脚本 `scripts/capacity_check.py` 顶部集中了上表的容量与预留，改平台时只改那一处。

---

## 4. 精度、性能和测试计划

### 4.1 精度观察点与策略

| 观察点 | 位置 | dtype 合同 | 有效区 |
| --- | --- | --- | --- |
| `P` | S0 输出 | BF16 输入、FP32 累加 | `[M,V]` |
| `v_new` | S1 输出 | FP32 计算、BF16 量化后交给 S2 | `[M,V]` |
| `dH` / `Kw` | S2 输出 | BF16 输入、FP32 累加 | `[K,V]` / `[K,K]` |
| `h_c` | S3 内 | 全程 FP32，chunk 间不降精度 | `[K,V]` |
| `M_c` | S3 输出 | FP32 | `[K,K]` |
| `m_c` | S4 内 | FP32xFP32、FP32 累加，chunk 间不降精度 | `[K,K]` |
| `hm` | 最终输出 | FP32，不再舍入 | `[K,V+K]` |

统一策略：`atol=1.5e-2 / rtol=2e-3 / max_abs_limit=0.05`（开发期自测口径；
库上看护走 ATK 双标杆，见 `tests/atk/pre_process_fwd_kernel_merged/`）。
量级依据：`m` 半边在 `atol=1e-6` 下与 CPU 标杆全过（强检查）；
`h` 半边存在 9.5e-3 的内在散布（`bf16(h)` 量化不连续在反馈环里放大，弱检查）。
分区至少覆盖 `ALL`、`h` 半边、`m` 半边、`head0`、`head_last`。
结构性错误、数值误差、padding/无效区与非确定性问题分别定位（分类见 `tests/atk/` 的用例分组）。

**验收顺序：先精度、后性能。** 04 阶段按 `S0 -> S1 -> S2 -> S3 -> S4` 逐 Stage 打通并与 CPU
标杆比对，整算子精度全部通过后才进入 05 的性能验收；性能不达标时先按 4.2.3 的瓶颈判定顺序
定位，只有确认是设计问题才回到 03。

**每条用例都要双过。** 4.3 的用例表既是精度用例集也是性能用例集——**39 条每一条都必须
同时满足精度阈值与 1.0x 性能目标**，不做"只保主 case 性能、边界 case 放行"的妥协。
因此 4.2 的目标按用例逐条建档，而不是只给一条 model case 的基线。

### 4.2 性能目标与统计口径

#### 4.2.1 目标框架

| 项目 | 取值 |
| --- | --- |
| 对标对象 | H20 上的上游 `pre_process_fwd_kernel_merged`，**逐用例同 shape、同 dtype、同调用粒度** |
| 目标倍率 | `T_npu <= 1.0 x T_h20`，**对 4.3 的 39 条正向用例逐条成立** |
| 判定指标 | `msprof` `op_summary` 的 `Task Duration(us)` 中位数与分位数 |
| 采集口径 | 预热/采样次数、统计量、是否含上下游 kernel、`T_win` 与 CP `world_size` 的对应关系 —— **待用户提供**（5.2 第 1~4 项） |
| 无 GPU 说明 | 本机无 NVIDIA 设备，H20 基线只能采用用户提供的实测数据 |

#### 4.2.2 逐用例建档表（04/05 填写，字段固定便于回归）

| 字段 | 说明 |
| --- | --- |
| 用例ID / 算法路径 / shape | 与 4.3 的用例表一致 |
| H20 基线 | 同一 shape、同一 precision 档位的实测 `Task Duration` |
| 昇腾实测 | 本算子同一 case 的 `Task Duration`，注明预热/采样次数 |
| 倍率 | `T_npu / T_h20`，判据 `<= 1.0` |
| 结论 | 达标 / 不达标（不达标须写瓶颈定位：Cube / 同步等待 / 搬运） |

#### 4.2.3 已有参考值与设计下界

| 项目 | 取值 |
| --- | --- |
| 已有参考值 | H20 `T=11264, HK=HV=32, K=V=128, BT=64`：default 1620.7 us / tf32x3 3404.7 us / ieee 18321.6 us |
| 昇腾设计下界 | 约 281 us（Cube 吞吐）；本设计 GM 读约 138 us，均低于 H20 default 基线 |
| 预期瓶颈判定顺序 | Cube（`M_c@m` 的 FP32 路径）-> 同步等待（5 段串行链）-> 搬运 |

除上述一条外，其余 35 条 case 目前**没有 H20 基线**，需要按 4.2.2 逐条采集（5.2 第 1~4 项）。
基线补齐前可以先用首版实现做"同 case 优化前后"的相对回归，但**验收结论必须用 H20 基线**。

`M_c@m` 的精度路径**已定稿为 FP32 原生**（见 2.8）：候选顺序 FP32 原生（基线）→ 单遍 HF32 →
BF16 三分拆（含 `M_c` 的 L0A 结构改动），后两者只在 04 实测"FP32 原生不被目标版本支持或吞吐
不达预期"时启用。`AFFINE_CHAIN_PRECISION = ieee` 的语义要求是 `m` 链（约 176 步）的有效尾数
约 20 位，FP32 原生直接满足，因此不再需要在"基线取 HF32"与"排除 HF32"之间二选一。

### 4.3 测试计划

用例表是 `docs/pre_process_fwd_kernel_merged_泛化用例.xlsx`，共 41 条正向用例 + 14 条非法输入用例，
由 `scripts/gen_cases.py` 生成；片内用量按固定规格一次算完（`scripts/capacity_check.py`），
`NT`、尾块行数与 `Nwork` 在工作簿里由公式给出。
用例粒度：`k` 在 `HK` 维、其余在 `HV` 维（成倍数），`K = V = 128`、`chunk_size = 64`；
`(B, HV)` 配对取自 `GDN泛化用例表.xlsx` 的真实场景。并行度按 **`Nseq × HV`** 算
（唯一形态：varlen 打包窗口，`Nseq = len(cu_seqlens)-1`），见 2.1.4。

| 分组 | 条数 | 覆盖 |
| --- | ---: | --- |
| 精度-窗口规模 | 10 | GDN / KDA 各 5 种 `T_win`：`1`（最小，尾块 1）、`1023`（短+尾块 63）、`4096`（整块）、`8191`（长+尾块 63）、`32767`（超长，`NT=512`） |
| 精度-变长 | 8 | GDN / KDA 各 4 种**打包**：2 段等长 `[0,256,512]`、3 段等长 `[0,512,1024,1536]`、3 段不等长且含 NT=1 的短段 `[0,64,320,1024]`、单段对照 `[0,16387]` |
| 精度-GVA | 7 | `1:2 / 1:3 / 1:4 / 1:8 / 1:32`，加两条 varlen+GVA 组合；只落在 GDN（GVA 仅存在于 g-only 路径） |
| 精度-dtype | 4 | GDN / KDA × gate `FP32` / `BF16` |
| 精度-并行度 | 6 | `Nseq × HV` 从 `1×8`（`Nwork=8`，用不满核）到 `64×8`（`Nwork=512`，18.3 波）；并行度与路径无关，GDN / KDA 交替取 |
| 精度-子区间 | 2 | **竞品调用形态**（2026-09-23 定稿）：张量 T 轴比窗口长，`cu_seqlens=[bos,eos]` 的 `bos>0`（GDN `[40,512)`、KDA `[444,512)`）；覆盖 contiguous 与 zigzag back 两种末段来源 |
| 性能 | 4 | 对齐 H20 的 `model-gk` / `model-g`（`T=11264`）、CP=2 窗口（`5632`）+GVA、`T=16384` 长窗口 |
| 非法输入 | 14 | `K!=128`、`V!=128`、`chunk_size!=64`、`HK>HV`、`HV%HK!=0`、`g`/`gk` 冲突或同缺、`bg`/`v` 非空（DPLR 不支持）、`cu_seqlens` 首元素非 0 / 非严格递增 / 末元素 > `T_win` / 含零长段、`cu_seqlens` 缺失、**`B != 1`**、空窗口、gate dtype 非法 |

**路径分布**：GDN 24 条 / KDA 15 条，量级相当（GDN 多出的 9 条 = GVA 7 条 + varlen+GVA 2 条；
GVA 只出现在 g-only 路径，DPLR 不支持、不进用例）。
**变长 10 条**（GDN/KDA 各 4 条 + 2 条 varlen+GVA），含 2 段/3 段打包、不等长段与单段对照。

**这 39 条同时是性能用例**（见 4.1 的验收顺序与 4.2 的逐用例建档）。按"预期不达标风险"分级，
占前三类的需要在 04 阶段就预留优化手段，不能等到 05 才发现：

| 风险 | 用例特征 | 为什么会有风险 | 预留手段 |
| --- | --- | --- | --- |
| **高** | `T_win=1` 与 `T_win=100`（`NT=1` / `NT=2`） | 固定开销主导：kernel launch + 窗口初始化（写 `m=I` 的 64 KiB）+ 3 次 MMAD，几乎没有东西可摊薄 | 窗口初始化在 `NT=1` 时是纯开销；评估"首 chunk 直接把 `M_c` 写进 `m` 区、跳过 `m = M_c @ I`" |
| **高** | `Nseq=1` 且 `HV=8/16`（`Nwork=8/16`） | **28 个 AIC 只用到 8/16 个**，机器利用率直接砍到 29%/57% | 段进 grid（`Nseq`，2.1.4 首选，零额外搬运）——**但只在该段的链"待用"时才算**（卡内切段全待用；跨卡窗口里的无关序列不算）；"打包 batch"本身**不增加并行度**（跨卡路径一次只算末段一条链，grid 不变）；不行再按列拆工作项 |
| 中 | 短窗口（`T_win=1023`）、`gate=FP32` | `NT=16` 仍有固定开销占比；`gk` 为 FP32 时读写字节翻倍 | 已计入 3.2 的 GM 流量；必要时评估 `gk` 的 BF16 化（需回 02 复核精度契约） |
| 低 | 长窗口（`T_win >= 4096`）、`B × HV >= 128` 的组合 | 固定开销被 `NT` 摊薄、核也铺得满，是设计下界最接近的一类 | —— |

用例表的 `layout` 列恒为 BNSD：布局已按用户要求与仓内其他 AscendC 算子统一，不设 BSND 分支
（见 5.2）。**DPLR 不纳入用例**（不支持，见本文开头的范围变更与 1.1）。

精度分区固定为 `ALL` / `h` 半边 / `m` 半边 / `head0` / `head_last`，其中 `m` 半边用更严的
`atol=1e-6` 复算；性能按 4.2 的口径采集。

---

## 5. 风险、兼容和回退方案

### 5.1 风险

1. **串行链长**：每 chunk 是 `S0 -> S1 -> S2 -> S3 -> S4` 五段串行，`NT=176` 时共 880 段。
   同一 head 的三个 Cube Stage 落在同一个 AIC 上顺序执行，依赖链无法靠换核掩盖，只能靠
`PIPE_PACK_OVERLAP`（不同工作项的 Cube 与 Vector 重叠）缓解；该候选在 04 阶段评估。
2. **AIV -> L1 手写 NZ**：UB -> L1 无随路 ND2NZ，需按分型手工搬运；若目标 CANN 版本限制该
   路径，退回 R14 的 GM 中转（每 chunk 追加 `v_new` + `L_c` + `H_c` + `M_c` 的 GM 往返）。
3. **`w` 的 A/B 双用**：依据是 NZ 分型索引对 A/B 操作数在本形状上一致（3.1 脚注），
   需在 04 用目标版本 `TileCopy` 复核；不成立时增加一份 `[BT,K]` BF16 副本（<= 32 KiB）。
4. **FP32 MMAD 的精度路径**：`M_c@m` 是全算子唯一 FP32 乘，路径选择影响性能但不影响 Stage
结构，在 04 阶段用核内循环实测决定。
5. **尾部负载不均**：`HV=32` 时 `Nwork=32` 落在 `AIC_NUM`=28 个 AIC 上，4 个工作项进入第二波；
`HV` 更大时（如 64/128）波次更多但每波更均匀。`CG` 的候选按 R20 在 04 阶段比较。
6. **短窗口的固定开销**：4.3 里有 `T_win=1`、`T_win=100` 两条（`NT=1`/`NT=2`），
   这类 case 的时间几乎全由 kernel launch + 窗口初始化 + 固定次数的 MMAD 构成。
   它们同样要满足 1.0x，因此**必须在 04 就把固定开销压到与 H20 同量级**；
   若做不到，优先砍窗口初始化（首 chunk 直写 `M_c`）而不是加 Stage。
7. **全用例性能承诺**：39 条正向用例每一条都要达标（4.1/4.2），没有"边界 case 放行"。
   这意味着尾块、varlen、GVA、小 `HV`、大 `HV` 全部在验收范围内，任何一条不达标都算未完成。
8. **`Nseq=1` + 小 `HV` 用不满核**（最可能导致整条用例不达标的一项）：并行度只有 `Nseq × HV`，
   单序列 + `HV=8` 时 28 个 AIC 里只有 8 个在跑。**阻断项，需要在 04 之前定**：优先吃满多序列
（接口本来就允许——一个打包窗口里带多段 `cu_seqlens`；零额外搬运，上游 kernel 的
   `MULTI_SEQS` 就是为这个场景准备的）；不行再评估按列拆工作项或 m 半边预算并行。详见 2.1.4。

### 5.2 未决问题

| # | 问题 | 阻塞对象 |
| --- | --- | --- |
| 0 | **是否吃满多序列**（一个打包窗口里的多段 `cu_seqlens`，`Nwork = Nseq × HV`）：并行度只有 `Nseq × HV`，`Nseq=1, HV=8` 时 28 个 AIC 只用 8 个。接口本来就允许多段（`api.md` 第 3 节），要定的是**验收范围**（多段是否进本轮）与 kernel 侧"段号进 grid 第三维"的实现。竞品的跨卡 `pre_process` 这一层是二维 grid（一次只一段），但它的**卡内**路径（`intracard_pre_scan`）就是把段号放进 grid 第三维的，所以这不是超出对标的扩展，而是对标竞品的另一条既有形态 | **04 之前**（2.1.4、5.1 第 8 条）；与第 3 项的口径联动 |
| 1 | 模型 case 的完整 shape/dtype 与 CP `world_size -> T_win`，以及每个模型的 `B` 与 `HV` | 第 4 章性能目标 |
| 2 | 模型实际 `precision` 档位（default 还是 tf32x3，差 2.1x） | 第 4 章目标值 |
| 3 | 1.0x 口径：单次 kernel 调用 vs 一个 rank 的整个 pre_process。**若采纳第 0 项吃多段**，这条从"可选"变成"必答"：竞品要按段多次调用、我们 1 次，只有按"一个 rank 的整个 pre_process 总时长"比才公平 | 第 4 章目标值；与第 0 项联动 |
| 4 | 性能统计口径（预热/采样次数、统计量） | 第 4 章、05 验收 |
| 5 | 测试落点（流程自带 `test/` vs `tests/atk/`） | 04/05 |
| 6 | R14 兜底（GM 中转）是否会被 AIV -> L1 限制触发 | 04 |
| 7 | ~~`hm` 的返回方式~~ **已定稿（2026-09-28）**：接入走 ctypes 后，`hm` 由 Python 侧 `_zeros` 预分配、作为输出张量传给 aclnn 再返回——与竞品"调用方预分配 `hm` buffer"一致，也与仓内 `npu_chunk_gated_delta_rule_fwd_h` 的 `h_out/v_new_out` 一致；**不新增 `hm_out` 参数** | 关闭 |
| 8 | ~~`M_c @ m` 的精度路径~~ **已定稿为 FP32 原生**（2026-09-22，见下方"已定稿"） | 关闭 |
| 9 | ~~CP layout / zigzag 是否一次吃两个 part~~ **已定稿（2026-09-23）**：支持子区间窗口后，zigzag 与竞品一样**每个 part 调一次**（各传该 part 末段的 `[bos,eos]`），不需要 `part_offsets` 或二维 `cu_seqlens`；整窗一次算（方案 A，`Nseq>1`）保留为可选扩展 | 关闭 |
| 10 | ~~变长窗口内含多段时 `hm` 的前导维~~ **已定稿为"链条数 = `Nseq`"**（2026-09-22，见下方"已定稿"） | 关闭 |
| 11 | **目标场景的"H 分布"与"每 part 待用链数"**：CP 下竞品跨卡路径的并行度 = 列块(≈4) × `HV`，它自己也不切 token 轴；打包 batch **不增加并行度**（`W \| B` 且等长时 pre_process 直接全空转，实测 `--preset aligned` 全 0）。所以 `HV=8/16` 的小头场景，唯一能提并行度的手段是**段进 grid**，而它只在"该段的链待用"时才不白算（卡内切段：全待用且关键路径 ÷S；跨卡窗口里的无关序列：不算）。需要框架侧给出：目标 case 的 `HV` 分布、每 part 待用链数、以及是否会走"卡内切段"（推理 prefill）。**若确实存在"小 H + 长序列 + 只走跨卡"的组合，则 04 必须把"按列拆工作项"作为备选实现出来（重复读 k/w，见 2.1.4 方案二）** | 04 之前；影响 4.2 目标可达性与 2.1.4 方案排序 |
| 12 | ~~窗口是否允许是张量 T 轴的"子区间"~~ **已定稿（2026-09-23）：支持**（`0 ≤ cu[0] < cu[-1] ≤ T`），与竞品调用 1:1 对齐、零拷贝零浪费；实现方式见 1.4.2（host 段基址按 `bos` + tiling 带 `T_full`，内核地址表达式不变形） | 关闭 |

> **第 0 项与第 10 项现在有实测支撑（2026-09-22，H20）**：竞品 kernel 的 `MULTI_SEQS`
> （段号进 grid dim2）与"每段单独调用"**逐位相等**（`max_abs=0.000e+00`），所以"一次调用吃
> 多段、每段一条链"不是新语义；竞品 CP 层"每 part 只喂末段"在多段 part 下也验证正确
> （唯一非零项被独立探针定量复现为分块口径带来的 bf16 量化差异）。

**已定稿（用户确认，2026-09-22）**

| 项 | 结论 | 依据/影响 |
| --- | --- | --- |
| `M_c @ m` 精度路径（原第 8 项） | **FP32 原生**（两侧 FP32、累加 FP32），不做 BF16 三分拆；回退候选顺序 单遍 HF32 → BF16 三分拆 | 精度按 CPU 契约标杆验收（FP32 原生比 H20 `default` 更接近真值）；H20 `default` 仅作性能对标口径（差 11×）。见 2.8、4.2.3 |
| DPLR 范围（原 A4，**2026-09-30 修订**） | **不支持**：`bg` / `v` 必须为空，非空在四个入口被拒；TilingKey 可达 4 个 | 见本文开头的范围变更、`api.md` §0/§2/§6；`USE_BG` 槽位与历史设计记录保留但 host 永不产生（1.1、3.2.1、4.3） |
| `hm` 前导维（原第 10 项） | **链条数 `Nseq = len(cu_seqlens)-1`**；等价说法"我们第 i 份 == 竞品对第 i 段单独调用" | 竞品"跨卡每一次调用"是 `[HV,K,V+K]`（无前导维，`MULTI_SEQS=False`）、"卡内切段"是 `[S_split,HV,K,V+K]`（`MULTI_SEQS=True`）；我们保留前导维 = 两者并集，`Nseq=1` 时与竞品跨卡形态逐字节相同。见 1.1、1.4、api.md 3.4 |
| 序列表达（2026-09-22 新增） | **只保留 varlen 打包窗口**：`B ≡ 1`、`cu_seqlens` 必给、`Nseq = len(cu_seqlens)-1`；等长 batch 由调用方打包成等长多段 | 与竞品 CP 契约一致（"CP expects `B == 1` for varlen"）；`B>1` 在 CP 下不可达（非 CP 下竞品 wrapper 直接 return）。见 `api.md` 3.2/6、`design.md` 1.1/1.2 |
| UB/L1 口径 | **硬上限**取平台实测值（L1 512 / UB 248 KiB）；**设计预算**（448 / 224）只是本设计预留，实际占用按**目标芯片 + 目标 CANN 版本**的组件实测重算 | 见 3.1、3.5；换芯片必须重算地址图 |
| Python 落地形态 | `from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged`（同时导出 `npu_` 前缀），实测调用示例见 `api.md` 3.5 | 与仓内 `_aclnn_ctypes.py` 的注册/调用约定一致 |

**已定稿（用户确认，2026-09-21）**：`K`、`V`、`chunk_size` 均写死为固定规格
（128 / 128 / 64），host 拦截其它值，依据见 3.4。**GVA 保留**，因此 `HK` 与 `HV`
可以不一致但必须成倍数（**不设上限**，采用 `npu_chunk_fwd_h` 的口径）；`k` 在 `HK` 维、
`w/v/u/g/gk` 在 `HV` 维，`S0`/`S2` 取 `k` 时按 `hk = hv // (HV/HK)` 复用，
输出 `hm` 与两个状态在 `HV` 维。

容量一次算完（3.1、3.4），分片余量充足，因此不再有"拆工作项 / 降列块宽 / 压行高"这类
档位取舍；`TilingKey` 只剩 6 个（3.2.1）。

**已定稿（用户确认）：输入输出布局统一为 BNSD `[B, H, T, D]`**，与仓内其他 AscendC 算子保持一致。
依据与取舍：

1. `npu_chunk_fwd_h`、`npu_chunk_gated_delta_rule_fwd_h`、`npu_chunk_gdn_bwd_intra`、
   `npu_chunk_kda_fwd_finalize` 的 canonical 布局都是 head-major `[B, HV, T, D]`，且
   `npu_chunk_fwd_h` 还显式覆盖 descriptor 为 `ACL_FORMAT_ND`；本算子接在 `chunk_fwd_h`
   之前、共用同一批输入张量，保持 BNSD 是**零转置**选择。
2. 上游竞品（以及 CPU 标杆）是 token-major `[B, T, H, D]`。这一层差异由
   调用方吸收，不进本算子契约；`docs/api.md` 第 3 节的 BNSD 约定同时是 01 阶段的已冻结接口。
3. 仓内 `npu_chunk_gated_delta_rule_bwd`、`npu_chunk_fwd_o`、`npu_chunk_kda_fwd_finalize`
   提供 `layout` / `output_layout` 参数接受 BSND，做法是**在 host 侧显式转置**，KDA bwd 还用
   `_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES` 按 token 分段来约束转置 workspace。
   本算子**不采用**这条路：它按窗口运行、窗口内 chunk 递推不可切，无法像 KDA bwd 那样分段摊薄；
   模型 case（`T=11264, HV=32, K=V=128`）的 `k/w/u/gk` 合计约 460 MB，一次
   `[B,T,H,D] <-> [B,H,T,D]` 的连续化转置按 1.5 TB/s 约需 613 us 往返，超过本算子自身
   约 281 us 的下界。若后续集成层确实需要 BSND 入口，应作为独立的适配层工作项重新评估，
   不改动 kernel 契约。

### 5.3 兼容与回退

- 公开接口、属性默认值、输出布局与 `docs/api.md` 一致；命名偏离（算子名不含 `catlass`）
  已按用户明确要求在 `docs/api.md` 第 8 节记录。
- 回退顺序：`SYNC_L1_HANDOFF -> SYNC_GM_QUEUE`（R14 兜底）；
  `PIPE_SERIAL -> PIPE_PACK_OVERLAP` 为正向候选，失败即恢复已验证的 `PIPE_SERIAL` 基线。
- 设计调整统一回写本文第 1/2/3/6 章再改代码。
- 交付物：`docs/design.md`（本文）、`docs/api.md`、
  `tests/atk/pre_process_fwd_kernel_merged/`
  （ATK 单算子验收工程，含 CPU 标杆 `scripts/pre_process_fwd_kernel_merged_cpu.py`）。

---

## 6. R01-R21 规则检查表

| 规则 | 结论 | 证据位置 |
| --- | --- | --- |
| R01 | 满足 | 2.1.2、2.1.4：先按数据依赖/计算类型/生命周期/精度观察点划 5 个 Stage，再映射 AIC（S0/S2/S4）与 AIV（S1/S3）；AIV 分工为同 head 分片并给出列段与 K 行范围 |
| R02 | 满足 | 2.1.2 前驱列、2.1.3：每条边的消费者都等待生产者；S2 与 S4 同为 Cube，但因 S4 读 S2 的 Cube 输出而保持两个 Stage |
| R03 | 满足 | 3.1、3.5：按 §1.1 平台参数的 L1 512 KiB / UB 248 KiB / L0A-B 64 KiB / L0C 256 KiB 核算，L1 与 UB 分别计预算；容量与预留的平台来源见 3.5 |
| R04 | 满足 | 2.2、3.1：`P` 由 Fixpipe 按列段写两个 AIV 的 UB，`P_ready` 在 Fixpipe 之后发布 |
| R05 | 满足 | 2.5、3.1：`h` 分片与 `dH`/`Kw` 行块驻 UB；`decay`、mask 计入 UB 峰值 |
| R06 | 不适用 | 采用 `SYNC_L1_HANDOFF`，Vector -> Cube 走 L1 直写，不经过 GM；R14 兜底路径见 5.1、5.3 |
| R07 | 满足 | 2.1.3、2.6：本设计唯一的 Cube -> Cube 跨 Stage 驻留是 `m`（S4 产出、下一 chunk 的 S4 消费，1 份放 L1）；`H_c`、`M_c` 是 Vector -> Cube 的 L1 交接，按 R06 的等价形式处理。各区的生产、消费与释放顺序见 2.1.3 |
| R08 | 满足 | 3.1.2、3.1.4：每 AIV 132 KiB，含跨 chunk 常驻的 `h`、`dH`/`Kw` 的 staging、`P` 与计算缓冲；对照设计预算 224 KiB（硬上限 248 KiB） |
| R09 | 满足 | 3.1：各 L1 区独立列出首次写入与最后消费；`m`、`H_c` 常驻，其余按 chunk 复用 |
| R10 | 满足（缓存策略待实测） | 3.2：`k`、`w` 每 chunk 只从 GM 搬一次（上游逐列块重复读，固定规格下是 4 次）；GM 流量逐项列出；L2 缓存参与方式在 04 按 API 支持与同条件性能确认 |
| R11 | 满足 | 3.1、2.7：各 UB 区标注生产、最后消费与复用事件（`MTE3_MTE2`、`MTE3_V`、`V_MTE2`），kernel 尾部排空 |
| R12 | 满足 | 2.3、2.5：每个 AIV 一次装入本分片全部数据，并用一次 VF 完成该 Stage |
| R13 | 满足 | 2.1.4、3.1：不同工作项（不同 head / 不同列段）无依赖，按可并行建模并使用互不重叠的地址 |
| R14 | 不适用（已保留兜底） | 3.2：本设计不需要 workspace；R14 作为 AIV -> L1 受限时的兜底保留，触发条件与代价见 5.1 第 2 条 |
| R15 | 满足 | 2.1.4 第 5 条：S1、S3 已分别是合并后的唯一 Vector Stage；S3 把 `h` 更新与 `M_c` 构造合并，拆开反而抬高 UB 峰值 |
| R16 | 满足 | 2.1.3：`h`、`m` 作为跨 chunk 复用结果在 UB/L1 驻留，并记录生产、消费与释放 |
| R17 | 满足（容量条件见证据） | 3.1：各区按 512 B 对齐连续排列，峰值与最大连续空闲区可计算 |
| R18 | 满足 | 2.1.4：所有 head 使用相同的列分片规则与算法路径，差异仅来自有效宽度与 tail |
| R19 | 满足 | 2.1.4、3.1.1~3.1.4：先算合并方案的同时存活量（L1 256 / UB 132 / L0C 128 KiB），容量足够后取依赖要求的最少 Stage；本设计未触发列段拆分，R14 兜底保留 |
| R20 | 满足（候选待实测比较） | 2.1.4、3.1、3.5：`Nwork = Nseq × HV`（`Nseq = len(cu_seqlens)-1`）、`blockDim=min(AIC_NUM,Nwork)`（`AIC_NUM` 由 host 读，不写死）、grid-stride；不拆列段（容量有 192 KiB 余量），`CG=1/2/4` 作为候选在 04 阶段比较波次、尾部负载与搬运 |
| R21 | 满足（待值域补充） | 1.3、4.1：`E(dg)` 处先判 mask 再求 `exp2`；`h`、`m` 全程 FP32 无跨 chunk 降精度；BF16 量化点固定在 S0 右操作数与 S1 输出；`decay` 对无效行取 1。输入值域与溢出界在 04 阶段按模型实测范围补写 |
