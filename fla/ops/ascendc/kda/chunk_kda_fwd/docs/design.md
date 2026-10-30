# ChunkKdaFwd 设计

## 目标

1. 顶层接口对齐非 CP 的 FLA `chunk_kda_fwd`。
2. 顶层 `chunk_kda_fwd` 不新增公开算子原型；chunk=64 的 A5 快路径复用既有 `ChunkKdaFwd`
   原型和外层 kernel 入口，chunk=32 也由公开 V2 入口路由到同一个 `ChunkKdaFwd` L0。
3. A2/A3/A5 使用同一数学定义；A5 保留 regbase 双发射特化。
4. 输入 layout 与输出 layout 解耦。
5. FwdH 同时服务 KDA 与 GDN，并支持可选 scalar gate、key-wise gate 和 `state_v_first`。

## L2 调度

```text
raw g -> ChunkKdaFwd[
    gate cumsum -> Prepare/Post-WU -> FwdH -> Finalize
] -> attn_out
```

`aclnnChunkKdaFwd` 做公开 layout 的连续化和必要视图转换。A5 的 BF16、chunk=64、K=V=128
dense 对齐快路径保持单次物理 `ChunkKdaFwd` L0；A5 其他多 chunk 场景将同一个私有 L0 按
Gate/Prepare、Post-WU、FwdH、Finalize 四个阶段依次提交，使阶段间通过物理 launch 边界重置事件状态。
A2/A3 和单 chunk 场景仍使用单次物理 L0。阶段选择仅使用私有 `stage` 属性，不增加公开属性、
接口字段或独立算子原型。

## 阶段职责

### KdaGateCumsum

将 raw/已激活 gate 转为 FP32 chunk-local log2 累计值：

```text
gk = cumsum(gate) / ln(2)
```

该算子同时保留独立 L2 接口供 GDN2 调用，输入和输出固定为 BNSD/NTD。

### Prepare

只读取 `q/k/v/gk/beta` 及变长元数据，产生：

```text
Aqk, Akk, qg, qg_scaled, w_seed, u_seed
```

矩阵计算和三角求逆使用 FP32 累积；公开中间量在写回时转为 q dtype。

### Post-WU

只读取 `k/gk/w_seed/Akk/u_seed`，产生：

```text
w, u, kg, v_new_seed
```

`Akk` 的 head 循环按 `H_v` 执行，GQA 映射只在读取 q/k head 时换算，避免按 `H_k` 重复或漏算。

变长尾块中 `w_seed` 与 `w` 可复用同一段 GM。两个 AIV 按 K 维列区间独占写回，并从高行向低行
处理，保证下三角计算依赖的源行在最后一次读取前不会被原位覆盖。`u_seed` 与 `u` 使用独立存储，
仍按行区间并行。

### FwdH state propagation

读取 `kg/w/u/gk` 和可选 `initial_state`，计算 chunk 间递推：

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
```

arch35 路径复用与 `ChunkGatedDeltaRuleFwdH` 相同的数学实现；其他场景在 `ChunkKdaFwd` 内嵌
共享 FwdH 实现。独立 GDN L0 原型继续保留给其他调用方，key-wise `gk` 固定使用 `exp2`。

### Finalize

只读取 `qg_scaled/Aqk/v_new/h`，计算：

```text
attn_out = qg_scaled @ h + Aqk @ v_new
```

kernel 内直接按 BSND/TND 写出 `attn_out`。供反向使用的中间量保持 BNSD/NTD。

## 状态布局

内部递推统一使用 `[...,K,V]`。`state_v_first=true` 时，L2 在进入 FwdH 前转置 initial state。
V2 组合的内部 `hCompute` 由共享 FwdH 原生写为 NT-first，Finalize 直接消费；
公开 `hOut` 同样为 NT-first，无需 chunk/head 转置。旧融合路径的 prepare/post_wu/FwdH/Finalize
也原生使用 NT-first，导出仅在 `state_v_first` 时转换末两维。`final_state` 按序列排列，与 FLA 顶层
输出一致。

## 重计算策略

L2 不理解 autograd 重计算策略。`final_state/gk/w/u/qg/kg/v_new/h` 是相互独立的
`OPTIONAL_OUTPUT`；非空指针表示导出，空指针表示不公开该结果。单 launch 路径为隐藏输出传递
固定 ABI 占位，并由 tiling 在 kernel workspace 中承接实际中间结果。A5 四段 launch 路径将
阶段间依赖的 `gk/w/u/qg/kg/v_new/h/final_state` 和私有 `qg_scaled/u_seed` 物化为 executor 内部张量，
使后续 launch 不依赖前一 launch 的 kernel workspace。公开输出存在时直接作为内部目标使用。

Python/legacy 包装层对齐 fla-org `chunk_kda_fwd` 提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

- `Aqk/Akk` 始终返回。
- `disable_recompute=false` 时不保留 `w/u/qg/kg/v_new`。
- `disable_recompute=true` 或 `return_intermediate_states=true` 时保留公开 `hOut`。
- `use_gate_in_kernel=false` 或 `disable_recompute=true` 时保留 `gk`。
- `final_state` 只在 `output_final_state=true` 时创建公开输出。

内部 `hCompute` 与公开 `hOut` 是两个生命周期：`hCompute` 是 FwdH 到 Finalize 的必需
NT-first 阶段结果；`hOut` 为空时，单 launch 路径由 kernel workspace 承接，四段路径由 executor
内部张量承接。`hOut` 非空时，L2 直接导出 NT-first 张量，仅在 V-first 时交换末两维。
该规则对齐非 CP 的低层 12 返回值接口；
第 12 项 `initial_state` 由 Python 层原对象透传。

反向 L2 norm 保存值（`q_hat/k_hat/q_rstd/k_rstd/beta_eff`）不进入上述 12 项返回值，而是
组合入口尾部的 5 个独立可选输出：调用方传入张量即导出、传空即不产出，同一次调用中两种
情形的计算结果逐位一致，只有是否落公开 GM 的区别。组合路径用 `ReuseOrAlloc` 直接复用
调用方张量承载 Prepare 的对应输出，不再额外分配；未传时仍按 Prepare 档位要求分配内部
张量，保证前向链（FwdH/Finalize）可用。对应的档位约束（`none/forward/recompute/save`）
与 `ChunkKdaFwdPrepare` 的 outputMask 一致，详见 `chunk_kda_fwd_prepare/docs/design.md`。

## 模板化方案与 tiling key

`ChunkKdaFwd` 只有一个外层 `op_kernel/chunk_kda_fwd.cpp` 入口和一个私有 L0 类型。A5 实现位于
`op_kernel/arch35/*.h`，host 侧 A5 模板选择位于 `op_host/arch35/*.h`。Prepare、Post-WU、
Finalize 的内部实现头与统一 kernel 入口同属 `chunk_kda_fwd/op_kernel/`，不存在对应的独立 L0
原型或 `.cpp` 入口。A5 四段路径只是用不同私有 `stage` 属性连续调用该入口。

- `tiling key=1`：非 chunk=64、K=V=128 场景的通用模板族。
- `tiling key=2`：chunk=64、K=V=128 模板族，包括 dense、tail 和 varlen。

两个 key 是同一 L0 的编译期场景变体，不是平台编号、独立算子或独立接口。A2/A3/A5
均生成两个 key；同一个 key 内再由编译架构选择根目录通用实现或 `arch35/` 实现。host 的
`SetTilingKey` 只检查 chunk、K、V，不检查 SoC。

在 arch35 上，key2 的 dense 对齐场景使用单 launch 融合流水和 arch35 FwdH。A5 多 chunk 的
tail/varlen 以及 key1 泛化场景使用四段 launch，并在 FwdH 阶段复用共享实现。其他架构在同一
key2 下使用其对应单 launch 实现。tiling key 和私有 `stage` 均不改变公开算子原型、输出契约
或数学定义。

## chunk32：Ascend950 单 kernel 大融合

### L2 路由与 L0 边界

`chunk_size=32` 不复用上述 `ChunkKdaFwd` 外层 L0，也不经过
`ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize` 三算子链。公开 Python 入口选择
`aclnnChunkKdaFwdV2`，其 chunk32 分支与 V1 的 chunk32 分支一样进入
`RunChunkKdaFwd32Fused`，再由 `l0op::KdaChunkForward32` 单次启动同一个
`ChunkKdaFwd` 的 key 32：

```text
fla_npu.ops.ascendc.chunk_kda_fwd(chunk_size=32)
  -> aclnnChunkKdaFwdV2
  -> RunChunkKdaFwd32Fused[Dense|Varlen]
  -> l0op::Exp(A_log) + l0op::KdaChunkForward32
  -> ChunkKdaFwd（key 32，一次 mix kernel launch）
```

host 负责连续化、公开 layout 转换、g/beta dtype 统一、state 形状适配和可选输出占位。
dense BSND 将等长 batch 合成整齐 `cu_seqlens` 后走 packed varlen 直读；dense BNSD 拍平
batch/head 后保持头连续。varlen 支持 BSND（要求 `B=1`）、TND 和 NTD，拒绝 BNSD。
`A_log` 的指数由 L0 host 的 `l0op::Exp` 完成，不是 fused kernel 内的 GM 预处理；
`KDA_FD32_SKIP_EXP=1` 是跳过该步骤、直接传已激活输入的诊断开关。

TilingKey 必须按实际 L0 区分：

| L0 / 路径 | TilingKey | 源码依据 |
| --- | --- | --- |
| 外层 `ChunkKdaFwd` 通用模板 | `1` | `op_host/chunk_kda_fwd_tiling.cpp` 的 `SetTilingKey(... ? 2 : 1)` |
| 外层 `ChunkKdaFwd` chunk=64、K=V=128 模板 | `2` | 同上 |
| 同一 `ChunkKdaFwd` 的 chunk32 分支 | `32` | `op_host/chunk_kda_fwd_tiling.cpp` 的 `SetTilingKey(32)` |

chunk32 使用独立 TilingData，但不引入第二个 L0/def；key 32 与 key 1/2 共用同一个
`chunk_kda_fwd` kernel 入口，先按 key 分发再解析各自 TilingData。fused 分支使用
`KERNEL_TASK_TYPE(32, KERNEL_TYPE_MIX_AIC_1_2)`。

### 128-token group 与任务划分

kernel 常量 `chunk_size=32`、`tokens_per_group=4*chunk_size=128`：一个流水 group 包含
4 个 32-token chunk，dense group 数为 `ceil(T/128)`，varlen 则逐序列计算。AIC:AIV 为
1:2，两个 AIV subcore 按以下方式分担一个 group：

- q/k/g/beta prepare 每次由一个 subcore 处理一个完整 32-token chunk；一个 group 内每个
  subcore 处理两个 chunk，subcore 0/1 分别对应行区间 0–31、64–95 与 32–63、96–127。
- v 修正和 o 写回按每个 chunk 的 16 行半块分给两个 subcore。
- state 按 K 维分成两个 64 行半幅，每个 subcore 独立持有和更新 `64x128`。

host 以“一个 batch/sequence 的一个 value head”为一个任务：dense 为 `B*HV` 个，varlen 为
`nSeq*HV` 个。`blockDim=min(taskCount, aicCoreNum)`，AIC 与 AIV 都以
`task=get_block_idx(); task+=blockNum` 静态步进领取任务。GVA 中 q/k head 由 value-head 任务号
除以 `qkGroup=HV/H` 映射；`A_log/dt_bias` 始终按真实 `HV` 寻址，batch 间共享参数。

dense、TND varlen、NTD varlen 分别使用 tiling `varlenMode=0/1/2`。varlen 的
`cu_seqlens` 由 host 校验并非递减、首项为 0、末项为 `T_total`，最多 1024 个序列，打包到
tiling `cuData`；kernel 逐任务解析 `[tok0,tok1)`，所以 varlen 的 group 数不由 host 的
`nChunks` 指定。空序列不进计算流水；请求 final state 时，有 initial state 的由 host 预填
`final=initial`，无 initial state 的由 kernel 写零。

NTD varlen 不做整调用 TND 物化：q/k/v/g 以
`(head*T_total+tok0)*dim` 定位，TND 则以 `(tok0*heads+head)*dim` 定位；两者 beta 在 kernel
中都按 `[HV,T_total]` 头连续读取，TND 的 beta 由 host 先转置。公开 `attnOut` 即使输入为
NTD 也固定为 sequence-major TND `[T_total,HV,V]`。

### AIC/AIV Stage 职责

`op_kernel/chunk_kda_fwd.cpp` 在 key 32 分支调用
`op_kernel/arch35/chunk_kda_fwd_chunk32/`，分别按 CUBE/VEC pass 编译实现。
AIC 使用裸 `mmad/fixpipe`，AIV 使用 MicroAPI；AIC 不读写 q/k/v/g/beta/state/o 的业务 GM，
这些 GM I/O 和 state 常驻均由 AIV 负责。两侧的 Stage 名称直接对应软件流水函数，而不是
多次 L0 launch。

| Stage 族 | 执行侧 | 职责与数据边界 |
| --- | --- | --- |
| `stage_prepare_g*` | AIV | 装载 raw g/q/k/beta，处理尾行；完成 safe gate、chunk-local gate 累计、beta sigmoid 和 q/k L2 normalization。归一化 q/k 写回 BF16 UB。 |
| `stage_prepare_qplus_kplus_kminus_kres*`、`stage_qplus_kplus*`、`stage_kminus_kres*` | AIV | 生成 qPlus/qPlusB、kPlus/kPlusB、kMinus、kRes，并以 BF16 NZ tile 经 UB→L1 交给 AIC。 |
| `stage_l0ab_kplus_kminus*`、`stage_kplus_kminus*`、`stage_ub_kplus_kminus*` | AIC | L1→L0A/L0B，计算 kd/ki 与 qd/ki 矩阵乘；L0C 以 FP32 累积，乘积以 FP32 fixpipe 到 UB。 |
| `stage_tril*`、`stage_tril_inv*`、`stage_masked_qk*` | AIV | 消费 FP32 乘积，完成 beta 缩放、严格下三角/单位阵构造、FP32 分块求逆及 masked qk 准备；供后续 mmad 的操作数转为 BF16。 |
| `stage_kplus_qplus_score*`、`stage_ub_kds*` | AIC | 以 L1 中的 state 计算 k/q 与 state 的 score；k@state score 以 BF16 交给 AIV 做 value 修正。 |
| `stage_v_corr*` | AIV | 计算修正后的 v，处理尾行，并将结果送到 L1。 |
| `stage_state_acc*`、`stage_l1_y*` | AIC | 计算下三角矩阵与修正 v 的乘积 y/stateAcc，结果转 BF16 L1 供 output/state 更新复用。 |
| `stage_iter_out*`、`stage_ub_1_iter*` | AIC | 累加 masked qk@y 到 attention out，并计算 kRes@y 的 state 增量；out 转 BF16 UB，state 增量保持 FP32 UB。 |
| `stage_iter_partial*`、`stage_iter_final*`、`stage_s1*` | AIV | 以 FP32 完成 state 衰减与增量加法，再写回 BF16 UB state；将本 group state 送到 L1 供后续 score 使用。 |
| `stage_out1*` | AIV | 等待 AIC out UB ready，按有效行钳制并写出 BF16 o；varlen 按 TND 公开布局寻址。 |
| `stage_ub_gamma*` | AIC | 将 AIV 送到 L1 的 FP32 gamma/beta 分发到两个 AIV 的 UB。 |

AIC 和 AIV 都按 prologue、steady、drain 组织跨 group 流水；Stage 的 `iter` 条件表达相邻
group 的生产/消费重叠，而不是数学上的额外递推。

### FP32 中间量与 BF16 cast 边界

`vec_prepare_qplus_kplus_kminus_kres` 的输入是已归一化并存为 BF16 的 q/k，以及 FP32 的
gate/cumsum/gamma。函数先显式将 q/k 从 BF16 cast 到 FP32；qPlus（含 scale）、kPlus、
kMinus、kRes 及其后半 chunk 的 anchor 运算全部在 FP32 寄存器中完成。只有最终写入 UB
时才 cast 回 BF16 并组包，随后以 16-bit NZ tile 搬到 L1。换言之，这些名字表示的是
“FP32 计算、BF16 输入/输出边界”的中间量，不是全程 BF16 算术。

AIC 接收 BF16 L0A/L0B 操作数，mmad 在 FP32 L0C 累积。跨回 AIV 时，kd/ki、qd/ki 乘积和
state `score_delta` 采用 FP32 NoQuant fixpipe；attention out 与 k@state score 则在 fixpipe
边界转为 BF16。state 常驻 UB 为 BF16 NZ，但每轮更新先转 FP32，完成 gamma 乘法与 FP32
增量加法后再转回 BF16。

### state、workspace 与跨核 flag 协议

每个任务开始，AIV 将公开的 FP32 ND initial state 转为 BF16 NZ，或在未提供时清零；每个
AIV subcore 的 `64x128` 半幅常驻 UB slot 23。state 在 group 间复用，并通过 UB→L1 提供给
AIC；末组完成后才聚集、转回 FP32 ND 并写 final state。final state 的暂存借用输入 UB 区，
写回后以 MTE3→V 自排干，避免下一任务装载覆盖暂存区。公开 state 仍是
`[N,HV,K,V]` FP32；Python 的 `state_v_first=true` 在进入 L2 前后转置，fused L2/L0 只接收
K-first。

生产路径不需要 GM workspace：`TilingForChunkKdaFwd32` 将 workspace size 置 0。kernel ABI
中的 `workspace` 仅接到 `tmp_gm` 供已注释的 debug dump 使用，正常 Stage 不通过它交换
中间量；中间数据驻留 UB/L1/L0，跨核生命周期由 flag 保证。

两侧 `SyncCV` 使用相同的 flag type。生产侧直接 `CrossCoreSetFlag`，消费侧的“伪 Set”先
按 buffer id 记录到软件 FIFO，真正 `WaitIntraBlock` 时排干 FIFO 并执行
`CrossCoreWaitFlag`。主要协议如下：

| flag type | 方向 | 保护的资源/生命周期 |
| --- | --- | --- |
| `FIXtoMTE3_UBReady` (2) | AIC FIX → AIV MTE3 | out UB 已生成，可写 GM。 |
| `FIXtoV_UBReady` (3) | AIC FIX → AIV V | FP32 乘积/score_delta 或 BF16 score UB 已可读。 |
| `MTE1toV_UBReady` (5) | AIC MTE1 → AIV V | gamma/beta 已由 L1 送到 UB。 |
| `VtoFIX_UBRelease` (6) | AIV V → AIC FIX | AIV 已消费 fixpipe UB，可复用。 |
| `VtoMTE1_UBRelease` (8) | AIV V → AIC MTE1 | gamma/beta UB 已消费。 |
| `MTE3toMTE1_L1Ready` (9) | AIV MTE3 → AIC MTE1 | q/k 派生量、state、v_corr 等 UB→L1 已完成。 |
| `MTE1toMTE3_L1Release` (12) | AIC MTE1 → AIV MTE3 | L0 装载已消费 L1 tile，AIV 可覆盖。 |
| `MTE3toFIX_UBRelease` (14) | AIV MTE3 → AIC FIX | out UB 已写 GM，可复用。 |

AIC 向两个 AIV 分别 set 基础 id 与 id+16，并在反向等待两个 id；每个 AIV subcore 各 set
一次、各 wait 自己一侧。部分 type 以计数奇偶切换相邻 flag id，配合 FIFO 防止流水覆盖。
`FIXtoMTE3_L1Release` (0) 与 `MTE3toMTE1_UBRelease` (11) 虽有定义，但当前 fused kernel
未使用，不属于实际 Stage 协议。

### chunk32 验证矩阵与验收记录

| 维度 | 覆盖/约束 |
| --- | --- |
| 平台与 L0 | Ascend950；`ChunkKdaFwd` key 32，一次 mix launch。 |
| dtype | q/k/v/o BF16；公开 V2 的 g/beta 各自可为 BF16/FP32，任一 FP32 时 host 将两者统一上转 FP32；A_log/dt_bias/state FP32。 |
| layout | dense BSND/BNSD；varlen BSND(B=1)/TND/NTD；NTD 原生 head-major 寻址，输出仍 TND。 |
| shape | K=V=128，T>0，128-token group 尾块，GVA `HV%H==0`；varlen 最多 1024 序列并覆盖空序列。 |
| state | 无 state、initial/final state、dense/varlen state；Python 覆盖 K-first/V-first 适配。 |
| gate/norm | safe gate、核内 gate/L2 norm/beta sigmoid、BF16 与 FP32 g/beta；chunk32 epsilon 固定为 `1e-6`。 |
| ATK 用例 | accuracy 40（id 0–33 layout/长序列/尾块/state，34–35 FP32 g/beta+GVA，36–39 TND/NTD varlen/空序列）；performance 2；mss 5。 |

已有验收记录为：全 FP32 q/k 准备中间计算版本在原始同精度 benchmark 下 40/40；正常数据
T=128/1024 额外回归 4/4。T=1 不记为“全部通过”。Stable/ctypes ABI 记录为 22 passed、
122 subtests passed、1 skipped。性能 case 数和通过记录不混作 accuracy 通过数。

## 性能设计

- Prepare 的右矩阵在 L1 驻留，避免 K/K^T 重复搬运和重复转置。
- AIC 使用 L1/L0 双缓冲组织 MTE2、MTE1、Cube、Fixpipe。
- AIV 使用输入 staging ping-pong，使下一 tile MTE2 与当前 tile VEC 重叠。
- A5 VEC 路径使用 regbase 双发射；数值主计算仍保持 FP32。
- inter-sub-chunk 合并使用独立 workspace 区域，避免阻塞主 tile 流水。

性能结论只使用 `msopprof`。目标回归 case 定义在 `tests/op_cases/chunk_kda_fwd.json`。

## 验证矩阵

- 平台：A2/A3/A5。
- dtype：FP16/BF16。
- layout：BSND/BNSD/TND/NTD。
- gate：raw/已激活、safe true/false。
- Shape：`K=V=64` 或 `K=V=128`（两档且必须同档，混合档与其它取值由参数校验拒绝），
  chunk=64（通用/三算子路径）或 chunk=32（Ascend950 fused），dense/varlen/tail/GVA。
- 属性：final state、重计算策略、`state_v_first`。
