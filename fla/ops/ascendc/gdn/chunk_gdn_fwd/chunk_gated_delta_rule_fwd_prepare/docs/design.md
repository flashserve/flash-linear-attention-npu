# ChunkGatedDeltaRuleFwdPrepare 设计

方案设计规则版本：V2

## 1. 目标

Prepare 将每个 chunk 的 Q/K 归一化、门控前缀和、下三角求逆和 W/U 计算合并到一次 kernel 启动。输出供后续状态更新与输出阶段使用。Python 和 ACLNN 契约见 [API 文档](api.md)。

kernel 入口在 `__NPU_ARCH__ == 9201` 时选择 arch92 的 MIX 1:1。arch35 使用 MIX 1:2，按任务奇偶分给两个 AIV，通常每包四个任务，GVA 比例为 3 时每包三个。以下描述 arch92 的单 AIV 三槽实现。

## 2. Stage 0–7 完整详设

### 任务与分核

一个任务处理一个序列的一个 chunk 和一个 value head。令 `G=HV/HK`，则 `hk=hv/G`（整除）。每个 chunk 从 head 0 起按最多三个连续的 hv 分包，槽号 `t=0,1,2`。包数为 `numChunks × ceil(HV/3)`。host 按平台核数设置 blockDim，每个核从自身编号起、以 blockDim 为步长取包。尾包仅遍历有效槽。变长模式按索引表确定序列、起始 token 和有效行数 M，块内计算宽度仍为 64 行。

包内同一 hk 的首个任务称为 owner。它准备 K 并计算一次 KKT，其他共享任务使用 owner 的 UB 槽。例如 HK=1、HV=4 时，包为 [0,1,2] 和 [3]，两个包分别由 head 0 和 3 准备 K。开启归一化时，仅 `hv % G == 0` 的任务写该 hk 的 Q̂、K̂ 和 rstd，允许两包分到不同核。

### 计算与数据流

下表中 K′ 为 S1 留在 UB 的归一化 K 或原始 K。g′ 是 chunk 内的门控前缀和，E(x) 在开启 use_exp2 时为 2ˣ，否则为自然指数。β 是本次选择的有效 beta。

| 阶段 | 执行单元 | 输入、计算与产物 |
| --- | --- | --- |
| S0 | AIV | 每核初始化一次 UB 中的 VCS 单位阵和索引，清零 L1 中各槽的左右叶块，并将 64×64 FP32 单位阵 I 写入 L1。 |
| S1 | AIV | owner 从 GM 取 K，可选归一化，上传 K′ 至 L1 并保留 UB 副本。各 hv 独立处理可选 fused gate、前缀和与 beta sigmoid；g′、β 驻留状态槽。尾块先清零 padding，再装入有效行。 |
| S2 | AIC | 等 owner 的 K′ 就绪，计算 KKT = K′ K′ᵀ（BF16 输入、FP32 累加），Fixpipe 写 owner UB，按每个共享任务的槽号通知 S3。 |
| S3 | AIV | 对 i>j 构造 Nᵢⱼ = −βᵢ E(clip(g′ᵢ−g′ⱼ,−50,50)) KKTᵢⱼ，其余位置为零。将 N 写入 L1，对 I−N 的两个 32×32 对角块求逆得到 D0、D1，分别嵌入两个初始为零的 64×64 叶块的左上、右下角。 |
| S4 | AIC | 以 I 预填 L0C，FP32 累加 Y = I + D1 × N。按槽将 Y 由 L0C 写入 GM workspace，再由 MTE2 搬回 L1，发出该槽的 S4 释放信号。 |
| S5 | AIC | 以 FP32 计算 A = D1 + Y × D0，即 (I−N)⁻¹，转 BF16 供 S7。在 output_a 打开时将有效行写 GM；满块从 GM 回装 L1，尾块或关闭输出时由 L0C 直接写完整 L1 tile。 |
| S6 | AIV | 从 GM 取 V，从 owner UB 取 K′，按行缩放后将 Kbg = K′ × β × E(g′)、Vb = V × β 写入各任务的 L1 槽，通知 S7。 |
| S7 | AIC | 用 L1 中的 A 分别计算 W = A × Kbg、U = A × Vb，FP32 累加后转 BF16。W 和 128 列的 U 留在各槽 UB，256 列的 U 拆为两次 128 列直写 GM。整包完成后通知 AIV。 |

### 片上地址与 workspace

以下区间采用 KiB 和左闭右开记法。UB 总容量 376，各槽的计算基址为 C=112t，状态基址为 S=336+12t。三个计算槽和三个状态槽共占 372 KiB，末尾 [372,376) 空闲。

| UB 相对区域 | 使用阶段与复用条件 |
| --- | --- |
| C+[0,16) | S0 临时零阵使用槽 0，之后 S2 在 owner 槽写 KKT，保留至所有共享任务完成 S3。 |
| C+[16,64) | S1 的 Q、K、Q̂ 各占 16 KiB。S3 从 +16、+24、+40 起依次放置 packed leaves、VCS 结果、完整 N。S6 从 +16 放 Kbg，从 +32 放 Vb（最大 32 KiB）。 |
| C+[64,80) | owner 的 K′ 从 S1 保留到本包最后一个共享任务完成 S6。 |
| C+[80,96)、C+[96,112) | 分别保留 W 和 U128，可跨越下一包 S1/S3，直到 AIV 等到 S7 信号并完成写回。 |
| S+[0,9)、S+[9,10) | 槽 0 前部存索引（+0）与 VCS 单位阵（+512 字节）。各槽从 +9 KiB 起依次放 g′、β、K rstd、Q rstd，每项为 64 个 FP32。 |

| 存储 | 区间与用途 |
| --- | --- |
| L1（512 KiB） | Y [0,48)、K′/N [64,112)、右叶 [128,176)、左叶 [192,240)，每项按槽步长 16。S2 读完 K′ 后 S3 在同址写 N。 |
| L1 驻留区 | A [256,280) 步长 8，Kbg [288,336) 步长 16，Vb [352,448) 步长 32；I 在 [480,496)。S5 的 A 与 S6 的两个右操作数各自保留到 S7。 |
| L0A、L0B（各 64 KiB） | 各有四个 16 KiB 窗口。低两槽供 S2 及 S4/S5 装载，高两槽由 S4/S5 的 FP32 乘法和随后 S7 的 BF16 乘法分时使用。 |
| L0C（256 KiB） | S2 按 owner 槽号奇偶选基址 0 或 64。S4/S5 按槽选基址 0、64、128，各容纳 16 KiB FP32 tile。S7 在基址 0、64 复用两个 32 KiB 的 64×128 FP32 结果区。 |
| GM workspace | host 分配系统 workspace 加上每核 3 × 16 KiB。用户区中字节偏移为 coreIdx × 48 KiB + t × 16 KiB，每槽保存一个 64×64 FP32 NZ 的 Y，从 S4 写出至回装 L1。arch35 每核为 4 × 16 KiB。 |

### 跨核 flag 与跨包写回

arch92 全部跨核 Set/Wait 使用 mode `0x4`，仅连接同一 block 的 AIC 与 AIV0。表中的上界按**每个 ID、每个方向**计算。相位之间的依赖将单个 4-bit 计数器的未消费数限制为 1。

| ID | Set（生产者 / pipe） | Wait（消费者 / pipe） | 配对、复用与未消费上界 |
| --- | --- | --- | --- |
| 0–2，t | S1 AIV / MTE3 | S2 AIC / MTE1 | 每包每个 owner 发送一次 K′ ready，非 owner 跳过此方向。每 ID ≤1。 |
| 0–2，t | S2 AIC / FIX | S3 AIV / V | S2 消费 owner 的 ready 后才发 KKT done。共享同一 KKT 的每个有效任务各收一次，使用自己的 ID。每 ID ≤1。 |
| 3–5，3+t | S3 AIV / MTE3 | S4 AIC / MTE1 | 每个有效槽的 N 和叶块就绪后发送，S4 累加前消费。每 ID ≤1。 |
| 6–8，6+t | 初始化或 S4 AIC / FIX | S1 或退出时 AIV / 默认 PIPE_S | AIC 为三个槽分别预发一次。每次使用该槽（含非 owner），S1 先消费，S4 dump 后归还。尾包空槽保留原信号，退出时每槽消费一次。每 ID ≤1。 |
| 9–11，9+t | S6 AIV / MTE3 | S7 AIC / MTE1 | Kbg 和 Vb 都已上传后发一次，S7 读前消费。每 ID ≤1。 |
| 12–14 | — | — | 保留，计数为 0。 |
| 15 | S7 AIC / FIX | AIV / 默认 PIPE_S | 每包发一次。下一包 S6 或本核退出前消费，发下一次 S7 必须先等下一包 S6。≤1。 |

同核处理连续两包时，AIV 可先执行新包的 S1/S3，上一包 W/U 仍驻留在独立 UB 窗口。进入新包 S6 前，AIV 等 ID 15，将上一包 UB 输出的有效行搬到 GM，再用核内 MTE3→MTE2、MTE3→V 事件约束搬运和覆盖顺序。末包单独等 ID 15 并写回。即使某个核未分到包，也需收齐初始化时发出的三个 S4 槽信号。

同步辅助函数和地址常量见 [arch92 common.h](../op_kernel/arch92/chunk_gated_delta_rule_fwd_prepare_common.h)，两侧循环与退出路径见 [arch92 kernel](../op_kernel/arch92/chunk_gated_delta_rule_fwd_prepare.h)。
