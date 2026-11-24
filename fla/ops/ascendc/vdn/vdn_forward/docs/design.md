# VDN 前向设计

方案规则版本为 [V2](../../../../../../docs/agents/03-方案设计.md)。公开入口在当前 PyTorch NPU stream 上依次提交因子生成、双向递推、衰减前缀和窗口读出。跨帧的递推构成依赖链，各帧的 Gram 和求逆可以独立批处理。F 为真实帧数，H 为本卡可见头数，S 为每帧 token 数，D=128；C 为当前块的有效帧数，W 为补齐后的工作帧数，N=WH。

## 分块与数据流

帧块上限 `frame_tile` 为正的 2 次幂。开启 `pad_tail` 时用零填充 K、V 和 β 至 W 帧。填充帧的 Gram 为零，M 为单位阵，只将前 C 帧的结果写入状态序列。关闭补齐时将余数分解为若干 2 次幂块。85 和 87 帧在显式主块 64 时均处理 64+32 个求逆槽，分别保存 64+21 和 64+23 个真实结果。核内任务以分块参数和实际核数计算，头数及帧数无需选择模型专用模板。

| 阶段 | 按执行顺序的计算 | 工作划分和输出 |
| --- | --- | --- |
| 1. 生成因子 | 对 K/V 做可选预处理，构造 `A=Kᵀdiag(β)K`、`B=Vᵀdiag(β)K`、`M=I+(A+Aᵀ)/2`，按指定求解器得到 `X=M⁻¹`，写出 `T=diag(α)X`、`U=BX`。 | 按帧块顺序提交；每块把帧和头展成 N 组 bmm。有效输出各为 [F,H,D,D] FP32。value_first 决定先提交哪一个 Gram。 |
| 2. 双向状态递推 | `P[0]=R[F]=seed`，向前 `P[t+1]=P[t]T[t]+U[t]`，向后 `R[t]=R[t+1]T[t]+U[t]`。 | VdnScan 在 AIC 上执行 2H 条链。job 的商选方向、余数选头；blockDim=min(2H,可用 AIC 数)。每条链由同一核按帧序写入一份状态库。 |
| 3. 窗口衰减前缀 | `L[0]=0`，`L[t+1]=L[t]+log(max(α[t],1e−12))`。 | PyTorch 在 GM 中按帧维做 cumsum，拼入首行零，得到 [F+1,H,D] FP32。 |
| 4. 合并与读出 | 对第 t 帧的闭区间 [lo,hi]，取 `a=exp(L[t+1]−L[lo])`、`b=exp(L[hi+1]−L[t])`。按 key 列广播，得到 `Z=P[lo]diag(a)+R[hi+1]diag(b)`；同块的 `QZᵀ` 写入最终输出。 | VdnBridge 将 C×H 对帧头分给 AIV，blockDim=min(CH,可用 AIV 数)。每对帧头独立写出 [D,D] 状态。状态先转为 Q 的 dtype，再按帧头分批读出。尾块只保存有效行。 |

短卷积按投影独立选择，空间 5×5 depthwise 后接长度 5 的中心时间卷积，每个帧块最多向两侧各取 2 帧 halo，clip 外补零。SiLU 应用于 Q/K/V，L2 归一化只应用于 Q/K。BF16 路径保留特征舍入点，包括加权 V、value Gram 和读出前的 Z。Q 的预处理推迟至阶段 4，以避免整段额外的 Q 副本。token-major 快路径将布局转换与激活/加权合并为同一 Triton kernel。[预处理实现](../../../../triton/triton_core/vdn_short_conv.py)

### 阶段 1 的两条求解路径

- **Cholesky**。每个 D×D 矩阵切为四个 32×32 主对角面板。Triton 面板按列生成 Cholesky 对角块及其逆；批量矩阵乘计算下面板并原位更新未处理的 Schur 补。已有的两个对角逆块按 `−L₂₂⁻¹L₂₁L₁₁⁻¹` 合并，宽度按 32→64→128 递增；最终 `X=L⁻ᵀL⁻¹`。当前实现每个面板直接启动前同步当前 stream，以保证和 torch_npu 队列的先后关系。[面板与合并实现](../../../../triton/triton_core/vdn_cholesky.py)
- **Newton–Schulz**。谱界 405、目标残差 1e−7，`c=2/407`。首步 `X₁=2cI−c²M` 逐元素完成，其余 11 步计算 `X←X(2I−MX)`。原生校正路径要求 FP32 矩阵乘、D=128、当前批量 N≥192，并由分块策略选择 32 MiB 校正缓冲预算内的求解器。Triton 融合对称化与首步，同时输出 −M 和 X₁；两个逆矩阵缓冲交替写回，另设等大的 FP32 校正矩阵。每一轮先用 bmm 写入 −MX，再调用 VdnAddDiagonal，最后用 bmm 写入下一个 X。[Gram 初始化](../../../../triton/triton_core/vdn_gram.py)
- **VdnAddDiagonal**。每个 D×D 校正矩阵切为两个含主对角线的 64×64 子块。AIV 的 job=2n+s，s∈{0,1}，首元素偏移为 `nD²+s×64×(D+1)`；blockDim=min(2N,可用 AIV 数)。每个 AIV 只搬入一次 2I 左上 64×64 子块，并用作两处对角子块的加数，其余两个子块保持原值。[原生校正](../../vdn_add_diagonal/op_kernel/arch22/vdn_add_diagonal_vec.h)

## GM 缓冲与搬运

| 对象 | 字节数 | 生产、消费与释放 |
| --- | --- | --- |
| T、U | 2×4FHD² | 阶段 1 逐块写入真实帧；阶段 2 预填状态库并完成递推后释放。完整块的 U 可直接作为 bmm 输出，补齐块只复制有效结果。 |
| 两份状态库 P、R | 2×4(F+1)HD² | 两端各保存一个 seed，其他行先写 U；Scan 在目标行累加乘积，最后由阶段 4 根据窗口边界读出。 |
| 原生 Newton 批量缓冲 | 每个 N×D×D FP32 缓冲为 4ND² | Gram 对称化后可释放原始 A；迭代时 −M、两个 X 和校正缓冲并存。各轮复用同一组存储，尾块寻址使用 W。 |
| 衰减与读出 | 前缀 4(F+1)HD；状态 C×H×D²×元素字节；输出 F×H×S×D×元素字节 | 每个 Bridge job 读 2 个 int64 端点、4 个 D 元 FP32 对数向量和 2 个 D² 元 FP32 状态，只写一个 D² 元状态。块级 bmm 接着读 Z 和 Q；补齐读出临时量仅覆盖本块。 |

F=87、H=7 时，T/U 合计 76.125 MiB，P/R 合计 77 MiB；W=64 时一个 Newton 方阵缓冲为 28 MiB。这些分项的生命周期有交叠。实际峰值还包括特征、卷积（开启时）、GEMM 临时缓冲、输出、框架分配和 aclnn workspace。三个原生子算子向 CANN 请求平台 `GetLibApiWorkSpaceSize()`，没有自行划分用于核间交接的 GM workspace。

## 片上存储与同步

以下为 kernel 内主动分配的**每核**字节地址，相互独立的存储区分别从 0 开始编号。

| 子算子与存储区 | 物理区间与用途 | 生产者 → 消费者 → 复用点 |
| --- | --- | --- |
| Scan，L1 | [0,65536) 当前状态， [65536,131072) 当前 T，均为 FP32 128² | TLOAD 从 GM 装入 → TMOV 装入 L0A/B → 下次迭代可覆盖。L0A、L0B、L0C 各用 [0,65536)，L0C 由矩阵乘写入。 |
| AddDiagonal，UB | [0,16384) 当前校正子块，[16384,32768) 复用的 2I 子块 | MTE2→V 的 event 0 后加法，V→MTE3 的 event 0 后回写原地址；本 job 的 PIPE_ALL barrier 后下个 job 可复用。 |
| Bridge，UB | [0,32768)、[32768,65536) 分别保存前后状态的 64×128 行块；[65536,67584) 为四个连续 512 B 的端点/衰减向量。转换区从 67584 起，BF16 到 83968，FP32 分配到 100352。 | MTE2→V 的 event 0 等待读入。Sub、Exp、按列 Mul 和 Add 间有 V barrier；BF16 模板再 cast。V→MTE3 的 event 0 后搬出，PIPE_ALL 等待缓冲重新可写。FP32 从前状态块直接搬出。 |

Scan 的每帧计算按 `TLOAD → barrier → TMOV → barrier → TMATMUL → barrier → 原子加 TSTORE → barrier → 恢复原子模式 → barrier` 执行。预写好的 U 参与单一生产者的 FP32 累加。下一个帧迭代从刚写好的状态读取；同一链的状态不跨物理核交接。Bridge 每个 job 从头到尾复用同一组缓冲，分两次处理 64 行。三种 kernel 均采用步长为实际 blockDim 的循环，单次 launch 中所有参与的核执行该算子的同一种工作。[Scan 内核](../../vdn_scan/op_kernel/arch22/vdn_scan_cube.h)、[Bridge 内核](../../vdn_bridge/op_kernel/arch22/vdn_bridge_vec.h)

跨阶段依赖由当前 stream 上的提交顺序承担。阶段 2 启动前完成所有因子块和状态库预填；阶段 4 每个块的 bmm 排在本块的 Bridge 后。T、U 和 P、R 是不同的 GM 区域；有效字节区间重叠的原生可写输入由 L2 拒绝。跨阶段没有跨核 ready/free 标志、异步交接队列或双缓冲流水。

## 测量

模型 case 为 85/87 帧、H=7、S=405、D=128、BF16 特征、FP32 Gram 与迭代、预计算文本状态；显式帧块 64，开启 SiLU、QK L2 和 value_first，关闭短卷积、HF32。复现选项、精度阈值和各模型结果见[测试说明](../../../../../../tests/atk/vdn_forward/README.md)。端到端同步前向包含 Python/Stable ABI 提交、原生执行和组合路径中的所有中间操作，TP 通信在本算子外。离线 profile 比较可按阶段聚合每个 bmm、预处理和状态 kernel 的 `op_summary` 中 `Task Duration(us)`；该指标与端到端同步 P50 分开报告。
