# Recurrent KDA 设计

当前规则版本：V2。公开参数见 [api.md](api.md)。
整体结构参考 PR #1083 的 efbe1866，针对 KDA 的向量 gate 和 State 布局实现独立计算路径。

## 目标与结构

整体采用参考 PR 的类式 constructor / Init / Process 组织、静态 UB、X=4 输入/out 双缓冲、三个 State bank 轮换、每 token 一个 VF 和 Mutex 同步。V-first 沿用 UnalignRegForStore 输出收集，K-first 在寄存器中累计完整 out 后连续写 UB。A5 的 V=128/V=256 统一使用同一 Engine，不沿用 x8-pr 调度器或 UB 地址规划；A2/A3 继续走兼容实现。

保留 KDA 原接口，包括向量 gate、128 维 dt_bias、K-first、空请求、非连续输入及各输出组合。host 依次完成形状检查、维度与 stride 填充、属性检查、分核和 UB 规划，复用既有公开接口校验。

### A2/A3 平台边界

A5 的 State view/别名前置校验和资源规划按实际架构启用。A2/A3 的通用 kernel 恢复为 PR 基线，保留 token-major 直接读取条件、BSND 的严格批间距、原分核公式和 ViewCopy 路径。公共结构化错误码继续保留，auto-sync 配置不在本次修改范围。

State 输入与输出沿用远程主线已修复的公开语义：非原位模式直接读取初态，仅写入有效目标槽，不克隆或预填整个输出池。

### State 输入与输出

`initial_state` 在所有输出组合下均须显式传入非 `None` 张量。非原位调用直接读取该输入，不修改其数据或版本计数；仅在 `output_final_state=true` 时分配独立 State 输出，不通过克隆输入后执行原位 kernel 实现。Python 对原位调用拒绝 `requires_grad` State，并更新输入版本计数；非原位调用对该输入使用 `detach` view。

| `inplace_final_state` | `output_final_state` | kernel 的 State 写回目标 | Python 返回的 State |
| --- | --- | --- | --- |
| true | false | `initial_state` | `None` |
| true | true | `initial_state` | 与输入同一张量 |
| false | false | 不写回 GM State | `None` |
| false | true | 独立的 `final_state` | 独立输出张量 |

开启 State 写回时，无 indices 则只写各非空请求的最终 State，有 indices 则写每个有效 token 对应槽位的 State。非原位输出不预填输入，因此未写入槽位（包括空请求对应槽位）的值未定义，不能要求其等于输入。原位模式的未写入槽位保持输入原值。空请求指 `cu_seqlens` 中相邻 offset 相等；q/k/v/g/beta/State 任一维长度为0的输入仍被 host 拒绝。`cu_seqlens` 末值小于 token capacity 时，`out` 的未计算尾部同样不保证初值。

aclnn 保留输入与 State 输出 view 的 stride/offset，kernel 直接读输入并写目标 view。原位调用若另外提供独立 `finalState` 且要求输出，aclnn 会在 kernel 后将更新后的输入 State 复制到该输出；Python 原位入口直接返回输入 alias。

## Stage 与分核

每个 AIV 完整处理一个请求的一个 value head，任务数为请求数乘 HV；任务按 core 数做 grid-stride 分配，hk=hv/(HV/H)。A5 的 blockDim=min(AIV 数,任务数)，不使用 AIC；A2/A3 保留基线先计算每核任务数、再确定 blockDim 的公式。B=32、HV=48 时共有 1536 个任务，56 个 AIV 每核 27 或 28 个任务。

V=128 每任务处理一个128列 V 分块；V=256 沿 V 分成 `[0,128)`、`[128,256)`，同一 AIV 依次完成两个分块的全部 token。每个分块复用原128×128 VF、三个 State bank 和输入/out 槽。K-first 从 GM 的 `[K,256]` 中逐 K 行搬入对应128列，UB 保持 `[K,128]`，输出按原 stride 写回，无转置。V-first 对应128个 V 行。请求元数据、A_log 和 bias 每任务解析或搬入一次；q/k/g/beta 每分块读入。两种 V 维度均按四 token 的 tile 顺序递推，支持跨任意多个 tile，末 tile 仅处理有效 token；UB 占用不随序列长度增加。保留 cu_seqlens 和 State 索引容量校验。

1. Scalar 解析请求、初态槽位和 head，MTE2 搬入初态与当前 tile 的 q/k/v/g/beta。
2. Vector 对每个 token 计算归一化、gate、prediction、State 更新和 out。
3. MTE3 按输出模式搬出 State，并在整个 tile 完成后批量搬出 out。

q/k 可选按 FP32 的 x/sqrt(sum(x*x)+1e-6) 归一化，q 乘 scale；仅在开启 beta sigmoid 时应用 sigmoid，并在同时开启 `allow_neg_eigval` 时将其结果乘 2，未开启 sigmoid 时直接使用 beta。alpha[k]=exp(g[k])，raw gate 时为 exp(-exp(A_log)*softplus(g[k]+bias[k]))，raw gate 同时开启 safe gate 时为 exp(lower_bound*sigmoid(exp(A_log)*(g[k]+bias[k])))；缺省 bias 按零处理，未开启 raw gate 时不使用 A_log/bias。softplus 使用 max(x,0)+log(1+exp(-abs(x))) 避免中间指数溢出。

两种布局计算相同的递推公式：prediction[v]=sum_k(State[v,k]*alpha[k]*k[k])，residual[v]=v[v]-prediction[v]，State_new[v,k]=State[v,k]*alpha[k]+residual[v]*beta*k[k]，out[v]=sum_k(State_new[v,k]*q[k])。公式中的 State 使用逻辑 [V,K] 下标，物理布局由路径决定。BF16 State 每 token 写回 UB 前 RINT 舍入，out 基于舍入前的 FP32 更新值；FMA、归约顺序和公共乘积的重排须通过原有精度阈值。

V-first FP32/BF16 保留两行粒度，采用两遍读取：先预计算 alpha*k，第一遍归约 prediction，使用 StoreUnAlign 收集到512 B scratch；批量读取 prediction 与 v，计算 residual 后以交织布局覆盖同一 scratch。第二遍重读原 State，从 scratch 逐行广播 residual，执行 State_new=State*alpha+residual*(beta*k)，沿用原 out 收集与写回。prediction 收集完成和 residual 写入完成后各执行一处 VST→VLD 屏障。FP32 每 token 读取128 KiB、BF16 每 token 读取64 KiB State UB，GM字节数不变。

K-first 参考 x8-pr 的 UpdateKFirstColumns，保持紧凑 [K,V] 存储，沿连续 V 维使用两个64元素寄存器计算，不转置或增加 State 行 padding。q/k/alpha 在此路径按前后64个 K 元素加载。先将 alpha*k 写入512 B系数 scratch，执行一次 VST→VLD 屏障，再逐 K 行广播系数、连续读取128个 V，使用 FMA 在两个寄存器中累加完整 prediction。随后原位计算 delta=(v-prediction)*beta，第二遍逐 K 行重读原 State，广播 alpha/k/q，更新 State 并在寄存器中累加完整 out。BF16 State 在寄存器中转 FP32，out 使用舍入前的更新值，再 RINT 转 BF16 写回 UB。最终 out 通过两组64元素转换和 Store 写入 UB，整个 tile 完成后再 DMA。第一遍读取结束后 prediction 原位变为 delta，系数 scratch 在当前 VF 之后可复用；GM 搬运、State bank 调度和 Mutex 不变。

## UB 与 workspace

| 区域 | 字节区间 | 用途 |
| --- | --- | --- |
| State 0 | [0,65536) | 当前/目标/下一任务初态，角色动态轮换 |
| State 1 | [65536,131072) | 同上 |
| State 2 | [131072,196608) | 同上 |
| 输入/out 槽 0 | [196608,202880) | q/k/v 各 1024 B，gate 2048 B，beta 128 B，out 1024 B |
| 输入/out 槽 1 | [202880,209152) | 同上 |
| dt_bias 槽 0 | [209152,209664) | 一个 head 的 128 个 FP32 bias |
| dt_bias 槽 1 | [209664,210176) | 下一有效任务的 bias，与槽 0 交替 |
| VF scratch | [210176,210688) | V-first：128个 prediction/residual 顺序复用；K-first：128个 alpha*k 系数 |

V-first/K-first FP32/BF16 均合计210688 B，即205.75 KiB。全部起点按32 B对齐，BF16 State 每 bank 有效数据为32 KiB，但保留64 KiB地址跨度。scratch只由当前VF的Vector访问。V-first 首遍StoreUnAlignPost后批量读 prediction，计算 residual 后覆盖同一 scratch，再逐行广播；K-first 存入 alpha*k 后按 K 行广播，第二遍不再使用该 scratch。两种布局的用途互斥，V-first 两次写后各执行一处 VST→VLD 屏障，K-first 系数写后执行一处，下一 token 可复用。两种布局每个128列 V 分块、每 token 的 State UB 读取为 FP32 128 KiB/BF16 64 KiB、写入为 FP32 64 KiB/BF16 32 KiB；V=256 的整个任务需计两个分块。L1、L0和算法workspace均为零；host 固定申请16 MiB系统workspace，kernel 不使用该 workspace 存放算法中间量。

输入由 MTE2 生产，当前 tile VF 消费；out 由 VF 生产，MTE3 消费。bias 由 MTE2 生产，当前任务的全部 VF 消费；两个槽按有效任务交替，空任务不推进槽号。State 在 VF 更新后可供下一 token 与 MTE3 并行读取。每 token 都在 UB 中读写完整 State 分块；仅当 `inplace_final_state || output_final_state` 时写回 GM，有 indices 时每 token 写回，无 indices 时只在请求末 token 写回。

State view 的末两维必须稠密，slot/head 维允许正向、互不重叠的间隔。对当前128×128分块，输入/输出的 `stride[2]==128` 时用单块 DataCopy，否则用128行 DataCopyPad。因此外层有间隔的 State 仍可使用单块搬运，而物理连续的 K-first V=256 也须逐 K 行搬运128列分块。Q/K/V 的受支持 view 按 tiling 中的实际 stride 直接读取，其余可安全连续化的 view 由 aclnn 预处理；gate/beta 和可选参数、元数据由 aclnn 连续化。使用默认 L2 策略。

## 同步与生命周期

engine 独占 UB 与 Mutex 编号：input[3,4]、out[5,6]、State[7,8,9]、parameter[10,11]，共九个。input 保护 MTE2→V→MTE2；out 保护 V→MTE3→V；State 保护 V 更新、MTE3 读取和 MTE2 覆盖。V 只读取旧 State 时允许 MTE3 同时读。parameter 分别保护两个向量 bias 槽的跨任务复用；未开启 raw gate 或未传入 dt_bias 时，不执行参数锁操作。

任务入口等待 V→MTE2 握手，选择 (current+1)%3 的 bank 搬入初态；然后向当前参数槽提交 bias 和首 tile 输入，Vector 持有该槽参数锁直到任务结束。相邻有效任务选择不同参数槽，下一任务输入搬入无需等待当前任务释放参数槽；再次复用同一槽时仍须等待其上一个消费者释放。每个 token 锁住目标 bank，执行 VF，再按需提交 State 搬出。tile 完成后释放输入/out 锁并提交整个 tile out。

indexed State 写回时，每个 bank 记录其尚未回收的快照目标槽位。Vector 获取目标 bank 的 Mutex 后，上一笔以该 bank 为源的搬出已完成，可清除对应记录。提交新快照前，若其 GM 槽位与其他未完成快照相同，在 MTE3 上插入 `PipeBarrier<PIPE_MTE3>()`，并清除此前记录，保证同址覆盖按 token 顺序完成；其他槽位继续重叠搬出。只靠不同 bank 的 Mutex 或同流水 Lock/Unlock 不能保证重叠 GM 写的完成顺序。记录在每个任务的 V 分块内维护；不同活跃请求不得共享写槽，同一任务的不同 V 分块写入不同 GM 区间。该等待不阻塞 Scalar。

倒数第二个 token VF 后发布 V→MTE2，保证下一工作项（下一 V 分块或下一任务）所需 bank 已完成最后一次 Vector 读取。T=1 时下一任务所用 bank 不参与当前 VF，可提前发布。下一工作项初态搬入允许与当前末 token VF 重叠；同一任务的下一 V 分块继续复用参数槽，下一任务参数搬入使用另一参数槽，覆盖旧槽时仍由该槽独立的 parameter Mutex 保护。结尾关闭未消费的握手，再由 V 按槽0、槽1依次获取并立即释放 output Mutex，等待两槽最后的 MTE3 搬出完成，替代入口的 PipeBarrier<PIPE_ALL>。最后一笔 tile out 位于此前全部 State 搬出之后，因此该等待覆盖本 AIV 的 State/out 写回；全部 MTE2 输入已在 VF 前由 input/parameter Mutex 消费。未使用的 out 槽处于释放状态，不增加等待；cu_seqlens 校验早退时尚未提交算子数据 DMA。核尾不新增 Mutex 编号，不使用 MTE3_S 或 PIPE_S。

沿用参考 PR 的逐 tile 提交方式，不额外引入 x8-pr 的 lookahead 输入预取。不同物理输入/out 槽可在复用保护下覆盖搬运与计算；是否形成有效流水重叠须由设备采集确认。

## 元数据与模板

cu_seqlens、indices、accepted 是设备 tensor，host 检查 shape/dtype 等描述信息，不读取内容，保持 graph replay 时读取当前设备值。A5 kernel 在提交算子数据 DMA 前检查全部 cu_seqlens 的首值、单调性和范围；二维 indices 还要求请求长度不超过 max_step。该检查失败时本核直接返回。每个非空请求另行检查初态及各 token 的 State 槽位和 accepted；失败时停止本核后续任务，但此前任务可能已经写出，不能保证输出完全未修改，也不在 host 同步报错。空请求跳过上述槽位检查及算子数据 DMA。

无 indices 时初态槽位为请求编号；有 indices 且无 accepted 时，使用请求首 token 对应的槽位。有 accepted 时使用 `begin+accepted-1` 对应槽位：packed 一维 indices 要求 `1<=accepted<=请求长度`，二维 speculative indices 要求 `1<=accepted<=max_step`，允许超过本次请求长度，但选中的初态槽位仍须在 State pool 内。每个有效 token 的写槽也须在 pool 内。同一请求可重复写槽，按前述 MTE3 顺序保护；不同活跃请求的写槽不得冲突，这一要求由调用方保证。

kernel 入口唯一显式模板参数为 `uint32_t KERNEL_MODE`，host 用 `GET_TPL_TILING_KEY(kernelMode)` 生成 TilingKey。mode 值0为 `RKDA_TPL_COMPAT`，仅由 A2/A3 选择；A5 的 V=128/V=256 均按 State 布局和融合条件选择下表的1～4。V 维度、State dtype、是否 indexed 以及是否写 State 不另增加 mode。

State、gate、beta 的 dtype 由算子 DataTypeList 对应的编译配置提供 `DTYPE_INITIAL_STATE`、`DTYPE_GATE`、`DTYPE_BETA` 宏；这些宏再作为 Engine/VF 的类型模板实参。它们不是入口额外的显式模板参数，也不编码到本算子的 `KERNEL_MODE` 中。

常用融合要求 norm、raw gate、bias、sigmoid 全开且 negative、safe 全关；其 `FUSED=true` 将这六项 VF 条件固定为编译期常量。其他组合使用 `FUSED=false`，VF 从 `a5VfFlags` 中解析运行时开关。State 写回、indices、请求长度等调度条件仍由 Scalar 按 tiling/设备元数据处理，融合模式并未将这些分支全部特化。

## 验证矩阵

沿用原 CPU 标杆、输入值域和精度标准。精度覆盖布局、dtype、gate 模式、tail、长序列、空请求、槽位和输出组合；补充同一 AIV 连续任务与跨 tile 复用。A5 可达四种模板模式，按下表核对，不能把 A2/A3 的 compat 计为第五种 A5 模式。

| 模板模式 | mode 值 | 布局 | 功能组合 |
| --- | --- | --- | --- |
| `RKDA_TPL_V_FIRST` | 1 | V-first | 预计算 gate，或不满足常用融合条件的组合 |
| `RKDA_TPL_K_FIRST` | 2 | K-first | 同上 |
| `RKDA_TPL_V_FIRST_FUSED` | 3 | V-first | norm、raw gate、bias、sigmoid 开启，negative/safe 关闭 |
| `RKDA_TPL_K_FIRST_FUSED` | 4 | K-first | 同上 |

每种模式均覆盖 V=128/V=256、FP32/BF16 State；通用模式补充 safe gate、negative、缺省 bias 和各功能开关的组合。长序列覆盖 T=9/17/65/1024、混合 cu_seqlens、空请求和逐 token State 写回，确认 V=256 跨 tile 处理且无提前退出。

模板覆盖记录需保存输入 case、host 实际产生的 TilingKey、模板模式及精度结果，不能只根据属性推算 key；诊断插桩仅用于记录选择，验收后恢复正式 host 库。检查各模式的内存与有效输出完整性；非原位 State 仅比较实际写入槽位。性能矩阵为 issue680 的36个 shape × V-first/K-first × FP32/BF16 State × final/indexed 两种写回，共288对基线/当前实现比较。实测完成情况以独立 output 记录中的样本、环境、构建身份和结果为准，测试脚本及原始 profiling 不纳入 PR。

## 实现索引

| 设计内容 | 实现位置 |
| --- | --- |
| State 分配、返回与版本计数 | [Python Stable 入口](../../../../../../torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py)、[Stable C++ launcher](../../../../../../torch_custom/fla_npu/csrc/src/stable_recurrent_kda.cpp) |
| view、连续化及原位独立输出复制 | [aclnn_recurrent_kda.cpp](../op_host/op_api/aclnn_recurrent_kda.cpp) |
| 分核、UB 和融合标志 | [recurrent_kda_tiling_processor.h](../op_host/recurrent_kda_tiling_processor.h) |
| mode 声明与 host 选择 | [recurrent_kda_tiling_key.h](../op_kernel/recurrent_kda_tiling_key.h)、[recurrent_kda_tiling.cpp](../op_host/recurrent_kda_tiling.cpp) |
| 入口模板、dtype 与 Engine 实例化 | [recurrent_kda.cpp](../op_kernel/recurrent_kda.cpp) |
| VF、State 搬运、元数据及同步 | [arch35/recurrent_kda.h](../op_kernel/arch35/recurrent_kda.h) |
