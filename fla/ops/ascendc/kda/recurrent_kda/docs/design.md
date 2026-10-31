# RecurrentKda 设计方案

方案设计规则版本：`V2`

接口和输入约束见 [api.md](api.md)，精度、性能及各轮验证范围见[测试记录](../../../../../../tests/atk/recurrent_kda/README.md)。本文描述当前实现；开发过程记录和流水图不随 PR 提交。

## Host 与平台路径

序列模式由是否传入 `cu_seqlens` 决定，gate 模式由是否传入 `A_log` 决定；未启用 raw gate 时忽略 `dt_bias`、`safe_gate` 和 `lower_bound`。A2/A3/A5 共用这些语义，保留各自的硬件计算和资源路径。

A5 面向 `K=V=128`。定长 X 按序列长度 S，变长按各请求最大有效长度选择，候选为1/2/4/8，上限8，全空取1。`cu_seqlens` 是 host 值依赖；改变元数据须重新 tiling。

所有平台的 packed 调用都在 host 校验 `cu_seqlens` 的首项、单调性和容量，保留 executor 自有快照，并禁用依赖旧元数据的 executor 缓存。A2/A3 的无 indices packed 路径也执行该校验，因此新增必要的 host 同步/D2H；固定长度且无 indices 不增加此开销。A5 还用快照计算最大有效长度及 X。

Python stable/ctypes 入口在 CU/indices 存在时先等待当前 PyTorch NPU stream，补足框架生产者到 ACLNN host 快照的就绪依赖。该处理局限于 recurrent_kda，不修改公共 runtime。直接 ACLNN 的元数据就绪责任由调用方承担，详见 API。

在生产者已就绪的前提下，host 元数据读取使用同步 D2H，不调用全设备同步，不新增 stream、event 或 pinned staging 的逐调用分配；host placement 元数据直接复制。runtime 同步复制的跨流行为以 A5 实测为准，不扩展为所有 CANN 版本的保证。

公开 host 对活跃 indices/accepted 做范围校验并保存 executor 自有快照，空请求和 speculative padding 跳过；此路径禁用旧 executor 缓存。无 indices 不增加 slot/accepted 快照步骤，packed 调用仍执行 CU 校验和快照。State 的逻辑 view 复制由同一 helper 覆盖非原位初始化与原位独立输出。

## A5 调度与 UB

单 Vector Stage 按 tile → token 执行。每个任务根据 bank 当时的可用状态选出两个 bank 交替递推，另一个可复用 bank 预取下一任务初态，角色不固定绑定到物理 bank 编号。逐 token 快照覆盖源 bank 前必须等待对应 MTE3 完成。输入/out 双缓冲；当前 tile 必要同步后先 Dispatch VF，再提交另一槽输入和可用第三 bank 的预取，覆盖前等待与尾部排空保留。

State 在 UB 保持接口 dtype，Cast 在寄存器中完成；BF16 写回使用 Cast_RINT，不保存 FP32 State 影子。当前 token 的 out 使用舍入前的 FP32 更新值。两个 FP32 out 寄存器保存完整 token，最后两次 Cast/Store 到 UB；整个 tile 完成后批量提交 MTE3。

V-first 的四个 V 行归约结果作为 out 收集粒度，计算分组不受四行约束。K-first 全程保持 K-first，GM 接口布局不变，UB State 每行增加32B间隙：FP32 pitch=136、BF16 pitch=144，沿连续 V 方向 Load/Store。

A5 资源峰值为 `199744+3072*X+2*max(128,32*X)` 字节；FP32 K-first 额外增加12288字节，BF16 K-first 与 V-first 无额外 bank 分配。不增加算法 workspace。

V-first/K-first 复用 `[eventOffset+256,eventOffset+768)` 的512B尾部预留区，分别传递 delta 或 prediction 系数。事件字占前36B，与 scratch 不重叠；此区不与 DMA 输入/out、State bank 重叠，State 预取不使用此区。保留专用 VST→VLD 屏障和逐 token 复用条件，UB 总峰值公式不变。

## V-first 寄存器计算

所有 X，包括 X=1 的 fused/general 分派，均采用两遍 State 读取。变长调用中的短请求和单 token 尾 tile 也执行两遍；公开 TilingKey 不变。每 token 将128个 v 批量转成 FP32，q 已包含归一化和 scale，alpha、beta 沿用门控及 sigmoid 规则。

第一遍只读当前 State bank，计算 `prediction[v]=Σ_k State[v,k]*(alpha[k]*k[k])`，按四行收集到两个 FP32 寄存器，再批量形成 `delta=v-prediction`。将 delta 以两次 VST 写入512B scratch，经 VST→VLD 屏障由第二遍读取；本 token 最后一行消费完成后，下一个 token 才能覆盖。

第二遍重读同一源 bank，使用每 token 预计算的 `beta*k`，通过 MulAddDst 计算 `State_next=alpha*State+delta*(beta*k)`。先用未舍入的更新值计算 out，再以接口 dtype 写另一 bank；State Store 先于 out 收集。prediction/out 点积用一条 Mul 和一条 MulAddDst 合并两个64列片段，再按原归约树求和。

BF16 第一遍将两行 Load/Cast、点积交错，再处理后两行；第二遍在寄存器读入后 Cast 为 FP32、写回前 Cast_RINT。FP32 第二遍以滚动双行更新，在生成低片段 out 乘积后提前 Store 并释放低片段 State，随后完成高片段点积和写回。两种调度均保留四个归约结果的收集方式、512B residual UB 传递及必要同步。

每 token State 有效 UB 读取量为 FP32 128KiB、BF16 64KiB，写入分别为64KiB/32KiB。两遍读取以增加 UB 读取和 BF16 转换换取更短的 State 寄存器生命周期，不减少逐行归约，不改变 GM 流量。乘法重排和 FMA 可能改变 FP32 舍入，按原 CPU 标杆、值域及混合容差验收。

## K-first 寄存器计算

FP32/BF16 均沿连续 V 方向两遍计算，每个 FP32 寄存器覆盖64个连续 V 元素；BF16 在 Load 后于寄存器转 FP32。两遍均从当前 State bank 读取，直接消费两个 FP32 v 寄存器，不需要 vFloat scratch。

每 token 在寄存器计算完整 `alpha*k`，两次 VST 写入512B scratch，经专用 VST→VLD 屏障后，第一遍沿 K 从 UB 广播系数，替代128次寄存器 Gather 和128次 lane 自增，累计 `prediction=Σ State*(alpha*k)`。系数在第一遍结束前全部消费，第二遍不再读取此区；下一 token 在前一 token 更新和必要屏障后复用。

随后两个向量计算 `delta=beta*(v-prediction)`。第二遍逐 K 计算 `State_next=alpha*State+delta*k`，先累计 `out=Σ State_next*q`，再以接口 dtype 写另一 bank；BF16 写回使用 Cast_RINT，不从已舍入的 BF16 结果计算当前 token 的 out。第二遍系数从 q/k/alpha 寄存器 Gather 广播。

连续访问省去 State Gather/Scatter、逐行归约和四行 out 收集。每 token State 有效 UB 读取量为 FP32 128KiB、BF16 64KiB，写入分别为64KiB/32KiB；GM 流量、bank 所有权和整 tile out 批量搬出规则不变。FP32 顺序累计及 FMA 舍入变化按原 CPU 标准验收。

## A2/A3 资源预算

A5 的计算分组、三 bank 容量及 X 预算不套用于 A2/A3。generic kernel 固定窗口 X=8，两槽 packed q/k/v/g 输入共 `2*X*(8*alignK+2*alignV)` 字节，beta 单槽为 `4*X*NV` 字节，另留64字节 scalarBuf 和2048字节 guard。

具名 FP32 窗口工作区为 `X*(4*alignV+12*alignK+4*NV)` 字节；加上每个 V 行的 State、broadTmp 和两个行结果后得到 tmpBuff 所需容量。host 先验证 tmpBuff 不超过扣除真实输入及 State/out 队列后的剩余容量，再只分配所需字节。

## 代码组织与验证

A5 纯数据 context 和集中布局常量在 `arch35/recurrent_kda_struct.h`；VF 启动/计算在 `recurrent_kda_vec.h`；任务、预取与写回在 `recurrent_kda_pipeline.h`，文件头给出 Stage、布局和同步表。

[测试记录](../../../../../../tests/atk/recurrent_kda/README.md)区分各次提交、候选回退、双平台验收、msprof kernel 性能及 host 调用链测量，不能将历史全量验收当作每个后续提交均重跑全量。A3、旧 CANN、整体 Example/ST、正式 NPU CI 及内部故障注入的未覆盖范围按该记录说明。
