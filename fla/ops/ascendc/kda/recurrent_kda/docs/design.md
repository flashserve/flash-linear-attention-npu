# RecurrentKda 设计方案

方案设计规则版本：`V2`

接口和输入约束见 [api.md](api.md)，精度、性能及各轮验证范围见[测试记录](../../../../../../tests/atk/recurrent_kda/README.md)。本文描述当前实现；开发过程记录和流水图不随 PR 提交。

## Host 与平台路径

`layout` 保持原接口语义：BSND 输入 rank4、TND 输入 rank3；是否传入 `cu_seqlens` 独立决定请求边界。无 CU 时 BSND 按 batch 行划分、TND 为单请求；有 CU 时均按展平 token 的 offset 划分。输出 shape 与 v 相同。`use_gate_in_kernel` 控制 raw gate，开启须传 A_log；关闭时 A_log、dt_bias 为空且 safe_gate=false。A2/A3/A5 共用这些公开语义。

设备 CU、indices、accepted 在每次执行中读取，host 只检查 shape/dtype 和不依赖内容的约束。删除私有 prepared_metadata 属性、host 元数据快照及值依赖 tiling，不冻结捕获后的请求边界或状态槽位。kernel 保留边界、容量和活跃槽位的防御检查，非法设备内容不执行相应任务；它不是同步 host 报错承诺。元数据生产者与算子依靠调用 stream 或调用方显式跨 stream 事件建立依赖。

A5 V=128 保留优化实现及长序列扩展，固定 X=8，有无 CU 均采用相同窗口容量。host 直接校验固定窗口所需 UB，不按长度枚举 X；kernel 的窗口、输入/参数/out/事件区偏移及槽间距均为编译期常量，不再统计最大请求长度、动态选择窗口或重建偏移。每个 tile 仍只计算实际有效 token，不补算到8个。每次执行及图回放仍读取当前 CU、indices、accepted，保留请求边界与槽位合法性检查，不增加 host 同步或元数据快照。A5 V=256 使用原 arch35 实现，每段长度不超过8。A2/A3 沿用 generic 计算及 UB 预算。

A5 tiling 预计算 head 比例和 VF 功能位；kernel 组合运行时 bank、slot 和有效 token 数启动 VF。参数槽保留完整地址。新增内部字段不改变公开 ACLNN/Python 参数，A2/A3 不使用 A5 功能位。

Q/K 归一化统一使用 `x/sqrt(sum(x*x)+1e-6)`，零向量仍输出零。A5 V=128 的 BF16 State 仍跨 token 按 BF16 舍入、out 使用当前寄存器中的未舍入值。非原位入口保留已修复的初态复制，避免从未初始化的 scratch 递推。

## A5 调度与 UB

单 Vector Stage 按 tile → token 执行。A5 V=128 的仅末态、逐 token 快照及不写 State 路径统一每 token 启动一次 VF，V-first/K-first、FP32/BF16 使用相同执行粒度；VF 内不再循环多个 token。输入仍按固定 X=8 双缓冲，单 token VF 按任务内 token 偏移选择当前输入/out 槽中的行，整个 tile 完成后批量搬出 out。每个任务根据 bank 当时的可用状态选出两个 bank 交替递推，另一个可复用 bank 预取下一任务初态，角色不固定绑定到物理 bank 编号。逐 token 快照覆盖源 bank 前必须等待对应 MTE3 完成。当前 tile 首个 VF 启动后提交另一槽输入和可用第三 bank 的预取，覆盖前等待与尾部排空保留。

State 搬出行描述符在每核任务循环外构建并复用。定长输入地址直接使用当前或预取任务已解析的 request 和起始 token，按各自 batch stride 与相对 token stride 计算 Q/K/V 地址，不再对展平 token 重复除余；packed 输入仍按全局 token stride 寻址。不新增按 token 展开的任务表或算法 workspace。

A5 V=128 使用静态 Tensor 空间管理：直接以 UB 地址0和编译期总长度构造 `LocalTensor<uint8_t>`，不创建 TPipe/TBuf，不调用 InitBuffer。各 State bank、双缓冲输入/参数/out 和尾部 scratch 沿用上述固定偏移，地址范围为 `[0,eventOffset+2048)`；host 容量校验和编译器预留区保持不变。入口显式调用 `InitSocState()`，所有任务清理后调用 `PipeBarrier<PIPE_ALL>()`。A2/A3 generic 和 A5 V=256 在各自分支内保留 TPipe，执行时不与静态路径混用。接口和约束参考[静态 Tensor 编程](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.html)。

A5 V=128 的核内跨流水同步使用 Mutex::Lock/Unlock，不使用 SetFlag/WaitFlag。13个 MutexID 通过独立于 TPipe 的 AllocMutexID/ReleaseMutexID 管理，不硬编码硬件编号。Lock/Unlock 在同一流水严格成对，同流水访存依赖继续用 LocalMemBar/PipeBarrier。接口参考 [Mutex](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/Mutex_ISASI.md) 和 [Lock约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/Lock.md)。A2/A3 generic 和 A5 V=256 不使用这套新 Mutex 调度。

| 资源 | ID份数 | 持锁范围与复用条件 |
| --- | --- | --- |
| State访问 | 3 | MTE2初始化目标bank；每个VF读源、写目标，两个bank均持锁至该VF结束；防止MTE2覆盖旧VF仍在读取的State |
| State快照 | 3 | MTE2初始化、VF写目标、MTE3读快照；VF读源不持此锁，允许VF读与旧快照读重叠 |
| 输入 | 2 | MTE2整tile搬入、V整tile消费；再次搬入相同槽前自动等待V读完 |
| 参数 | 2 | MTE2任务参数搬入、V整任务消费；保留跨token读取所需生命周期 |
| out | 2 | V整tile生成、MTE3整tile批量搬出；再次覆盖相同槽前自动等待DMA |
| tile控制 | 1 | V持锁覆盖完整tile，结束后Scalar获取并释放，保留原tile边界V→S依赖 |

State初始化按“访问锁→快照锁”获取，DMA后反序释放，再按参数槽获取/释放参数锁。每个单token VF在外部先获取源访问锁、目标访问锁、目标快照锁，VF后反序释放。MTE3仅获取对应bank的快照锁，复制完成后释放。这样MTE2覆盖等待旧VF与旧DMA，Vector覆盖等待旧DMA，而Vector读源不等待同一bank上的旧DMA读。每个Mutex在所有者切换后建立对应流水依赖，不用同流水的连续Lock代替访存屏障。

indexed 路径当前任务校验后立即提交首态、参数和首tile输入，再解析下一任务。首VF由输入、参数和State访问锁等待MTE2就绪；下一任务非法时不启动当前任务VF，外层统一通过Scalar获取全部资源，排空已提交搬入。仅末态路径保留原lookahead和Scalar循环提交位置；两种模式均每token一次VF。下一初态预取继续选择当前两个递推bank以外的可用bank，未采用三bank轮换或末token才预取的候选。

逐token快照由VF释放目标快照锁后交给MTE3，下一次Vector覆盖目标bank前重新获取该锁。相邻快照目标槽重复时保留MTE3单流水屏障；更早快照在其源bank再次被覆盖前完成。仅末态模式在最后一个token后搬出State；out始终在整个tile完成后批量搬出。末快照与out可跨任务在途，Scalar只更新已提交资源的逻辑预留；实际DMA完成由资源Mutex保护，不在Scalar上回收旧DMA后才提交当前快照。

每tile开始V获取输入/out槽，结束释放输入/out槽及控制锁，再由Scalar获取/释放控制锁。参数锁从任务首VF之前持有到末VF之后。所有正常、全空、非法元数据和尾任务出口均回到统一清理：Scalar依次获取并释放13个资源，确认最终所有者完成后ReleaseMutexID，再退出UB生命周期。无State写回时不获取MTE3快照锁，但VF和初态搬入仍按相同访问规则保护bank。

State 在 UB 保持接口 dtype，Cast 在寄存器中完成；BF16 写回使用 Cast_RINT，不保存 FP32 State 影子。当前 token 的 out 使用舍入前的 FP32 更新值。两个 FP32 out 寄存器保存完整 token，最后两次 Cast/Store 到 UB；整个 tile 完成后批量提交 MTE3。

V-first 的四个 V 行归约结果作为 out 收集粒度，计算分组不受四行约束。K-first 全程保持 K-first，GM 接口布局不变，UB State 每行增加32B间隙：FP32 pitch=136、BF16 pitch=144，沿连续 V 方向 Load/Store。

A5 V=128 固定 X=8，资源峰值为 `199744+3072*8+2*256=224832` 字节；FP32 K-first 额外增加12288字节，即237120字节。BF16 K-first 与 V-first 无额外 bank 分配。不增加算法 workspace。

V-first/K-first 复用 `[eventOffset+256,eventOffset+768)` 的512B尾部预留区，分别传递 delta 或 prediction 系数。不再使用前36B的 token/done 事件字；此区不与 DMA 输入/out、State bank 重叠，State 预取不使用此区。保留专用 VST→VLD 屏障和逐 token 复用条件，UB 总峰值公式不变。

## V-first 寄存器计算

完整8-token tile 和不足8个 token 的尾 tile 均采用单 token VF、两遍 State 读取。VF 分派仅区分融合功能和 State 布局，不按完整/尾 tile 分派多 token VF；公开 TilingKey 不变。每 token 将128个 v 批量转成 FP32，q 已包含归一化和 scale，alpha、beta 沿用门控及 sigmoid 规则。每次 VF 重新加载所需 head 参数，不能将历史每 tile 一个 VF 的参数复用收益沿用到当前结构。

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

A5 纯数据 context 和集中布局常量在 `arch35/recurrent_kda_struct.h`；VF 启动/计算在 `recurrent_kda_vec.h`；任务、预取与写回在 `arch35/recurrent_kda.h`，文件头给出 Stage、布局和同步表。

[测试记录](../../../../../../tests/atk/recurrent_kda/README.md)区分各次提交、候选回退、双平台验收、msprof kernel 性能及 host 调用链测量，不能将历史全量验收当作每个后续提交均重跑全量。A3、旧 CANN、整体 Example/ST、正式 NPU CI 及内部故障注入的未覆盖范围按该记录说明。
