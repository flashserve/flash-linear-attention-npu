# RecurrentKda 设计方案

方案设计规则版本：`V2`

公开参数见 [api.md](api.md)

## Host 与平台路径

保留 Python、stable、ctypes、ACLNN、L0 和 schema 的原有签名、顺序、默认值及返回别名，不新增 `prepared_metadata`。`layout` 决定物理维度：BSND 为四维，TND 为三维；`cu_seqlens` 独立决定请求边界。无 `cu_seqlens` 时 BSND 按 batch 行划分、TND 为单请求；有 `cu_seqlens` 时均在展平 token 上划分请求，输出 shape 与 v 相同。

`use_gate_in_kernel` 决定 g 的解释，不能由 A_log 是否存在替代：开启时须传 A_log，可传 dt_bias；关闭时 A_log、dt_bias 必须为空且 safe_gate=false。safe_gate 的 lower_bound 保留原约束。Nk、Nv 不超过256，Nv 是 Nk 的整数倍；这些公开语义由 A2/A3/A5 共用。

host 校验 shape、dtype、stride 和不依赖设备内容的容量约束。`cu_seqlens`、indices、accepted 在每次执行时从设备读取，不做 D2H、host 值快照或捕获后固定请求计划。kernel 保留请求边界和槽位合法性检查；设备非法内容的处理遵循原入口约定，不将设备防御检查描述为同步 host 报错。元数据生产者与算子使用调用 stream 或调用方显式跨 stream 事件建立依赖，ACLGraph replay 使用本轮数据。

A5 的 K=V=128 路径支持长序列，host/kernel 共用编译期常量 `RKDA_A5_WINDOW_TOKENS=8`。保留的 `tiling.windowSize` 固定写8，不按 layout、shape、最大请求长度或 `cu_seqlens` 内容选窗，也不重建 UB 偏移。每 tile 有效长度为 `min(8,end-tileStart)`，不补算无效 token。A5 V=256 保留每段长度不超过8的原路径；A2/A3 保留原 generic 长序列路径和独立资源预算。

Q/K/V 使用各自真实 stride，支持原接口允许的 causal_conv1d 切片等非连续输入。直接读取、连续化及拒绝重叠 descriptor 的判定沿用原入口，不能将 BSND 有 `cu_seqlens` 误作 TND 或弃用组合。tiling 可预计算 head 比例、功能位和不变搬运描述符；动态 bank 归属和 DMA 完成依赖仍由 kernel 管理，不引入全序列任务表或算法 workspace。

## A5 Stage 与任务映射

单个 Vector Stage 覆盖预处理、State 递推和输出。任务为一个请求的一个 value head，共 `B_request*Nv` 个，按 AIV 数量作 grid-stride 分配；`hk=hv/(Nv/Nk)`，使用整数除法。同一 State 的 token 串行，不同任务由不同 AIV 或同核后续任务处理。

目标循环为 `tile → token → V行块`：每个 tile 启动一次顶层 VF，VF 内处理该 tile 的全部有效 token。块行数 R 不固定为4，可分别为 V-first BF16、V-first FP32、K-first BF16、K-first FP32 选择寄存器组织；不能以改变分组为由改变数学语义。完整窗口和尾窗口均按有效 token 执行。

Main Scalar 读取元数据、准备描述符并提交 VF/DMA；Vector 完成所有数值计算。输入和 out 双缓冲，参数使用当前/后继任务双槽。算法 workspace、L1 和 L0 使用量均为0，系统 workspace 另计。性能口径为 msprof `op_summary` 的目标 kernel `Task Duration(us)`，调用链连续化等额外开销另列。

## 寄存器计算与输出

Q/K、gate、beta 和 State 的数值中间结果均采用 FP32。目标 Q/K 归一化沿用固定接口基线 A5 的正范数除法、零范数不除：

```text
q_hat = FP32(q); k_hat = FP32(k)
if use_qk_l2norm_in_kernel:
    q_norm = sqrt(sum(q_hat*q_hat)); k_norm = sqrt(sum(k_hat*k_hat))
    if q_norm > 0: q_hat /= q_norm
    if k_norm > 0: k_hat /= k_norm
q_scaled = scale*q_hat
```

此目标不通过加入 epsilon 改变微小范数输入；与当前代码的差异见文末。beta 按开关执行 sigmoid，只有开启 sigmoid 且 allow_neg_eigval=true 时再乘2。预计算 gate 使用 `alpha=exp(g)`；raw gate 使用 `x=g+dt_bias`（无偏置时 x=g）、`a=exp(A_log)`，普通分支为 `alpha=exp(-a*softplus(x))`，safe 分支为 `alpha=exp(lower_bound*sigmoid(a*x))`，softplus/sigmoid 保留稳定实现。

每个 V 行在寄存器内完成完整更新，行内 K 维归约：

```text
decayed[k] = State_prev[v,k]*alpha[k]
prediction = sum_k(decayed[k]*k_hat[k])
delta = beta_eff*(value[v]-prediction)
State_new[k] = decayed[k]+delta*k_hat[k]
out[v] = sum_k(State_new[k]*q_scaled[k])
```

每个 State 元素每 token 在 UB 中逻辑读一次、写一次，衰减、prediction、delta 和更新结果不写数值 scratch。跨行交错独立 Load/Cast、计算及 Store，控制寄存器生命周期；归约顺序和乘加重排需单独核验精度，不能仅依据静态指令数认定收益。

BF16 State 在 GM 和 UB 中只保留 BF16。读入寄存器后 Cast 为 FP32，写 UB 前 Cast_RINT，不保存 FP32 State 影子；当前 token 的 out 使用舍入前的 FP32 更新值，后续 token 使用已经舍入的 State。FP32 State 直接加载和写回。

128个 out 元素按 V 行位置收集在两个64元素 FP32 寄存器中，不限制每次处理四行。完整 token 计算结束后，两个寄存器分别转换为 BF16，以两次 Store 写入当前 tile 的 out 槽。整个 tile 的 VF 完成后再批量搬出有效 token 的 out，不逐 token 启动 out DMA。

## State 布局与槽位

State 在 GM 和 UB 中保持接口布局：V-first 为 `[V,K]`，K-first 为 `[K,V]`，均紧凑存放。V-first 连续读写 V 行；K-first 用 Gather 读入 V 行，在寄存器中计算后 Scatter 写回，不设置转置缓冲。BF16 相邻 V 元素成对解包、打包，写回不覆盖未更新元素。

无 indices 时，初态取当前请求对应的槽；有 indices 时，默认取首 token 对应的槽，传入 accepted 后改取第 accepted 个 token 对应的槽。accepted 只选择初态，不截短递推。

需要写 State 时，无 indices 仅写末态，有 indices 按 packed 或 speculative 索引逐 token 写回。重复槽按 token 顺序写入，覆盖前保护在途访问；空请求跳过。非原位初态复制及输出别名规则沿用原接口。

## UB 静态空间与生命周期

A5 V=128 使用静态 Tensor：按编译期字节地址、元素类型和元素数量构造 LocalTensor/子视图，不通过 TPipe、TBuf、TQue、TQueBind 或 TBufPool 分配，也不与独立静态地址混用。完成目标平台所需运行环境初始化，核对基础 API 的保留空间及 Mutex ID，guard 不能作为数值中间区。A2/A3 和 A5 V=256 不强制迁移。

固定 X=8，每个 State bank 预留完整 FP32 容量64 KiB；BF16 内容实际占32 KiB。两种 dtype、两种 layout 分支互斥，目标采用同一份地址布局，所有区间为32 B对齐的半开区间：

| UB 区间（KiB） | 大小 | 对象 | 生命周期 |
| --- | ---: | --- | --- |
| `[0,64)` | 64 KiB | State bank 0 | 递推、搬出或初态预取，旧访问结束后复用 |
| `[64,128)` | 64 KiB | State bank 1 | 同上，不固定角色 |
| `[128,192)` | 64 KiB | State bank 2 | 同上，不固定角色 |
| `[192,196)` | 4 KiB | qIn 双槽 | MTE2 写入至 VF 最后一次读取 |
| `[196,200)` | 4 KiB | kIn 双槽 | 同上 |
| `[200,204)` | 4 KiB | vIn 双槽 | 同上 |
| `[204,212)` | 8 KiB | gIn 双槽 | 原 dtype 输入，按最大4 B预留 |
| `[212,212.5)` | 0.5 KiB | betaIn 双槽 | 每 token 占32 B行间隔，单槽256 B |
| `[212.5,213.5625)` | 1.0625 KiB | A_log/dt_bias 双槽 | 当前及后继任务参数，各任务内全部 tile 复用 |
| `[213.5625,217.5625)` | 4 KiB | out 双槽 | VF 写入至 MTE3 读取完成 |
| `[217.5625,219.5625)` | 2 KiB | guard reserve | 对齐和实现安全边界，无数值 tensor |

A_log 单槽预留32 B，dt_bias 为 `128*4=512 B`，双槽共 `2*(32+512)=1088 B`。总峰值为 `196608+3072*8+512+1088+2048=224832 B=219.5625 KiB`。按历史可用248 KiB预算，连续余量为29120 B；host 必须核对实际可用容量。短请求、尾 tile 不缩减空间，K-first 不另增 bank 容量，算法 workspace 为0。

三个 bank 均参与当前任务递推。cur 是当前有效 State，next 是本次更新目标，正常按 `A→B→C→A` 轮换，初始编号由任务交接决定。只有当前任务最后一个 token 确定 cur/next 后，剩余 bank 才用于下一任务初态预取；非末 token 不提前占用专用预取 bank。覆盖前由实际覆盖流水等待旧 VF 和 MTE3 完成，预取初态保留到对应任务消费。

下一任务以预取 bank 作为初始 cur，重新建立三 bank 轮换，不要求任务入口将所有 bank 统一排空。单 token 任务同样先确定源/目标，再选择第三个 bank；若其旧访问尚未结束，Scalar 仍可提交预取，MTE2 通过 Mutex 延迟覆盖。

## Mutex 同步与提交顺序

A5 优化路径统一使用 `Mutex::Lock<pipe>` / `Unlock<pipe>`，参考 [Mutex 接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Mutex_ISASI.md)。Lock 等待指定流水，Unlock 在该流水前序操作完成后释放；Scalar 提交 Unlock 不代表访问已经完成。构建禁止启用 `--cce-auto-sync=on`，跨流水依赖手动管理，同流水保留必要访存屏障。

| 资源 | 份数 | 使用与复用条件 |
| --- | ---: | --- |
| State 访问 | 3 | MTE2 覆盖等待旧 VF，Vector 源/目标访问保护至完成 |
| State 快照 | 3 | Vector 写目标、MTE3 读快照、MTE2 覆盖；Vector 只读源不等待旧 MTE3 只读 |
| 输入槽 | 2 | MTE2 写入后 Vector 读取，MTE2 再覆盖前等待消费完成 |
| 参数槽 | 2 | MTE2 写入、Vector 整任务读取，任务末次读取后才可覆盖 |
| out 槽 | 2 | Vector 整 tile 生成、MTE3 批量读取，Vector 再覆盖前等待 DMA |

12份资源锁按生命周期映射，获取顺序固定、同流水 Lock/Unlock 配对，避免漏释放、过早释放和循环依赖。采用框架分配时 AllocMutexID/ReleaseMutexID 配套；自行管理时核对合法范围及系统保留，不能硬编码冲突 ID。不得以同流水连续 Lock 代替访存屏障。

Scalar 不执行 `Lock<PIPE_S>`、V_S/MTE2_S/MTE3_S 等待、完成标志轮询或变相全流水等待。正常交接、无效任务清理和出口同样遵守该规则；由实际消费者/覆盖者的 Vector、MTE2、MTE3 等待。出口按平台要求保证已提交访问完成和锁配对，不在 Scalar 上先排空旧任务再提交当前任务。

1. 当前任务必要检查完成后，尽早提交初态、参数和首 tile 输入，将下一任务准备与搬入重叠。已有预取初态由 Vector 等待就绪。
2. 每 tile 一次 VF，输入/out 槽在整个 tile 使用期间受保护，参数跨该任务全部 tile 复用。Scalar 可向另一输入槽提交下一 tile；任务尾 tile 可提前搬入下一任务首 tile 的 q/k/v/g/beta。
3. 下一任务参数预取独立使用另一参数槽，不等待下一初态预取。初态预取仅安排在末 token，目标为 cur/next 之外的 bank，由 MTE2 等待该 bank 的旧访问，尽量与末 token 计算重叠。
4. Scalar 按逻辑顺序提交逐 token State DMA，MTE3 等待 Vector 完成相应结果；out DMA 等待整个 tile 的 VF 完成。旧末态/快照/out 可以与后继计算在途，覆盖它们的源地址前由覆盖流水等待。
5. Scalar 按任务长度和轮换关系推导资源归属，不查询实时完成状态。不变 stride、行宽和 State 搬运描述符可在任务循环外缓存，每次仅更新 token 偏移、有效长度和槽位。

顶层 tile VF 内的逐 token 资源交接必须在实际访问边界完成，不能将 tile 启动或结束当成所有 token 的 State 就绪时刻；多 token VF、快照和末 token 预取的 Mutex 编排需结合生成指令验证。

## A2/A3 资源预算

A5 的三 bank、静态 Tensor、Mutex 和寄存器分组不套用于 A2/A3。generic kernel 固定窗口 X=8，两槽 packed q/k/v/g 输入共 `2*X*(8*alignK+2*alignV)` 字节，beta 单槽为 `4*X*NV` 字节，另留64字节 scalarBuf 和2048字节 guard。

具名 FP32 窗口工作区为 `X*(4*alignV+12*alignK+4*NV)` 字节；加上每个 V 行的 State、broadTmp 和两个行结果后得到 tmpBuff 所需容量。host 先验证 tmpBuff 不超过扣除真实输入及 State/out 队列后的剩余容量，再只分配所需字节。共用入口仍保留 A2/A3 的公开参数语义、长序列处理及真实 vStep，不能因 A5 预算调整减少 generic 实际需要的双槽或删除容量校验。

## 当前实现与目标的差异

下表以本次文档修改前代码为准，仅说明落地状态，不以现实现反向修改目标：

| 项目 | 当前实现 | 本文目标 |
| --- | --- | --- |
| VF 粒度 | 仅末态、逐 token 快照及不写 State 均为每 token 一次 VF | 每 tile 一次顶层 VF，内部逐 token |
| State 数值流程 | V-first/K-first 两遍读取 State，复用 guard 中512 B scratch 传递 residual 或系数 | 每元素逻辑读一次、写一次，中间结果驻留寄存器 |
| K-first UB/计算 | 每行增加32 B间隙，FP32 pitch=136、BF16 pitch=144，沿连续 V 两遍计算 | 紧凑 K-first，按 V 行 Gather/Scatter，BF16 成对解包/打包 |
| UB 峰值 | V-first/BF16 K-first 为224832 B；FP32 K-first 为237120 B | 两种 layout/dtype 统一预留224832 B，guard 不存数值中间结果 |
| Q/K 归一化 | A5 与 A2/A3 使用 `x/sqrt(sum(x*x)+1e-6)` | A5 按固定接口基线正范数归一化、零范数不除 |

三 bank 轮换、末 token 第三 bank 初态预取、参数与 State 预取分离、固定 X=8、静态 Tensor、资源 Mutex 和整 tile out 批量搬出已在当前代码中具备。原文中“两 bank 交替、第三 bank 提前预留”和 Scalar 清理等待的旧描述不再作为目标。

## 代码组织与验证范围

A5 纯数据 context 和集中布局常量在 `arch35/recurrent_kda_struct.h`；VF 启动/计算在 `recurrent_kda_vec.h`，寄存器辅助函数在 `recurrent_kda_regbase.h`；任务、预取与写回在 `arch35/recurrent_kda.h`。这些文件仍按当前实现组织，不能仅因文档修改宣称目标已经落地。

后续实现对齐须分别验收四条 dtype/layout 路径，覆盖仅末态、逐 token 快照、无 State 写回，单 token、空请求、尾 tile、长序列和跨任务连续执行。重点核对三 bank 覆盖、旧快照在途、重复目标槽、末 token 预取、输入/参数槽回绕及退出的锁配对，并通过实际编译命令和生成指令确认没有自动同步或 Scalar 等待。

以固定 main 接口基线做 BSND/TND × `cu_seqlens` 有/无、gate 参数组合、零/微小范数、State dtype/layout、别名及非法输入差分；ACLGraph replay 改变 `cu_seqlens`/indices/accepted 后须消费新值。K-first Gather/Scatter 地址、BF16 打包舍入、寄存器峰值和 State 逻辑读写量单独检查，A2/A3 长序列及 A5 V=256 兼容路径回归。

性能按相同设备、软件、输入、开关和采样口径比较目标 kernel，快照与仅末态分别记录。历史测试结论不能替代以上目标变更后的验收；本次仅更新两份文档，不新增性能结论。
