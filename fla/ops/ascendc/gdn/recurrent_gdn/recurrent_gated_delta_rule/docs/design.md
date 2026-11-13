# RecurrentGatedDeltaRule 统一实现

方案设计规则版本：V2

## 1. 目标

接口契约见 [api.md](api.md)。将既有融合实现迁入 arch22，移除历史 A2/A3 kernel，
保留两个 host ABI。该迁移不改变融合实现的计算、任务划分、地址规划和同步协议。
`useV2` 仅区别 host 参数契约和 dtype 校验，不再选择不同 kernel。A2/A3 与 A5
分别编译 arch22 和 arch35，但各架构内部的 V1/V2 host ABI 进入同一实现。
CPU 标杆独立实现递推及开关预处理，比较 out 和完整 State；固定原输入值域及混合容差。
性能采用 msprof `Task Duration(us)`，本次没有额外性能提升目标。

## 2. A2/A3 Stage 0–N 完整详设

令 B 为请求数，T 为总 token 数，HK/HV 为 QK/value head 数，K=V=128。
任务域为 `[0,B*HV)`：`batch=task/HV`，`hv=task%HV`，`hk=hv/(HV/HK)`。
所有计算均为 Vector，沿用既有 AIV-only 分核；不使用 Cube、L1、L0 或私有 GM workspace。
host 按可用 AIV 数限制 blockDim，任务按连续区间均匀分配。
例如 B=2、HK=2、HV=4 时共 8 个独立任务，每个任务独占一个请求和 value head。

### Stage 0：Vector，输入准备

每次窗口最多 8 token。Q/K/V 按各自 stride 搬入双 raw bank，转为 FP32。
Q/K 可选归一化并对 Q 乘 scale。beta 与 g 按窗口收集、转换，完成可选 sigmoid/
softplus 门控；使用非正指数实现稳定 sigmoid，softplus 修正小尾部舍入损失。
g=None 时不读取 gate。空序列不进入计算及窗口同步。

### Stage 1：Vector，逐 token 递推与写回

初始 State 由 accepted token 对应索引读入，按 State stride 搬入 FP32 bank。
每 token 顺序为 `S_decay=exp(g_eff)*S_prev`、`delta=(v-S_decay@k)*beta_eff`、
`S_next=S_decay+outer(delta,k)`、`out=S_next@(q*scale)`；这里的行点积由 Vector
乘法和固定归约实现。状态在两份 FP32 bank 间交替，归约临时值复用已消费的 source bank。
BF16 checkpoint 放到已消费 source bank 的上半区，FP32 checkpoint 直接使用目标 bank，
只有 checkpoint 舍入，递推状态不量化。每个有效 token 的 out/State 各写回一次。

UB 总量沿用原固定布局 162240 字节，另检查 2048 字节保护余量。
两份 State bank 位于 `[0,65536)` 和 `[65536,131072)`；BF16 快照分别复用每 bank 的
后 32768 字节。后续为两份 raw bank、三份 FP32 Q/K/V 窗口、delta/out/reduction、
broadcast、beta/gate/临时向量、两份 BF16 输出和 metadata。
具体半开区间由 arch22 头文件连续的 `k*Offset` 常量唯一确定，搬迁保持所有常量不变。

### 搬运与同步

每 task 的初始 State 从 GM 读一次，每窗口 Q/K/V 各读一次；beta/gate/indices
按窗口收集。MTE2→V ready 按 raw bank 与 Q/K/V 分配；V→MTE2 free 保护 raw bank
覆写。V→MTE3 ready 与 MTE3→V free 分别按 State bank 和输出 bank 配对。
task 复用前使用 V→MTE2 与 MTE3→MTE2 闭环，metadata 使用 S→V 同步。
所有 EventID 由 AllocEventID 分配，在 task 尾等待 pending，再在 kernel 尾 ReleaseEventID。
V 内读写复用使用 PIPE_V barrier；跨 pipe 依赖由硬事件表达。
窗口尾只消费有效 count，后续窗口复用遵循同一 ready/free 协议。

### 验证范围

200 条合法精度用例，覆盖 8 个模板 dtype 组合、12 个合法开关组合、变长/零长度序列、
accepted token、State padding/offset/转置、QKV 行 padding、窗口 8/9/16/17 边界。
确定性和内存各使用 50 条代表用例；额外验证 gk 拒绝与两个 host ABI 的有效控制。

## 3. A5 arch35 流水

A5 任务域同样为 `[0,B*HV)`，每个 work 只处理一个 batch 的一个 value head。
每 tile 为 4 token，Q/K/V、beta 和可选 g 使用双输入 buffer；递推 State 使用三个
FP32 bank 轮换。MTE2 预取下一 tile 或下一 task 的初始 State，Vector 顺序推进 token，
MTE3 写回已完成 token 的 out 和 checkpoint。State 为 BF16 时额外使用一份 State IO
buffer，BF16 舍入不反馈到后续 token。

A5 使用静态 UB Tensor，显式 mutex 管理 MTE2/Vector/MTE3 所有权，不依赖
`--cce-auto-sync=on`。VF 按 tiling dtype 与融合属性模板化，单次调用完成 gate/beta
处理、两行 State 并行更新、output dot-product 和 State 写回。rank-1/rank-2 State
索引、accepted token、Q/K/V stride 和 State stride 均直接由统一 tiling 数据驱动。

A5 回归使用 223 条 ATK 矩阵，并使用 issue 679 的 36 个模型 shape 检查性能；最终
验收数据必须来自合并后的统一算子，不能沿用 standalone V2 的历史记录。
