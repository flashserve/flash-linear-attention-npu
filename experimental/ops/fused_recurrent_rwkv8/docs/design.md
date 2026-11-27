# FusedRecurrentRwkv8 实验迁移设计

方案设计规则版本：V2（既有实现迁移审阅；完整运行验收未完成）

## 1. 目标

以 [接口契约](api.md) 和 PR #446 `049eb0d` 的纯 CPU golden 为基线，将已有算子整体迁至
PR #954 的一级实验目录，保留原 A2 实现并增加 A5 编译目标。本轮不改变数学语义，不进行
性能优化，不把编译结果解释为 A5 运行资格。

调用链为独立 Python 包 → 实验 Stable ABI launcher → 绝对路径实验 `libcust_opapi.so` → aclnn → L0 →
host tiling → 纯 AIV kernel。实验入口、vendor、清单、安装目录均与稳定制品分离。
不复制稳定算子注册表，也不借稳定 GDR 的 availability 判断 RWKV8 能否调用。

已确认的差异：目录和测试导入路径、实验 Python 入口、A5 OpDef 配置、host UB 容量估计。
CPU golden、TilingKey、分核与递推顺序保持不变。A2/A5 共用现有计算实现，首轮 A5
编译已经通过，不需要 MicroAPI 重写。另修复 `chunk_len` int64→uint32 的明确溢出风险：
host 与 Python 拒绝越界，kernel 的 sane 检查拒绝 0；不改变合法范围内的计算。
编译诊断与验证结果记录于 [验证记录](../../../validation.md)。

## 2. Stage 0 完整详设

### Stage 0：Vector，逐 token 状态递推

沿用 #446 的纯 AIV 实现，`work=b*H+h`，一个逻辑 block 拥有该 head 的整个 state，
`blockDim=B*H`，key 为 0。输入/输出与状态公式由接口契约定义。这里的单 AIV 分工是
历史纯 Vector kernel；不引入 R01 的 AIC+双 AIV 协作，也不在编译迁移时改成 R20
grid-stride 或改变浮点归约次序。该历史分核策略在 A5 的执行与性能需后续验证。

令 `Kp=next_power_of_two(K)`，最低为 8；记 `A(x)=32*ceil(x/32)`。一份 state 在 UB 中
以 `S^T` 行主序驻留，行 j 是连续 V 个 FP32 元素；所有输入 Cast 到 FP32 后递推。
每 token 搬入 q/w/k/z/b 的 K 元素及 v 的 V 元素，计算 decay 与 q 的 scale；对 state×z
做 K 维成对折叠得到 sa，原位更新 state，再对 state×q 做相同折叠得到 o。输出按输入
dtype Cast，按 flag 写 sa 与快照；reverse 只改变 token 访问顺序。

以 K=V=128 为资源核算例，Kp=128；FP32 的 UB 分布如下，字节单位、半开区间，
TPipe 按声明顺序分配，各对象只有一份：

| UB 区间 | 对象 | 首次写入 / 最后消费 |
| --- | --- | --- |
| [0,65536) | state，128×128 FP32 | 初始化 / 最后 token 的 o 与快照 |
| [65536,131072) | mat，128×128 FP32 | 系数广播 / 当前归约或更新及 o 写回 |
| [131072,135168) | tmp8，128×8 FP32 | 列广播第一级 / 同次列广播第二级 |
| [135168,139264) | q,w,k,z,b,decay,e,qs，各 512 B | 当前 token 搬入或生成 / 当前递推消费 |
| [139264,140288) | v,sa，各 512 B | 当前 token 搬入或生成 / 更新与可选写回 |
| [140288,142080) | 低精度 staging，可选七份各 256 B | 搬入或输出 Cast / 对应 Cast 或 MTE3 |

通式为 `2*A(4*Kp*V)+A(4*Kp*8)+8*A(4*K)+2*A(4*V)`，低精度额外
`5*A(2*K)+2*A(2*V)`。原 #446 host 仍使用旧单 state 预算，本次修正为上述实际分配。
硬件 UB 上限由 `PlatformAscendC::GetCoreMemSize(UB)` 查询。最大需要 142080 B；
无 L1/L0、无 GM workspace、无跨核通信、无 ping/pong 或多个同时存活 work。

| Tensor | 每 work 的 GM 访问 | 驻留 / 复用 |
| --- | --- | --- |
| q,w,k,z,b | 每 token 各读一次，合计 5*K*sizeof(io) | 当前 token FP32 UB |
| v | 每 token 读一次，V*sizeof(io) | 当前 token FP32 UB |
| initial_state | 存在时读 K*V*4 B 一次，逐行 DataCopyPad | state 到最后 token |
| o | 每 token 写 V*sizeof(io) | mat 归约后写回，MTE3 完成后复用 |
| sa | flag 打开时每 token 写 V*4 B | sa 更新后到当前写回完成 |
| s | flag 打开且命中快照时写 K*V*4 B | state 逐行写回，不转置 |

同步沿用原实现：MTE2 输入就绪后 `PipeBarrier<PIPE_ALL>()`；低精度 Cast 后再次 barrier；
系数计算后 barrier；o/sa/s 写出前等待 Vector，低精度输出 Cast 后等待；token 末等待
MTE3 再复用 mat 与 staging。Vector 指令间采用原编译器同步，未关闭 auto-sync。
不存在跨核 flag/event、ready/free 或 slot 回绕；Pad 行初始化为 0，避免参与 K 折叠。
无空任务；非法 shape 由 host 拒绝，kernel 的 sane 检查为二次防御。

```text
init state and Kp-K pad rows
for token in requested direction:
    copy six inputs; wait MTE2; cast if needed
    compute decay and scaled q; wait Vector
    sa = pairwise_fold(state * broadcast(z))
    state = state * broadcast(decay) + broadcast(b) * sa + broadcast(k) * v
    o = pairwise_fold(state * broadcast(scaled_q)); wait Vector
    write selected sa/s; cast o if needed; write o
    wait MTE3 before next token reuses buffers
```

R21：保留原 FP32 `exp(-exp(w))` 次序、K 维成对求和与输出量化；不引入新的代数变换。
输入生成继续使用原归一化 kk、z=-kk、b=kk*a 约束，避免无约束状态指数发散。
极端值和非有限值的完整运行结果仍属于待补回归，不以静态分析代替精度门禁。
