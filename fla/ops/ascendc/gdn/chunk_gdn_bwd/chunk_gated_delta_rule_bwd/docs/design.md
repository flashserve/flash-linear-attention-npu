# BSND/TND 转置消除

规则版本：V2。本设计优化已有融合 Bwd 的数据搬运，公开接口和数学公式沿用
[算子说明](../README.md)。主线基线为 `fb8651651b1bfbe4d58a005d3e3b62c5d86f4e0f`。

## 布局边界

`layout` 同时决定 q/k/v 输入和 dq/dk/dv 输出。BSND/TND 每次调用消除这六个张量的
显式转置，两种路径合计覆盖十二处；不能把十二次计入单次调用。

以元素为单位，外部张量的地址为：

```text
head-major:     ((b*H+h)*T+t0)*D，token 行距 D
sequence-major: ((b*T+t0)*H+h)*D，token 行距 H*D
```

q/k/dq/dk 的 H 为 Hk，v/dv 的 H 为 Hv，D=128。地址计算使用 64 位。
packed 的物理 b=0，t0 为原序列起点加 localChunk*64；搬运行数不跨越该序列尾部。
GM 的跨 token 行距和数学矩阵转置分别处理，片上矩阵布局与乘加顺序保持一致。

dO、g、beta、betaRaw、dBeta/dG、rstd、state 和中间 workspace 各自沿用原布局。
尤其 rstd 仍为 BNS，不能从 sequence-major 的 q/k offset 推导地址。

## 分阶段实现

| 执行链 | 外部张量搬运 | 内部布局 |
| --- | --- | --- |
| arch22 RecomputeWUFwd | Vector 按布局读取 k/v | W/U 及 Cube 输入保持 head-major |
| arch22 ChunkBwdDvLocal | Cube 按布局读取 q/k | dO/g/dvLocal 保持 head-major |
| arch35 Intra | Cube 读取 q/k，Vector 读取 v | W/U/dvLocal 保持 head-major |
| FwdH | Cube 按布局读取 k，含 arch22 gated reuse 分支 | W/U/h/vNew 保持原布局 |
| DHU | Cube/Vector 按布局读取 q/k | dO/g/W/dvLocal/dh/dv2 保持原布局 |
| arch35 Finalize | 按布局读取 q/k/v，S6 写 dv，S15 写 dq/dk | 标量梯度与归约 workspace 保持原布局 |
| arch22 Finalize | dq 子流程读取 q/k，WY 读取 k/v 并写最终 dv | 内部 dq/dk/dkWy/dw 保持 head-major |
| arch22 A2FinalizeQkTail | 读取原 q/k，合并梯度及可选 L2Norm 后写最终 dq/dk | rstd 与内部梯度分别寻址 |

DHU 对空序列不执行递推，有初态时直接将对应 dht 写到 dh0；无 dht 时沿用清零结果。
两种状态布局均按相同物理布局搬运，复用原 q 输入、FP32 暂存和输出缓冲及事件协议，
不增加 workspace、事件或跨核同步。旧实现会将该区域留为零，此处按已有接口契约修正。

保留原 GVA 归约顺序、BF16 阶段落盘、指数换算和状态编号。仅 W/U 的空序列 metadata
继续压缩，其它阶段仍使用原序列编号。公开八个返回槽位不变。

内部追加默认 false 的 `qkv_sequence_major` 属性，并通过 L0 重载、tiling 和模板选择传递。
旧 L0 重载、ACLNN ABI 和独立子算子的默认行为保留。TilingKey 使用
`ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL` 和 `GET_TPL_TILING_KEY`。

## 资源与同步

继续使用已有 L1/L0/UB tile、buffer slot 和 ready/free 生命周期。跨步输入直接进入原 tile，
输出直接消费原 tile，不新增全张量 GM 中转、不新增跨核同步、不改变 EventID 或 Fixpipe 次数。
原 buffer 释放和发布位置在对应跨步搬运完成后执行。

`DataCopyExtParams` 的 GM 行间隔以字节为单位，UB 行间隔遵循平台 API 的块单位。
连续公开输出直接写入；arch22 非连续输出写同布局连续暂存，再通过 ViewCopy 回填。
不支持的 DMA 行间隔或不可表示的字节地址在 host 拒绝，并报告参数、实际值和期望。

## 验证方法

同平台、同一数学输入比较 head-major 与 sequence-major 的逐字节输出，覆盖四种布局、
T=1/63/64/65、G=1/2/3/4、多 batch、packed 空序列、状态布局、归一化、sigmoid、
混合标量 dtype、两种指数模式、非连续输入及可追踪数值。
公开 C ABI 另验证非法参数和 arch22 非连续输出的 storage offset。

独立精度使用 CPU FP64 golden、CPU FP32 benchmark 和 ATK 原生 L1 双标杆；保留原值域。
失败保存输出，并从最早偏离的 Stage 定位。测试入口见
[ATK 说明](../../../../../../../tests/atk/chunk_gated_delta_rule_bwd/README.md)。

性能用同一环境下的旧主线包和新包分别采集 msprof。完整 executor 时间取其计算与转换
task 的 `Task Duration(us)` 之和，排除输入构造；BSND/TND 核对六次转置消失，
BNSD/NTD 检查回归。逐平台测试结果以实际验收记录为准，不能从共享源码推导设备通过。
