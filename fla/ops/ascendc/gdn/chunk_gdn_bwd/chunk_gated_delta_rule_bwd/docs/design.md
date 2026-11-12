# BSND/TND 转置消除

规则版本：V2。本设计优化已有融合 Bwd 的数据搬运，公开接口和数学公式沿用
[算子说明](../README.md)。主线基线为 `56a21dc3c5d12d17e8045e26dfcd8dcf226b7ae8`。

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
| arch22/arch35 Intra | Cube 读取 q/k，Vector 在 arch22 读取 k/v、arch35 读取 v | W/U/dvLocal 保持 head-major |
| FwdH | Cube 按布局读取 k，含 arch22 gated reuse 分支 | W/U/h/vNew 保持原布局 |
| DHU | Cube/Vector 按布局读取 q/k | dO/g/W/dvLocal/dh/dv2 保持原布局 |
| arch35 Finalize | 按布局读取 q/k/v，S6 写 dv，S15 写 dq/dk | 标量梯度与归约 workspace 保持原布局 |
| arch22 Finalize | dq 子流程读取 q/k，WY 读取 k/v 并写最终 dv | 内部 dq/dk/dkWy/dw 保持 head-major |
| arch22 A2FinalizeQkTail | 读取原 q/k，合并梯度及可选 L2Norm 后写最终 dq/dk | rstd 与内部梯度分别寻址 |

A2/A3 与 A5 统一使用 Intra → FwdH → DHU → Finalize。Intra 原生处理 BF16/FP32 gate、
useExp2 和空序列，不增加自然指数转换或重编号桥接。保留各平台原 GVA 归约顺序、BF16 阶段
落盘、指数计算和状态编号，公开八个返回槽位不变。WU/DvLocal 独立算子的接口和实现沿用 main。

四个小算子的原 L0 接口末尾追加 `const char *layout = "BNSD"`，OpDef 追加同名
默认 `BNSD` 的可选字符串属性。主算子显式传入自身 layout；省略参数的原调用继续
使用 head-major。每个 L0 只保留一个声明和定义，不增加 v2 或重载。NTD/TND 使用
现有 B=1 的四维物理描述；host 将四种布局解析为枚举值并写入 tiling data。
layout 只控制原始 q/k/v 与最终 dq/dk/dv，其余张量沿用各自布局。公开 ACLNN C ABI
保持不变；L0 默认参数保留源码调用兼容，使用方需要重新编译。
TilingKey 使用
`ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL` 和 `GET_TPL_TILING_KEY`。
布局不增加 TilingKey 模板维度。arch22 Finalize 的 DQ/WY Cube、Vector 和最终
Q/K 写回使用 tiling data 中的运行时 layout 计算 GM offset 和行跨度。

## 资源与同步

本 PR 仅承担布局转置消除。Intra 的既有 UB 预留、FwdH 的共享 tiling/前缀表、
空序列状态与空子块搬运、GM 清零边界及 Finalize 模板键沿用基线，相关原有缺陷另行跟踪。
因此先前包含这些修复的内存/空序列验证结论不能作为当前版本的通过依据。

arch22 Intra 的每个 AIV 只持有 32 行 K/V 原始缓冲。连续和跨步搬运均将有效行数
限制为 `max(0, min(validTokens - part*32, 32))`，无有效行时不发起 DMA，避免首个半片
将完整 64 行 chunk 读入 32 行 UB。

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

本次独立测试快照使用 CPU FP64 golden、CPU FP32 benchmark 和 ATK 原生 L1 双标杆；
保留原值域和阈值。CPU 链依次计算 Intra 的 W/U/dvLocal、FwdH 的 h/vNew、
DHU 的 dh/dh0/dv2，再将 dv2 作为 Finalize 的 du 输入。
A2 的 Intra 先将缩放后的 K/V 及 KQ score 落盘 BF16；A5 先将列缩放后的 A
和带 gate 的 score 落盘 BF16，FP32 benchmark 分别保持对应边界。FP64 golden
不增加这些量化，DHU 的 gate 指数计算也保持 FP64。失败保存输出，并从最早
偏离的 Stage 定位。分支中的 ATK 文件保留既有基线；验证快照及其校验值随
测试记录单独归档。

性能用同一环境下的旧主线包和新包分别采集 msprof。完整 executor 时间取其计算与转换
task 的 `Task Duration(us)` 之和，排除输入构造；BSND/TND 核对六次转置消失，
BNSD/NTD 检查回归。逐平台测试结果以实际验收记录为准，不能从共享源码推导设备通过。

### 小算子 layout 的运行时传递

四个 L0 原型只保留一个声明，在末尾追加 `const char *layout = "BNSD"`；
原有省略参数的源码调用继续选择 BNSD。公开 ACLNN C 接口和 Python 签名沿用既有契约。
内部 L0 的返回类型、参数顺序及默认值单独记录在 `tests/stable_public_api.json`
的 `internal_l0_signatures`，由现有 Stable coverage 门禁一并校验。

host 将 BNSD、BSND、NTD、TND 分别解析为四个布局枚举值并写入 tiling data。
kernel 根据运行时布局值选择 Q/K/V 的 GM offset 和行跨度；内部 W/U/H、workspace
和 dO 的既有布局规则保持一致。布局不进入 `ASCENDC_TPL_ARGS_DECL`、
`ASCENDC_TPL_SEL`、`GET_TPL_TILING_KEY` 或 kernel/helper 的模板参数。
此前的 QKV 布尔属性、context 标记和布局模板维度全部移除。

## 参数错误上报

融合入口的公开参数校验、四个 L0 的 layout 校验及相应 tiling 的 layout 校验直接使用
`op_common/log/log.h` 的预定义 Error Message 宏，不增加日志包装函数或 guard 宏。
dtype 使用 `OP_LOGE_WITH_INVALID_INPUT_DTYPE`；rank 使用
`OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE` 或对应的具名参数宏；shape、属性及跨参数约束
选择对应的 `OP_LOGE_FOR_*` / `OP_LOGE_WITH_INVALID_ATTR`，报告实际值和期望值或具体原因。
`EZ0002`、`EZ0005`、`EZ0007` 等消息标识与 ACLNN 数值返回码分别验证；L0 原有
空指针返回、tiling 的 `GRAPH_FAILED` 和公开入口的既有返回语义保持一致。
此次只调整 host 错误诊断，不改变 kernel、TilingKey、workspace、布局寻址或计算公式。
