# 算子优化术语说明

本文解释本仓在 AscendC 算子开发和性能分析中常用的优化术语。这里的术语优先服务代码阅读、方案评审和 AI agent 排查，不替代 CANN 或芯片官方文档。

## A5 regbase 改写

A5 regbase 改写通常指在 A5 目标路径上，将热路径 vector 标量循环或普通 AscendC vector API 改成寄存器级 MicroAPI/regbase 形式，使数据显式进入 vector register 后再做连续计算。

判断是否已经完成时，不只看代码里是否出现 `RegTensor`、`MicroAPI` 或 `UpdateMask`，还要看热路径是否真的把 load、compute、store 的主体迁到 regbase。若只有少量辅助逻辑使用 regbase，而主要逐元素计算仍靠普通 `LocalTensor` vector 指令或标量 `GetValue/SetValue`，只能算部分完成。

做 regbase 改写时要同时检查 mask、tail、dtype cast、非对齐 load/store 和同步关系。A5 路径常常要求配合双发射设计，否则只是 API 替换，未必能释放 vector 吞吐。

## 双发射

本文中的双发射指在 vector regbase 热循环内，显式展开两组互相独立的 vector 计算流，让 load、计算和写回可以在更长的无依赖指令序列中交叠。典型形态是同一轮循环内存在 `offset0` 和 `offset1` 两个 unroll，分别使用独立的 mask、寄存器变量和中间结果，例如 `maskFull0/maskFull1`、`vX0/vX1`、`vY0/vY1`。

判断是否做了双发射，重点看三点：

- 是否有两个独立 unroll，而不是同一组寄存器连续复用。
- 两个 unroll 之间是否没有 RAW/WAW 依赖，mask、load cache、临时结果和 store 目标都能分开追踪。
- 指令顺序是否能形成 load/compute 的交叠，例如先准备两路数据，再分别执行同类计算，而不是完整做完 `offset0` 后才开始 `offset1`。

只把循环步长改成 2、复制一段代码，或在两个 unroll 中复用同一个 regbase 临时变量，都不应算真正完成双发射。tail 路径也要能回退到正确的一路或短 mask 处理。

## CV 通路

CV 通路是 CUBE 和 VEC 之间通过片上本地存储交换中间张量的通路，目标是避免中间结果绕 GM 或被迫拆成多个松散 kernel。分析时要写清楚方向和层级，不能只写“已使能 CV”。

常见方向包括：

- `L0C->UB`：CUBE 产生的 L0C 结果直接搬到 UB，供 AIV 做 scale、mask、activation、cast、清脏区等 vector 后处理。
- `UB->L1`：AIV 在 UB 上生成或修饰后的中间数据搬到 L1，供后续 CUBE/Catlass 作为左矩阵或右矩阵继续消费。
- `L1->UB`：已经在 L1 的中间数据或驻留数据搬到 UB，供 AIV 做 vector 修饰，再写回本地层级或输出。

判断是否使能 CV 通路，要确认 producer 和 consumer 都在片上本地层级闭环，并且中间结果没有先落 GM 再读回。若代码仍通过 workspace GM 串接两个阶段，只能算逻辑上融合，不能算 CV 通路完成。CV 通路还必须有清晰同步：CUBE/Fixpipe/MTE 写完后 VEC 才能读，VEC 写完后后续 MTE/CUBE 才能消费。

## 浅融合

浅融合指多个相关计算被放在同一算子、同一 aclnn L2 流程或同一 kernel 中，但调度粒度仍是阶段级的。比如需要计算 `Aqk` 和 `Akk` 两类矩阵时，浅融合可能是先把所有 chunk 的 `Aqk` 任务做完，再开始所有 chunk 的 `Akk` 任务。

浅融合通常能减少接口调用、减少显式中间输出或统一 workspace 管理，但它不一定能保持 chunk 内热数据。因为阶段之间存在全局或大范围边界，前一阶段读入的 Q/K/V、mask、gate 或中间 tile 可能已经离开 L1/UB，甚至从 L2 cache 中被替换。

## 深融合

深融合指在没有真实前后数据依赖的情况下，把原本分属不同阶段的计算下沉到同一个 chunk、tile 或 task 循环内完成。这里的“没有真实依赖”是关键：后一部分计算不需要以前一部分的输出作为输入，只是复用相同输入、相同 chunk 元信息或相邻中间数据。

仍以 `Aqk` 和 `Akk` 为例，如果同一个 chunk 内两者都需要读取相同的 K 或相同的元信息，并且二者结果互不依赖，就可以在处理该 chunk 时同时安排两部分计算，而不是等全部 `Aqk` chunk 完成后再做 `Akk`。这样可以进一步实现 L1/UB 驻留，也可以尽量让热数据在 L2 cache 中被连续消费，降低数据离开 L2 后再次从 GM 拉取的概率。

深融合不是无条件把代码塞进一个大循环。设计前要确认：

- 两部分计算之间没有必须等待的结果依赖。
- chunk、head、layout、tail 和 varlen 映射能在同一循环中一致表达。
- L1/UB/L0 资源足够，不会因为强行融合导致 tile 变小、double buffer 失效或同步更重。
- profiling 上主要瓶颈确实来自搬运、重复读写、L2 热数据流失或阶段边界等待。

如果后一阶段必须消费前一阶段的输出，这属于 producer-consumer 阶段融合，不应简单称为“无依赖深融合”。这种场景重点是 workspace slot、ready/free flag 和 CV 通路，而不是并列计算下沉。

## L1 驻留

L1 驻留指某个 chunk 或 tile 的张量搬入 L1 后，在同一个 chunk/task 生命周期内被复用多次，避免每次 matmul 或 vector 修饰前都从 GM 重新搬入。常见场景是一块 tensor 先作为第一次 matmul 的左矩阵或右矩阵，随后在同一个 chunk 内又作为第二次 matmul 的左矩阵或右矩阵。

只有存在复用才讨论 L1 驻留。一个 tensor 搬到 L1 后只被消费一次，即使路径经过 L1，也不应标成“已做 L1 驻留”；它只是正常搬运。是否完成驻留，不取决于使用 Catlass tail 层接口还是手写流水，只取决于热数据是否真的保留在 L1 并被后续计算复用。

检查 L1 驻留时要看：

- 同一块数据是否有至少两次本地消费。
- 两次消费之间是否没有写回 GM 再读回。
- L1 buffer 生命周期、double buffer 槽位和同步是否覆盖所有 tail、空 chunk 和 varlen 分支。
- 若通过 Catlass 实现，tail 层接口是否支持同一 L1 tile 在多个 matmul 之间保留；若手写实现，也要有等价的生命周期和同步保证。

## UB 驻留

UB 驻留指 AIV 热路径中的中间数据在 UB 内连续被多个 vector 阶段或本地搬运阶段复用，避免写出到 GM 或重复从 GM/L1 拉入。典型例子是 gate、scale、mask、exp、cast 等连续 vector 操作共用同一批 UB 数据，或 AIV 生成的中间结果直接通过 `UB->L1` 进入后续 CUBE 路径。

判断 UB 驻留时，重点看数据所有权和生命周期。使用 `LocalTensor` slice、`TBuf` 或手工 UB ring buffer 时，必须能说明哪个阶段写、哪个阶段读、什么时候可以复用。`PipeBarrier<PIPE_V>()` 只能约束 V pipe 内顺序，不能替代 MTE/V、V/MTE 或 CUBE/VEC 之间的硬事件。

UB 容量较小，强行驻留可能挤掉 double buffer 或让搬运粒度变碎。因此 UB 驻留要和 profiling 结合判断：若算子主要是 VEC bound 或 MTE bound，并且热数据存在多次本地消费，收益通常更明确；若 CUBE bound 明显，单纯扩大 UB 驻留未必是优先项。

## 热数据和 L2 cache

热数据指短时间内会被多个阶段、多个 matmul 或多个 vector 片段重复读取的数据，例如同一 chunk 的 Q/K/V、gate、mask、局部 score 或 layout 转换后的 tile。L2 cache 不是显式可编程的 scratchpad，但调度顺序会影响热数据是否还留在 cache 中。

深融合、L1 驻留和 UB 驻留都可以看作让热数据被更近、更早地复用。若把所有 chunk 的一个阶段做完再进入下一阶段，热数据更容易从 L1/UB 生命周期中结束，也更容易在 L2 中被后续任务替换；若在 chunk 内连续完成多个无依赖计算，热数据命中 L2 或保留在 L1/UB 的机会更大。

## Bound 与优化收益

评估某个优化特性时，要先看算子主要 bound：

- MTE bound：优先考虑深融合、L1/UB 驻留、CV 通路、搬运合并、减少 GM 中转和改善 layout 连续性。
- VEC bound：优先考虑 A5 regbase 改写、双发射、增大 vector repeat 粒度、减少 scalar `GetValue/SetValue` 和减少无意义 cast。
- CUBE bound：优先考虑 tile shape、Catlass 路径、L0/L1 复用和矩阵计算排布；单纯做 regbase 或 UB 驻留收益可能有限。
- wait bound：优先检查 AIC/AIV 生产消费距离、cross-core flag、MTE3 写回、double buffer 和 slot 复用协议。

人力有限时，建议以算子为颗粒度评估收益。一个算子如果要动，通常应把它涉及的 regbase、双发射、CV 通路、L1/UB 驻留和深融合作为一个优化包评估，而不是孤立地只做单个特性。收益最高的目标通常是能缓解当前主 bound、且能让多个热路径共享同一次重构的算子。
