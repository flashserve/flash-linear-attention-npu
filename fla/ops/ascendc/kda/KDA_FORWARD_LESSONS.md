# AscendC 融合算子 AI Agent 开发踩坑与注意手册

本文档面向后续开发 AscendC 融合算子的工程师或 AI agent。KDA forward 是本文的主要实践样本，但结论不局限于 KDA：凡是同时涉及 L2/L0 拼接、AIC cube、AIV vector、workspace、layout、cross-core flag、数值稳定性和性能 profiling 的融合算子，开发前都应该先读一遍。

这不是普通变更说明，而是把一次复杂算子开发中反复出现的方向性、结构性、同步、精度、性能和验证问题沉淀成可复用经验。文档中的 KDA 例子用于解释问题，不代表这些规则只适用于 KDA。

## 1. 总原则

复杂融合算子不是一个 kernel 里随手写循环就能做好的功能拼接。开始写代码前，先拆清楚三类问题：

- 数学语义：三方对标实现或论文/公式定义了“应该算什么”和“输出契约是什么”。
- NPU 转写：本仓已有模块定义了“在 AscendC 上应该怎么组织 cube/vector/搬运/同步”。
- 工程边界：dtype、layout、workspace、异常拦截、文档、测试和 PR 交付必须一致。

推荐开发顺序：

```text
需求目标
  -> 三方对标语义
  -> 本仓 NPU 可复用模块
  -> 数据依赖图和并行边界
  -> L2/L0 分层设计和 workspace 规划
  -> cube/vector 分工与流水协议
  -> 单算子小 shape 精度
  -> 组合路径目标 shape 精度
  -> 特殊值/极端值域/尾块/变长
  -> msopprof 性能定位
  -> 回归用例和文档同步
```

不要反过来做。尤其不要在精度第一现场还没定位时就调性能、改阈值、缩小输入 range，或者把 cube 计算搬到 vector/scalar 上兜底。

## 2. 必须区分的输入来源

### 2.1 三方对标实现

三方对标实现用于回答“数学上应该是什么”：

- KDA forward 的公式。
- `safe_gate` 的 raw gate 到 `gk` 的转换。
- 返回 tuple 的顺序和中间量命名。
- `final_state` 固定 `fp32` 的语义。
- `initial_state=None` 时的行为。

不要把三方 Triton 的代码结构直接照搬到 NPU。它提供的是公式和对标输出，不提供 AscendC 的最优流水。

### 2.2 本仓 NPU 可参考模块

本仓已有实现用于回答“AscendC 上应该怎么转写”：

- GDN `ChunkGatedDeltaRuleFwdH`：跨 chunk 状态传播和 `h/final_state` 语义。
- GDN `solve_tri`：三角矩阵求逆、MCH/MXR 类 blocked solve 的思路。
- Catlass cube GEMM：矩阵主路径。
- causal conv layout：BSND/TND 兼容输入到 BNSD/NTD 性能布局的边界设计。
- 现有 cross-core flag 封装和 AIC/AIV 分阶段协议。

### 2.3 开发者约束

开发者输入通常决定“不能走哪些歧路”：

- 目标矩阵计算必须用 cube/Catlass。
- AIV 不做 scalar 热路径，不用 `GetValue/SetValue` 写主流程。
- 数据搬运用 `DataCopy/DataCopyPad`，尽量成行成块搬运。
- 有数据依赖的计算尽量驻留 UB，减少 GM 往返。
- cube 核和 vector 核要尽量并行，不能人为串行化。
- 精度问题要用 CT dual/CT viz 或逐阶段输出定位，不能只看最终 pass/fail。
- 性能结论以 `msopprof` 为准，Python wall time 只能作为粗略体感。

## 3. 融合算子设计哲学

### 3.1 先画数据依赖图，再谈融合

融合不是把所有代码塞进一个 kernel。正确的第一步是把计算拆成：

```text
无数据依赖的大并行部分:
    按 batch/head/chunk/tile 最大并行度切分。
    优先交给 AIC cube 或 AIV 大块 vector。

有串行依赖的状态传播:
    单独抽成阶段或专门 kernel。
    通过更细粒度的任务划分、负载均衡和 workspace 驻留优化。

只负责格式或边界的辅助部分:
    L2 里拼 L0、ViewCopy、Cast、LayoutSwap。
    不要污染核心计算 kernel 的热路径。
```

KDA 的实践例子：

```text
chunk 内 Aqk/Akk/qg/kg/w/u:
    无跨 chunk 依赖，应该按 chunk/head 并行，矩阵主路径走 Catlass cube。

h_next/final_state:
    有跨 chunk 依赖，复用 GDN fwd_h 的状态传播思路，避免拖慢 chunk 内大并行计算。

layout/cast/output copy:
    在 aclnn L2 层串 L0 辅助算子，边界清晰。
```

### 3.2 cube/vector 分工要从硬件算力出发

设计时先问三个问题：

```text
这是矩阵乘、矩阵求逆、矩阵三角 solve 吗？
    是 -> 优先 cube/Catlass/blocked solve。

这是逐元素激活、scale、mask、clamp、exp、cast、pad 吗？
    是 -> AIV vector，大块 DataCopyPad + repeat。

这是少量标量元数据、shape、offset、cu_seqlens 吗？
    是 -> 可以标量读取，但不能进入热路径内层循环。
```

硬件上 cube 算力远高于 vector。目标矩阵计算即使“不完整”“带脏数据”“需要 mask”，也应该优先想办法 pad/clean 后继续用 cube，而不是退回 vector 或 scalar。

### 3.3 搬运效率决定 vector 上限

搬运设计的基本原则：

- 输入 layout 尽量让热路径读写连续。
- 一次搬运尽量覆盖整行、整 tile 或至少多个 cache line。
- `DataCopyPad` 用来处理尾部和对齐，不要因此把 full chunk 路径拆成逐元素搬运。
- vector 指令使用 repeat time 处理大块数据，避免在 `for d` 内发大量小 VEC 指令。
- double buffer 必须真的让 MTE 和 VEC/CUBE 重叠；只定义 ping-pong 变量但每步都 wait，流水仍然是串行。

### 3.4 有脏数据不等于必须小颗粒度切分

尾块、mask、变长序列场景里，常见错误是为了避开脏数据，把 full tile 切成很多小循环。正确思路通常是：

```text
读入完整 tile
  -> 对无效区 pad 成中性值
  -> cube/vector 照常大块计算
  -> 写回时 DataCopyPad 或按有效区搬出
```

只有在公开语义要求中间量无效区精确清零时，才需要额外 clean；即便如此，也应尽量用 vector 大块清理。

### 3.5 编译期模板化和运行时 tiling 分工

模板参数用于消除热路径分支：

```cpp
template <typename T, bool SAFE_GATE, int V_DIM>
class Kernel;
```

运行时 tiling 数据用于规模、offset、workspace 和调度：

```cpp
tiling.batch
tiling.seqLen
tiling.layout
tiling.dataType
tiling.safeGate
```

推荐方式参考 GDN：host tiling 写入 dtype/属性字段，kernel 入口按字段选择模板实例；模板内部使用 `if constexpr` 裁剪路径。不要为了每个 dtype/属性组合滥用 tiling key，除非算子框架或模板 registry 明确要求这样做。

## 4. KDA 实例的能力边界

可以声明：

- KDA forward 正向路径。
- `K=128, V=128/256, chunk_size=64/128` 的 `fp16/bf16` 场景。
- BSND/TND 兼容 layout，以及 BNSD/NTD 性能 layout。
- `gk` 外部输入路径。
- `KdaGateCumsum` safe-gate raw path 的基本语义。
- `initial_state=None` 和传入 `initial_state` 的预留/透传语义。
- `final_state` 为 `fp32`。

不能擅自声明：

- KDA backward 已完成。
- 高 `K/V` 非 chunk 对齐变长序列已优化。
- 所有中间量无效区都有公开语义。
- sanitizer 已覆盖 race/mem/init/sync，除非实际跑过并确认命中 sanitizer kernel。

## 5. 设计阶段常见错误

### 5.1 把接口拼接误当融合算子

错误方向：

```text
Python/PyTorch 层调用多个已有 torch op 拼出 KDA
```

问题：

- 功能看似串起来了，但核心计算不在 AscendC 融合 kernel 内。
- 性能边界不成立，PR 范围也不清晰。
- 无法在 AscendC L2/L0 层控制 workspace、layout、dtype 和同步。

合理方案：

```text
PyTorch API
  -> aclnn L2
      -> KdaLayoutSwap12 / Cast
      -> ChunkKdaFwd stage 1
      -> ChunkKdaFwd stage 3
      -> ChunkGatedDeltaRuleFwdH
      -> ChunkKdaFwd stage 2
      -> ViewCopy / KdaLayoutSwap12
```

L2 可以拼 L0，但核心矩阵准备、求逆、post-WU、output 都必须在 AscendC L0 算子中完成。

### 5.2 不拆 chunk 内并行和 chunk 间依赖

KDA 有两类完全不同的依赖：

```text
chunk 内:
    Aqk, Akk, qg, kg, w, u
    可按 chunk/head 并行

chunk 间:
    h_next, final_state
    依赖前一 chunk 状态
```

错误做法：

- 把所有计算放进一个串行循环。
- 或者把有依赖的 `h` 状态传播当作普通 chunk 并行任务。

合理做法：

- `ChunkKdaFwd stage 1/3` 处理 chunk 内可并行项。
- 复用 GDN `ChunkGatedDeltaRuleFwdH` 串起状态传播。
- `ChunkKdaFwd stage 2` 在 `h/v_new` 可用后计算 `o`。

### 5.3 把 `kg` 和 `gk` 混淆

命名语义：

```text
gk: log2 空间下 key gate 累积值
kg: key-gated k 中间张量
```

`kg` 不是 `gk` 的笔误。kernel 中常见语义为：

```text
qg = q * exp2(gk)
kg = k * exp2(-gk)
```

状态传播结合 chunk 的 `g_last` 后等价于：

```text
kg_state = k * exp2(g_last - gk)
```

文档、接口和测试里第一次出现缩写时必须写清楚语义，否则后续 review 很容易把 `kg` 当成拼写错误。

### 5.4 对 `useSplitForward` 的误解

`useSplitForward=true` 不是为了“多写几个 stage 显得复杂”，而是为了主流 `bf16, K=128, V=128, chunk_size=64` 场景：

```text
q dtype != fp32
chunk_size == 64
K * V >= max(4 * 64 * 64, 64 * (K + V))
```

该路径可以复用 cube 主路径和 GDN 状态传播。不满足模板的 shape 应在 host 层明确拦截，不能走 `stage=0` scalar 路径兜底。

## 6. AscendC 编码红线

### 6.1 目标矩阵计算必须走 cube

错误模式：

```cpp
for (uint64_t i = 0; i < curT; ++i) {
    for (uint64_t j = 0; j < curT; ++j) {
        float acc = 0.0f;
        for (uint64_t d = 0; d < K; ++d) {
            acc += q[i][d] * k[j][d] * gate[i][j][d];
        }
        Aqk[i][j] = acc;
    }
}
```

现象：

- 小 shape 能过，目标 shape 性能完全不达标。
- `msopprof` 看到 scalar/VEC 占比异常，AIC cube 利用率低。
- 优化空间越来越小，因为一开始就选错主路径。

正确方向：

- AIV 准备 `qg/kg/w seed`。
- AIC 使用 Catlass GEMM 计算 `Aqk/Akk/post-WU/output`。
- 非目标 fallback 可以保 correctness，但不能作为目标路径。

### 6.2 AIV 永远不要在热路径做大批 scalar

错误模式：

```cpp
float x = tensor.GetValue(idx);
float y = Exp2Scalar(x);
tensor.SetValue(idx, y);
```

正确模式：

```cpp
DataCopyPad(rowUb, gm[offset], rowParams, padParams);
SetFlag<HardEvent::MTE2_V>(event);
WaitFlag<HardEvent::MTE2_V>(event);

Muls(tmpUb, rowUb, LN2, count);
Exp(tmpUb, tmpUb, count);
Mul(outUb, inUb, tmpUb, count);

SetFlag<HardEvent::V_MTE3>(event);
WaitFlag<HardEvent::V_MTE3>(event);
DataCopyPad(gmOut[offset], outUb, outParams);
```

### 6.3 不要用 scalar fallback 修精度

精度偏差的正确定位顺序是：

```text
检查输入语义
  -> 检查 layout/offset
  -> 检查 mask/无效区
  -> 检查 exp/gate 值域
  -> 检查 solve/矩阵路径
  -> 检查同步和 UB 生命周期
```

错误方向是把 cube 矩阵计算搬回 vector/scalar，因为这样通常只是绕开了脏数据、mask 或同步问题，同时毁掉性能。

### 6.4 `inf * 0` 不是 0

mask 不能简单依赖乘 0：

```cpp
masked = expVal * mask;  // expVal=inf, mask=0 时可能得到 nan
```

合理方向：

- 在指数前 clamp 到对等语义的安全范围。
- 或在 mask 前先把非法/非有限值过滤到 0。
- 对照 GPU/Triton 的 `exp2` 或 safe gate 语义，不要自己发明不同的数值范围。

## 7. 同步和 UB 生命周期踩坑

### 7.1 `MTE3->MTE2` 不能保护 VEC 复用

真实问题：

`KdaGateCumsum` 曾在每行 `acc` 写回 `gk` 后只做：

```cpp
DataCopy(gk[t], acc, k);
SetFlag<HardEvent::MTE3_MTE2>(event);
WaitFlag<HardEvent::MTE3_MTE2>(event);
```

这只能说明 MTE2 不会立刻复用同一 UB，并不能说明 VEC 不能复用。下一 task 开始时：

```cpp
Duplicate(acc, 0.0f, k);
```

可能覆盖 MTE3 仍在读取的 `acc`。

修复：

```cpp
DataCopy(gk[t], acc, k);
SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
SetFlag<HardEvent::MTE3_V>(mte3ToVEvent);
WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent);
```

现象特征：

- 不是所有行都错，而是某些 chunk 的最后一行错。
- 常见坏值为 0，因为下一 task 初始化把 `acc` 清零。
- 只发生在同一个 core 需要继续处理下一个 task 的场景。
- 短 shape 可能完全测不出。
- 下游表现为 `g_i - g_j` 正跳几百，`Aqk/Akk/w/kg` 爆成极大值或 NaN/Inf。

为什么 `T=16384 BSND` 可能没测出：

- 如果当时测的是 `chunk_kda_fwd` 主算子，`gk` 已经作为外部输入传入，并没有走 `KdaGateCumsum safe_gate`。
- 如果测的是普通小 `g_step`，尾行错成 0 也只是很小误差，不会引发 `exp2` 爆炸。
- 如果只看 `o/final_state` 或性能，没有逐元素检查 `gk` chunk 尾行，就会错过第一现场。

回归用例要求：

```python
raw = torch.randn(1, 1536, 2, 128, dtype=torch.bfloat16, device="npu")
gk = torch.ops.npu.npu_kda_gate_cumsum(
    raw, 64,
    A_log=a_log,
    dt_bias=dt_bias,
    use_gate_in_kernel=True,
    safe_gate=True,
    lower_bound=-5.0,
)
ref = cpu_safe_gate_chunk_cumsum(raw, a_log, dt_bias, 64)
assert_close(gk.cpu(), ref, rtol=2e-3, atol=2e-3)
```

关键不是 `T` 必须很长，而是：

- task 数要大于可用 AIV core 数，让同一个 core 连续处理多个 task。
- `K=128` 覆盖目标行宽。
- `safe_gate=True` 覆盖大负累积值域。
- 对比对象必须是 `gk` 本身。

### 7.2 VEC 临时复用后要闭合生命周期

曾出现过类似模式：

```cpp
Broadcast(tmpUb, gkScale, ...);
Mul(calcUb, calcUb, tmpUb, ...);

// 后面又把 tmpUb 当作 MTE2 目的地址
DataCopy(tmpUb, gmInput, ...);
Add(tmpUb, calcUb, tmpUb, ...);
```

如果中间没有 `V_MTE2` 事件闭合，MTE2 可能覆盖 VEC 尚未完成的临时值，或 VEC 读到 MTE2 尚未写完的数据。

规则：

- 同一 UB slot 每次换 owner，都要有对应事件。
- `PipeBarrier<PIPE_V>()` 只约束 V pipe 内顺序，不替代 MTE/V 之间的硬事件。
- 双缓冲时，每个 slot 的 free/ready 生命周期必须闭环。

### 7.3 Cross-core flag 必须计数平衡

错误模式：

```text
AIV: 只有有效 row 才 set ready
AIC: 对所有 tile 都 wait ready
```

结果：

- partial chunk 或变长序列下 timeout。
- 小 dense case 正常，大 shape 或随机 cu_seqlens 卡死。

正确做法：

- 通过 tiling 保证 AIC/AIV 参与核数量匹配。
- 无有效数据的 subblock 如果被 AIC 等待，也必须发送空 payload 的 ready，或在 tiling 中让 AIC 不等待它。
- flag id 需要集中管理，不要散落魔法数字。
- 不允许 producer 连续 set 同一个 flag 而 consumer 还没 wait，必要时用 reverse/free flag。

### 7.4 元数据不能按元素走热点 MTE 流水

异步 AICore 错误经常在后续第一个同步点才暴露。例如测试代码在 `_stat()` 中调用 `torch.npu.synchronize()` 时看到 507014，并不表示统计函数有问题；应先定位同步点之前最后下发的 kernel 和 stage。

一种高风险写法是对每个 chunk 反复搬运单个 `int64` 元数据：

```cpp
for (uint64_t task = coreIdx; task < taskCount; task += coreNum) {
    DataCopyPad(metaUb, chunkIndicesGm[task * 2], eightBytes, pad);
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    SetFlag<HardEvent::V_S>(scalarEventId);
    WaitFlag<HardEvent::V_S>(scalarEventId);
    // 再从 UB 取 seq/localChunk，继续当前 task。
}
```

这种实现即使没有数学错误，也会产生以下问题：

- 每个 task 为 8 字节数据下发一次 `DataCopyPad`，有效载荷远小于 MTE cacheline。
- 元数据读取把 MTE2、VEC 和 Scalar 事件串成细粒度控制链，长序列下指令和事件数量随 chunk 数线性增长。
- 调试时常表现为小 shape 正常，长序列在 AIV timeout；异常在 Python 后续同步处才被观察到。
- 为修复 timeout 继续叠加 barrier 只会扩大串行区，不能解决搬运粒度和元数据归属错误。

更合理的方案是把不变控制信息留在 host/tiling 边界：

```text
L2:
  校验 cu_seqlens/chunk_indices
  -> 规范化为 (seq_id, local_chunk)
  -> chunk_map = (seq_id << 16) | local_chunk
  -> 同时保存每条序列的 seq_start/seq_end

Kernel:
  packed = tiling.chunk_map[flat_chunk]
  seq = packed >> 16
  local_chunk = packed & 0xffff
  start = tiling.seq_start[seq] + local_chunk * chunk_size
  end = min(start + chunk_size, tiling.seq_end[seq])
```

这里的 scalar 只用于任务索引和 offset 推导，不承担矩阵或向量数值计算。容量必须由 tiling data/tiling stack 的实际上限推导并在 Python、L2 和文档三处统一拦截；超过容量时按完整序列边界拆分请求。

定位这类问题时建议按以下顺序缩小范围：

1. 开启 launch blocking 或在 L2 增加仅用于调试的 stage stop，确定首个出错 kernel。
2. 固定输入多次运行，区分确定性死等、未初始化读取和随机 race。
3. 检查热点循环里的小块 `DataCopyPad`、事件对和跨核 ready/free 计数。
4. 用 memcheck/synccheck 前确认实际对象带 sanitizer 符号，日志出现 `Start ... sanitizer on kernel ...`。
5. 修复后回到原长序列、原 layout、原 dtype 和原调用链验证，不能只用缩小后的定位 shape 收尾。

## 8. Gate、指数和数值范围

### 8.1 `gk` 合法值域

safe gate 下：

```text
gate = lower_bound * sigmoid(x), lower_bound=-5
gate <= 0
gk = cumsum(gate * 1/ln2, within chunk)
```

所以同一 chunk 内：

```text
gk[t+1] <= gk[t]
g_i - g_j <= 0 for causal i >= j
```

如果看到：

```text
step_max = max(gk[t+1] - gk[t]) >> 0
```

通常说明：

- `gk` 某行没写对。
- layout/offset 错。
- UB 写回被覆盖。
- chunk 边界错。

### 8.2 先查 `gk`，再查 `Aqk/Akk`

下游爆炸链路：

```text
gk tail row = 0, expected = -367
  -> g_i - g_j = +367
  -> exp2(g_i - g_j) huge
  -> Aqk/Akk/w/kg huge
  -> solve/output/state 出 NaN/Inf
```

如果最终看到的是 `o/final_state` NaN，不要直接改 `ChunkKdaFwd` 或 solve。先做：

```python
print(gk.isfinite().all())
for each chunk:
    print(max(gk[1:] - gk[:-1]))
compare(gk, cpu_gate_reference)
```

## 9. Layout 和接口边界

### 9.1 BNSD/NTD 是性能 layout

如果上游 causal conv 已经转成 BNSD/NTD，KDA 内部不应再做 `KdaLayoutSwap12`。

现象：

- `msopprof` 中 layout swap 占比大。
- 大 head 场景性能被搬运吃掉。

正确策略：

```text
BSND/TND: 兼容输入，L2 转 BNSD 内部 layout
BNSD/NTD: 性能输入，L2 reshape/view 后直通 kernel
```

### 9.2 `return_intermediate=True` 要明确有效区语义

中间量不是只有 shape 对就完事。需要明确：

- `Aqk/Akk/w/u/qg/kg/v_new/h` 的公开 layout。
- 无效 token、padding 行、partial chunk 脏区是否需要清零或屏蔽。
- `h` 的 chunk 维顺序。
- `g` 返回的是 `gk` 累积 gate，还是 raw gate。

如果中间量无效区出现极值，不要直接判定主输出错误。先确认公开语义是否要求比较该区域。

### 9.3 用户输出不能随便当 split path 中间量

错误模式：

```cpp
ChunkKdaFwd(..., userOutO, userOutAqk, ...);
Muls(userOutAqk, scale);
ChunkGatedDeltaRuleFwdH(... userOutKg, userOutW, ...);
```

问题：

- 用户输出和 executor-owned 临时 tensor 在 L0 链里行为不同。
- 可能触发 tiling/workspace 推导异常。

正确模式：

- split path 内部全部用 executor-owned BNSD temporaries。
- 最后一步 `ViewCopy` 或 layout swap 回用户输出。

### 9.4 L0 读取必须注册为输入

错误模式：stage 1 把 `Akk/w_pre/v_new` 写到某个输出槽，stage 3 又把同一地址只作为“输出”传入并在 kernel 内读取。代码地址看似连通，但 executor 看不到 stage 3 对这些张量的读依赖，可能在 `return_intermediate=False` 时提前复用 workspace。

典型现象：

- `return_intermediate=True` 精度正常，改成 `False` 后出现整块结构性误差。
- 同一 kernel 单独运行正常，L2 拼接后错误。
- 增加导出或调试 copy 后问题暂时消失。

正确做法：

```text
stage 1 outputs: w_pre, Akk, v_new_seed
stage 3 inputs:  stage_qg=w_pre, stage_aqk=Akk, stage_v_new=v_new_seed
stage 3 outputs: w, u, kg
```

任何 kernel 读取的 GM tensor 都必须是该 L0 的 `OP_INPUT`，写入的 tensor 必须是 `OP_OUTPUT`。需要原地语义时，也要让图构建层看见真实的读写依赖，不能靠地址别名或导出选项维持生命周期。

### 9.5 连续分核不能用 `subBlockIdx` 代替有效行判断

两个 AIV subblock 按连续范围切分 `curT` 行时，不能写成：

```cpp
if (subBlockIdx >= curT) {
    return;
}
```

例如 `curT=1, subBlockNum=2` 时，连续切分可能得到 AIV0 的 `[0, 0)` 和 AIV1 的 `[0, 1)`；上述判断会让真正持有唯一有效行的 AIV1 退出，造成尾块 `qg/w` 整行未生成。典型现象是只在 `T % chunk_size = 1` 或相似极短尾块出现结构性误差，完整 chunk 全部正常。

正确方式是先校验 `subBlockIdx < subBlockNum`，再计算：

```cpp
rowBegin = curT * subBlockIdx / subBlockNum;
rowEnd = curT * (subBlockIdx + 1) / subBlockNum;
if (rowBegin == rowEnd) {
    // 无数值工作，但仍按协议完成需要的 ready/free 握手。
}
```

对所有 `1..chunk_size-1` 尾长做回归，尤其覆盖 1、跨 16/32/64 tile 边界及 `chunk_size-1`。

### 9.6 多槽流水必须同时约束数据和 credit

只做 `producer set ready -> consumer wait ready` 不足以保护 ring buffer。生产者可能在消费者尚未读完时绕回并覆盖同一 slot。可靠协议应为：

```text
producer wait free(slot)
producer write payload(slot)
producer set ready(slot)
consumer wait ready(slot)
consumer read/compute payload(slot)
consumer set free(slot)
```

队列深度、flag 数量和 workspace slot 数必须一一对应。任务尾部要把握手次数补齐或显式 drain，后续阶段复用 flag id 前要证明上一协议已经完全排空。workspace 应按 core 数和固定队列深度分配，避免随 token/chunk 数线性增长。

## 10. 精度定位流程

### 10.1 不要只看最终输出

KDA 的逐阶段定位顺序：

```text
1. KdaGateCumsum: gk
2. Stage 1: qg/kg/Aqk/Akk/w/u
3. Stage 3: post-WU 后的 w/u/kg
4. GDN fwd_h: v_new/h/final_state
5. Stage 2: qg @ h, Aqk @ v_new, o
```

每一步都应该能单独 dump 或以 `return_intermediate=True` 验证。

### 10.2 结构性错误和数值误差要分开

结构性错误特征：

- 误差集中在固定行、固定 chunk 尾行、固定 head。
- 输出有明显块状、条纹状、周期性。
- 正负号或维度映射整体错。
- 多次固定输入运行结果不一致。
- 出现 NaN/Inf 或极大值。

数值误差特征：

- 误差随机分散。
- CT dual 显示 test 和 benchmark 在同一数量级。
- 没有固定 layout/offset 模式。

结构性错误必须修 kernel。数值误差可用 dual benchmark、MCH/MXR 迭代次数、fp32 workspace 等方式评估取舍。

### 10.3 CT 工具使用

大 tensor 直接画图很慢，可以 sampling：

```bash
ct viz ... --sc 4096
```

但如果 kernel 没有产出数据，问题不是 CT 慢，而是：

- kernel timeout。
- op launch 失败。
- cross-core flag 死等。
- 内存越界或同步缺失。

双标杆用来判断相对 NPU 是否劣于可接受 benchmark：

```text
golden:    CPU/Triton 高精度参考
benchmark: CPU/Triton fp32 中间计算并 cast 到目标 dtype
test:      NPU 输出
```

不要把 `diff_thd` 调大来制造通过。阈值和双标杆语义必须和用例规格一致。

### 10.4 固定输入多跑

怀疑同步、UB 生命周期或 race 时，固定随机种子多跑：

```text
same input, same shape, run N times
compare bitwise
```

如果结果不一致，优先怀疑：

- 缺少跨 pipe event。
- cross-core flag 不平衡。
- UB slot 被提前复用。
- GM workspace 写区重叠。

## 11. 性能定位流程

### 11.1 性能结论以 msopprof 为准

Python wall time 包含：

- Python 调度。
- torch extension 初始化。
- op graph 构建。
- 同步位置差异。

KDA 性能结论必须看 `msopprof`：

```text
OpBasicInfo
PipeUtilization
AIC/AIV wait
MTE2/MTE3/VEC/CUBE 占比
```

### 11.2 先看 bound，再改代码

常见 bound 和对应方向：

```text
ScalarBound:
    有 GetValue/SetValue 或逐元素循环，必须向 vector/cube 改。

MTE2Bound:
    搬运太碎、重复读、layout 不连续、DataCopyPad 粒度小。

VECBound:
    AIV 做了过多本应 cube 做的矩阵工作，或 vector 指令 repeat 粒度太小。

AIC wait:
    AIC 等 AIV 准备，检查流水、flag、producer-consumer 队列。

AIV wait:
    AIV 等 AIC 或 MTE，检查 cube 输出、MTE3 写回和双缓冲。
```

### 11.3 不要在 for 循环里塞大量小指令

坏模式：

```cpp
for (d = 0; d < K; ++d) {
    DataCopyPad(one_element);
    Exp(one_element);
    DataCopyPad(one_element);
}
```

好模式：

```cpp
DataCopyPad(rowUb, rowGm, rowBytes);
Exp(rowUb, rowUb, repeatCount);
Mul(outUb, rowUb, otherUb, repeatCount);
DataCopyPad(outGm, outUb, rowBytes);
```

原则：

- 一次搬运尽量覆盖整行或长 tile。
- 使用 repeat time 让 vector 指令处理大块数据。
- double buffer 要让搬运和计算能互相掩盖，而不是只声明了 ping-pong 变量。

## 12. Partial Chunk 和 Varlen

partial chunk 是当前最容易出错的区域。

危险点：

- `curT < chunk_size` 时，cube tile 仍可能读取完整 64x64。
- 无效行如果没 pad 成中性值，会进入 `Aqk/Akk/solve/output`。
- AIV 跳过无效任务，但 AIC 仍等待 flag。
- `chunkSlot` 和 `globalTokenStart` 混用，导致 scratch/h 写错槽。

正确设计要求：

```text
globalTokenStart: 用于 q/k/v/gk/beta/o 读写
chunkSlot:        用于 h/scratch/final_state compact chunk 槽
curT:             当前 chunk 有效 token 数
```

partial chunk 的选择：

1. 用 dedicated partial path，显式 pad/clean 到完整 tile，并保持 AIC/AIV flag 平衡。
2. 或用 vector correctness path，但仅用于非目标性能场景，不能污染目标 full chunk 路径。

不要为了尾块正确，把 full chunk 路径也改成逐元素 scalar。

## 13. MCH/MXR/MXH 求逆经验

`Akk` 是：

```text
Akk = inv(I + tril(k_i * k_j * exp2(g_i - g_j) * beta_i, -1))
```

它不是普通 elementwise 修补能解决的对象。经验：

- MCH 迭代次数影响精度和性能。迭代少性能好但可能放大小值域误差；迭代多性能差且在部分矩阵上也可能不稳定。
- MXR/MXH 的收益来自 L0C double buffer、本地驻留和指令流水，而不是仅替换公式名字。
- 如果引入 MXR/MXH，必须同时看 `Akk`、`sampled_o` 和 `final_state`，不能只看 final。
- 如果 sampled_o 小值域相对误差明显差，先拆 `qg @ h` 和 `Aqk @ v_new` 两项定位。

伪代码定位：

```python
o_state = qg @ h_prev * scale
o_local = Aqk @ v_new
o = o_state + o_local

compare(o_state_npu, o_state_ref)
compare(o_local_npu, o_local_ref)
compare(Akk_npu, Akk_ref)
```

## 14. 测试矩阵设计

小 shape 不能证明目标 shape 正确，长 shape 也不能替代针对性观测点。

推荐测试矩阵：

```text
基础语义:
    BSND/TND/BNSD/NTD
    fp16/bf16/fp32 where supported
    initial_state None / provided
    return_intermediate false / true

gate:
    external gk
    KdaGateCumsum normal g_step
    KdaGateCumsum safe_gate raw path
    safe_gate large K=128 multitask tail row

目标性能:
    B=1, H_K=1,  H_V=2,  T=16384, K=128, V=128, chunk=64
    B=1, H_K=32, H_V=64, T=4096,  K=128, V=128, chunk=64
    B=1, H_K=32, H_V=32, T=65536, mean_len=1024 变长序列, where supported

边界:
    tail length 1..63
    cu_seqlens 随机扰动
    gk 极端负值但合法 safe_gate 范围
    fixed input repeated run
```

关于 `T=16384 BSND` 的教训：

- shape 足够长不等于覆盖所有路径。
- 如果输入是外部 `gk`，不会测到 `KdaGateCumsum`。
- 如果不是 safe_gate 大值域，尾行错误不一定放大成 NaN。
- 如果不看 `gk` 逐元素 diff，问题会被下游输出掩盖。

所以测试设计必须同时覆盖：

```text
路径: 是否真的经过目标 kernel
值域: 是否能放大隐藏问题
观测点: 是否观察第一现场
调度: 是否触发多 task/core 复用
```

## 15. 文档和 PR 交付

每次修改下面内容时，必须同步更新设计文档、测试说明和公开接口说明：

- layout 支持范围。
- dtype 支持范围。
- `safe_gate`、`initial_state`、`return_intermediate` 等属性语义。
- 预留但不支持的参数。
- 中间量返回顺序。
- 性能目标和已知限制。

公开 PR/issue/报告中不要写：

- 内部服务器、IP、用户名。
- 绝对路径、临时目录、日志路径。
- token、环境路径、内部问题单来源。

只写：

- 做了哪些测试。
- 结果是否通过。
- 失败场景和公开可理解的原因。
- 已知限制。

## 16. 修改前后检查清单

修改前：

- 明确这次改的是语义、性能、同步、layout、dtype 还是测试。
- 找到三方对标公式和本仓可参考实现。
- 确认是否会影响 L2 op_def、PyTorch wrapper、返回 tuple 和文档。
- 确认当前工作树是否有无关改动，避免混进提交。

修改中：

- hot path 不使用 `GetValue/SetValue`。
- 矩阵主路径不退回 scalar/vector。
- 每个跨 pipe buffer 复用都有事件闭环。
- 每个 cross-core flag 都有匹配 wait/set。
- partial chunk 不读取未定义脏行。
- 用户输出不作为 split path raw intermediate。

验证中：

- 先 gate-only、stage-only，再 full KDA。
- 先小 shape，再目标 shape。
- 精度失败必须看 CT dual/CT viz 或逐阶段 diff。
- NaN/Inf 必须追第一处非有限或第一处极大值。
- 固定输入多跑验证同步类问题。
- 性能用 msopprof 看 bound。

提交前：

- `git diff --check`。
- 只 stage 相关文件。
- 文档与代码能力边界一致。
- 新增问题必须有能稳定触发的回归用例。
- PR 描述只写公开测试项和结果，不暴露内部环境。

## 17. 已修复问题速查

### 17.1 Gate cumsum 尾行被写 0

根因：

- `KdaGateCumsum` 每行 `acc` 写回后缺少 `MTE3->V`。
- 下一 task 的 `Duplicate(acc, 0)` 覆盖 MTE3 仍在读取的 UB。

现象：

- `gk` 前若干 chunk 尾行变 0。
- `step_max` 出现几百的正跳。
- `Aqk/Akk/w/kg` 极大或 NaN。
- `o/final_state` 只是下游症状。

修复：

```cpp
SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
SetFlag<HardEvent::MTE3_V>(mte3ToVEvent);
WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent);
```

回归：

- `safe_gate=True`
- `T=1536`
- `H_V=2`
- `K=128`
- `chunk_size=64`
- gate-only 对 CPU reference。

### 17.2 `final_state` dtype 不一致

根因：

- op_def 或 L2 按 `q` dtype 分配 `final_state`。

现象：

- executor nullptr。
- dtype 对比失败。
- `final_state` 误差或类型和三方对标不一致。

修复：

- `initial_state/final_state` 固定 `fp32`。
- op_def、L2 分配、kernel GlobalTensor 类型一致。

### 17.3 BNSD/NTD 被重复 layout swap

根因：

- 没把 BNSD/NTD 识别为性能 layout。

现象：

- `KdaLayoutSwap12` 在 msopprof 中占比异常。
- 已经连续的输入被再次搬运。

修复：

- BSND/TND 才转 layout。
- BNSD/NTD reshape/view 后直通。

### 17.4 partial chunk flag 不平衡

根因：

- AIV 对无效行 skip，但 AIC 仍等待。

现象：

- 随机 cu_seqlens 或尾块 shape timeout。
- 小 dense case 通过。

修复方向：

- dedicated partial path。
- full tile pad/clean。
- 空任务也参与 flag 协议，或 tiling 让 paired side 不等待。

## 18. 后续重点

1. 为 high `K/V` non-aligned 变长序列实现 dedicated partial-chunk 性能模板；正确性路径继续保持完整 tile 补中性值、仅回写有效区。
2. 对 `return_intermediate=True` 的 BSND/TND/BNSD/NTD 中间量逐项定义有效区语义。
3. 对新增流水路径跑 sanitizer race/mem/init/sync，确认实际命中 sanitizer kernel。
4. 若继续探索 MXR/MXH，必须同时提交精度、性能和流水驻留证据，不能只提交公式替换。
