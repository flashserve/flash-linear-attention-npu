# chunk_kda_bwd：A2/A3 L0 编排改造方案

> 历史阶段设计。当前 A5 V2 已同步单 layout 契约，路由及 ABI 以 [api.md](api.md) 为准；下文“V2 定义不变”的结论已被本次非兼容升级替代。

方案设计规则版本：V2。输出约定以 [api.md](api.md) 为准：dq/dk 跟随输入 dtype，五个梯度统一使用 output_layout。以下保留原编排设计，输出适配已按该约定更新。

本方案以本轮明确的七条原则为准：**只改 A2/A3 的执行逻辑，A5 保留原入口和 V2 两条路径。** 下文展示接口与编排。

## 1. 改造范围

| 项目 | 本次方案 |
| --- | --- |
| Python | 保留 chunk_kda_bwd / npu_chunk_kda_bwd 名称和已有参数，末尾增加四个可选参数；后续确认 use_gate_in_kernel 默认且必须为 True，其余默认值不变 |
| A2/A3 | 所有 implementation 模式统一调用 aclnnChunkKdaBwd，在 cpp 内直接编排 L0 |
| A2/A3 保存模式 | disable_recompute=True：Prepare → Dhu → Finalize |
| A2/A3 重计算模式 | disable_recompute=False：Recompute → FwdH → Prepare → Dhu → Finalize |
| A5 原入口 | 继续调用原融合算子 |
| A5 V2 | 对外入参、进入条件及原有计算流程保持不变 |
| 底层算子 | 复用已交付的 Prepare/Finalize；本次同时给 Recompute 增加 input_layout 预留入参及强校验，计算 kernel 不变 |

代码中的准确参数名是 disable_recompute / disableRecompute，以下统一采用该拼写。True 表示不重计算，False 表示需要重计算。

## 2. 新 Python 接口

公开的 chunk_kda_bwd 是 npu_chunk_kda_bwd 的别名。当前函数定义在 [Stable wrapper](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py) 和 [ctypes wrapper](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py)，两处同步增加末尾四项：

```python
def npu_chunk_kda_bwd(
    q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale,
    *, raw_g=None, A_log=None, dt_bias=None,
    initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None,
    chunk_size=64, safe_gate=True, lower_bound=-5.0,
    use_gate_in_kernel=True, disable_recompute=True,
    use_exp2=True, state_v_first=False,
    implementation="auto", q_rstd=None, k_rstd=None,
    input_layout="BNSD",               # 新增
    output_layout="BNSD",              # 新增
    use_beta_sigmoid_in_kernel=False,  # 新增
    allow_neg_eigval=False,            # 新增
):
    ...
```

返回仍是 (dq, dk, dv, db, dg, dh0, dA, dbias)。旧调用省略这四项时，仍按默认 head-major 布局、关闭 beta sigmoid 反向处理，输入约定保持不变；A2/A3 输出遵循 api.md，不再保留 FP32 dq/dk 和 head-major db/dg。

门控按后续确认要求 use_gate_in_kernel 默认且必须为 True，Stable 与 ctypes 在平台和实现模式分流前校验，None 归一化仍使用 True；显式 False 在 Python 入口抛出 ValueError。Stable C++ dispatcher 和原 ACLNN 入口也增加对应校验，V2 保留已有约束。所有平台适用该门控限制，A5 的路由和计算实现不变。调用方仍须提供 raw_g 和 A_log，dt_bias 保持可选；本次不改 Finalize kernel。

| 新参数 | 含义 |
| --- | --- |
| input_layout | 有重计算时传 Recompute，仅允许归一化后的 BNSD/NTD；无重计算时传 Prepare，沿用其布局范围。默认 BNSD，packed 时归一化为 NTD；BSND 在 packed 时归一化为 TND |
| output_layout | dq/dk/dv/db/dg 的返回布局 |
| use_beta_sigmoid_in_kernel | 是否由 Finalize 完成 beta sigmoid 链式反向 |
| allow_neg_eigval | 开启 beta sigmoid 反向时，使用前向 beta=2×sigmoid(x) 对应的导数 |

四个参数不作为选择 V2 的条件。本次只在 A2/A3 启用新增能力。A5 默认值调用保持原行为；非默认新增能力不通过扩展 V2 入参实现，超出既有能力时明确报不支持，不静默忽略。

## 3. 路由规则：先平台，再保留 A5 原判断

| 平台 | implementation / rstd / 重计算 | 总入口 |
| --- | --- | --- |
| A2/A3 | legacy、auto、optimized；有无成对 rstd；保存或重计算 | 一律 aclnnChunkKdaBwd |
| A5 | 原规则判定需要优化路径，例如 optimized、auto 下带成对 rstd 或要求重计算 | aclnnChunkKdaBwdV2 |
| A5 | 原规则判定普通路径 | aclnnChunkKdaBwd |

A5 继续使用现有 [选择函数](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_kda_policy.py) 的判断和校验，包括原有非法组合处理，不重写其优先级。A2/A3 应在调用该 A5 选择逻辑之前完成平台分流，避免被 A5 专用校验拦截。

```python
# 路由示意，省略共同参数校验和实际参数列表。
if soc in ("A2", "A3"):
    # legacy/auto/optimized 均到这里。
    # rstd、新属性、disable_recompute 传给原总入口。
    call_aclnnChunkKdaBwd(...)
else:
    # A5 保留当前选择函数与调用方式。
    optimized = existing_A5_selection(...)
    if optimized:
        call_aclnnChunkKdaBwdV2(...)  # 原入参不变
    else:
        call_aclnnChunkKdaBwd(...)
```

Stable C++ adapter 也要保持这一边界，不能在 A2/A3 上看到 optimized 就转入 V2。ctypes 采用相同规则。

A2/A3 的“所有模式、任何入参组合走同一路径”指不再按模式切换总入口；张量、布局、rstd 配对等仍需合法。legacy 不应成为 A2/A3 阻止重计算或成对 rstd 的理由。A5 原来的限制保留。

## 4. 新 aclnnChunkKdaBwd 定义

在原 GetWorkspaceSize 的 stateVFirst 后、输出前，新增 qRstdOptional、kRstdOptional 和四个属性。原业务参数顺序保持。第二阶段执行函数不变：

```cpp
aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, double lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const char *inputLayout,     // 新增
    const char *outputLayout,    // 新增
    bool useBetaSigmoidInKernel, // 新增
    bool allowNegEigval,         // 新增
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbOut, const aclTensor *dgOut,
    const aclTensor *dh0OutOptional, const aclTensor *dAOutOptional,
    const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaBwd(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

两点区别：

- aclnnChunkKdaBwd 的新参数服务于 A2/A3 编排；A5 普通路径的调用点也需按新原型补 null rstd 和四个默认值，但设备执行仍调用原融合算子。
- aclnnChunkKdaBwdV2GetWorkspaceSize / aclnnChunkKdaBwdV2 的公开定义不变，不向 V2 增加四个属性。

C/C++ 调用方与配套适配层需要同步重编译，Python 调用的兼容性由默认参数保证。

## 5. 在 aclnn_chunk_kda_bwd.cpp 中直接调用 L0

### 5.1 总体结构

```cpp
// 编排示意，省略实际参数和错误检查。
if (isA2OrA3) {
    // inputLayout 已按 dense/packed 归一化。
    // defaultInputLayout 为内部 head-major：dense BNSD，packed NTD。
    const char *prepareInputLayout = inputLayout;
    if (!disableRecompute) {
        // 重计算只允许 head-major，禁止 BSND/TND。
        CHECK_COND(IsBnsdOrNtd(inputLayout), ACLNN_ERR_PARAM_INVALID,
                   "Recompute input_layout must be BNSD or NTD.");
        // 和 A5 V2 一样，在总层给 packed 张量增加 batch=1 视图。
        const auto *q4 = BatchView(q, packed, executor);
        // k/v/g/beta/A 和各输出同样按 A5 V2 建立视图，不搬数据。
        // 即使 inputLayout="NTD"，Recompute 实际收到的仍是 rank-4 BNSD。
        l0op::ChunkKdaBwdRecompute(q4, ..., inputLayout, ...);
        l0op::ChunkFwdH(...);
        prepareInputLayout = defaultInputLayout;
    }

    // 有重计算传默认布局，无重计算传外部 inputLayout。
    l0op::ChunkKdaBwdPrepare(..., prepareInputLayout, ...);
    l0op::ChunkGatedDeltaRuleBwdDhu(...);
    l0op::ChunkKdaBwdFinalize(..., outputLayout, ...);
    // 五个梯度由 Finalize 直写；仅保留直接 ACLNN 的 BF16 db 类型兼容及参数梯度适配。
} else {
    // A5：保留当前原融合算子调用。
    l0op::ChunkKdaBwd(...);
}
```

这些调用都放在 aclnnChunkKdaBwdGetWorkspaceSize 创建的同一个 executor 中，最终统一查询 workspace、通过第二阶段执行。方案不再引入额外的编排函数层。

### 5.2 保存与重计算

| disableRecompute | 缓存处理 | Recompute 的 input_layout | Prepare 的输入布局 |
| --- | --- | --- | --- |
| true | 读取前向保存的 gk/w/qg/kg/vNew/h，校验缓存完整；不调用 Recompute/FwdH | 不适用 | 传总接口的 inputLayout |
| false | Recompute → FwdH 恢复上述缓存；Aqk/Akk 仍保留 | 传归一化后的 inputLayout，仅 BNSD/NTD 合法 | 传内部默认 BNSD；packed 为 NTD |

Recompute 当前没有 layout 入参。本次同时扩展其接口，增加 input_layout，但只对该字符串做强校验：仅接受 BNSD 和 NTD。总层参考 A5 V2，先为 packed 输入/输出建立 batch=1 视图，因此即使传 inputLayout="NTD"，Recompute 实际收到的也是 rank-4 BNSD 张量。Recompute 不按此参数判断 rank、重建视图或转换布局，原计算及输出保持。具体接口改法见 5.5。

因此两条链的布局传递是：

- 有重计算：校验 inputLayout 为 BNSD/NTD → 总层按 A5 V2 建立 batch 视图 → Recompute(inputLayout) → FwdH → Prepare(默认布局) → Dhu → Finalize。
- 无重计算：Prepare(inputLayout) → Dhu → Finalize。

inputLayout 传给路径中的第一个 L0 后，不再沿后续算子重复传递；Recompute 本轮只校验该值，不提供布局转换能力。实现时仍需核对 dO 等未由 Recompute 生成的外部张量，确保交给 Prepare/Dhu/Finalize 时符合各自输入契约；必要的实际布局适配不能只靠修改布局字符串代替。

原入口中“必须 disableRecompute=true”“gk 永远非空”的校验要按 A2/A3 分支调整；保存模式仍严格要求缓存，重计算模式按参考流程恢复缓存。A5 原入口的现有校验保留。

### 5.3 新参数怎样传到小算子

| 参数 | A2/A3 总层处理 | 实际消费者 |
| --- | --- | --- |
| inputLayout | 有重计算仅接受 BNSD/NTD，总层先建立 BNSD 视图、再透传参数，后续 Prepare 用默认布局；无重计算才传 Prepare | Recompute 仅校验字符串值域，不按它处理张量；Prepare 按其已有契约处理布局 |
| outputLayout | 按目标布局分配 dq/dk/dv/db/dg | Finalize 直接写回 |
| qRstd/kRstd | 成对校验并透传，参考 A5 V2 | Finalize 内完成 Q/K norm 反向 |
| useBetaSigmoidInKernel | 透传，不在总层再做一次 sigmoid 反向 | Finalize |
| allowNegEigval | 与 beta sigmoid 开关一起校验、透传 | Finalize |

具体遵循更新后的小算子契约：

- Recompute 新增 input_layout，仅在 Host 校验字符串是否为 BNSD/NTD；张量按总层提供的 rank-4 BNSD 视图计算，不按字符串处理 rank、视图或寻址。重计算路径的 FwdH 和 Prepare 使用默认布局。
- 无重计算时，Prepare 接收总接口的 inputLayout，并将输出统一为 head-major，再交给 Dhu/Finalize。Prepare 的实际输入张量仍须符合其 layout 契约，不能混用不同物理布局。
- Finalize 输入固定 head-major；dq/dk/dv/db/dg 均按 outputLayout 直接写回。总接口保留这五个输出的布局。
- 没有 rstd 时 q/k 受 inputLayout 控制；有成对 rstd 时 q/k 是前向保存的 q_hat/k_hat，固定 head-major，不能再按外部布局误转置。其他前向缓存保持原布局。
- beta 是前向有效 beta。Finalize 负责一次链式梯度：普通 sigmoid 乘 beta×(1-beta)，2sigmoid 乘 beta×(1-beta/2)。Dhu 不新增这些参数。

dq/dk 直接按输入 dtype 分配；不增加输出选项或计算模式。旧调用方按 api.md 更新输出约定。

### 5.4 中间量和必要保护

沿用已有 Prepare → Dhu → Finalize 的中间量关系：

| 中间量 | 生产者 | 消费者 |
| --- | --- | --- |
| dAqk / dqRaw | Prepare | Finalize |
| dvLocal | Prepare | Dhu |
| dh / dvScan | Dhu | Finalize |

所有临时 tensor 由同一 executor 分配，保持到最后消费者完成；每次调用检查 L0 的实际返回值。缺省偏置和未返回参数梯度所需的临时输出参考 V2 处理，不能与仍在使用的 dg 输出混用。各 L0 的片上资源、分核、数学公式和同步不在本次重新设计。

### 5.5 Recompute 接口改造：预留 input_layout，暂不扩展布局能力

本节是本次新增的开发范围。不是等待同事再增加接口，而是和总接口改造一起完成。

**接口位置：**L0 和独立 ACLNN GetWorkspaceSize 都在 lowerBound 后、wOut 前增加 const char *inputLayout，原有参数相对顺序不变。L0 目标定义为：

```cpp
const std::array<const aclTensor *, 5> l0op::ChunkKdaBwdRecompute(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, bool useGateInKernel, bool useExp2,
    double lowerBound,
    const char *inputLayout,  // 新增，仅校验和预留
    const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *qgOut, const aclTensor *kgOut,
    const aclTensor *gkOut,
    aclOpExecutor *executor);
```

独立 aclnnChunkKdaBwdRecomputeGetWorkspaceSize 同样插入 inputLayout；它原有的属性集合保持，不额外补入只在 L0 存在的 useGateInKernel。第二阶段 aclnnChunkKdaBwdRecompute 执行签名不变。独立 Python Recompute 入口末尾增加可选 input_layout，缺省保持原调用，Stable/ctypes 同步传参；本次不扩展独立入口原有的 rank 支持范围。

| 校验项 | 要求 |
| --- | --- |
| 字符串 | 仅精确接受 BNSD/NTD；拒绝空指针、空串、BSND/TND、其他值 |
| 传 BNSD 或 NTD | 都允许实际 rank-4 BNSD 张量；NTD 表示总层原始 packed 布局，不能据此要求传给 Recompute 的张量必须是 rank 3 |
| rank/shape | 保留 Recompute 原有 rank-4 等校验，不与 input_layout 建立新的 rank 对应关系 |
| packed 元数据 | 沿用 A5 V2，视图增加 batch=1 后仍传原 cu_seqlens/chunk_indices |
| 总接口重计算 | 在追加 Recompute/FwdH 任务前拒绝 BSND/TND，不先转置后放行 |
| 无重计算 | 不受 Recompute 这项限制，Prepare 继续使用其支持的布局 |
| 输出布局 | 不改变 Recompute 现有输出顺序，也不因此收紧总接口 output_layout 的独立能力 |

实现分工：

1. L0、独立 L2 的声明/定义、DFX，以及两套调用适配层增加参数。L0 必须校验，因为大融合直接调用 L0；只在独立 L2 校验不够。
2. OpDef 在原四项属性后追加 input_layout，默认 BNSD；L0 将其作为属性传给 launcher。Host tiling 只增加该字符串的相同值域校验，覆盖直接图调用；原有 shape 校验保留，不按 input_layout 推导或约束 rank。
3. 校验结束后，不把 input_layout 放进 device tiling data，不改 kernel 入参、计算公式、寻址、分核、workspace 或 TilingKey，也不按它选择转置路径。
4. batch 视图适配全部放在反向大融合总层，复用 A5 V2 的 BatchView 做法。packed 的 [H,T,D] 变成 [1,H,T,D]，beta 等标量输入和各输出同样建立对应视图，不发生数据搬运，变长元数据不变。Recompute 的 L0/L2 内不新增视图或 rank-3 适配逻辑。
5. A5 V2 的公开签名与路由不变，仅在内部 Recompute L0 调用点补原行为对应的 BNSD 默认值；A2/A3 总接口在完成相同 batch 视图适配后，仍透传归一化后的 BNSD/NTD。两种字符串都走原来的计算路径。

这里的 NTD 是接口预留值，不表示 Recompute 新增了直接读取 rank-3 张量的能力。直接调用独立 Recompute 时仍需满足原有张量契约；不能仅把字符串改为 NTD 就传入未适配的 rank-3 张量。

总接口仍保留 input_layout 的统一参数形式，但本轮重计算输入支持范围明确为 head-major。以后扩展 Recompute 的 BSND/TND 支持时，再去掉对应强校验并实现真实布局读取，本次不提前放开。

## 6. 修改文件与顺序

| 文件 | 要改什么 |
| --- | --- |
| [共享 Python 策略](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_kda_policy.py) | A2/A3 平台分流，新增布局/开关校验；A5 原选择函数保留 |
| [Stable wrapper](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py)、[ctypes wrapper](../../torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py) | 新增四个默认参数；A2/A3 统一原入口，透传 rstd/属性；A5 调用保持 |
| [Stable adapter](../../torch_custom/fla_npu/csrc/src/stable_chunk_kda_bwd.cpp) | schema、输出分配和原 L2 参数顺序同步；A2/A3 不再因 optimized 转 V2；A5 V2 分支保留 |
| [原 L2 头文件](../../fla/ops/ascendc/kda/chunk_kda_bwd/op_host/op_api/aclnn_chunk_kda_bwd.h)、[cpp](../../fla/ops/ascendc/kda/chunk_kda_bwd/op_host/op_api/aclnn_chunk_kda_bwd.cpp) | 扩展原签名、DFX/校验；A2/A3 直接编排五个或三个 L0；A5 保留原融合调用 |
| [CMake](../../fla/ops/ascendc/kda/chunk_kda_bwd/op_host/CMakeLists.txt) | 为 A2/A3 补 Recompute/FwdH 依赖，保留 A5 现有依赖、宏与原/V2 构建路径 |
| Recompute L0/L2、OpDef、Host tiling 及独立适配层 | 按 5.5 增加 input_layout，仅增加 BNSD/NTD 字符串校验；Recompute 内不做视图适配，原计算不变；总层沿用 A5 V2 的 BatchView 后透传，后续 Prepare 用默认布局 |
| V2 相关文件 | 作为 A2/A3 重计算和 norm 衔接的参考；保留公开原型、原路由与计算逻辑 |

如果新 Recompute/Prepare/Finalize 原型要求 A5 调用点同步补参数，只补原行为对应的默认值，不改变 V2 对外入参，也不改变 A5 计算顺序。独立处理接口编译适配与 A2/A3 功能改造，避免把本次修改扩大为 A5 重构。

实施顺序：先固定 Recompute 新接口及 BNSD/NTD 强校验 → 接 Python/适配层参数与平台路由 → 扩展原总 L2 → 接保存模式 → 接重计算 → 验证新属性 → 做 A5 不变性回归。

## 7. 验证要点

| 要验证的事 | 核查方式 |
| --- | --- |
| 省略四个参数仍满足新输出约定 | 旧调用与显式四默认值逐输出比较 shape、dtype、布局、None 和数值 |
| A2/A3 路由统一 | legacy/auto/optimized × 有无成对 rstd × 保存/重计算，均确认调用原 aclnnChunkKdaBwd |
| 保存模式不重计算 | 任务序列有 Prepare/Dhu/Finalize，没有 Recompute/FwdH |
| 重计算顺序正确 | 任务序列为 Recompute/FwdH/Prepare/Dhu/Finalize；与同输入保存模式、独立 CPU 标杆对齐 |
| 有重计算的布局传递 | BNSD/NTD 经总层视图适配后，Recompute 实际输入均为 rank 4；inputLayout 原值透传，FwdH/Prepare 使用默认布局 |
| Recompute 预留参数强校验 | BSND/TND、空串、空指针、未知值在 Host 拒绝；覆盖总接口、直接 L0、独立 L2 和图调用 |
| NTD 字符串配 rank-4 视图 | 相同的合法 rank-4 BNSD 输入分别传 BNSD/NTD，两者都通过新增校验，不因 NTD 字符串要求 rank 3 |
| 重计算布局别名 | packed 默认 BNSD 归一化为 NTD 后允许；BSND 归一化为 TND 后仍拒绝，不静默转置 |
| Recompute 内部不改计算 | 不新增视图处理、device layout 字段、布局专用 key 或转置任务；总层 BatchView 与 A5 V2 一致，增加参数前后结果保持 |
| 无重计算的布局传递 | 使用非默认外部布局，确认 Prepare 收到 inputLayout，输出统一为内部布局后交给 Dhu/Finalize |
| 新增布局生效 | dense/packed 的输入/输出布局组合，特别核对外部 dO 等张量的实际布局、db/dg 跟随 output_layout 及 rstd 场景 |
| norm/beta 反向正确 | rstd 成对与不成对，beta off/sigmoid/2sigmoid，确认没有重复反向 |
| A5 不受影响 | 普通调用仍走原融合；原优化条件仍进 V2；原合法/非法组合及输出保持 |
| ABI 配套 | Stable、ctypes 和 OPP 的原 L2 新参数顺序一致，V2 公开签名不变 |
| 完整 backward 正确且稳定 | 按仓库 ATK 流程验证精度、tail、packed、奇数 head、确定性、内存与完整调用性能 |

“同一路径”不意味着相同 kernel 数：保存与重计算、布局适配本来就有任务差异。验证以总入口选择和实际 L0 任务序列为准，不以 implementation 名称推断执行路径。

本次不修改算子实现、不跑卡。上一版 api.md/design.md 改动已恢复，上一版临时 validation.md 已删除；本文件为新的独立方案。
