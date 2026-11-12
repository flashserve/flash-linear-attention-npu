# Finalize 输出布局、dtype 与片上转换

设计规则版本：V3。接口见 [api.md](api.md)。布局直写、梯度 dtype 和参数片上转换
共用原有 FP32 数学计算与 workspace。下面分别说明布局、写回、类型转换和 raw tiling。

## 布局与内部总接口适配

独立 Finalize 的 `output_layout` 控制 dq/dk/dv/db/dg 全部五个逐 token 梯度，
dA/dbias 不受其控制。op 定义另有内部可选 string 属性 `model_layout`，默认空字符串
表示 standalone，非空仅接受 `BSND/BNSD/TND/NTD`。属性总数为 11，内部 L0 函数签名
在末尾追加 `modelLayout`；独立 ACLNN 不新增公开参数，继续通过默认值进入 standalone。

A2/A3 使用现有 outputTokenMajor：token-major 地址为 `(b*T+t)*H+h`，
head-major 地址为 `(b*H+h)*T+t`。dg 按 D=128 展开，db 为标量。
token-major 行间跳过 H-1 个 head；head-major 行间无间隔。
db 的 UB 每行保持 32 字节对齐，用已有 DataCopyPad 写回，无额外 Cast/Transpose。

A5 的 GatePost 和 dBeta 写回复用 FinalizeOutputOffset，并按布局标记选择 GM stride。
总接口内部将同一公开 layout 传给 `output_layout` 与 `model_layout`：

- q/k/v 及 dq/dk/dv 随 layout；成对提供 rstd 时，q/k 固定为 head-major q_hat/k_hat。
- 公开 do/raw_g/beta 及 dg/db 固定 token-major。A5 do 由 ACLNN 直接传给 Prepare/Dhu
  进行原生 stride 读取；普通 beta 转置一次供 Recompute/Finalize 共用，因此 Finalize 的 beta 输入仍为
  head-major，raw_g 则保持 token-major。
- outputTokenMajor 控制 q/k/v 梯度，gateOutputTokenMajor 控制 db/dg；混合布局路径的
  Gate 梯度始终为 token-major，独立路径则由 `output_layout` 决定。

布局只由显式字符串确定，输出 shape 仅用于校验。`T == H` 时不得交换 H/T 或根据同形
维度猜测布局；非法 `model_layout` 在 host 报告中给出参数名、实际值和合法集合。

验证设计覆盖 dense/packed、独立调用的两种输出顺序、总接口四种 layout、非对齐尾块、
奇数 head、H=T、raw_g/beta 四种 dtype 和 norm/sigmoid 分支。预期值由独立 CPU 数学
标杆生成；比较布局转换只存在于测试侧，不改变数学公式、输入值域、精度阈值或公开契约。

## 历史 A2/A3 布局定向验证

2026-10-28，在 Ascend910B3 / CANN 9.2.0 上重建 wheel 后，24 组独立 Finalize
用例的七个输出与旧实现按目标布局排列后的结果逐元素一致，12 组默认布局检查通过。
覆盖 dense B=2、packed 含空序列、T=65 尾块、H=3、H=T=4、norm 开关及 BF16/FP32 beta。
这是当时写回布局等价性验证，不替代当前实现的完整 ATK 或 A5 设备验收。

## Gradient dtype writeback

All gate arithmetic, chunk partials and parameter reductions remain FP32.
The A2/A3 final gate stage casts dg with CAST_RINT into the retired raw-gate
input slot before DMA, then waits for MTE3 before that slot is reused by MTE2.
The A5 gate stage uses its retired raw-gate tile at UB[176,208) KiB for the
BF16 result; the existing stage7 MTE3-to-MTE2 event protects its lifetime.
No UB allocation or workspace size changes are needed.

Stage 12 reduces and writes dA/dbias in FP32, retaining the original
FP32 parameter reduction and writeback path. A_log/dt_bias are FP32 inputs.
The output descriptors select writeback types; no extra launch is introduced
in A2/A3 or A5 V2. A5 legacy retains its FP32 device ABI and L2 converts its
outputs as a compatibility fallback; it does not gain the V2 fusion benefit.

Validation compares all four raw_g/beta dtype combinations with FP32 A_log/dt_bias against
FP32-output baseline results rounded once to each requested output dtype.
Cover both standalone output orders, the total-interface layouts, saved/recompute,
dense/packed, empty sequence and partial chunks. Separately compare the independent
CPU mathematical golden. These are coverage requirements, not a device-pass claim.

## 原始 gate 参数的片上类型转换

Finalize 的 raw_g/beta 支持 BF16/FP32，dg/dbeta 跟随对应输入；A_log/dt_bias 及其梯度
固定 FP32。FP32 累计量 gk、d_aqk、dq_raw、归约 workspace 不变，独立 Finalize 的
output_layout 仍控制五个逐 token 梯度。

A2/A3 Stage11 将 BF16 raw_g 读入既有 raw 槽，在每个 tile 的临时区尚未使用时扩展为
FP32（compact 使用 [184,192) KiB product 区，其他路径使用 [128,136) KiB q 区），
再复制回 raw 槽，避免重叠 Cast；A_log 和 dt_bias 直接按 FP32 读取。双缓冲事件继续保护
MTE2/V/MTE3 生命周期。A5 在 Stage11 VF 加载 BF16 寄存器后转换 FP32，沿用原有 FP32
链式求导；BF16 全掩码覆盖全部 128 个元素，不能复用只覆盖一半 lane 的 FP32 掩码。
Stage12 的参数跨 chunk 归约及写回均保持 FP32。

总接口 saved/recompute 均直接透传原始 raw_g，不为其插入独立 GM FP32 Cast；
Recompute 在 Vector 读入 BF16 raw_g 后升精度。Finalize 接收原始 raw_g，普通 beta 已在
ACLNN 中转换为内部 head-major。A5 legacy 单体实现保留原有支持范围，融合覆盖
A2/A3 staged 和 A5 V2/独立 Finalize。

验证应覆盖 raw_g/beta 四种 dtype 组合、dense/packed、输出布局、saved/recompute、
可选 bias、空序列和尾块，并检查 BF16 dg/db 写回及 FP32 参数梯度。性能结论必须来自
当前实现的独立实测，不能由 Cast 数量或历史结果推断。

## A5 raw tiling 与 host 检查

A5 `ChunkKdaBwdFinalizeTilingData` 的逻辑结构大小为 **184 字节**。host 在写入结构后
将 raw tiling size 设置为 `sizeof(TilingData)+8`，即 **192 字节**，与设备侧
`opParaSize` 一致。Arch22 保留自己的 tiling 数据和处理路径。

host 测试通过 `static_assert` 固定 184 字节逻辑 ABI，并在 A5 成功路径检查 192 字节
raw size。typed getter `GetTilingData<T>()` 会重置 raw size，所以必须先读取并保存
`GetRawTilingData()->GetDataSize()`，再使用 typed getter 检查布局字段。
raw tiling 修复、DFX、host/ATK 资产和文档更新均属于正确性与可验证性工作，
不作为 O1/O2 性能机制。

## 总接口内部 FP32 beta_eff / BF16 db

A2/A3 总接口按原始 beta dtype 分配 db，通过现有 model_layout 调用 Finalize。
输入 BetaT 与输出 DTYPE_D_BETA 分离，原 FP32 数学和 UB 生命周期保持不变，
仅最终 pack/store 使用输出类型。独立 Finalize API 继续要求 db dtype 与其 beta 输入一致；
新增组合只供融合 sigmoid 的总接口路径使用。

## A2/A3 WY 中间量 FP32 精度修复（issue 1156）

公开接口和输出 dtype 不变。Stage0 的 `dW = dv_scan @ h.T` 保留 FP32 累加结果；
Stage1 的 `kE = k * exp2(gk)`、Stage3 的严格下三角 Zb、Stage4 的
`Tza = Zb @ akk.T` 不再经过 BF16 舍入。akk 在 Stage1 从 BF16 无损扩展为 FP32，
Stage2/4/6 使用关闭 HF32 的 FP32 MMAD。dA_log 的正负项抵消会放大 WY 链路的
连续 BF16 舍入误差，因此连贯提高这些中间量的精度，而不修改最后的归约或测试阈值。

### 任务和片上资源

保留 `HEADS_PER_WORK=2`、work 到 chunk/head 的映射及原有两代 workspace 流水。
每个 AIV 仍执行原有 head 任务，奇数 head 的空 owner 参与同步而不读写张量。
没有新增 kernel launch，也没有修改 host 的 usedCoreNum 选择。

每个 head 的 L1 lane 为 224 KiB，两 lane 合计 448 KiB。下表单位均为 KiB，区间为
相对 lane 起点的半开区间；owner 1 加 224。FP32 的 NZ fractal 为 16×8，BF16 为 16×16。

| 区间 | 对象与类型 | 生命周期 |
|---|---|---|
| [0,8)、[8,16) | akk BF16，两代 | Stage0 加载/消费；沿用 akkFree 对代的保护 |
| [16,32)、[32,48) | akk FP32，两代 | Stage1 GM 产生、LoadKe 加载，Stage2/4/6 消费，Stage6 释放 |
| [48,80) | dW FP32 | Stage0 产生，Stage2 最后消费 |
| [80,112) | kE FP32 | Stage1 GM 产生、LoadKe 加载，Stage2 最后消费 |
| [80,96) | Zb FP32，复用 kE 区 | Stage2 结束后 LoadZb 加载，Stage4 消费 |
| [48,64) | Tza FP32，复用 dW 区 | Stage4 产生，紧接着的 Stage6 消费，不需要跨 work 保存两份 |
| [48,112) | Stage8 FP32 AQ/AK/Kneg | Stage6 后复用 dW/kE 区，Stage8 全部消费后进入下一次 Stage0 |
| [112,144) | h BF16 | Stage0 消费后允许预取下一 work |
| [144,176) | dh BF16 | 同上，最后读者包括 CUBE_RH |
| [176,192) | v_new BF16 | Stage0 消费，受 l1Free 保护 |
| [192,208) | dv_scan BF16 | Stage0 消费，受 l1Free 保护 |
| [208,224) | v BF16 | Stage0 消费，受 l1Free 保护 |

Stage8 与下一 work 的 state 预取可并行，二者地址不重叠。Stage8 复用区也不覆盖任一
akk bank。该复用保持双 head 调度，避免将精度提升变成按单 head 串行执行。

UB 总量仍为 192 KiB。Stage1 的 kE 输出 ping/pong 由各 8 KiB BF16 扩展为各 16 KiB
FP32，位于 [152,168)、[168,184)；这时此前 RH 计算的临时量已经结束。
两份输出 DMA 均完成后，akk 输入使用 [88,96) BF16，cast 输出使用 [104,120) FP32；
它们复用已消费的 gate 输入区，不覆盖 k、exp2(gk)、RH 等驻留量。
Stage3 的 Zb 输出 ping/pong 使用 [0,8)、[8,16) FP32，保留每 slot 的 MTE3→V 释放闭环。

### GM 中转与同步

每 head 两个 bank，每 bank 512 KiB，单 Cube core 共 2 MiB；其他已存在的 workspace
对象保持原 offset。新增/扩展区域如下，单位为 KiB：

| 区间 | 对象 | 写者 → 读者 |
|---|---|---|
| [384,416) | kE FP32 [64,128] | Stage1 AIV → LoadKe AIC |
| [416,432) | Zb FP32 [64,64] | Stage3 AIV → LoadZb AIC |
| [432,448) | akk FP32 [64,64] | Stage1 AIV → LoadKe AIC |
| [448,480) | dW FP32 [64,128] | Stage0 Fixpipe → Stage0 MTE2 |
| [480,496) | Tza FP32 [64,64] | Stage4 Fixpipe → Stage4 MTE2 |
| [496,512) | 对齐余量，无数据 | 无 |

A2 的现用 Fixpipe L0C→L1 路径不能直接输出 FP32，因此 dW/Tza 先以 ND FP32 写 GM，
再以 ND→NZ 搬入 L1。这两次中转分别增加 32/16 KiB GM 写和等量 GM 读，保留 FP32
范围而不引入 FP16 溢出风险。akk 增加 8 KiB BF16 GM 读、16 KiB FP32 GM 写/读；
kE/Zb 相比原路径各增加 16/8 KiB GM 写和等量读。

新增一个 AIC 本地 `FIX_MTE2` event：每次 Fixpipe 写出后 Set，随后 Wait 才从 GM
读取；MTE2→MTE1 的现有 inputReady event 确认 L1 数据可读。两个 owner 顺序复用该
event，使用后不留 pending flag，End 释放它。

Stage1 将 kE 和 akk 写完后才发布原有 KE_READY。LoadKe 对两者发起 MTE2，然后发布
keReady；Stage2 必须在第一次读取 akk 之前等待 keReady，不能仍等到计算 ZW 才等待。
这样同时保护完整块与需要 padding 的尾块。其余跨 work ready/free 计数不变：
Stage2 消费 dW/kE → 前一代 Stage4/6 消费 Zb/Tza → Stage8 复用 [48,112) → 下一 Stage0。

有效 rows/columns 仍由 ResolveChunk 和 CopyToL1 处理；akk 的无效行列在 L1 清零，
kE 的无效行在 UB 清零。输出布局、norm、beta mode、空序列及参数归约语义沿用原设计。

性能和精度验证结果见同次修复的验收记录；A2 设备验证不能替代 A3/A5 验收。
