# Finalize 输出布局、dtype 与片上转换

设计规则版本：V3。接口见 [api.md](api.md)。布局直写、梯度 dtype 和参数片上转换
共用原有 FP32 数学计算与 workspace。下面分别说明布局、写回和类型转换的实现。

A2/A3 使用现有 outputTokenMajor：token-major 地址为 `(b*T+t)*H+h`，
head-major 地址为 `(b*H+h)*T+t`。dg 按 D=128 展开，db 为标量。
token-major 行间跳过 H-1 个 head；head-major 行间无间隔。
db 的 UB 每行保持 32 字节对齐，用已有 DataCopyPad 写回，无额外 Cast/Transpose。

A5 的 GatePost 和 dBeta 写回复用 FinalizeOutputOffset，并使用相同的布局标记
选择 GM stride。总接口 V2 删除原先 token-major 到 head-major 的适配。

验证覆盖 dense/packed、两种输出顺序、非对齐尾块、奇数 head、H=T、BF16/FP32 beta、
norm/sigmoid 分支。预期值由原有 CPU 计算或已验证基线按目标布局排列；不得改变
数学公式、输入值域或精度阈值。原版 CI example 不修改，用于检查旧调用方式。

## 定向验证

2026-10-28，在 Ascend910B3 / CANN 9.2.0 上重建 wheel 后，24 组独立 Finalize
用例的七个输出与旧实现按目标布局排列后的结果逐元素一致，12 组默认布局检查通过。
覆盖 dense B=2、packed 含空序列、T=65 尾块、H=3、H=T=4、norm 开关及 BF16/FP32 beta。
这是写回布局等价性验证，不替代完整 ATK 或 A5 设备验收。

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
Cover both output orders, saved/recompute, dense/packed, empty sequence and
partial chunks. Separately compare the independent CPU mathematical golden.

### A2 dtype 定向验证（2026-10-28，历史 BF16 参数扩展版本）

179.166.1.3 / Ascend910B3 / CANN 9.2.0，独立 wheel 构建、安装通过。
72 组用例在 Stable 与 ctypes 两入口共 144 次调用通过；比较全部有效输出，
仅将旧版基线的 db/dg/dA 转换为新约定 dtype，结果逐元素一致。
覆盖 8 种 gate/beta/A_log 类型组合、dense/packed、保存/重计算、
BNSD/BSND/NTD/TND、尾块、空序列、norm、sigmoid/2-sigmoid 和 H=17。
3 项错误梯度 dtype 均被 ACLNN 拒绝；真实前向缓存接反向的公开接口
回归 16/16 通过。独立 CPU 标杆的递推、自梯度、beta 模式与布局自检通过。

验证目录为 `/home/chenlinxin/wys/pr993-dtype-20261028`，wheel SHA256 为
`a8f99a088a716236a61e4a4323582238725e47910736418c3d45e6aa99e05cec`。
本轮按用户要求只验证 A2，未验证 A3/A5 设备，未执行完整 ATK、性能或 sanitizer 验收。

## 原始 gate 参数的片上类型转换

Finalize 的 raw_g/beta 支持 BF16/FP32，dg/dbeta 跟随对应输入；A_log/dt_bias 及其梯度固定 FP32。FP32 累计量 gk、d_aqk、dq_raw、归约 workspace 不变，output_layout 仍控制五个逐 token 梯度。

A2/A3 Stage11 将 BF16 raw_g 读入既有 raw 槽，在每个 tile 的临时区尚未使用时扩展为 FP32（compact 使用 [184,192) KiB product 区，其他路径使用 [128,136) KiB q 区），再复制回 raw 槽，避免重叠 Cast；A_log 和 dt_bias 直接按 FP32 读取。双缓冲事件继续保护 MTE2/V/MTE3 生命周期。A5 在 Stage11 VF 加载 BF16 寄存器后转换 FP32，沿用原有 FP32 链式求导。Stage12 的参数跨 chunk 归约及写回均保持 FP32。

总接口 saved 模式直接透传原始参数，不再插入 Finalize 输入 Cast。recompute 分支单独创建重算所需 FP32 参数，Finalize 仍接收原始输入；重算算子的 Cast 不属于本次消除范围。A5 legacy 单体实现保留原有支持范围，本次融合覆盖 A2/A3 staged 和 A5 V2/独立 Finalize。

验证：对照修改前 FP32 Finalize，覆盖 raw_g/beta 四种 dtype 组合（A_log/dt_bias 固定 FP32）、packed/dense、输出布局、saved/recompute、可选 bias 和尾块；A2 真机比较结果与既有输出按目标 dtype 舍入后的结果。模型长序列使用相同输入和 profiler 配置，按 OP Type 比较 Cast 数量及 Finalize/全链路耗时。A5 单独编译检查，真机结果不得由 A2 结果替代。

### Cast 融合定向验证（2026-10-28，历史 BF16 参数扩展版本）

A2 / Ascend910B3 / CANN 9.2.0：146 组 × Stable/ctypes 两入口，共 292 次调用，
全部输出与 e6896d9b 基线按目标 dtype 舍入后逐元素一致，4 项错误输出 dtype 被拒绝。
覆盖 16 种参数 dtype 组合、dense/packed、保存/重算、两种输出顺序、norm、
sigmoid/2-sigmoid、空序列、尾块、H=17、可选 bias。真实前向缓存接反向、
独立 Finalize 与总接口对比 32/32 通过；新旧 BF16 模型 smoke，以及 BF16/FP32
小算子与总接口 smoke 均逐元素一致。独立 CPU 标杆自检通过。

相同模型模拟输入 T=131072、H=4、D=128、31 段、2066 chunks，预热 5 次、采样 10 次：

| gate | 每轮 Cast | Cast 耗时合计（μs） | 全链路 Task Duration 合计（ms） |
| --- | --- | --- | --- |
| BF16 | 6 → 2 | 287.362 → 2.278 | 25.307 → 25.142 |
| 外层转 FP32 | 7 → 4 | 652.858 → 654.958 | 20.825 → 20.882 |

该表不是端到端 wall time；FP32 模式未测到整体加速。剩余 Cast 为前向参数转换，
以及 FP32 模式外层 gate.float()/CastBackward；两种模式 Transpose 均保持 5 次。
BF16 模式 Finalize 自身 8.135 → 8.205 ms，片上转换增加约 71 μs，
但删除的独立输入/输出 Cast 合计约 285 μs。

验证目录 `/home/chenlinxin/wys/pr993-fused-cast-20261028`。A2 wheel SHA256：
`e33db38fdef0b55c792c1801cd77e94c610d6c6fde06e91acebd2ef17a32ece4`。
A5 交叉编包通过，wheel SHA256：
`a04b4f36047dcc3fa8d7033eaa5f86719eaf13ad05cc6954c44c59fe8282591c`。
A3/A5 真机、完整 ATK、sanitizer 和双平台 quick CI 尚未验收。

## FP32 参数约定回归（2026-10-29）

本轮撤回 A_log/dt_bias 的 BF16 扩展，两者及其梯度固定 FP32；raw_g/beta
四种 dtype 组合、布局直写和片上转换保留。前向源码及 Python 前向函数与 PR 基线一致。

A2 / Ascend910B3 / CANN 9.2.0：完整 KDA 前后向 wheel 编包安装通过。
42 组反向 × Stable/ctypes 两入口共 84 次调用，全部输出与修改前 FP32 参数路径
逐元素一致；覆盖 saved/recompute、dense/packed、两种输出顺序、norm、sigmoid、
空序列、尾块、H=17 和可选 bias。4 项非法梯度 dtype、4 项 BF16 参数输入检查通过。
前向四布局 × 四种 g/beta 组合 × 两入口 × 总接口/Prepare，共 64 次对照通过，
并拒绝 BF16 A_log/dt_bias；12 组可选参数及保存模式检查通过。
8 组真实前向缓存接反向验证通过，每组检查两种输出布局及独立 Finalize 与总接口一致。

原 CI example 4/4 通过：定长/变长前后向精度、大 head 尾块和空序列 guard；example 文件未修改。

A5 Prepare/Finalize 定向交叉编译通过。Prepare 仅在验证目录选取覆盖四种
g/beta 组合及 safe-gate/softplus 的 8 个 tiling key，生产 selector 未修改；
该 wheel 不作为完整发布包。A3/A5 实卡、完整 ATK、sanitizer 与双平台 quick CI 未完成。

验证记录目录名为 `pr993-fp32-params-20261029`；A2 wheel SHA256：
`45cbd2edff6aaeca9411bb2403e06d9a7d1550443ba4d55c8a8a867f2c283c22`。
A5 定向验证 wheel SHA256：
`a22d4b7b0faee83d782d83401799c9f1fc736df8722648843690d3187ca4c963`。


## 总接口内部 FP32 beta_eff / BF16 db

A2/A3 总接口按原始 beta dtype 分配 db，通过现有 model_layout 调用 Finalize。
输入 BetaT 与输出 DTYPE_D_BETA 分离，原 FP32 数学和 UB 生命周期保持不变，仅最终 pack/store 使用输出类型。
独立 Finalize API 继续要求 db dtype 与其 beta 输入一致；新增组合只供融合 sigmoid 的总接口路径使用。
