# KDA 反向优化设计

## A2/A3 输出直写

设计规则版本：V2。本轮调整 Finalize 输出寻址及总接口的输出适配，Stage、tiling、workspace
地址和同步不变，接口以 [api.md](api.md) 的 A2/A3 输出约定为准。

- Stable C++ 与 ctypes 直接按 q/k dtype 分配 dq/dk，Finalize 直写。
- dq/dk/dv/db/dg 统一按 output_layout 分配，Finalize 直接写回，删除总接口中的布局转置。
- db/dg 按原始 beta/raw_g dtype 分配，dA/dbias 固定 FP32，A2/A3 和 A5 V2 由 Finalize
  写回时转换；内部 FP32 工作张量不会改变对外输出 dtype。A5 legacy 保留
  FP32 kernel ABI，通过 L2 转换输出。
- Finalize 的 A2/A3 写回使用 outputTokenMajor 选择输出地址和行间 stride；A5 总接口
  另外区分 q/k/v 梯度与固定 token-major 的 Gate 梯度寻址。
- A5 V2 的公开 dg/db 固定 token-major，dq/dk/dv 随单一 layout；输入 Gate 的必要适配见
  “A5 V2 单 layout 适配”。
- C++ 绑定在 legacy/auto/optimized 全部模式下识别 A2/A3；A5 分支按 PR993 契约分配输出。

## 原 ACLNN 入口的形状描述

`aclnnChunkKdaBwdGetWorkspaceSize` 在调用 L0 前，将所有非空输入和输出的
`OriginalShape` 统一为 `ViewShape`，避免一维存储描述导致 tiling 误判逻辑 rank。
该操作原地更新 descriptor，不改变 StorageShape、stride、offset 或数据布局；
连续布局要求及 dense/varlen 的 rank 校验保持不变。V2 入口不受本次修改影响。

## 执行流程

V2 由三个阶段组成，保存中间量模式跳过前向重计算：

| 阶段 | 工作 |
|---|---|
| Prepare | 生成 dAqk、dq_raw 及后续阶段所需数据 |
| Dhu | 反向扫描状态，生成 dh 与 dv_scan |
| Finalize | 合并 dq/dk/dv/db，计算 Gate 与参数梯度；可融合 Q/K 归一化反向 |

重计算模式先恢复 gk/w/qg/kg/v_new/h 等中间量，再执行同一反向链路。
A2/A3 与 A5 V2 总接口直接将原始 dtype 的 raw_g 传给 Recompute；BF16 在其 Vector
读入后转为 FP32，移除总接口的独立 Cast 及 FP32 GM 临时张量。FP32 输入路径、
计算公式和 kernel 同步不变；回归覆盖 BF16/FP32、四种 layout、定长/变长及 L2Norm 开关。
保存模式直接读取前向的 chunk-major h；Prepare 与 Finalize 使用对应偏移，
Finalize 对 h、dh 分别寻址，Dhu 和内嵌 state_scan 均产生 NT-first dh。
重计算模式由 ChunkFwdH 直接写出 chunk-major h，不再增加 Transpose/Contiguous；
保存模式不增加转换。h 与 dh 均按 dense `[B,NT,H,K,V]` 或 packed `[totalNT,H,K,V]` 分配。
入口约束见 [接口说明](api.md)。

## 数值处理

- 按 32 行分带，使用首尾 Gate 值的中点平移指数，并在指数计算前屏蔽无效位置。
- 矩阵计算使用 BF16 高位与残差补偿，累加及归约使用 FP32。
- 严格下三角 dA 放大 2^16 后计算，再乘 2^-16 还原；对角项单独处理，不计入 Gate 梯度。
- 保留标准 DFX 初始化，避免 BF16 冷启动转换异常。

## 存储与流水

- Prepare 使用 128 KiB UB。
- Finalize 使用每 AIV 248 KiB UB、每 AIC 128 KiB L1；双头窗口内每个 AIV 处理一个头。
- Q/K/beta 常驻片上，矩阵数据直接组织为 NZ，按分带写回输出。
- **O1**：A5 Finalize 的 Stage2、Stage5 dAqk、Stage10、dW residual 和 residual 五组 VF
  在同一成对循环内交错两组独立操作 A/B，循环外保留独立单尾；逐行/逐块算术、掩码及
  写回顺序不变。
- **O2**：Stage7 将 `exp2(g-center)` 的 FP32 bandExp 保存在已释放的 qPos 高位区
  [80,96) KiB，Stage9 按同布局复用；下一 band 才复用 qPos，UB 峰值、算术顺序和
  跨核旗均不变。
- **O5**：Stage3 计算 `exp2(g)` 后，将同一 FP32 结果暂存在当前 band 尚未产生最终结果的
  dg 区，Stage4 直接读取，去掉重复的 `Muls(LN2)+Exp`。两个 band 的区间不相交，
  Stage4 随后仍按原顺序写入最终 dg；不新增 UB，也不改变逐行计算和归约顺序。
- 本轮性能机制为上述 O1/O2、O5，以及后文 O4 原生 token-major do 输入。A5 raw tiling、
  DFX、host/ATK 资产和文档调整均为非性能修改，不据此分配性能收益。普通 beta 的
  既有转置不属于 O4，也不计入其收益。
- A5 host 的 `ChunkKdaBwdFinalizeTilingData` 逻辑大小为 184 字节，raw tiling size 设为
  `sizeof(TilingData)+8`，即 192 字节，与设备 `opParaSize` 一致；Arch22 路径不变。
- dg 及参数梯度部分和写入 workspace，采用固定顺序归约，不使用原子累加。
- 融合反向入口在同一块私有 workspace 内依次铺设 dv0/dq_raw/dAqk/dh/dv_scan/dAkk 六个跨阶段缓冲、
  Kernel B 与 Kernel C 区域；各区域按 512 字节对齐，跨阶段偏移以 512 字节为单位写入 tiling，
  因此在 tiling payload 大小不变的前提下，单次发射可寻址的私有 workspace 上限由 4 GiB 提升到 2 TiB。
- q_rstd/k_rstd 成对提供时，在 Finalize 内完成归一化反向，不增加第四个 kernel。
- A5/V2 的 dq/dk/dv 按公开 layout 写回，dg/db 固定按 token-major 写回；Finalize 分别
  使用 outputTokenMajor 与 gateOutputTokenMajor 选择地址和 stride。独立 Finalize 的
  output_layout 仍控制全部五个梯度，内部 model_layout 默认空字符串表示独立调用。
  输出 shape 仅用于校验，`T == H` 时不猜测布局；这些布局与 ABI 修正不属于 O1/O2。

Python 层统一处理参数校验、空序列压缩和规范 chunk 元数据。

### 通用 state_scan 初始化同步

通用融合路径的 state UB 使用 ping/pong 缓冲。正常加载通过
`MTE3_MTE2 → MTE2 DataCopy → MTE2_V` 保护复用；零初始化不执行
MTE2 DataCopy，因此等待缓冲可复用后仍需显式 Set/Wait `MTE2_V`，
将此前 MTE3 读取完成的依赖传给 Vector，再执行 Duplicate。
复用既有的逐缓冲事件，不新增事件资源或更改内存布局。

### WY/Intra 的掩码与源缓冲生命周期

WY 的逐行 Duplicate 使用两个 uint64_t 的位掩码数组，FP32 的高位掩码
显式置零，避免 API 读取数组外的元素。WY/Intra 的 FP32 Store 先用 Adds
将源 Plane 复制到输出队列；Adds 后执行 PIPE_V 屏障，再允许调用方覆写源
Plane。输出队列的事件继续负责队列缓冲与 MTE3 之间的同步。

WY 的 EndFusedMmadPhase 消费最后一个 FIX_M 信号后，通过 M_FIX 的
Set/Wait 将消费完成传回 FIX，再允许 BeginFusedMmadPhase 重发 FIX_M。
这使阶段重置的发送端等待接收端消费完成，避免 FIX 流水连续发送同一事件。

Finalize 的输入输出及源码入口见
[算子说明](../../chunk_kda_bwd_finalize/README.md)。


## 原始 beta 与 beta_eff 分离（A2/A3）

接口以 [api.md](api.md) 为准。本次数学公式、FP32 累加、分核、tile、workspace 和同步不变。
总接口从 beta descriptor 分配 db，从 beta_eff（融合）或转换后的 beta（不融合）选择计算输入。
Prepare/Dhu 不新增 beta 输入；Recompute/Finalize 共享所选有效 beta。
Finalize 注册额外两种组合：raw_g 为 BF16/FP32、有效 beta FP32、db BF16；原四种组合保留。
仅融合且 model_layout 路径可解耦有效 beta 和 db dtype，独立小接口的校验保持原约定。
A2/A3 vector 写回改用输出 DTYPE_D_BETA，使用已有每行 32 字节对齐的 UB 暂存和 DataCopyPad；输入仍用 BetaT。
梯度先在 FP32 中完成 sigmoid 导数，再在写回时一次 CAST_RINT 至 BF16，不新增 GM 临时张量或独立 Cast。
CPU 公式不变；新增 BF16 db 的预期为旧 FP32 梯度按模型 beta dtype 舍入。验证需覆盖原始 beta 两种 dtype、融合开关/scale=2、保存/重计算、四种 layout、两套 Python 后端、独立 Finalize 回归与非法缓存。


## A5 V2 单 layout 适配

本节扩展总接口；历史 legacy 的流水和 workspace 说明不代表 V2 实现。
公开契约以 [api.md](api.md) 为准。前向及 A5 重计算 H 限制保持原样。

| 阶段 | 当前设计 |
|---|---|
| Python/Stable/ctypes | 共用单一 layout 的校验和输出分配；A5 auto/optimized 路由 V2，legacy 拒绝新契约 |
| ACLNN V2 | 按 v 与 layout 解析维度；缓存固定 head-major；do 以原始 token-major 直接传给 Prepare/Dhu，普通 beta 转置一次供 Recompute/Finalize；A5 保留 sigmoid 参数并拒绝非默认值 |
| Prepare | 接收原始 token-major do；四维变长通过 B=1 的 reshape view 进入现有 NTD kernel |
| Dhu | 与 Prepare 共用原始 token-major do；内部可选标志控制实际行距，独立调用默认 head-major |
| Recompute | reserved 位分别控制 raw_g、q/k、v 的 stride 读取；两条 Vector 分支一致，缓存写出不变；BF16 raw_g 片上升精度，接收已转换的 head-major 普通 beta |
| Finalize | q/k、v、raw_g 分别按标志读取，有 rstd 时 q/k 固定 head-major；Cube 两处 v 读取同步调整；内部 beta 为 head-major，dq/dk/dv 随 layout，公开 dg/db 固定 token-major |

公开 do/raw_g/beta 始终为 token-major，dg/db 同样固定 token-major；dg 跟随 raw_g
dtype（BF16/FP32），db 跟随 beta dtype，A_log/dt_bias 及其梯度固定 FP32。
总接口不增加 input_layout/output_layout 参数；独立 Finalize 的 output_layout 语义不变。

不改变数学公式、累加 dtype、tile、分核、UB/L1 布局、workspace 或事件协议。
GM→UB 每行连续 128 元素，非连续行使用 DataCopyPad；GM→L1 保持原 CATLASS
搬运与尾块清零，只调整实际 GM leading dimension。beta 的 BF16/FP32 转换仍在片上完成。

验证分三层：保留独立小接口的 output_layout 组合为布局等价基线，使用独立 CPU
递推标杆检查数学结果，再检查真实前向缓存接入。覆盖定长/三维及四维变长、四种 layout、
`T == H`、尾块、空段压缩、L2Norm、raw_g/beta 四种 dtype、保存/重算、双 Python 后端及
直接 ACLNN 拒绝检查。当前 ATK 入口保持 BSND/TND，CPU golden 保持 head-major，
输出在比较前转回该排列；资产设计不替代实际设备结果。

BF16 raw_g 的 Stage 11 转换使用 BF16 全掩码，覆盖全部 128 个元素；此前误用 FP32 掩码会遗漏部分 BF16 lane，影响 dg/dA/dbias。本次同步修正，不改变 FP32 路径。

## A5 V2 原生 token-major do 输入

模型输入 do 始终为 BSND/TND，与 qkv 的 layout 无关。旧路径先在总接口执行
`SwapHeadToken + Contiguous`，生成 head-major dOHead，再由 Prepare 和 Dhu 顺序共用。
新路径直接下传原始 do，不再生成 dOHead：

```text
旧：do GM -> Transpose/Contiguous -> dOHead GM -> Prepare/Dhu L1/L0
新：do GM -- 原生 stride 读取 ----------> Prepare/Dhu L1/L0
```

Prepare 用 vNew、Dhu 用 dv 推导规范 B/H/T/V，总接口内部 L0 末尾传入
`doTokenMajor=true`，op_def optional attr 显式选择新寻址；独立小接口保持默认 false
和 head-major，公开 Python 与总 ACLNN 签名不变。T==H 也只由该标志消歧。
dense 地址为 `((b*T+t)*H+h)*V`，token 行距为 H*V，head 间距为 V；
packed 的 b 为 0。四维变长 Prepare 去 B=1 view、Dhu 补 B=1 view，均不搬运数据。

L1/L0 tile、UB/workspace、cache hint、分核和事件协议保持不变。A5 host 分别设置
Prepare/Dhu raw tiling 为 `sizeof(TilingData)+8`，并拒绝超出 uint32 行距的输入；
不支持的平台和非法布局由带参数上下文的结构化报错拦截。默认 false 路径、A2/A3
以及 legacy 的紧凑 tiling 不增加字段、不改变寻址。

普通 beta 仍由总接口转置一次为 head-major，供 Recompute/Finalize 共用；本项不改变
该数据流，也不计入 beta 的性能或显存变化。do 的独立性能与显存结果以 ATK README
和最终 wheel 的实测记录为准。
