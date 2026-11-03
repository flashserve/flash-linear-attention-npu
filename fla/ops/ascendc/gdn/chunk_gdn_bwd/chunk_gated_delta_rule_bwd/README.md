# ChunkGatedDeltaRuleBwd

## 功能

`ChunkGatedDeltaRuleBwd` 实现 Gated Delta Rule 的分块反向计算。A5 上依次调度
`ChunkGdnBwdIntra`、`ChunkGatedDeltaRuleFwdH`、`ChunkGatedDeltaRuleBwdDhu` 和
`ChunkGatedDeltaRuleBwdFinalize`。当前实现支持定长和变长序列、GVA、可选初始状态、
Q/K L2Norm 反向以及 beta sigmoid 反向。

A2/A3 使用已有 `RecomputeWUFwd` 和 `ChunkBwdDvLocal` 生成内部 W/U 与局部 Value 梯度，
再调用相同的 FwdH、DHU 和 arch22 单 kernel Finalize。W/U 保留在 executor 内部。
前两个 A2/A3 组件使用自然指数，`useExp2=true` 时在内部将 FP32 gate 乘以 ln(2)；
后续算子仍接收原始 gate 和原始 `useExp2` 属性。

A2/A3 新增前序和非连续输出暂存仅在其私有分支执行。纯 `ascend950` 构建不编入
这些新增 Host helper，也不加入 W/U、local-dV 两项额外依赖；A5 保持原有 Host 流程。
含 A2/A3 的混合 SoC 构建仍共用 Host 对象，并在运行期选择平台分支。

用于组合一致性对比的公开算子链依次由以下算子组成：

1. `ChunkGdnBwdIntra`（`chunk_gdn_bwd_intra`），重计算 W/U 并生成局部 Value 梯度；
2. `ChunkGatedDeltaRuleFwdH`（`chunk_gated_delta_rule_fwd_h`），重计算分块状态和新 Value；
3. `ChunkGatedDeltaRuleBwdDhu`（`chunk_gated_delta_rule_bwd_dhu`），反向扫描状态梯度；
4. `ChunkGatedDeltaRuleBwdFinalize`（`chunk_gated_delta_rule_bwd_finalize`），生成最终输入梯度。

本算子不包含独立 kernel，而是在单个 ACLNN executor 中直接组合上述 L0 算子。公开 NPU
算子链仅用于检查组合顺序、参数传导和布局转换；独立精度标杆使用四个阶段对应的 CPU
reference，避免 DUT 与标杆复用同一 kernel。

## 输入

令 `B` 为物理 batch size，`Hk` 为 q/k 头数，`Hv` 为 v 头数，`T` 为 token 数，
`K=128`，`V=128`，`N` 为逻辑序列数，`Nc` 为 chunk 总数。

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | 由 `layout` 决定；BF16 | Query |
| `k` | 必选 | 与 q 同 shape/dtype | Key |
| `v` | 必选 | 由 `layout` 决定；BF16 | Value |
| `g` | 必选 | `[B,T,Hv]`；BF16/FP32 | 前向保存的 chunk 内累计门控值，固定 BSN |
| `beta` | 必选 | `[B,T,Hv]`；BF16/FP32 | 前向使用的 Delta 系数，固定 BSN；内部按 g/beta 的共同较高精度计算 |
| `A` | 必选 | `[B,Hv,T,64]`；BF16 | 前向保存的 chunk 内系数矩阵，固定为 BNSD |
| `dO` | 必选 | `[B,T,Hv,V]`；与 v 同 dtype | 输出 `O` 的上游梯度，固定 BSND |
| `initialStateOptional` | 可选 | `stateVFirst=false` 时为 `[N,Hv,K,V]`，否则为 `[N,Hv,V,K]`；BF16 | 前向初始状态；非空时输出 `dh0` |
| `dhtOptional` | 可选 | 与 initial state 的状态布局相同；BF16 | 最终状态的上游梯度 |
| `qRstdOptional` | 可选 | `[B,Hk,T]`；FP32 | Q L2Norm 反向保存值，固定 BNS |
| `kRstdOptional` | 可选 | `[B,Hk,T]`；FP32 | K L2Norm 反向保存值，固定 BNS |
| `betaRawOptional` | 可选 | `[B,T,Hv]`；与 beta 同 dtype | beta sigmoid 变换前的输入，固定 BSN |
| `aLogOptional` | 预留 | 当前不限制 | 当前仅保留接口槽位，不参与计算 |
| `dtBiasOptional` | 预留 | 当前不限制 | 当前仅保留接口槽位，不参与计算 |
| `cuSeqlensOptional` | 可选 | `[N+1]`；INT64 | 变长序列累计长度，需与 `chunkIndicesOptional` 同时提供 |
| `chunkIndicesOptional` | 可选 | `[2*Nc]`；INT64 | canonical sequence-major chunk 索引 |

四种 Q/K/V 布局均使用四维输入：

| layout | q/k | v |
| --- | --- | --- |
| `BNSD`、`NTD` | `[B,Hk,T,K]` | `[B,Hv,T,V]` |
| `BSND`、`TND` | `[B,T,Hk,K]` | `[B,T,Hv,V]` |

`layout` 只控制 q/k/v 和 dq/dk/dv。g、beta 固定为 BSN；可选的 q/k rstd 固定为 BNS；
dO 固定为 BSND，可选的 betaRaw 固定为 BSN，与 q/k/v 的 `layout` 无关。

`Hv` 必须能被 `Hk` 整除。变长模式要求物理 `B=1`；`cuSeqlensOptional` 必须从 0 开始、
以 `T` 结束且单调不降，`chunkIndicesOptional` 按逻辑序列和 chunk 顺序排列。
相邻相等的累计长度表示空序列，不产生 chunk。A2/A3 仅为无状态 W/U 重计算准备剔除空序列并
重编号的内部 metadata；其它阶段保留原序列编号和状态布局。提供 initial state 时，空序列
的 `dh0` 等于对应 `dht`，未提供 dht 时为零；未提供 initial state 时 `dh0` 仍为空。
`qRstdOptional/kRstdOptional` 仅在 `useQkL2normInKernel=true` 时提供；`betaRawOptional`
仅在 `useBetaSigmoidInKernel=true` 时提供。

## 输出

接口固定返回八个梯度槽位：

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `dq` | 必选 | 与 q 同 shape/dtype | Query 梯度 |
| `dk` | 必选 | 与 k 同 shape/dtype | Key 梯度 |
| `dv` | 必选 | 与 v 同 shape/dtype | Value 梯度 |
| `dBeta` | 必选 | `[B,T,Hv]`；与 beta 同 dtype | Delta 系数梯度，固定为 BSND |
| `dG` | 必选 | `[B,T,Hv]`；与 g 同 dtype | 门控值梯度，固定为 BSND |
| `dh0` | 可选 | 与 `initialStateOptional` 同 shape/dtype | 初始状态梯度；未提供 initial state 时为空 |
| `dALog` | 预留 | 当前为空 | 当前不计算，ACLNN 输出描述符必须为 `nullptr` |
| `dDtBias` | 预留 | 当前为空 | 当前不计算，ACLNN 输出描述符必须为 `nullptr` |

`dALog` 和 `dDtBias` 不使用 shape 为 `[0]` 的占位 Tensor；Torch 接口对应槽位固定返回
`None`。

在 A2/A3 上，`dq/dk/dv` 和非空 `dh0` 支持非连续输出 view。L2 为需要连续写入的 L0 分配临时输出，
计算结束后通过 ViewCopy 写入原 view；BNSD/NTD 的连续输出保持直接写入。
BSND/TND 仍按原布局转换后写回，`dh0` 保持 `stateVFirst` 指定的状态布局。
A5 沿用原有输出处理，本次不扩展其非连续输出支持范围。
未提供 initial state 时，即使提供 dht，公开 `dh0` 仍为空。

## 属性

| 名称 | 当前支持范围 | 说明 |
| --- | --- | --- |
| `layout` | `BNSD/BSND/NTD/TND` | 仅控制 q/k/v 及 dq/dk/dv 的公开布局 |
| `scale` | 有限浮点数 | Query 缩放因子，通常为 `K**-0.5` |
| `chunkSize` | `64` | 分块大小 |
| `useExp2` | `true/false` | gate 分别使用以 2 为底或自然指数语义 |
| `useGateInKernel` | 仅 `false` | 当前要求传入前向已处理的门控值 |
| `useQkL2normInKernel` | `true/false` | 为 true 时必须提供 q/k rstd |
| `useBetaSigmoidInKernel` | `true/false` | 为 true 时必须提供 beta raw 输入 |
| `stateVFirst` | `true/false` | 控制 initial state、dht 和 dh0 的末两维为 `[V,K]` 或 `[K,V]` |
| `returnIntermediateStates` | 预留 | 当前不改变公开返回值或 executor 内部依赖 |

未支持的属性组合会返回参数错误，不会静默切换到其他实现。

## 支持范围

- 实现支持 Ascend 910B、Ascend 910C（`ascend910_93`）和 Ascend 950；
  910B/910C 按 `DAV_2201` 共用 arch22 路径，验证结论以对应设备、版本的 ATK 记录为准。
- 支持 BF16 主输入；完整 Bwd 的 g/beta 可分别选择 BF16 或 FP32，包含混合 dtype。
  executor 先将二者统一为共同较高精度（任一为 FP32 则均用 FP32），再调用内部 Finalize。
  独立调用 [Finalize](../chunk_gated_delta_rule_bwd_finalize/docs/api.md) 时，g/beta 必须同为 BF16 或同为 FP32。
  dq/dk/dv 分别跟随 q/k/v；dG/dBeta 分别转回原始 g/beta dtype，混合输入时两路标量梯度的 dtype 也可不同。
- 支持 `K=128`、`V=128`、`chunkSize=64`。
- 支持 `Hv/Hk in {1,2,3,4}`，覆盖 MHA 和 GVA。
- 支持固定长度和变长序列；变长模式要求物理 `B=1`。
- 支持 `BNSD`、`BSND`、`NTD` 和 `TND` 四种四维公开布局。
- 支持可选 initial state、dht、Q/K L2Norm 反向和 beta sigmoid 反向。
- `aLogOptional`、`dtBiasOptional`、`dALog` 和 `dDtBias` 当前仅保留接口槽位。

本组合的规格受 FwdH 等前序组件约束，仍为 `K=V=128`、`chunkSize=64`、
`Hv/Hk in {1,2,3,4}`。独立 A2/A3 Finalize 的 V=256、chunkSize=128 或更大头比例支持
不代表整个 Bwd executor 已支持这些规格。

A2/A3 组件复用尚需验证两个数值边界：RecomputeWUFwd 与 A5 Intra 的 BF16 中间舍入位置不同；
ChunkBwdDvLocal 对 gate 差取 `min(delta,0)`，与非单调 gate 的直接指数公式可能不同。
新增路径保留原 CPU 标杆和混合精度阈值，不能据前向有效 gate 的通过结果宣称已覆盖任意 gate 值域。

## 测试

测试分为以下两层：

1. 组合一致性测试：分别调用四个公开 NPU 算子，对比 `dq/dk/dv/dBeta/dG/dh0`，验证
   L2 组合顺序、可选参数传导、公开布局转换和 `stateVFirst` 状态转换。
2. ATK 独立精度测试：组合 `ChunkGdnBwdIntra`、`ChunkGatedDeltaRuleFwdH`、
   `ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize` 已有的 CPU reference，
   使用 `mixed_tolerance_bm` 对比全部有效梯度。

用例需要覆盖 `BNSD/BSND/NTD/TND`、`Hv/Hk in {1,2,3,4}`、完整 chunk、尾块、变长序列、
有无 initial state、`stateVFirst=true/false`、Q/K L2Norm 开关和 beta sigmoid 开关。
`aLogOptional/dtBiasOptional` 需要覆盖非空传入但不参与计算；`dALog/dDtBias` 必须验证为
空输出而不是 shape 为 `[0]` 的 Tensor。
