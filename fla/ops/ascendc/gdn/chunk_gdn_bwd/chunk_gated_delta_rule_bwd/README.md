# ChunkGatedDeltaRuleBwd

## 功能

`ChunkGatedDeltaRuleBwd` 实现 Gated Delta Rule 的分块反向计算。所有支持的 SoC 统一
依次调度 `ChunkGdnBwdIntra`、`ChunkFwdH`、`ChunkGatedDeltaRuleBwdDhu`
和 `ChunkGatedDeltaRuleBwdFinalize`。当前实现支持定长和变长序列、GVA、可选初始状态、
Q/K L2Norm 反向以及 beta sigmoid 反向。

A2/A3 不再组合 `RecomputeWUFwd` 和 `ChunkBwdDvLocal`，而是与 A5 一致直接调用融合算子
`ChunkGdnBwdIntra` 重计算内部 W/U 与局部 Value 梯度；W/U 保留在 executor 内部。
BF16/FP32 gate、`useExp2` 与 varlen（空序列无 chunk 记录，天然跳过）均由该内核原生
处理，不再做自然指数转换或空序列重编号。`recompute_w_u_fwd`、`chunk_bwd_dv_local`
两项构建依赖暂随 arch22 包保留，executor 已不再调用。

A2/A3 在 ACLNN 层处理非连续输出暂存和输出重叠隔离，具体规则见[输出存储关系](#输出存储关系)。
公共 Host 对象保留 A2/A3 与 A5 的实现，并在运行期按 `NpuArch` 选择平台分支。

本算子在单个 ACLNN executor 中依次调度以下 L0 算子。

1. `ChunkGdnBwdIntra`（`chunk_gdn_bwd_intra`），重计算 W/U 并生成局部 Value 梯度；
2. `ChunkFwdH`（`chunk_fwd_h`），重计算分块状态和新 Value；
3. `ChunkGatedDeltaRuleBwdDhu`（`chunk_gated_delta_rule_bwd_dhu`），反向扫描状态梯度；
4. `ChunkGatedDeltaRuleBwdFinalize`（`chunk_gated_delta_rule_bwd_finalize`），生成最终输入梯度。

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
相邻相等的累计长度表示空序列，不产生 chunk，各阶段统一使用原始序列编号和状态布局，
空序列因无 chunk 记录被内核天然跳过。提供 initial state 时，空序列
的 `dh0` 等于对应 `dht`，未提供 dht 时为零；未提供 initial state 时，`dh0` 仍为空。
`qRstdOptional/kRstdOptional` 仅在 `useQkL2normInKernel=true` 时提供；`betaRawOptional`
仅在 `useBetaSigmoidInKernel=true` 时提供。

## 输出

接口固定返回八个梯度槽位：

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `dq` | 必选 | 与 q 同 shape/dtype | Query 梯度 |
| `dk` | 必选 | 与 k 同 shape/dtype | Key 梯度 |
| `dv` | 必选 | 与 v 同 shape/dtype | Value 梯度 |
| `dBeta` | 必选 | `[B,T,Hv]`；与 beta 同 dtype | Delta 系数梯度，固定为 BSN |
| `dG` | 必选 | `[B,T,Hv]`；与 g 同 dtype | 门控值梯度，固定为 BSN |
| `dh0` | 可选 | 与 `initialStateOptional` 同 shape/dtype | 初始状态梯度；未提供 initial state 时为空 |
| `dALog` | 预留 | 当前为空 | 当前不计算，ACLNN 输出描述符必须为 `nullptr` |
| `dDtBias` | 预留 | 当前为空 | 当前不计算，ACLNN 输出描述符必须为 `nullptr` |

`dALog` 和 `dDtBias` 不使用 shape 为 `[0]` 的占位 Tensor；Torch 接口对应槽位固定返回
`None`。

在 A2/A3 上，`dq/dk/dv` 和非空 `dh0` 支持非连续输出 view。L2 为需要连续写入的 L0 分配临时输出，
计算结束后通过 ViewCopy 写入原 view。连续 BSND/TND 的 `dq/dk/dv` 由 kernel 直接写入公开布局；
非连续输出保留同布局暂存到 view 的回填。布局搬运详见 [设计](docs/design.md)。`dh0` 保持
`stateVFirst` 指定的状态布局。未提供 initial state 时，即使提供 dht，公开 `dh0` 仍为空。
A5 沿用原有输出处理，设备验证状态见对应版本的 ATK 记录。

### 输出存储关系

A2/A3 对 `dq/dk/dv/dBeta/dG` 五个 Finalize 梯度逐个进行重叠判断。BNSD/NTD 与 BSND/TND 的连续
`dq/dk/dv` 在 Host 可判定其与公开输入及其他四个梯度均不重叠时直接写入。
Host 同时检查实际地址、view offset、连续性、字节跨度和 storage 绑定关系，无法排除重叠时按可能重叠处理。
相应输出由 ACLNN 按需申请独立临时目标，Finalize 完成后以显式 `CopyNpuToNpu` 回写，
后续再完成公开输出的布局与 dtype 转换。该路径会增加临时存储及搬运任务，workspace 以本次
`GetWorkspaceSize` 返回值为准。

五个梯度的回写顺序为 `dq → dk → dv → dBeta → dG`，对应独立 Finalize 的
`dq → dk → dv → dbeta → dg`。这五个输出的 view 相互重叠时，重叠字节最终保留顺序较后输出写入的值。
同时保留五个完整梯度需要五个输出 view 互不重叠。逐项判定和回写流程见
[Finalize 输出存储关系](../chunk_gated_delta_rule_bwd_finalize/docs/api.md#输出存储关系)，
executor cache 的隔离方式见[设计说明](../chunk_gated_delta_rule_bwd_finalize/docs/design.md#63-workspace-结构与搬运)。

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

未支持的属性组合会返回参数错误，不会静默切换到其他实现。

Python 入口的 `return_intermediate_states` 是既有调用的兼容参数，其取值会被忽略。

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

A2/A3 与 A5 复用同一条 `ChunkGdnBwdIntra` 组合链。两个平台的 Intra 仍有不同的
BF16 存储边界：arch22 先量化缩放后的 K/V 和 KQ score，arch35 先量化列缩放后的 A
和带 gate 的 score。因此本次独立测试快照的 CPU FP32 benchmark 按平台对齐中间
存储，CPU FP64 golden 保持高精度数学语义；分支中的 ATK 文件保留既有基线。
非单调 gate 值域的覆盖结论仍以对应设备、版本的 ATK 记录为准，不能据前向有效
gate 的通过结果宣称已覆盖任意 gate 值域。

## 测试

### 精度

[ATK 用例与执行命令](../../../../../../tests/atk/chunk_gated_delta_rule_bwd/README.md) 提供仓内测试入口。
NPU 节点调用公开融合接口，CPU 节点独立串联上述四个阶段的标杆，比较全部有效梯度。
仓内基础用例使用 `mixed_tolerance_bm`。ATK 说明中的 A2 专项回归使用 `cv_fused_double_benchmark` 双标杆。

用例需要覆盖 `BNSD/BSND/NTD/TND`、`Hv/Hk in {1,2,3,4}`、完整 chunk、尾块、变长序列、
有无 initial state、`stateVFirst=true/false`、Q/K L2Norm 开关和 beta sigmoid 开关。
`aLogOptional/dtBiasOptional` 需要覆盖非空传入但不参与计算；`dALog/dDtBias` 必须验证为
空输出而不是 shape 为 `[0]` 的 Tensor。

### 性能

[Issue #661](https://github.com/flashserve/flash-linear-attention-npu/issues/661) 的 A2/A3 性能验收使用
Python 入口 `fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_bwd`。两组典型用例共用 `B=1`、
`Hk=16`、`Hv=32`、`K=V=128` 和 `chunk_size=64`。Q/K/V 使用 BF16，g/beta 使用 FP32。

| 场景 | T | 序列长度 |
| --- | --- | --- |
| 定长 | 11264 | 11264 |
| 变长 | 65536 | 平均 1K |

在 A2、A3 和 A5 上使用相同输入和调用参数。变长场景固定并记录 `cu_seqlens`。
使用 `msprof` 采集 `op_summary`，按 `Task Duration(us)` 汇总完整融合调用的计算和布局转换任务。
记录各平台的软件版本、设备信息和原始采样，以 A5 的同口径结果作为对照。
