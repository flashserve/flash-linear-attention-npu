# Arch22 设计伪代码

本目录给出 A2/A3（Arch2201）的 `chunk_kda_fwd_prepare` 八 Stage 设计伪代码。它与 Arch35
共享 `V0 -> V1 -> C2 -> V3 -> C4 -> C5 -> V6 -> C7` 数学边界和 S=4 causal-prefix
语义，但使用独立的资源、搬运和同步合同。该目录不进入 CMake、没有设备 kernel/Host ABI，
因此仍不能作为可构建或可调用的算子支持声明。

## 执行映射

- kernel 形态为一组 AIC 配两个 AIV；四 head 逻辑组拆成两个 pair wave。
- AIV0 处理 group-local head 0/2，AIV1 处理 1/3。每个 pair 的缺失 tail head 仍参加
  mode `0x2` collective 的 dummy arrive/wait，但不计算地址、不访问 GM/UB。
- 同一 AIV 依次执行 `V0(slot0) -> V1(slot0) -> V0(slot1) -> V1(slot1)`，保证唯一
  shared arena 中的 G 在 V1 最后 reader 前不被下一 head 覆盖。
- Prepare 保持 chunk-first 分核；chunk task 不足时才增加完整 HK cohort pack 维度，同一 HK
  映射的全部 HV 不跨 workgroup。

## 固定资源账本

- 普通 UB 使用 `[0,0x2E000)` 共 184 KiB：private0 72 KiB、shared 40 KiB、private1
  72 KiB；`[0x2E000,0x30000)` 保留给 CANN。V1 的 72 KiB score 全部落当前 private，
  score MTE3 drain 时下一 private 可继续 V0/V1。
- L1 保留四份 72 KiB current lane 和四份跨 Cube stage resident，峰值 `0x5C000`，
  不因物理 pair wave 缩减用户要求的四份常驻空间。
- L0C 只有两条 64 KiB physical lane。C2/C4/C5/C7 分别取得 stage-use ticket，
  每个 stage 的最后 Fixpipe reader 完成后才归还 lane。
- L0A/L0B 也使用两条 physical lane，但具有独立的 operand generation。每个有效 C2 band、
  C4、C5、C7 按实际 AIC 发射顺序在所属 lane 上递增；同一 epoch 的全部 operand 与 MMAD reader
  完成后，才能通过 `CubeToMte1OperandReuse` 允许 MTE1 覆盖。L1/L0C ticket 不能替代该合同。
- 每 workgroup 使用四个 workspace slot，单 slot `0x22400`，总 stride `0x8A000`。control page
  的 `[0x89000,0x89080)` 保存四条 32 Byte Q/K cache generation/state record，后续 `0x80`
  Byte 是统一格式 pad，其余为 reserve。

## Arch22 数据通路

- C2 只计算 tail 中有效的 `ceil(M/16)` 个 band，将 compact Aqk/Akk 通过 Fixpipe 写 GM；
  V3 仅 MTE2 读取已定义的 compact 矩形。Arch22 不假定 L0C 可以直写 AIV UB。
- C4 将 FP32 `T` 写入 GM relay；C5 stage 入口通过已命名的 FIX->MTE2 依赖重载到 L1，
  并独立等待 C4 的 X1 MTE2 resident ready，不在 C4 内消费当前 stage 的 Cube 输出。
- `validRows<=32` 时不存在 q10，C5 不启动无效 MMAD，也不伪造 Fixpipe 完成；它在消费本 stage
  的 L0C credit 后通过 control path 发布 Akk ready 和 next credit。该 tail 下 V3 同时省略 VCS
  GM drain，C4 省略 VCS load、T MMAD 和 T GM relay，只保留必要的 L1/L0C control credit 闭环。
- V3 不初始化 q10，C5 是 q10 的唯一 writer。Current 的 q01 是公开 Akk 输出；Fused 只在
  `M>32` 的完整 ND->Cube-ready 转换 fallback 中 relay q01，top-only 不物化无人消费的 q01。
  C7 full 从标准 row-major relay 一次转换完整 Akk；top-only 只把有效 q00 直接装成 tight 32x32
  Cube operand，禁止从不完整的 64x64 NZ 视图按 row-major offset 派生 q00。
- V6/C7 的两个 RHS plane 使用固定的 16 KiB 物理间距。`M<=32` 时每个 plane 只传前 8 KiB，
  C7 做两次 `32x128x32` MMAD；`M>32` 时各传 16 KiB并做两次 `64x128x64` MMAD。
- V0 将固定 64 个 FP32 `betaEff`（256 Byte）写入 context，V3/V6 各重载完整 256 Byte，
  但 VF 只消费 `validRows`；这是 Arch22 UB 容量证明后的 GM relay 例外。
- 每个 HK cohort 仅首个 HV 从 GM 装入 raw Q/K 并在 V0 做一次所选 norm；其他 HV 通过
  level-triggered cache-ready 状态读取同一份已舍入 Qhat/Khat。每个 mapped HV 的 V6 完成
  Q/K MTE2 source read 并发布 `V6RhsReady`；C7 按 pair 顺序汇聚完整 cohort 后由 AIC 归还
  一次 cache free。逻辑最后 HV 不能自行 free，因为另一 AIV 可能尚未完成 cache 读取。

## 同步边界

workspace、private UB、shared UB、L1、L0A/L0B operand、L0C 使用互不混淆的 owner ticket。
L0A/L0B 的两个 bank 是同 AIC 核内 epoch，不额外发布跨核 flag。跨 AIC/AIV 的
score/raw/VCS/payload/RHS/slot credit 以 pair collective 表达：两个 AIV 各 arrive/wait 一次，
AIC 每 pair wait/publish 一次。逻辑 SyncPoint 最终必须通过少量有反向 credit 的物理 channel
复用，不能把每个枚举机械映射成独立 flag，也禁止 `PIPE_ALL`。

Q/K cache ready 不属于单消费者 pair collective。它在 control record 中按 generation 发布，
同 cohort 的多个 AIV/HV 可 acquire 读取且不清除状态。V6 的 pair collective 汇聚两个 AIV，
C7 依次观察覆盖 cohort 的全部 pair 后，由 AIC coordinator 单独归还 free generation。

这些核间 ticket 不替代核内 pipe 依赖。Vector 明确保留 `MTE2->V`、`V->MTE3`；Cube 明确保留
`MTE2->MTE1`、`Cube/M->MTE1` operand-release、`Cube/M->Fixpipe`，并为 GM relay、payload overlay 与 C7 fill/load WAW 单独
具名。Fixpipe->MTE2 relay 只在真实 Fixpipe producer 存在时启用：C4 T -> C5，以及
`M>32` 的 C5 q10 -> C7；top-only C5 control path 不伪造该事件。它们仍是 **PROPOSED**
设备边，必须按目标 CANN 头文件验证成对 event。

## 实现准入

下列事项仍标为 **PROPOSED**，正式 Arch22 kernel 编码前必须针对目标 CANN 版本查头文件并做
最小编译：S=4 变 N MMAD、Fixpipe compact stride、GM->L1 ND2NZ、FP16/BF16 cast、8 KiB VF
scratch、mode `0x2` flag ID/深度及全部 HardEvent。`AkkStorage::Fp32Internal` 当前明确拒绝。
host C++17 语法检查只验证伪代码接口闭合，不等价于 A2/A3 设备编译、精度或性能验证。
