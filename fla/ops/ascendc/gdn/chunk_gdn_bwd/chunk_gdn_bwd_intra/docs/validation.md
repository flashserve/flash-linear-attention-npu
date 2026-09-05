# ChunkGdnBwdIntra 开发期验证记录

> 本文件仅用于方案设计、实现和测试期间的过程门禁，不是最终交付件。
> 正式验收通过后，将有效结论收敛到算子 ATK README，并删除本文件及所有引用。

[API 文档](api.md) | [设计文档](design.md)

本文记录测试计划、设备实测和实现一致性，不定义接口或方案。

## 1. 验证与性能计划

设计通过后按 `docs/agents/04-operator-development.md` 和 `05-operator-testing.md`
进入实现。验证顺序固定为：

1. 以用户 CPU 标杆复现 `Gate`、A_bg/A_beta、W/U、S/D/dv 语义；D 分支直接验证
   `Gate(FP32(g[s])-FP32(g[t]))`，不使用指数相除或正负指数乘积。覆盖默认 exp2、自然指数、
   固定/变长、`M=1/32/33/64`、G=1/2/3/4、空 sequence 与尾 chunk。
2. 验证 ACLNN/ATB 接口、workspace size、BNSD shape/dtype、`chunk_indices` 和非法参数。
3. 增加固定长度 `G=4` 调度验证用例：`B=1,T=256,HK=24,HV=96,K=V=128,BT=64`、
   BF16、`use_exp2=True`。该用例检查每个 chunk 的 Stage 0 只计算 `HK=24` 份 Score，
   每份 Score 恰好服务 4 个 value head；Stage 1/2 各产生 `HV=96` 份结果，并逐项与
   CPU 标杆比较 w/u/dv_local。同步检查同一 `HV` 轴切片内四个 `r` 共用一个 Score：
   Stage 0 只搬一次 q/k、只做一次 MMAD，再把 Score 的前后 `BT/2` 行分别交付
   AIV0/AIV1；两个 AIV 都处理 `r=0/1/2/3`，各自只计算 A_bg/A_beta/D 的 `BT/2` 行。
   同时检查每个 `r` 的两个 GM 行半片都 ready 后 Stage 2 才搬入 L1、ring workspace 前后
   `BT/2` 行地址不重叠、两个 AIV
   各自末次消费后释放 Score，以及第一份 head ready 后 Stage 2 不等待其它三份 head。
   `CG=4` 时，输入 T 轴包含 `Nchunk=4` 个 chunk，输入 `HV=96` 轴包含
   `NhvSlice=ceil(96/4)=24` 个连续切片，因此 `Nwork=4*24=96`、
   `blockDim=min(96,AIC_CORE_NUM)`，每个 `ring_slot` 保存 4 个 value-head slot；当
   `AIC_CORE_NUM>=96` 时 A_bg/A_beta/D 三段 workspace 各为 3 MiB，总计 9 MiB。
4. 增加 `Nwork>blockDim` 的用例，使同一 MixBlock 依次处理多个 `(chunk,HV 轴切片)`；
   检查同一 `ring_slot` 必须先收到 `workspace_free`，才能写入后续 `(chunk,HV 轴切片)`，并逐 chunk/head 对比
   CPU 标杆，确认不存在跨任务数据污染。
   另用 `G=1,CG=4` 检查每个 AIV 有四个独立 Score 半片槽，AIC 可以在不等待当前
   `HV` 轴切片内 `score_free` 返回的情况下连续发布 r0/r1/r2/r3。
5. 在 A5 对模型 case `B=1,T=11K,HK=16,HV=32,K=V=128` 使用 `CG=4`，记录
   精度、workspace、wave/尾任务、Stage 0/1/2 流水和设备耗时；另增加 `G=3,CG=3`
   用例验证同一参数化模板。模型 case 的验收目标为设备耗时不超过 375 us。
6. 重点 profiling Stage 1 单次 VF 的寄存器压力、直接差值 GateDelta 计算和三段 ring workspace 带宽，
   以及 Stage 2 三次 MMAD 的 L0C 复用等待。若精度或性能要求需要改变分组，必须回写
   [设计文档](design.md)并重新完成整体方案评审。

## 2. A5 实现验证记录

### 2.1 泛化精度验证

2026-09-04 使用 ATK 26.8.8 的 `mixed_tolerance_bm` 模式，以 FP64 CPU 标杆对
Stage 2 三个正式输出进行混合容差比较。16 个用例分四批执行，每批 4/4 通过，合计
16/16 通过；测试时关闭 GM 初始化，避免测试框架的额外显存占用影响算子验证。
CPU 标杆、输入生成、16 个用例的逐项覆盖矩阵和执行命令统一记录在
[ATK README](../../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)，此处只保留实测结论。

对曾用于定位长时间无结果问题的固定长度组合
`B=1,T=65,HK=3,HV=6,K=V=128` 另做单用例复核，w/u/dv_local 的最大绝对误差分别为
`1.2079e-4`、`1.2095e-4`、`7.6257e-6`，均满足 `rtol=5e-3, atol=5e-3`。
该用例的算子调用和同步均正常返回；此前无结果发生在算子调用前的设备驱动互斥等待，
不属于算子执行超时。

### 2.2 已否决的优化候选

| 候选 | 精度结果 | 结论 |
| --- | --- | --- |
| 两级 work ring，将下一 work 的 Stage 0/1 与当前 Stage 2 交错 | 安全串行版本精度通过；实际交错版本因复合 L1 event 复用出现死锁 | 串行版本模型 case 为 526.428 us，慢于 514.651 us 基线；完整撤回 |
| Stage 0 对连续 leader 的 q/k 使用两条批量 ND2NZ | 单次 launch 正确；G=2、T=1024、HV=8 同进程重复 10 次后 `u max abs=0.609375` | 跨 launch 事件状态不稳定，未进入性能测试；完整撤回 |
| 将 work 线性化改为 HV 切片优先、chunk 次优先 | G=2、T=1024、HV=8 同进程重复 10 次精度通过 | 模型 case 50 次均值 510.199 us，对比 512.564 us 仅提升 0.46%；收益不足，完整撤回 |
| Stage 2 record 使用深度 2 的预载窗口 | G=2、T=1024、HV=8 同进程重复 10 次精度通过 | 模型 case 50 次均值 511.766 us，对比 512.564 us 仅提升 0.16%；收益不足，完整撤回 |
| `G=2` 利用 Stage 0 空置槽做跨 work 双 bank Score 前视 | 单 work `T=64` 精度通过；多 work `T=1024` 设备执行超时 | 插入的下一 work Stage 0 与当前 Stage 2 共享 MTE/L0 事件状态，发生死锁；完整撤回 |
| 将 `l1_record_free` 从 VF 前移到 MTE3 覆盖前，并在等待前预计算两个 Vector slot | 单 work `T=64` 精度通过；多 work `T=1024` 的 `w max abs=0.129867`、`dv_local max abs=0.239592` | 跨 work 地址复用不满足当前事件语义；完整撤回 |
| VF 后将三个 ND 半片转排为紧凑 NZ，把每个 head 的 MTE3 从 12 次降为 3 次 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 1.437 ms，转排开销远大于 MTE3 收益；完整撤回 |
| 当前 Stage 2 完成 `r0/r1` 后插入下一 work 的双 bank Stage 0，再放行下一 work Vector | 单 work `T=64` 精度通过；多 work `T=1024` 设备执行超时 | Score MMAD helper 不能在当前 Stage 2 生命周期内再次进入；完整撤回 |
| 默认 exp2 路径以 `1/exp(g*ln2)` 复用正指数，替代负指数计算 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 514.993 us，慢于 512.568 us 基线；A5 除法开销更高，完整撤回 |
| 删除旧 `workspace_free` work 级事件，仅保留逐 `r` 的 `l1_record_free` | G=2、T=1024、HV=8 的 `w max abs=0.209942`、`dv_local max abs=0.347656` | 同编号跨核事件跨 work 复用需要代际隔离，逐槽事件不足以取代该事件；完整撤回 |
| full-chunk VF 省略尾列越界比较与清零 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 513.152 us，与 512.568 us 基线无显著收益；避免增加模板体积，完整撤回 |
| `G=2` 双 bank q/k 预取，直接跨 Stage 2 保留 Stage 0 helper 的 MTE2->MTE1 event | G=2、T=1024、HV=8 设备执行超时 | 两个 MMAD helper 的本地事件状态冲突；改为无本地 event 的 MTE2 发射并在消费前以 `PIPE_MTE2` 收口 |
| `G=2` 双 bank q/k 预取，使用无本地 event 的 MTE2 并在消费前以 `PIPE_MTE2` 收口 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 515.938 us，屏障成本抵消预取收益；完整撤回 |
| `G=2` 独立 TilingKey，将 head 映射中的除法、取模和分支常量化 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 514.865 us，慢于 512.568 us 基线；完整撤回 |
| `G=2` 双代 Score/L1 事件，在当前 Stage 2 前完整执行下一 work 的 Stage 0 | G=2、T=1024、HV=8 精度通过，误差保持 `w=0.000961065,u=0,dv_local=0.00216728` | 模型 case 50 次均值 514.050 us，对比 512.568 us 基线无显著收益；AIC 关键路径未缩短，完整撤回 |

当前 G=1 实现基线保持三段独立 GM workspace，并将 Stage 0 Score 以 FP32 写入 AIV UB。
该实现仍使用整 slot `MTE3_MTE2` 和整 chunk ready，尚未满足 2.4 定义的目标同步方案；
以下精度和性能仅作为同步重构前基线，不能作为新方案验收结果。目标实现必须拆分
input/output free、按 `r` 发布 `workspace_ready`、由最后一次 workspace->L1 的
`PIPE_MTE2` 提前返回 `workspace_free`，并保留 Stage 2 双 L0C 的 `FIX_M` ready/free。

精度与确定性结果（BF16 主 dtype）：

```text
G=1, gate=BF16/FP32, use_exp2=true/false, T=64/96:
  同一进程连续 1000 次通过
  w max abs <= 9.62e-4, u max abs = 0, dv_local max abs <= 2.17e-3
G=4, gate=BF16/FP32, use_exp2=true/false, T=64/96:
  代表组合连续 500 次通过
```

以下是旧 `H=96,T=8192/16384` 实现的历史设备时间，不是当前模型 case 的性能验收数据：

| shape | fused | recompute | dv_local | baseline sum | fused / baseline |
|---|---:|---:|---:|---:|---:|
| `B=1,T=8192,H=96,K=V=128` | 2.160 ms | 1.358 ms | 0.956 ms | 2.314 ms | 0.934 |
| `B=1,T=16384,H=96,K=V=128` | 4.355 ms | 2.691 ms | 1.880 ms | 4.571 ms | 0.953 |

在独立 `W_KB/W_VB/W_D` GM 中转约束下，旧 shape 每个 head/chunk 的 AIC GM 读取量
约为 `96 KiB`：Stage 0 的 q/k 为 32 KiB，Stage 2 的 A、Kb、Vb、D、d_o 为 64 KiB。
两个基线合计约为 `104 KiB`，融合只减少一次 8 KiB 的 A 搬入，对应流量下界比例约
`96/104=0.923`；实测 0.934 与该下界一致。因此 `fused/baseline <= 0.5` 与当前三个
独立 GM 边界不相容。继续冲击 0.5 前必须返回方案设计阶段，评审取消或绕过至少部分
Vector-to-Cube GM 中转；不能在实现阶段静默违反已评审 workspace 约束。

## 3. 开发参考与实现一致性追溯

开发阶段读取的分块独立开发参考为 `CHUNK_INDEPENDENT V1`，来源提交
`a78efa10258e4021bf248ef0d4c0d9d38a136e06`。本算子采用其中的独立 Stage 接口、生产者/
消费者所有权、ready/free 生命周期和有限环形 GM workspace 原则；参考中指向的示例文件
在当前提交不存在，因此不复制其代码，绝对 L1/UB 地址、`HV` 轴连续切片和同步编号均以
[设计文档](design.md)为准。

当前追溯表是开发门禁，不是验收结论。标记为“待完成”的项目完成并补齐实测证据前，
不得进入完整测试或性能验收。

| 设计项 | 代码位置与实际实现 | 验证用例或 profiling 证据 | 状态 |
| --- | --- | --- | --- |
| 三个 Stage 的 `r=0..CG-1` 映射 | `chunk_gdn_bwd_intra_fast_g1.h` 的 AIC/AIV `Process`；当前仍按整 chunk 发布 Stage 1 ready，目标是两个 AIV 都遍历全部 `r`，并分别发布前后 `BT/2` 行 ready | 待记录 `CG=3/4` 的 AIC 顺序及两个 AIV 行半片顺序 | 待完成 |
| `Process<CG>` 统一逻辑模板和 shape 命中 | 当前实现尚未形成 2.4 中按输入 `HV` 轴连续切片的统一骨架；目标为 `G=3` 取 `CG=3`，其它支持场景取 `CG=4` | 待完成 `G=3,CG=3` 和其它 `G,CG=4` 的精度、workspace 与性能验证 | 待完成 |
| Stage 0/2 L1 操作区 | 当前 MMAD helper 仍使用连续内部 stage，尚未对应 2.1/2.3 的绝对区间 | 待检查 Stage 0 按 `r` 固定预留的 `CG` 个 32 KiB 槽、Stage 2 的 `CG` 个 56 KiB 槽，以及 `CG=4` 时 352 KiB、`CG=3` 时 264 KiB 的峰值 | 待完成 |
| AIV 驻留槽和 Vector ping/pong | 当前 TPipe 队列深度为 1，尚未对应 2.2 的 `CG` 份 Score 半片槽及两个 Vector ping/pong 槽 | 待用 `G=1,CG=4` 证明四份 Score 可连续发布，并用 `msprof op` 证明 MTE2/V/MTE3 重叠 | 待完成 |
| Score 与 Stage 1 ready/free | Score ready/free 已闭环；当前 prep ready 仍是整 chunk 粒度，目标为两个 AIV 分别按 `r` 发布 GM 半片 `workspace_ready`，AIC 等同一 `r` 的两个 ready | `T=64/1088/4096,G=1` 精度通过；逐 chunk 误差为 0 | 部分完成 |
| Stage 1 slot 复用 | 当前使用整 slot `MTE3_MTE2`；目标按不重叠地址拆为 input_free 和 output_free，使下一轮 MTE2 不等待上一轮 MTE3 | 待采集 MTE2/V/MTE3 重叠证据 | 待完成 |
| workspace 公式和 slot 生命周期 | 当前实现仍按旧的整 chunk ring 计算；目标为 A_bg/A_beta/D 各按 `blockDim*CG*8 KiB` 分配，两个 AIV 写入同一 `rho` 的前后 `BT/2` 行，AIC 将该 `rho` 的三个 record 搬入 L1 后立即返回两份 credit | 现有 `T=4096,G=1` 仅作为旧实现基线；新公式待 `CG=3/4` 验证 | 待完成 |
| ping/pong 实际重叠 | 尚无两份独立 UB 存储及其 ready/free 证据 | 待采集 `msprof op` 流水 | 待完成 |
