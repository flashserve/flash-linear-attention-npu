# ChunkKdaFwdPrepare A2 设计

融合 Q/K norm、gate prefix、因果 score、块三角逆和 W/U；接口与输出策略见
[API](api.md)，A5 实现见 [design.md](design.md)。chunk 间状态递推由后续算子负责。

## 调度与计算策略

chunk 固定 64，K/V 固定 128。一个 AIC 配两个 AIV，每个 AIV 处理一个完整 value head；
按 chunk 与 value-head pair 组成 work，使用 grid-stride 调度和双 slot workspace。
GVA 的 Q/K head 为 `floor(hv / (HV/HK))`；奇数 head 的 inactive lane 仍参与事件协议。
Q/K 检查点仅由对应 Q/K head 的首个 value head 写出。

| key | norm | gate | 叶内求逆 |
| --- | --- | --- | --- |
| 0 | Identity | Precomputed | FP32 64 阶 |
| 1 | L2 | Precomputed | FP32 64 阶 |
| 2 | Identity | Softplus | 两份 FP32 32 阶 |
| 3 | L2 | Softplus | 四份 BF16 Cube 16 阶 |
| 4 | Identity | SafeSigmoid | 两份 FP32 32 阶 |
| 5 | L2 | SafeSigmoid | 四份 BF16 Cube 16 阶 |

六个数学 key 共用 BF16 inverse/RHS relay；归一化、gate/prefix、score 后处理和 Cube
累加保留 FP32。g/beta 的四种 BF16/FP32 输入组合各包含这六个 key。
beta 模式、四档输出策略和 Exp/Exp2 由调用内不可变的 tiling 字段控制，不增加数学 key。
Precomputed FP32 gate 保留 TwoSum residual，避免相减消去误差。

## 流水

| Stage | 输入与计算 | 产物与交接 |
| --- | --- | --- |
| 0 Prepare | 按公开 layout 分块 DMA，norm、gate/prefix | UB Q/K/G/beta |
| 1 ScorePayload | FP32 Exp/乘法、BF16 Cast | GM Payload；MTE3 完成后 score-ready |
| 2 Scores | ND2NZ、LoadData、Mmad、Fixpipe | FP32 Raw；raw-ready 与 L1 scoreReleased 分别发布 |
| 3 RhsAndOutputs | 当前 RHS 与公开输出 | BF16 RHS；DMA 退休后复用 UB |
| 4 LowerAndAqk | Raw 后处理、适用 key 的 FP32 叶逆 | Lower/Aqk/叶逆；store 完成后 lower-ready |
| 5 InversePair | 叶逆与分块合并 | resident X；每轮 FIX_MTE1 后读取 |
| 6 PreloadWu/Wu | 同一 X 乘两份 RHS | W/U；全部消费者完成后释放 slot |

Stage 0 的 Q/K/g 有效 r 行由 DMA/Cast 完整覆盖，仅清零 `[r,64)` 尾行；beta 仍清零
全部 64 行。保留 V→MTE2、MTE2→V 事件及完整 64 行内部矩阵。
布局直接映射到 tile 的 DMA stride，不创建整张输入的 transpose 临时张量。

## 资源

地址由 [resource.h](../op_kernel/arch22/chunk_kda_fwd_prepare_resource.h) 统一定义，
host 进行 workspace 乘加溢出检查，device 使用相同上界。

| 资源 | 上界与用途 |
| --- | --- |
| 每 AIV UB | `0x2f800`（190 KiB）：Q/K、FP32 G、work、V、small、mask |
| 每 AIC L1 | 192 KiB，两 lane 各 96 KiB；score 与 inverse/RHS 按生命周期复用 |
| L0A/L0B/L0C | 各 64 KiB；A/B 两 lane 各 32 KiB，C 受 FIX_M 保护 |
| 每 lane GM | `0x26000`（152 KiB），分区见下表 |
| 每 core GM | 两 slot、每 slot 两 lane，共 `0x98000`（608 KiB） |

| GM 相对地址 | 大小 | 数据 |
| --- | --- | --- |
| `0x00000` | 72 KiB | Payload；适用 key 在 raw-ready 后复用低 8 KiB 交接叶逆 |
| `0x12000` | 32 KiB | FP32 Raw |
| `0x1a000` | 8 KiB | BF16 Lower |
| `0x1c000` | 8 KiB | Eye |
| `0x1e000` | 16 KiB | RhsK |
| `0x22000` | 16 KiB | RhsV |

`laneBase = core*0x98000 + slot*0x4c000 + lane*0x26000`。
总 workspace 为平台 API workspace 加 `usedCoreNum*0x98000`。
关闭的公开输出不作为 scratch，六个 key 使用相同资源上界。

## 同步约束

- scoreReleased 只释放 score 的 L1 读取租约，raw-ready 才表示 FP32 Raw 已可读。
- UB 覆盖前等待末次 MTE3 消费退休；Lower、RHS 的 ready 覆盖相应 store。
- resident X 保留到 W/U 两个消费者完成；L1 在末次 MTE1 后释放，GM slot 在 FIX 完成后释放。
- slot 复用等待上一代消费者的 free；inactive lane 同样执行配套握手。

实现入口见 [common.h](../op_kernel/arch22/chunk_kda_fwd_prepare_common.h) 与
[vector.h](../op_kernel/arch22/chunk_kda_fwd_prepare_vector.h)。
