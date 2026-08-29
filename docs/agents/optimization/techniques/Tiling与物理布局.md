# Tiling 与数据布局

本文描述 row tile、K 维累加、TilingKey、ND/NZ 和 consumer-friendly 物理布局的通用方法。

## 容量预算先于 Tile

Host tiling 先扣除固定内存，再选择 tile：

```text
UB = resident
   + input/output ping-pong
   + state ping-pong
   + direct-path slots
   + API temporary/guard
   + compute temporary

L1 = resident operands
   + L1A/L1B scratch ping-pong
   + per-head/group direct slots
```

所有预算使用真实 dtype、layout、对齐和 API 地址跨度。只按有效元素数计算可能低估 NZ 稀疏跨度或内部临时空间。

## Vector Row Tiling

从满足向量和搬运对齐的较大 row 开始，逐步缩小直到所有 UB 区域可容纳：

```text
row = candidate_max
while input + output + compute + resident + guard exceeds UB:
    row = next_smaller_aligned_value
```

- 不同矩阵路径可以使用不同 row tile。
- row tile 覆盖完整 K/V 维度时，更容易形成连续搬运和规则地址递推。
- tail row 使用独立有效长度和 mask，不改变完整 row 的快速路径。
- 两个 Vector subblock 按各自真实 UB 预算计算，不假设可共享 UB。

## K 维分块累加

完整 K 维无法一次放入 L0 时，沿归约维分块并累加到同一 L0C：

```text
for kTile in reduction tiles:
    MMAD(accumulate into the same L0C)
Fixpipe once after the final kTile
```

- 第一块设置正确的初始化语义，后续块只累加。
- 只有最后一块完成后发布 M->Fixpipe ready。
- 不要把归约维分块误写成多个独立输出。
- 不要在 K 循环中重复 Fixpipe 同一逻辑结果。

## Consumer-Friendly 布局

producer 可以直接生成 consumer 连续读取的物理布局，例如在矩阵侧完成等价转置或按目标 NZ 组织输出。采用前必须证明：

- 数学语义保持等价。
- consumer 的 stride、mask 和有效区定义完整。
- 公开输出仍满足接口要求。
- workspace 字节布局和测试 reference 使用相同语义。

不要让 Vector 为矩阵转置执行大范围散写，也不要为了连续访问改变公开 tensor 语义。

## ND/NZ 与直连

ND->NZ 或 UB->L1 直连必须核实当前 CANN 和目标 SOC 公共 API 的真实实现：

- API 是否实际经过内部临时 buffer。
- `dstNzC0Stride` 等参数产生的真实物理跨度。
- 临时 UB/LCM、scalar、VEC 和 MTE3 成本。
- 内部 helper 是否属于稳定可用接口。

固定稀疏 NZ stride 下，按 C0 列的 strided copy 有时比通用转换更稳定；紧凑 NZ 可以增加单次搬运，但要同步调整容量、slot 生命周期和信号表。

## TilingKey

TilingKey 只编码会改变编译期布局、指令路径或容量规划的维度，例如 dtype、固定 K/V 档位、chunk size 和真实计算分支。

- 连续运行时值留在 TilingData，不为单个 shape 生成模板。
- Host 生成 key 与 kernel 模板参数一一对应。
- 不同 SOC 仍复用同一个 L0 和模板体系。
- 新模板声明性能优势域和完整功能范围，不把边界 shape 变成长期 fallback。
- 所有模板覆盖完整/尾 chunk、所选完整/尾 head window 和支持的 dtype/layout。
