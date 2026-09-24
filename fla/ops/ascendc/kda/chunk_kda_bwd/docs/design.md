# KDA 反向优化设计

## 执行流程

V2 由三个阶段组成，保存中间量模式跳过前向重计算：

| 阶段 | 工作 |
|---|---|
| Prepare | 生成 dAqk、dq_raw 及后续阶段所需数据 |
| Dhu | 反向扫描状态，生成 dh 与 dv_scan |
| Finalize | 合并 dq/dk/dv/db，计算 Gate 与参数梯度；可融合 Q/K 归一化反向 |

重计算模式先恢复 gk/w/qg/kg/v_new/h 等中间量，再执行同一反向链路。
保存模式直接读取前向的 chunk-major h；Prepare 与 Finalize 使用对应偏移，
Finalize 对 h、dh 分别寻址，Dhu 及其 head-major dh 不变。
重计算模式生成 h 后单独转为 chunk-major；保存模式不增加转换。
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
- dg 及参数梯度部分和写入 workspace，采用固定顺序归约，不使用原子累加。
- 融合反向入口在同一块私有 workspace 内依次铺设 dv0/dq_raw/dAqk/dh/dv_scan/dAkk 六个跨阶段缓冲、
  Kernel B 与 Kernel C 区域；各区域按 512 字节对齐，跨阶段偏移以 512 字节为单位写入 tiling，
  因此在 tiling payload 大小不变的前提下，单次发射可寻址的私有 workspace 上限由 4 GiB 提升到 2 TiB。
- q_rstd/k_rstd 成对提供时，在 Finalize 内完成归一化反向，不增加第四个 kernel。

Python 层统一处理参数校验、空序列压缩和规范 chunk 元数据。

Finalize 的输入输出及源码入口见
[算子说明](../../chunk_kda_bwd_finalize/README.md)。
