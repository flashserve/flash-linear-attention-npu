# 性能调优与 Bound 分析

性能优化从语义、所有权和流水正确性开始，再根据 profiling 判断主要 bound。Python wall time 只用于 liveness，不作为算子性能结论。

## 推荐顺序

每轮只改变一个可测因素：

1. 固定数学语义、有效输出和 task 所有权。
2. 确认 chunk 依赖模型、head window 和 fixed/varlen 映射。
3. 完成 workspace、carry、slot 和内存容量方案。
4. 闭环 stage DAG、ready/free、ping/pong 和 tail 配平。
5. 删除不必要的清零、cast、重复计算和 GM 镜像。
6. 驻留重复使用的小 tensor 和 L1/L0 operand。
7. 在目标 SOC 支持时评估直连数据通路。
8. 将 wait 下沉到第一条真实消费。
9. 调整 row tile、双缓冲和窗口内 overlap。
10. 最后优化向量循环、寄存器依赖和双发。

## Head Window 调优

窗口大小应由服务时间和完整流水决定：

1. 测量 AIC/Cube 相关生产阶段归一到单个 head 的 `T_C`，以及单个 AIV/Vector subblock 处理完整 head 的 `T_V`。
2. 确认可并行且负载均衡的完整 head owner 数 `N_V`；只有 owner 相互独立时，才用 `T_V / N_V` 近似聚合 Vector 消费间隔。
3. 两个 Vector subblock 各自承包完整 head 时先测 2-head，并与 1-head、4-head 固定对比。
4. 当 `T_C` 不小于聚合 Vector 消费间隔，或 2-head 已形成稳定 overlap 时，保留较小窗口。
5. 只有 Cube 明显更快且 2-head timeline 存在可被额外在飞 head 隐藏的空洞、背压或排空间隔时，才保留 4-head。

扩大窗口不会改变由 Vector 决定的稳态吞吐上限，还会增加 slot、event、容量和尾窗口复杂度。跨 head/group 驻留带来的独立收益应单独 A/B，不与流水领先收益混在一起。

每轮执行：

```text
记录方案变化和预期 bound
  -> Release 编译
  -> 精度和回归
  -> 固定环境 profiling
  -> 比较 Task Duration 和完整 pipe
  -> 只保留稳定且无泛化劣化的改动
```

## 消除无效工作

- producer 完整覆盖时删除预清零。
- 同一中间量在有效生命周期内只计算和搬运一次。
- 小 tensor、广播 scalar、mask 和地址步长移出热循环。
- 累加首项直接覆盖，后续项再累加。
- Tiling 可推导的 shape、offset 和 layout 参数在初始化阶段缓存。
- helper 不重复执行热路径地址推导和分支判断。

## 向量循环

连续依赖尽量融合为单趟向量循环。通用要求包括：

- 循环计数和地址规则递增，满足硬件循环条件。
- full mask、广播值和不变量移到循环外。
- 对齐主路径使用 full mask，尾块使用独立 tail mask。
- 相邻迭代使用独立 load/compute/store 依赖链时再尝试双发。
- 自动展开不稳定时才手工构造双链，并保留奇数尾块。
- 同时核算数据、mask 和地址寄存器，避免寄存器压力抵消收益。

目标 SOC 支持的 RegBase/VF 能力和限制见对应 SOC 文档。

## 按 Bound 选择方向

pipe 字段可能重叠，不能相加得到 Task Duration。

| 现象 | 优先检查 |
|---|---|
| AIC scalar 高 | task/varlen 映射、热循环地址计算、动态 layout、过细 helper |
| AIC MTE2 高 | GM->L1 重复搬运、resident、wait 位置、tile 粒度 |
| AIC MTE1 高 | L1->L0 重复搬运、L0 resident、L0A/L0B slot |
| Fixpipe 高 | L0C 多次落地、GM 中转、copyout 和 direct-path tile |
| AIV scalar 高 | 逐元素标量、循环条件、地址计算和 helper 控制流 |
| AIV VEC 高 | 重复 cast/exp、循环不变量、粒度、融合和双发 |
| AIV MTE2/MTE3 高 | row tile、重复 workspace、非连续 layout、过多 copyout |
| Task 高但单 pipe 不高 | AIC/AIV wait、head-window slot 背压、跨核 flag、任务粒度 |

VEC 时间下降而 Task Duration 不变，通常说明总耗时已由 MTE 或同步主导，应停止继续展开向量循环，转向数据搬运和 producer-consumer overlap。

## 性能基线

固定记录：

- shape、dtype、分支、chunk size 和 head ratio。
- fixed/varlen、sequence 数量和随机 seed。
- 相同物理设备、软件版本和 Release 编译参数。
- 相同 warmup、repeat 和 launch 设置。
- Task Duration、AIC/AIV 全部 pipe 字段和利用率。

锚点 shape 用于证明目标收益；额外矩阵用于证明泛化。不能用更换 case、减少覆盖或只报告局部 pipe 改善制造性能提升。
