# 算子优化检查清单

本清单用于性能设计和实现交付。通用工程交付同时执行上层 [`../operator-checklist.md`](../operator-checklist.md)。

## 依赖和执行模型

- [ ] 已判定 chunk 间有无 carry，且没有把 chunk 内依赖误判为跨 chunk 依赖。
- [ ] 数学语义、有效输出、递推方向和状态边界已经固定。
- [ ] 每个中间量的 producer、consumer、location、dtype/layout、生命周期和复用键已形成 DAG。
- [ ] 独立轴、归约 owner 和 task 所有权明确。
- [ ] 默认 4-head window 使用两个 bank 和 8 个逻辑 per-head workspace slot。
- [ ] 完整窗口和 1 至 3 个 tail head 使用一致的调度和同步模型。
- [ ] 调整窗口大小时已提供容量、同步、性能和泛化证据。

## 内存和数据通路

- [ ] workspace 每个 segment/slot 的地址键、ready、free 和覆盖点明确。
- [ ] workspace bank 与 UB/L1/L0 ping/pong 分开建模。
- [ ] carry、累加和公开输出 dtype 的转换点明确。
- [ ] UB/L1/L0 预算包含 resident、scratch、双缓冲、direct slot 和 API 临时空间。
- [ ] 小 tensor 已评估 UB resident，重复使用的数据只搬运和计算一次。
- [ ] L1/L0 跨 head/group 驻留按最小复用键设计。
- [ ] L1 resident 在最后一次 L1->L0 后释放，L0 resident 在最后一次 MMAD 后释放。
- [ ] producer 完整覆盖时删除预清零；可能读取 padding/残留时只清零真实范围。
- [ ] 直连与 GM fallback 保持相同语义、stage 和有效区。

## 流水和同步

- [ ] AIC 的 L1A/L1B、L0A/L0B 和 L0C 生命周期完整。
- [ ] 同一逻辑 L0C 结果只执行一次 Fixpipe copyout。
- [ ] AIV input、output、state 和 direct slot 使用独立或已证明的生命周期。
- [ ] 每个核内事件的 producer pipe、consumer pipe、slot、方向和复用点明确。
- [ ] 每条跨 AIC/AIV 数据边的 set/wait、ready/free 按 head 和分支配平。
- [ ] owner 与非 owner Vector subblock均完成必要协议。
- [ ] 4-head window 不依赖只支持 2 次积压的旧 raw flag 假设。
- [ ] wait 紧贴第一条真实 RAW 消费，独立搬运先执行。
- [ ] tail、空 payload、跳过计算和 kernel 收尾均归还 free。

## Tiling 和模板

- [ ] 实现结构符合上层 [`../operator-coding-standard.md`](../operator-coding-standard.md)，A5 路径同时符合 [`soc/a5.md`](soc/a5.md) 的编码规范。
- [ ] row tile 按真实 UB 预算和 API 临时空间生成。
- [ ] 归约维分块只在末块后执行一次 Fixpipe。
- [ ] 物理布局调整与数学语义、workspace 和 reference 一致。
- [ ] ND/NZ 直连已核实目标 CANN/SOC 的真实实现和地址跨度。
- [ ] TilingKey 只包含影响编译期结构的维度，没有按单个 shape 过度模板化。
- [ ] 所有 SOC 仍复用同一 L0 定义、原型和 L2 调用路径。

## 性能和验证

- [ ] 每轮只修改一个可测因素，并记录预期 bound。
- [ ] Release 编译和完整精度矩阵通过。
- [ ] 固定环境比较修改前后 Task Duration 和全部 pipe 字段。
- [ ] 4-head 完整窗口、tail、多 window 复用、grouped/GVA 和 fixed/varlen 已覆盖。
- [ ] 额外 shape 用于验证泛化，没有替代固定性能锚点。
- [ ] 流水图确认预期 overlap 和 wait 位置。
- [ ] race、越界、未初始化和同步修改已执行对应 sanitizer，并确认命中目标 kernel。
- [ ] 没有通过缩小 range、跳过 case、放宽阈值或更换性能 case 制造通过结论。
