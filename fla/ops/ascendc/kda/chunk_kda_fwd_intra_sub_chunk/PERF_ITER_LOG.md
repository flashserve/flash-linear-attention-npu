# ChunkKdaFwdIntraSubChunk · PERF 快照

> **完整迭代档案已迁至 [`ITERATION_LOG.md`](ITERATION_LOG.md)**（时间线、否决清单、sim、提交锚点）。  
> 本文只保留**当前已验证最优**基线，避免与设计文档双写。

## 当前已验证最优

| 项 | 值 |
|----|-----|
| Shape | `B=1,T=8192,H=HV=32,K=128,BT=64,bf16` |
| Task Dur med | **2.075 ms**（裸 msprof，n=4，2026-07-25） |
| 硬目标 | ≤ 1.5 ms（未结案） |
| Cube | 路径 A：`L1A_DBUF=1` + `FIX_MTE2_DBUF=1`，`WIN_L1=0` |
| Vector | MTE2 merge + V-A 卫生 + FwdSub 行广播 Mul + Add-fold（粗同步） |
| 精度 | full suite PASS；H32 multi-iter OK（无 profiler） |

采集口径：[`MSPROF_GUIDE.md`](MSPROF_GUIDE.md)。设计落法：[`DESIGN.md`](DESIGN.md) §3.3 / §6。
