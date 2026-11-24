# VdnBridge 接口

`aclnnVdnBridgeGetWorkspaceSize(prefix, suffix, bounds, logs, start, out, workspaceSize, executor)` 准备执行计划；`aclnnVdnBridge(workspace, workspaceSize, executor, stream)` 在调用方 stream 执行。

| 参数 | dtype | shape / 取值 |
| --- | --- | --- |
| prefix、suffix | FP32 | `[F+1,H,128,128]`，双向状态，首行或末行包含初始状态 |
| bounds | INT64 | `[F,2]`，每帧闭区间 `0 <= lo <= f <= hi < F` |
| logs | FP32 | `[F+1,H,128]`，零行起始的 alpha 对数前缀和 |
| start | int64 | 当前输出帧的全局起点，`0 <= start <= F-C` |
| out | FP32 / BF16 | `[C,H,128,128]`，调用方分配；`1 <= C <= F` |

全部张量在同一 910B 设备、ND 连续布局，H 和 F 均为正整数。out 的有效字节与任何输入均须互不重叠；同一 allocation 的非重叠连续视图可用。状态库总字节数、视图大小和地址偏移须可表示。函数按 out 的 dtype 在写回前转换。设备核限制越界窗口索引以保护 GM 访问，调用方负责提供合法窗口。`count`、`out_bf16` 为 L0 内部属性，由 L2 从 out 推导。空指针返回 ACLNN_ERR_PARAM_NULLPTR，形状、类型、布局或区间非法返回 ACLNN_ERR_PARAM_INVALID。
