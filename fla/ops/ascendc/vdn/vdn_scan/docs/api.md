# VdnScan 接口

```cpp
aclnnStatus aclnnVdnScanGetWorkspaceSize(
    const aclTensor *transition, const aclTensor *prefix, const aclTensor *suffix,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnVdnScan(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

| 参数 | 连续 FP32 张量的形状 | 含义 |
| --- | --- | --- |
| `transition` | `[F,H,128,128]` | 每帧、每个头的状态转移矩阵 |
| `prefix` | `[F+1,H,128,128]` | `prefix[0]` 为前向初值，其余行预填本帧注入项 |
| `suffix` | `[F+1,H,128,128]` | `suffix[F]` 为反向初值，其余行预填本帧注入项 |

对每个头，前向按 `f=0..F-1` 计算 `prefix[f+1] += prefix[f] @ transition[f]`，反向按 `f=F-1..0` 计算 `suffix[f] += suffix[f+1] @ transition[f]`。三个张量须属于 NPU 上的公开紧凑格式，各自覆盖的字节区间须互不相交。允许共享底层存储的非重叠连续视图。帧数满足 `1 <= F < INT32_MAX`，头数满足 `1 <= H < INT32_MAX / 2`。调用方从传入的 `prefix`、`suffix` 读取结果。

两段式接口先返回工作区需求及执行器，再提交到指定 stream。成功返回 `ACLNN_SUCCESS`。必选指针为空返回 `ACLNN_ERR_PARAM_NULLPTR`，dtype、shape、存储布局或别名不满足契约时返回 `ACLNN_ERR_PARAM_INVALID`。执行器或内核启动失败时向上返回相应的运行时错误。
