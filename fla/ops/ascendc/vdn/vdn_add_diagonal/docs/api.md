# VdnAddDiagonal 接口

```cpp
aclnnStatus aclnnVdnAddDiagonalGetWorkspaceSize(
    const aclTensor *correction, const aclTensor *twoEye,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnVdnAddDiagonal(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

| 参数 | 连续 FP32 张量的形状 | 含义 |
| --- | --- | --- |
| `correction` | `[N,128,128]` | 存放 `-M @ X`，调用完成后存放 `2I - M @ X` |
| `twoEye` | `[128,128]` | 值为两倍单位矩阵，跨矩阵共享 |

两个张量须位于 NPU 且为公开紧凑格式，逻辑存储区间须互不相交。允许共享底层存储的非重叠连续视图。支持 `1 <= N <= INT32_MAX / 2`。调用方须传入具有指定值的 `twoEye`，内核直接复用其左上角的 64 × 64 对角块。调用方从传入的 `correction` 读取结果。

两段式接口先返回工作区需求及执行器，再提交到指定 stream。成功返回 `ACLNN_SUCCESS`。必选指针为空返回 `ACLNN_ERR_PARAM_NULLPTR`，dtype、shape、存储布局或别名不满足契约时返回 `ACLNN_ERR_PARAM_INVALID`。执行器或内核启动失败时向上返回相应的运行时错误。
