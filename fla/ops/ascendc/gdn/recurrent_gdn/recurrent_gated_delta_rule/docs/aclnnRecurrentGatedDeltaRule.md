# aclnnRecurrentGatedDeltaRule

完整签名、参数约束、融合语义与兼容性规则统一见 [api.md](api.md)。
当前仅有一套两段式 ACLNN 接口：原接口名采用完整融合参数表，替换历史精简签名。

第一段 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` 校验描述符并准备执行器，
第二段 `aclnnRecurrentGatedDeltaRule` 在指定 stream 上执行。
调用方需要在第一段与第二段之间按返回大小申请 device workspace，待 stream 执行结束后释放。
State 原地更新；out 由调用方提供，不与输入或 State 重叠。

支持 A2/A3、A5（Ascend 950）和 A6（Ascend 960DT，`ascend960dt`）。

以下片段假定已创建符合 [api.md](api.md) 的 device tensor、stream 和输出描述符：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices,
    g, nullptr, numAcceptedTokens, aLog, dtBias, scaleValue,
    useQkL2NormInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval,
    out, &workspaceSize, &executor);
// Check status before allocating workspace or executing.
if (status != ACLNN_SUCCESS) return status;

void *workspace = nullptr;
if (workspaceSize != 0) {
    aclError error = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (error != ACL_SUCCESS) return error;
}
status = aclnnRecurrentGatedDeltaRule(workspace, workspaceSize, executor, stream);
aclError syncStatus = aclrtSynchronizeStream(stream);
if (workspace != nullptr) aclrtFree(workspace);
if (status != ACLNN_SUCCESS) return status;
if (syncStatus != ACL_SUCCESS) return syncStatus;
```

不使用融合预处理时，将 aLog/dtBias 传为 nullptr，四个 bool 传为 false；
beta、State、g 的 BF16/FP32 支持、packed/table 索引和 view 支持仍然可用。
完整的无融合调用示例见
[`../examples/test_aclnn_recurrent_gated_delta_rule.cpp`](../examples/test_aclnn_recurrent_gated_delta_rule.cpp)。

编译示例时，将本包 vendor 的 `op_api/include` 放在 CANN include 路径之前，
并加入仓库的 `common/include/external`，链接本包的 `libcust_opapi.so`。
CANN 自带的同名头文件采用历史精简签名。
