# aclnnCausalConv1d

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR/Ascend 950DT</term>       |   √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |   √  |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |   √  |
| <term>Atlas 200I/500 A2 推理产品</term>          |   ×  |
| <term>Atlas 推理系列产品</term>                    |   ×  |
| <term>Atlas 训练系列产品</term>                    |   ×  |

## 功能说明

- 接口功能：完成因果一维卷积（Causal Conv1d）计算，支持前向计算（runMode=0）和状态更新（runMode=1）两种运行模式。
- 计算公式：

  Causal Conv1d 是一种因果一维卷积算子，常用于序列建模中。在每个时间步 $t$，根据当前输入 $x\_t$、卷积权重 $w$ 和历史状态，计算卷积输出 $y\_t$。

  $$
  y\_t = \text{Activation}\left(\sum\_{j=0}^{W-1} w\_j \cdot x\_{t-j} + b\right)
  $$

  其中，$W$ 为卷积核宽度（支持2、3、4），$w\_j$ 为卷积权重，$b$ 为偏置（可选），$\text{Activation}$ 为激活函数（可选，SiLU）。当 `activationMode=0` 时不使用激活函数，`activationMode=1` 时使用 SiLU 激活函数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnCausalConv1dGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCausalConv1d"接口执行计算。

```cpp
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    aclTensor       *convStates,
    const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices,
    const aclTensor *initialStateMode,
    const aclTensor *numAcceptedTokens,
    int64_t         activationMode,
    int64_t         padSlotId,
    int64_t         runMode,
    aclTensor       *y,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

## aclnnCausalConv1d

- 返回值
  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnCausalConv1d默认确定性实现。
- 卷积核宽度 $W$ 仅支持2、3、4。
- 特征维度 $D$ 需为16的倍数。
- runMode=0（前向计算模式）时，卷积核宽度为编译时已知常量，支持FnRolling快速路径优化。
- runMode=1（状态更新模式）时，卷积核宽度为运行时参数。
- 当存在 initialStateMode 输入时，算子需要额外的 workspace 用于初始状态同步。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        if (i >= 5) {
            break;
        }
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *convStatesDeviceAddr = nullptr;
    void *queryStartLocDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *convStates = nullptr;
    aclTensor *queryStartLoc = nullptr;
    aclTensor *y = nullptr;

    int32_t batchSize = 2;
    int32_t seqLen = 4;
    int32_t dim = 64;
    int32_t width = 3;
    int32_t stateLen = width;

    std::vector<int64_t> xShape = {batchSize * seqLen, dim};
    std::vector<int64_t> weightShape = {width, dim};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> convStatesShape = {batchSize, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batchSize + 1};
    std::vector<int64_t> yShape = {batchSize * seqLen, dim};

    std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int16_t> weightHostData(GetShapeSize(weightShape), 1);
    std::vector<int16_t> biasHostData(GetShapeSize(biasShape), 0);
    std::vector<int16_t> convStatesHostData(GetShapeSize(convStatesShape), 0);
    std::vector<int64_t> queryStartLocHostData = {0, seqLen, batchSize * seqLen};
    std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(convStatesHostData, convStatesShape, &convStatesDeviceAddr, aclDataType::ACL_BF16, &convStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, &queryStartLocDeviceAddr, aclDataType::ACL_INT64, &queryStartLoc);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    int64_t runMode = 0;

    ret = aclnnCausalConv1dGetWorkspaceSize(x, weight, bias, convStates, queryStartLoc, nullptr, nullptr, nullptr,
                                            activationMode, padSlotId, runMode, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCausalConv1d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1d failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(yShape, &yDeviceAddr);

    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(convStates);
    aclDestroyTensor(queryStartLoc);
    aclDestroyTensor(y);

    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(convStatesDeviceAddr);
    aclrtFree(queryStartLocDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
