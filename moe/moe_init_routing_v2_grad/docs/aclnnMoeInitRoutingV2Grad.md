# aclnnMoeInitRoutingV2Grad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>    |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |


## 功能说明

-   **算子功能**：[aclnnMoeInitRoutingV2](aclnnMoeInitRoutingV2.md)的反向传播，完成tokens的加权求和。
-   **计算公式**：

    $$
    gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnMoeInitRoutingV2GradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeInitRoutingV2Grad”接口执行计算。

* `aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(const aclTensor *gradExpandedX, const aclTensor *expandedRowIdx, int64_t topK, int64_t dropPadMode, int64_t activeNum, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeInitRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeInitRoutingV2GradGetWorkspaceSize

-   **参数说明**：
    - gradExpandedX（aclTensor\*，计算输入）：表示Routing过后的目标张量，要求为一个2D/3D的Tensor，2D shape为Dropless场景的[B\*S\*K, H]或者Active场景下的[A, H]，3D shape为Drop/Pad场景下的[E, C, H]，数据类型支持FLOAT16、BFLOAT16、FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - expandedRowIdx（aclTensor\*，计算输入）：表示token按照专家序排序索引，一维Tensor，shape为[B\*S\*K]；元素值在Drop/Pad场景下范围为[-1, E\*C)，其他场景范围为[0, B\*S\*K)，且值除-1外唯一不重复，数据类型支持INT32，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - topK（int64\_t，计算输入）：topK值，Host侧的整型，必须大于0，且能被expandedRowIdx的0轴大小整除。
    - dropPadMode（int64\_t，计算输入）：表示场景是否为Drop类，Host侧整型，取值范围为[0, 1]，0表示Dropless场景，1表示Drop/Pad场景。
    - activeNum（int64\_t，计算输入）：表示场景是否为Active场景，Host侧整型，值范围大于等于0，当dropPadMode为0时生效，0表示非Active场景，大于0表示Active场景，Active场景下gradExpandedX的0轴大小必须等于activeNum值。
    - out（aclTensor\*，计算输出）：表示Routing反向输出，2D的Tensor，shape为[B\*S, H]；数据类型支持FLOAT16、BFLOAT16、FLOAT32，输出类型与输入gradExpandedX一致，[数据格式](../../../docs/zh/context/数据格式.md)要求为ND，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

      shape符号说明：

      B: batch size; S: tokens数量; H: hidden size, 即每个token序列长度; K: 即topK, token被处理的专家数
      A: activeNum值; E: expert num, 即专家数; C: expert capacity, 表示专家处理token数量的能力阈值

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错:
    161001(ACLNN_ERR_PARAM_NULLPTR): 输入和输出的Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 输入和输出的数据类型不在支持的范围内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. dropPadMode的属性值不是0和1。
                                          2. topK小于等于0。
                                          3. activeNum小于0。
                                          4. gradExpandedX不是2D/3D，或者dropPadMode为1时，gradExpandedX不是3D。
                                          5. dropPadMode和activeNum都为0时，gradExpandedX和expandedRowIdx的0轴大小不相等。
                                          6. dropPadMode为0且activeNum大于0时，gradExpandedX的0轴与activeNum大小不相等。
                                          7. out和gradExpandedX的尾轴大小不相等。
                                          8. out的0轴不等于expandedRowIdx的0轴大小除以topK。
    ```

## aclnnMoeInitRoutingV2Grad

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeInitRoutingV2GradGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeInitRoutingV2Grad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2_grad.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> gradExpandedXShape = {4, 2};
    std::vector<int64_t> expandedRowIdxShape = {4};
    std::vector<int64_t> gradXShape = {2, 2};
    void* gradExpandedXDeviceAddr = nullptr;
    void* expandedRowIdxDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    aclTensor* gradExpandedX = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclScalar* k = nullptr;
    aclScalar* dropPadMode = nullptr;
    aclScalar* activeNum = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> gradExpandedXHostData = {0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4};
    std::vector<int32_t> expandedRowIdxHostData = {2, 0, 1, 3};
    std::vector<float> gradXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    int32_t kValue = 2;
    int32_t dropPadModeValue = 0;
    int32_t activeNumValue = 0;

    // 创建输入 aclTensor
    ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT, &gradExpandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    k = aclCreateScalar(&kValue, aclDataType::ACL_INT32);
    CHECK_RET(k != nullptr, return ret);
    dropPadMode = aclCreateScalar(&dropPadModeValue, aclDataType::ACL_INT32);
    CHECK_RET(dropPadMode != nullptr, return ret);
    activeNum = aclCreateScalar(&activeNumValue, aclDataType::ACL_INT32);
    CHECK_RET(activeNum != nullptr, return ret);
    // 创建输出 aclTensor
    ret = CreateAclTensor(gradXOutHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRoutingV2Grad第一段接口
    ret = aclnnMoeInitRoutingV2GradGetWorkspaceSize(gradExpandedX, expandedRowIdx, kValue, dropPadModeValue, activeNumValue, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeInitRoutingV2Grad第二段接口
    ret = aclnnMoeInitRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto gradXSize = GetShapeSize(gradXShape);
    std::vector<float> gradXData(gradXSize, 0);
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr, gradXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < gradXSize; i++) {
        LOG_PRINT("gradXData[%ld] is: %f\n", i, gradXData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradExpandedX);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyScalar(k);
    aclDestroyScalar(dropPadMode);
    aclDestroyScalar(activeNum);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradExpandedXDeviceAddr);
    aclrtFree(expandedRowIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```