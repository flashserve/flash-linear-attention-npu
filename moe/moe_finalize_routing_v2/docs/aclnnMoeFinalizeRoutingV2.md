# aclnnMoeFinalizeRoutingV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term> Ascend 950PR/Ascend 950DT</term>                  |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

-   **接口功能**：MoE计算中，最后处理合并MoE FFN的输出结果。
-   **计算公式**：

    $$
    expertid=expertIdx[i,k]
    $$
    
    $$
    out(i,j)=x1_{i,j}+x2_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num\_rows},j}+bias_{expertid,j}))
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeFinalizeRoutingV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeFinalizeRoutingV2”接口执行计算。

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor *expandedX,
    const aclTensor *expandedRowIdx,
    const aclTensor *x1Optional,
    const aclTensor *x2Optional,
    const aclTensor *biasOptional,
    const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional,
    int64_t          dropPadMode,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeFinalizeRoutingV2GetWorkspaceSize

-   **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1522px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 315px">
    <col style="width: 231px">
    <col style="width: 210px">
    <col style="width: 121px">
    <col style="width: 210px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>expandedX</td>
        <td>输入</td>
        <td>公式中的expandedX ，MoE的FFN输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>drop less场景：(NUM_ROWS * K, H)，<br>drop pad场景：(E, C, H)。</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expandedRowIdx</td>
        <td>输入</td>
        <td>公式中的expandedRowIdx。</td>
        <td>-</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(NUM_ROWS * K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>x1Optional</td>
        <td>输入</td>
        <td>公式中的x1，表示第一个共享专家。</td>
        <td>-</td>
        <td>与expandedX一致。</td>
        <td>ND</td>
        <td>与out一致。</td>
        <td>√</td>
    </tr>
    <tr>
        <td>x2Optional</td>
        <td>输入</td>
        <td>公式中的x2，表示第二个共享专家。</td>
        <td>-</td>
        <td>与expandedX一致。</td>
        <td>ND</td>
        <td>与out一致。</td>
        <td>√</td>
    </tr>
    <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>公式中的bias，表示偏置量。</td>
        <td>-</td>
        <td>与expandedX一致。</td>
        <td>ND</td>
        <td>(E, H)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>公式中的scales。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>(NUM_ROWS, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>expertIdxOptional</td>
        <td>输入</td>
        <td>公式中的expertIdx。</td>
        <td>Tensor中的值取值范围是[0, E-1]。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(NUM_ROWS, K)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>dropPadMode</td>
        <td>输入</td>
        <td>表示是否支持丢弃模式,expandedRowIdx的排列方式。</td>
        <td>取值范围为[0, 3]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>-</td>
        <td>与expandedX一致。</td>
        <td>ND</td>
        <td>(NUM_ROWS, H)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
      - expandedX要求是一个2D/3D的Tensor，支持的数据类型为FLOAT16、BFLOAT16、FLOAT32，支持drop less和drop pad场景。
      - scalesOptional：混合精度模式下，支持 expandedX 为 BFLOAT16 时 scalesOptional 为 FLOAT32；非混合精度模式下，数据类型要求与expandedX一致。
      
    - <term>Ascend 950PR/Ascend 950DT</term>：
      - expandedX要求是一个2D/3D的Tensor，支持的数据类型为FLOAT16、BFLOAT16、FLOAT32，支持drop less和drop pad场景。
      - scalesOptional数据类型可以与expandedX不一致。
-   **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 253px">
    <col style="width: 140px">
    <col style="width: 762px">
    </colgroup>
    <thead>
        <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>传入的必选输入、必选输出或者必选属性，是空指针。</td>
        </tr>
        <tr>
        <td> ACLNN_ERR_PARAM_INVALID </td>
        <td> 161002 </td>
        <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2"> 561002 </td>
        <td>多个输入tensor之间的shape信息不匹配。</td>
        </tr>
        <tr>
        <td>输入属性和输入tensor之间的shape信息不匹配。</td>
        </tr>
    </tbody></table>

## aclnnMoeFinalizeRoutingV2

-   **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1179px"> <colgroup>
    <col style="width: 169px">
    <col style="width: 130px">
    <col style="width: 880px">
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMoeFinalizeRoutingV2GetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody>
    </table>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. 确定性计算：
     - aclnnMoeFinalizeRoutingV2默认确定性实现。

2. NUM\_ROWS：表示行数；  
K：表示从总的专家E中选出K个专家；  
H：表示hidden size，即每个token序列长度，为列数；  
E：表示expert num，即专家数，E需要大于等于K；  
C：表示expert capacity，即专家处理token数量的能力阈值。  

1. expandedRowIdx：当dropPadMode参数值为0、2时，Tensor中的值取值范围是[0,NUM_ROWS * K-1]；当dropPadMode参数值为1、3时，Tensor中的值取值范围是[-1, E\*C - 1]。

2. 在x1Optional参数未输入的情况下，x2Optional参数也不能输入。

3. scalesOptional不存在时，K为1。

4. biasOptional存在时，expertIdxOptional必须同时存在。

5. dropPadMode的取值与含义对应如下：
     - 0：drop less 场景，expandedRowIdx按**列**排列（与[aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting.md)输出格式对应）。
     - 1：drop pad 场景，expandedRowIdx按**列**排列（与[aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting.md)输出格式对应）。
     - 2：drop less 场景，expandedRowIdx按**行**排列（与[aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)输出格式对应）。
     - 3：drop pad 场景，expandedRowIdx按**行**排列（与[aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)输出格式对应）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2.h"
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
    auto  ret = aclInit(nullptr);
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> expandedXShape = {3 * 2, 4};
  std::vector<int64_t> x1Shape = {3, 4};
  std::vector<int64_t> x2OptionalShape = {3, 4};
  std::vector<int64_t> biasShape = {2, 4};
  std::vector<int64_t> scalesShape = {3, 2};
  std::vector<int64_t> expandedExpertIdxShape = {3, 2};
  std::vector<int64_t> expandedRowIdxShape = {3 * 2};
  std::vector<int64_t> outShape = {3, 4};
  void* expandedXAddr = nullptr;
  void* x1Addr = nullptr;
  void* x2OptionalAddr = nullptr;
  void* biasAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expandedExpertIdxAddr = nullptr;
  void* expandedRowIdxAddr = nullptr;
  void* outDeviceAddr = nullptr;
  
  aclTensor* expandedX = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2Optional = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expandedExpertIdx = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                                     0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
  std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
  std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
  std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
  std::vector<float> outHostData(12, 0.0f);
  int64_t dropPadMode = 0;
  // 创建expandedX aclTensor
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr,
                        aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2Optional aclScalar
  ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建totalWeightOut aclTensor
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建expandedExpertIdx aclTensor
  ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                        aclDataType::ACL_INT32, &expandedExpertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedRowIdx aclTensor
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr,
                        aclDataType::ACL_INT32, &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建Out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeFinalizeRoutingV2第一段接口
  ret = aclnnMoeFinalizeRoutingV2GetWorkspaceSize(expandedX, expandedRowIdx, x1, x2Optional, bias, scales,
                                                  expandedExpertIdx, dropPadMode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeFinalizeRoutingV2第二段接口
  ret = aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2 failed. ERROR: %d\n", ret); return ret);

  // 4.（ 固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(expandedX);
  aclDestroyTensor(x1);
  aclDestroyTensor(x2Optional);
  aclDestroyTensor(bias);
  aclDestroyTensor(scales);
  aclDestroyTensor(expandedExpertIdx);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(expandedXAddr);
  aclrtFree(x1Addr);
  aclrtFree(x2OptionalAddr);
  aclrtFree(biasAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expandedExpertIdxAddr);
  aclrtFree(expandedRowIdxAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

