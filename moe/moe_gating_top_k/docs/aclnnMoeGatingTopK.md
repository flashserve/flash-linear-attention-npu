# aclnnMoeGatingTopK

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term> Ascend 950PR/Ascend 950DT</term>                  |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：MoE计算中，对输入x做Sigmoid或者SoftMax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
-   计算公式：

    对输入做Sigmoid或者SoftMax：

    $$
    if normType==1:
        normOut=Sigmoid(x)
    else:
        normOut=SoftMax(x)
    $$

    如果bias不为空：

    $$
    normOut = normOut + bias
    $$

    对计算结果按照groupCount进行分组，每组按照groupSelectMode取max或topk2的sum值对group进行排序，取前kGroup个组：

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$

    根据上一步的groupId获取normOut中对应的元素，将数据再做TopK，得到expertIdxOut的结果：

    $$
    y,expertIdxOut=TopK(normOut[groupId, :],k=k)
    $$

    对y按照输入的routedScalingFactor和eps参数进行计算，得到yOut的结果：

    $$
    yOut = y / (ReduceSum(y, dim=-1)+eps)*routedScalingFactor
    $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnMoeGatingTopKGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeGatingTopK”接口执行计算。
```Cpp
aclnnStatus aclnnMoeGatingTopKGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *biasOptional, 
  int64_t          k, 
  int64_t          kGroup, 
  int64_t          groupCount, 
  int64_t          groupSelectMode, 
  int64_t          renorm, 
  int64_t          normType, 
  bool             outFlag, 
  double           routedScalingFactor, 
  double           eps, 
  const aclTensor *yOut, 
  const aclTensor *expertIdxOut, 
  const aclTensor *outOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```
```Cpp
aclnnStatus aclnnMoeGatingTopK(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnMoeGatingTopKGetWorkspaceSize

-   **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
    <col style="width: 146px">
    <col style="width: 110px">
    <col style="width: 301px">
    <col style="width: 219px">
    <col style="width: 328px">
    <col style="width: 101px">
    <col style="width: 143px">
    <col style="width: 146px">
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
        <td>x</td>
        <td>输入</td>
        <td>待计算入参，对应公式中的x。</td>
        <td>无</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>与输入x进行计算的bias值，对应公式中的`bias`。</td>
        <td>shape值与x最后一维相等。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>k</td>
        <td>输入</td>
        <td>topk的k值，对应公式中的`k`。</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kGroup</td>
        <td>输入</td>
        <td>分组排序后取的group个数，对应公式中的`kGroup`。</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupCount</td>
        <td>输入</td>
        <td>分组的总个数，对应公式中的`groupCount`</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSelectMode</td>
        <td>输入</td>
        <td>分组排序方式。</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>renorm</td>
        <td>输入</td>
        <td>renorm标记。</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>输入</td>
        <td>norm函数类型。</td>
        <td>无</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outFlag</td>
        <td>输入</td>
        <td>表示是否输出norm操作结果。</td>
        <td>无</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>routedScalingFactor</td>
        <td>输入</td>
        <td>计算yOut使用的routedScalingFactor系数，对应公式中的`routedScalingFactor`。</td>
        <td>无</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>输入</td>
        <td>用于计算yOut使用的eps系数，对应公式中的`eps`。</td>
        <td>无</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>对x做norm、分组排序topk后计算的结果，对应公式中的`yOut`。</td>
        <td>数据类型与x需要保持一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>输出</td>
        <td>对x做norm、分组排序topk后的索引，对应公式中的`expertIdxOut`。</td>
        <td>shape要求与yOut一致。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outOut</td>
        <td>输出</td>
        <td>norm计算的输出结果，对应公式中的`normOut`。</td>
        <td>shape要求与x保持一致。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>-</td>
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
    </tbody>
    </table>  

-   **返回值：**

    <p>aclnnStatus：返回状态码，具体参见<a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。</p>
    <p>第一段接口完成入参校验，出现以下场景报错：</p>
    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>计算输入和计算输出是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_TILING_ERROR</td>
        <td>561002</td>
        <td>
        x的shape不满足要求。<br />
        x和biasOptional的shape不匹配。<br />
        k的大小不在1到x_shape[-1] / groupCount * kGroup之间。<br />
        kGroup的大小不在1到groupCount之间。<br />
        每个group的专家数按32对齐之后<br />
        计算输入参数的值不满足要求。<br />
        </td>
      </tr>
    </tbody></table>  

## aclnnMoeGatingTopK

-   **参数说明：**
    <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
    <col style="width: 173px">
    <col style="width: 112px">
    <col style="width: 668px">
    </colgroup>
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeGatingTopKGetWorkspaceSize获取。</td>
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

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnMoeGatingTopK默认确定性实现。

* 输入shape限制：
    * x最后一维（即专家数）要求不大于2048。 
* 输入值域限制：
    * 要求1 <= k <= x_shape[-1] / groupCount * kGroup。
    * 要求1 <= kGroup <= groupCount，并且kGroup * x_shape[-1] / groupCount的值要大于等于k。
    * 要求groupCount > 0，x_shape[-1]能够被groupCount整除且整除后的结果大于groupSelectMode，并且整除的结果按照32个数对齐后乘groupCount的结果不大于2048。
    * renorm仅支持0，表示先进行norm操作，再计算topk。
* 其他限制：
    * groupSelectMode取值0和1，0表示使用最大值对group进行排序, 1表示使用topk2的sum值对group进行排序。
    * normType取值0和1，0表示使用Softmax函数，1表示使用Sigmoid函数。
    * outFlag取值true和false，true表示输出，false表示不输出。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k.h"
#include <iostream>
#include <vector>
#include <random>

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

std::vector<float> GenerateRandomFloats(int64_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> result(count);
    for (auto& num : result) {
        num = dist(gen);
    }
    return result;
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
  // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {3, 256};
  std::vector<int64_t> biasShape = {256};
  std::vector<int64_t> outShape = {3, 8};
  std::vector<int64_t> expertIdOutShape = {3, 8};
  std::vector<int64_t> normOutShape = {3, 256};

  void* inputAddr = nullptr;
  void* biasAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* normOutAddr = nullptr;
  
  aclTensor* input = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* normOut = nullptr;
  
  std::vector<float> inputHostData = GenerateRandomFloats(GetShapeSize(inputShape));
  std::vector<float> biasHostData = GenerateRandomFloats(GetShapeSize(biasShape));
  std::vector<float> outHostData(GetShapeSize(outShape));
  std::vector<int32_t> expertIdOutHostData(GetShapeSize(expertIdOutShape));
  std::vector<float> normOutHostData(GetShapeSize(normOutShape));

  // 创建expandedPermutedRows aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedPermutedRows aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expertForSourceRow aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedSrcToDstRow aclTensor
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedSrcToDstRow aclTensor
  ret = CreateAclTensor(normOutHostData, normOutShape, &normOutAddr, aclDataType::ACL_FLOAT, &normOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeGatingTopK第一段接口
  ret = aclnnMoeGatingTopKGetWorkspaceSize(input, bias, 8, 4, 8, 1, 0, 1, false, 1, 1, out, expertIdOut, normOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeGatingTopK第二段接口
  ret = aclnnMoeGatingTopK(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopK failed. ERROR: %d\n", ret); return ret);

  // 4.（ 固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(bias);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(normOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputAddr);
  aclrtFree(biasAddr);
  aclrtFree(outAddr);
  aclrtFree(normOutAddr);
  aclrtFree(expertIdOutAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

