# aclnnMoeGatingTopKSoftmaxV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term> Ascend 950PR/Ascend 950DT</term>               |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

-   接口功能：MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topK操作；如果renorm=1，先对x的输出做topK操作，再进行Softmax操作。其中yOut为softmax的topK结果；expertIdxOut为topK的值的索引结果，即对应的专家序号；如果对应的行finished为True，则专家序号直接填num\_expert值（即x的最后一个轴大小）。
-   计算公式：
1. renorm = 0,

    $$
    softmaxResultOutOptional=softmax(x,axis=-1)
    $$

    $$
    yOut,expertIdxOut=topK(softmaxResultOutOptional,k=k)
    $$

2. renorm = 1

    $$
    topkOut,expertIdxOut=topK(x, k=k)
    $$

    $$
    yOut = softmax(topkOut,axis=-1)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeGatingTopKSoftmaxV2”接口执行计算。

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(
    const aclTensor *x, 
    const aclTensor *finishedOptional, 
    int64_t          k, 
    int64_t          renorm, 
    bool             outputSoftmaxResultFlag, 
    const aclTensor *yOut, 
    const aclTensor *expertIdxOut, 
    const aclTensor *softmaxResultOutOptional,
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize

-   **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
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
        <td>公式中的x，待计算的输入。</td>
        <td>要求是一个2D/3D的Tensor，每一维大小应不大于int32的最大值2147483647。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>finishedOptional</td>
        <td>输入</td>
        <td>公式中的finished，表示该行是否参与运算。</td>
        <td>shape为x_shape[:-1]。</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>k</td>
        <td>输入</td>
        <td>topK的k值，专家数。</td>
        <td>0 <= k <= x的-1轴大小，且k不大于1024。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>renorm</td>
        <td>输入</td>
        <td>Softmax和TopK的顺序。</td>
        <td>只支持取值0和1。0表示先计算Softmax，再计算TopK；1则顺序相反。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>outputSoftmaxResultFlag</td>
        <td>输入</td>
        <td>表示是否输出softmax的结果。</td>
        <td>当renorm=0时，true表示输出Softmax的结果，false表示不输出；当renorm=1时，该参数不生效，不输出Softmax的结果。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>对x做softmax后取的topK值。</td>
        <td>其shape的非-1轴要求与x的对应轴大小一致，其-1轴要求其大小同k值。</td>
        <td>要求与输入x一致。</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>softmaxResultOutOptional</td>
        <td>输出</td>
        <td>计算过程中Softmax的结果。</td>
        <td>shape要求与x一致。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>rowIdxOut</td>
        <td>输出</td>
        <td>公式中的scales。</td>
        <td>hape要求与yOut一致。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>x</td>
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
        <td rowspan="8"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="8"> 561002 </td>
        <td>多个输入tensor之间的shape信息不匹配。</td>
        </tr>
        <tr>
        <td>输入属性和输入tensor之间的shape信息不匹配。</td>
        </tr>
        <tr>
        <td>x的shape维度不为2或3。</td>
        </tr>
        <tr>
        <td>x与finishedOptional的shape不匹配。</td>
        </tr>
        <tr>
        <td>k的值小于0或大于x-1的轴的大小。</td>
        </tr>
        <tr>
        <td>k的值大于1024。</td>
        </tr>
        <tr>
        <td>renorm的值不是0或1。</td>
        </tr>
        <tr>
        <td>softmaxResultOutOptional的数据类型不在支持的范围内。</td>
        </tr>
    </tbody></table>

## aclnnMoeGatingTopKSoftmax

-   **参数说明：**
    <table>
            <thead>
                <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
                <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。</td></tr>
                <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
                <tr><td>stream</td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
            </tbody>
        </table>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeGatingTopKSoftmaxV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
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
  std::vector<int64_t> inputShape = {3, 4};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> softmaxResultOutOptionalShape = {3, 4};

  void* inputAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* softmaxResultOutOptionalAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* softmaxResultOutOptional = nullptr;

  std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> softmaxResultOutOptionalHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // 创建expandedPermutedRows aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expertForSourceRow aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建expandedSrcToDstRow aclTensor
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建softmaxResultOutOptional aclTensor
  ret = CreateAclTensor(softmaxResultOutOptionalHostData, softmaxResultOutOptionalShape, &softmaxResultOutOptionalAddr, aclDataType::ACL_FLOAT, &softmaxResultOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3.调用CANN算子库API，需要修改为具体的算子接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnMoeGatingTopKSoftmaxV2第一段接口
  ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(input, nullptr, 2, 0, true, out, expertIdOut, softmaxResultOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnMoeGatingTopKSoftmaxV2第二段接口
  ret = aclnnMoeGatingTopKSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(softmaxResultOutOptional);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputAddr);
  aclrtFree(outAddr);
  aclrtFree(expertIdOutAddr);
  aclrtFree(softmaxResultOutOptionalAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```