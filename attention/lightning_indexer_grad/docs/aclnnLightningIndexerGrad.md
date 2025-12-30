# aclnnLightningIndexerGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A3 训练系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |


## 功能说明

- 算子功能：训练场景下，实现LightningIndexer反向，其中输入有Query, Key, Weights, Dy, Indices，反向主要利用正向计算的Indices从Key中提取TopK序列从而降低Matmul计算量。

- 计算公式：
  LightningIndexer反向计算公式如下：

  $$
  S = Relu(Matmul(Query, Gather(Key, Indices)))
  $$

  $$
  Y = Dy*Weights
  $$

  $$
  dW = Reduce(S * dy)
  $$

  $$
  dQ = Matmul(ReluGrad(Y, S), Gather(Key, Indices))
  $$

  $$
  dK = ScatterAdd(Matmul(ReluGrad(Y, S), Q), Indices)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLightningIndexerGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLightningIndexerGrad”接口执行计算。

```c++
aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *dy,
  const aclTensor   *spareIndices,
  const aclTensor   *weights,
  const aclTensor   *actucalSeqLengthsQuery,
  const aclTensor   *actucalSeqLengthsKey,
  int64_t           headNum,  
  char              *layout,
  int64_t           sparseMode,
  int64_t           preTokens,
  int64_t           nextTokens,
  bool              determinstic,
  const aclTensor   *dQuery,
  const aclTensor   *dKey,
  const aclTensor   *dWeights,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnLightningIndexerGrad(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  aclrtStream stream)
```
### aclnnLightningIndexerGradGetWorkspaceSize

- **参数说明：**

<table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的query。</td>
      <td>shape支持[B, S1, N1, D]/[T1, N1, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>shape支持[B, S2, N2, D]/[T2, N2, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>shape支持[B, S1, N1, D]/[T1, N1, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输入</td>
      <td>公式中的sparseIndices。</td>
      <td>shape支持[B, S1, K]/[T1, K]。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的weights。</td>
      <td>shape支持[B, S1, N1]/[T1, N1]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actucalSeqLengthsQuery</td>
      <td>输入</td>
      <td>表示query每个Batch S的累加和长度。</td>
      <td>TND排布时需要输入，其余场景输入nullptr。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actucalSeqLengthsKey</td>
      <td>输入</td>
      <td>表示key每个Batch S的累加和长度。</td>
      <td>TND排布时需要输入，其余场景输入nullptr。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>输入</td>
      <td>代表head个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>代表query/key的数据排布格式。</td>
      <td>当前支持TND/BSND。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse模式。</td>
      <td>支持取值0/3。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>用于稀疏计算 ，表示slides window的左边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>用于稀疏计算 ，表示slides window的右边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>determinstic</td>
      <td>输入</td>
      <td>表示当前是否支持确定性计算。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dQuery</td>
      <td>输出</td>
      <td>dQuery梯度。</td>
      <td>数据类型与query一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dKey</td>
      <td>输出</td>
      <td>dKey梯度。</td>
      <td>数据类型与key一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dWeights</td>
      <td>输出</td>
      <td>dWeights梯度。</td>
      <td>数据类型与weights一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>


- **返回值：**

返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
<table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
<col style="width: 319px">
<col style="width: 144px">
<col style="width: 671px">
</colgroup>
<thead>
  <tr>
    <th>返回码</th>
    <th>错误码</th>
    <th>描述</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>query、key、weights、sparseIndices、weights的数据类型和数据格式不在支持的范围内。</td>
  </tr>
  <tr>
    <td>input_layout输入的类型不在支持的范围内。</td>
  </tr>
</tbody>
</table>


### aclnnLightningIndexerGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 144px">
  <col style="width: 125px">
  <col style="width: 700px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLightningIndexerGradGetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明

- 确定性计算：
  - aclnnLightningIndexerGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- inputLayout支持TND/BSND。
- 关于数据shape的约束，以Layout的BSND举例。其中：
  
  - B（Batchsize）：取值范围为1\~1024。
  - N（Head-Num）：取值为64。
  - G（Group）：取值为64。
  - S1（Seq-LengthQ）：取值范围为1\~128K。
  - S2（Seq-LengthK）：取值范围为topK\~128K。
  - D（Head-Dim）：取值为128。
  - TopK：取值为2048


## 调用示例

通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_grad.h"

#define CHECK_RET(cond, return_expr)                   \
    do {                                               \
        if (!(cond)) {                                 \
            return_expr;                               \
        }                                              \
    } while (0)

#define LOG_PRINT(message, ...)                        \
    do {                                               \
        printf(message, ##__VA_ARGS__);                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T> 
void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
        aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
        aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


void FreeResource(aclTensor *q, aclTensor *k, aclTensor *dy, aclTensor *sparseIndices, aclTensor *weights, 
                  aclTensor *dQuery, aclTensor *dKey, aclTensor *dWeights, void *qDeviceAddr, void *kDeviceAddr, 
                  void *dyDeviceAddr, void *sparseIndicesDeviceAddr, void *weightsDeviceAddr, void *dQueryAddr, 
                  void *dKeyAddr, void *dWeightsAddr, uint64_t workspaceSize, void *workspaceAddr, int32_t deviceId, 
                  aclrtContext *context, aclrtStream *stream)
{
    // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (dy != nullptr) {
        aclDestroyTensor(dy);
    }
    if (sparseIndices != nullptr) {
        aclDestroyTensor(sparseIndices);
    }
    if (weights != nullptr) {
        aclDestroyTensor(weights);
    }
    if (dQuery != nullptr) {
        aclDestroyTensor(dQuery);
    }
    if (dKey != nullptr) {
        aclDestroyTensor(dKey);
    }
    if (dWeights != nullptr) {
        aclDestroyTensor(dWeights);
    }

    // 释放device资源
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (dyDeviceAddr != nullptr) {
        aclrtFree(dyDeviceAddr);
    }
    if (sparseIndicesDeviceAddr != nullptr) {
        aclrtFree(sparseIndicesDeviceAddr);
    }
    if (weightsDeviceAddr != nullptr) {
        aclrtFree(weightsDeviceAddr);
    }
    if (dQueryAddr != nullptr) {
        aclrtFree(dQueryAddr);
    }
    if (dKeyAddr != nullptr) {
        aclrtFree(dKeyAddr);
    }
    if (dWeightsAddr != nullptr) {
        aclrtFree(dWeightsAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // query、key、dy、sparseIndices、weights对应的shape值，并重新gen data，再执行
    int64_t batch = 2;
    int64_t s1 = 3;
    int64_t s2 = 2048;
    int64_t d = 128;
    int64_t g = 64;
    int64_t n2 = 1;
    int64_t topK = 2048;

    std::vector<int64_t> qShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> kShape = {batch, s2, n2, d};
    std::vector<int64_t> dyShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> sparseIndicesShape = {batch, s1, topK};
    std::vector<int64_t> weightsShape = {batch, s1, n2 * g};
    std::vector<int64_t> dQueryShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> dKeyShape = {batch, s2, n2, d};
    std::vector<int64_t> dWeightsShape = {batch, s1, n2 * g};

    int64_t headNum = 64;
    int64_t sparseMode = 3;
    char layoutStr[] = "BSND";
    bool deteminstic = true;
    int64_t preToken = 65536;
    int64_t nextToken = 65536;

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *dyDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *dQueryAddr = nullptr;
    void *dKeyAddr = nullptr;
    void *dWeightsAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *dy = nullptr;
    aclTensor *sparseIndices = nullptr;
    aclTensor *weights = nullptr;
    aclTensor *dQuery = nullptr;
    aclTensor *dKey = nullptr;
    aclTensor *dWeights = nullptr;

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1.0);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1.0);
    std::vector<aclFloat16> dyHostData(GetShapeSize(dyShape), 1.0);
    std::vector<int32_t> sparseIndicesHostData(GetShapeSize(sparseIndicesShape), 1);
    std::vector<aclFloat16> weightsHostData(GetShapeSize(weightsShape), 1.0);
    std::vector<aclFloat16> dQueryHostData(GetShapeSize(dQueryShape), 1.0);
    std::vector<aclFloat16> dKeyHostData(GetShapeSize(dKeyShape), 1.0);
    std::vector<aclFloat16> dWeightsHostData(GetShapeSize(dWeightsShape), 1.0);

    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // 创建acl Tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT16, &dy);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT16, &weights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    
    ret = CreateAclTensor(dQueryHostData, dQueryShape, &dQueryAddr, aclDataType::ACL_FLOAT16, &dQuery);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dKeyHostData, dKeyShape, &dKeyAddr, aclDataType::ACL_FLOAT16, &dKey);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dWeightsHostData, dWeightsShape, &dWeightsAddr, aclDataType::ACL_FLOAT16, &dWeights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnLightningIndexerGrad第一段接口
    ret = aclnnLightningIndexerGradGetWorkspaceSize(
        q, k, dy, sparseIndices, weights, nullptr, nullptr, headNum,
        layoutStr, sparseMode, preToken, nextToken, deteminstic, dQuery, dKey, dWeights, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
            return ret);
    }

    // 调用aclnnLightningIndexerGrad第二段接口
    ret = aclnnLightningIndexerGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGrad failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult<aclFloat16>(dQueryShape, &dQueryAddr);
    PrintOutResult<aclFloat16>(dKeyShape, &dKeyAddr);
    PrintOutResult<aclFloat16>(dWeightsShape, &dWeightsAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                  sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);

    return 0;
}
```