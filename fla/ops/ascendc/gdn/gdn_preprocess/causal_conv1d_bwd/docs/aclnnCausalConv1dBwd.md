# aclnnCausalConv1dBwd

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

- 接口功能：完成因果一维卷积（Causal Conv1d）的反向传播计算，输出输入梯度 `dx`、权重梯度 `dw`、偏置梯度 `db` 和初始状态梯度 `dh0`。
- 支持固定长度序列和变长序列：
  - 固定长度：输入为 `BSND`/`BSH` 或 `BNSD` layout。
  - 变长序列：输入为 `TND` 或 `NTD` layout，并通过 `queryStartLocOptional` 描述每条序列的起止位置。
- 支持输出梯度 layout 转换：`inputLayoutOptional` 只指示 `yOptional`、`dy` 的输入物理排布，`x`、`dxOut` 始终使用逻辑 layout。
- 计算公式：

  令卷积核宽度为 $W$，特征维度为 $D$，前向预激活输出为 $y$，上游梯度为 $dy$。当 `activation=0` 时：

  $$
  g_t = dy_t
  $$

  当 `activation=1` 或 `activation=2` 时，按 SiLU/Swish 反向计算有效梯度：

  $$
  \sigma(y_t)=\frac{1}{1+e^{-y_t}}
  $$

  $$
  g_t = dy_t \cdot \sigma(y_t) \cdot (1 + y_t \cdot (1 - \sigma(y_t)))
  $$

  反向传播计算：

  $$
  dx_t = \sum_{i=0}^{W-1} g_{t+i} \cdot weight_{W-1-i}
  $$

  $$
  dw_{W-1-i} = \sum_{b,t} g_{b,t+i} \cdot x_{b,t}
  $$

  $$
  db = \sum_{b,t} g_{b,t}
  $$

  当 `initialStateOptional` 存在时，算子会把序列开头依赖的历史状态纳入 `dw` 和 `dh0` 计算；当 `dhtOptional` 存在时，算子会把最终卷积状态梯度反传到序列尾部的 `dx`。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnCausalConv1dBwdGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnCausalConv1dBwd` 接口执行计算。

```cpp
aclnnStatus aclnnCausalConv1dBwdGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *yOptional,
    const aclTensor *weight,
    const aclTensor *dy,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *queryStartLocOptional,
    int64_t activation,
    char *inputLayoutOptional,
    const aclTensor *dxOut,
    const aclTensor *dwOutOptional,
    const aclTensor *dbOutOptional,
    const aclTensor *dh0OutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnCausalConv1dBwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

如需以 `aclTensor` 形式传入 `queryStartLocOptional`，可使用 `aclnnCausalConv1dBwdTensorGetWorkspaceSize`，除 `queryStartLocOptional` 类型为 `const aclTensor *` 外，其余参数含义与 `aclnnCausalConv1dBwdGetWorkspaceSize` 相同。

## aclnnCausalConv1dBwdGetWorkspaceSize

- 参数说明：

  | 参数名 | 输入/输出 | 描述 |
  | :-- | :--: | :-- |
  | x | 输入 | 前向输入 tensor。固定长度 shape 为 `[B, T, D]`，变长 shape 为 `[totalTokens, D]`，始终使用逻辑 layout。 |
  | yOptional | 输入 | 前向预激活输出 tensor。物理 layout 由 `inputLayoutOptional` 指定，shape 与 `dy` 相同。`activation=0` 时可为 `nullptr`；`activation=1` 或 `activation=2` 时必须提供。 |
  | weight | 输入 | 卷积核权重 tensor，shape 为 `[W, D]`，数据类型与 `x` 相同，format 支持 `ND`。`W` 为卷积核宽度，`D` 为逻辑特征维度。 |
  | dy | 输入 | 前向输出梯度 tensor。数据类型、format、shape 与 `x` 相同。 |
  | initialStateOptional | 输入 | 初始卷积状态 tensor，shape 为 `[B, W, D]`，数据类型与 `x` 相同，format 支持 `ND`。可为 `nullptr`。 |
  | dhtOptional | 输入 | 最终卷积状态梯度 tensor，shape 为 `[B, W, D]`，数据类型与 `x` 相同，format 支持 `ND`。可为 `nullptr`。 |
  | queryStartLocOptional | 输入 | 变长序列起止位置数组，类型为 `aclIntArray`，元素类型为 `int64_t`，长度为 `B+1`。`TND`/`NTD` layout 下必须提供；固定长度 layout 下可为 `nullptr`。 |
  | activation | 输入 | 激活反向模式，`int64_t` 类型。`0` 表示不使用激活函数；`1` 表示 SiLU；`2` 表示 Swish，当前与 SiLU 等价。 |
  | inputLayoutOptional | 输入 | 输入 layout 字符串，可为 `nullptr`，默认值为 `BSND`。支持 `BSND`、`BSH`、`TND`、`BNSD`、`NTD`，其中 `BSH` 等价于 `BSND`。 |
  | dxOut | 输出 | 输入梯度 tensor。输出采用逻辑 layout：固定长度为 `[B, T, D]`，变长为 `[totalTokens, D]`。数据类型与 `x` 相同，format 支持 `ND`。 |
  | dwOutOptional | 输出 | 权重梯度 tensor，shape 为 `[W, D]`，数据类型与 `weight` 相同，format 支持 `ND`。 |
  | dbOutOptional | 输出 | 偏置梯度 tensor，shape 为 `[D]`，数据类型与输入相同，format 支持 `ND`。 |
  | dh0OutOptional | 输出 | 初始状态梯度 tensor，shape 为 `[B, W, D]`，数据类型与 `x` 相同，format 支持 `ND`。当不需要 `dh0` 或未提供 `initialStateOptional` 时可为 `nullptr`。 |
  | workspaceSize | 输出 | 返回用户需要在 Device 侧申请的 workspace 大小。 |
  | executor | 输出 | 返回 op 执行器，包含算子计算流程。 |

- Layout 与 shape 说明：

  | inputLayoutOptional | `x` shape | `yOptional/dy` shape | `dxOut` shape | `queryStartLocOptional` |
  | :-- | :-- | :-- | :-- | :-- |
  | `BSND` / `BSH` | `[B, T, D]` | `[B, T, D]` | `[B, T, D]` | 可为 `nullptr` |
  | `BNSD` | `[B, T, D]` | `[B, N, T, Dh]`，`D=N*Dh` | `[B, T, D]` | 可为 `nullptr` |
  | `TND` | `[totalTokens, D]` | `[totalTokens, D]` | `[totalTokens, D]` | 必须提供 |
  | `NTD` | `[totalTokens, D]` | `[N, totalTokens, Dh]`，`D=N*Dh` | `[totalTokens, D]` | 必须提供 |

  对于变长序列，`queryStartLocOptional` 必须满足：

  $$
  queryStartLoc[0]=0
  $$

  $$
  queryStartLoc[B]=totalTokens
  $$

  且 `queryStartLoc[i+1] >= queryStartLoc[i]`。

- 返回值：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnCausalConv1dBwd

- 参数说明：

  | 参数名 | 输入/输出 | 描述 |
  | :-- | :--: | :-- |
  | workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。当 `workspaceSize` 为 0 时可传 `nullptr`。 |
  | workspaceSize | 输入 | workspace 大小，由第一段接口 `aclnnCausalConv1dBwdGetWorkspaceSize` 获取。 |
  | executor | 输入 | op 执行器，由第一段接口 `aclnnCausalConv1dBwdGetWorkspaceSize` 获取。 |
  | stream | 输入 | acl stream 流。 |

- 返回值：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - `aclnnCausalConv1dBwd` 默认确定性实现。
- 数据类型：
  - `x`、`yOptional`、`weight`、`dy`、`initialStateOptional`、`dhtOptional` 的数据类型必须一致。
  - `dxOut`、`dh0OutOptional` 的数据类型必须与 `x` 一致。
  - `dwOutOptional`、`dbOutOptional` 的数据类型与输入相同。
  - `FLOAT16`、`BFLOAT16` 场景下，算子内部按 `FLOAT` 进行主要累积计算，所有输出在写回时转换为输入数据类型。
- shape：
  - `weight` 必须为二维 tensor，shape 为 `[W, D]`。
  - `yOptional` 和 `dy` 必须采用相同输入 layout 和相同 shape；`x` 始终采用逻辑 layout。
  - `initialStateOptional`、`dhtOptional`、`dh0OutOptional` 的逻辑 shape 为 `[B, W, D]`，与正向算子的状态 layout 一致。
  - `BSND`/`TND` layout 下逻辑特征维 `D` 必须为 16 的倍数；`BNSD`/`NTD` layout 下最后一维 `Dh` 必须为 16 的倍数，逻辑特征维为 `D=N*Dh`。
  - 不支持空序列，固定长度场景下 `T > 0`，变长场景下 `totalTokens > 0`。
- layout：
  - `inputLayoutOptional` 仅影响 `yOptional`、`dy` 的输入物理排布。
  - `dxOut` 始终按逻辑 layout 输出，不输出 `BNSD` 或 `NTD` 物理排布。
  - `BNSD` 和 `NTD` layout 下，`D=N*Dh`，且 `Dh` 需要满足 16 对齐约束。
- 激活函数：
  - `activation` 仅支持 `0`、`1`、`2`。
  - `activation=1` 或 `activation=2` 时必须提供 `yOptional`。
- 变长序列：
  - `TND`/`NTD` layout 下必须提供 `queryStartLocOptional`。
  - `queryStartLocOptional` 必须是 `int64_t` 数组，长度为 `B+1`，首元素为 0，末元素为 `totalTokens`，且单调非递减。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_bwd.h"

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
    return ACL_SUCCESS;
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
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t B = 2;
    int64_t T = 128;
    int64_t D = 256;
    int64_t W = 4;
    int64_t activation = 0;
    char inputLayout[] = "BSND";

    std::vector<int64_t> xShape = {B, T, D};
    std::vector<int64_t> weightShape = {W, D};
    std::vector<int64_t> stateShape = {B, W, D};
    std::vector<int64_t> dxShape = {B, T, D};
    std::vector<int64_t> dwShape = {W, D};
    std::vector<int64_t> dbShape = {D};
    std::vector<int64_t> dh0Shape = {B, W, D};

    std::vector<float> xHost(GetShapeSize(xShape), 1.0f);
    std::vector<float> yHost(GetShapeSize(xShape), 0.0f);
    std::vector<float> weightHost(GetShapeSize(weightShape), 0.5f);
    std::vector<float> dyHost(GetShapeSize(xShape), 2.0f);
    std::vector<float> initialStateHost(GetShapeSize(stateShape), 0.0f);
    std::vector<float> dhtHost(GetShapeSize(stateShape), 0.0f);
    std::vector<float> dxHost(GetShapeSize(dxShape), 0.0f);
    std::vector<float> dwHost(GetShapeSize(dwShape), 0.0f);
    std::vector<float> dbHost(GetShapeSize(dbShape), 0.0f);
    std::vector<float> dh0Host(GetShapeSize(dh0Shape), 0.0f);

    void *xAddr = nullptr;
    void *yAddr = nullptr;
    void *weightAddr = nullptr;
    void *dyAddr = nullptr;
    void *initialStateAddr = nullptr;
    void *dhtAddr = nullptr;
    void *dxAddr = nullptr;
    void *dwAddr = nullptr;
    void *dbAddr = nullptr;
    void *dh0Addr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *y = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *dy = nullptr;
    aclTensor *initialState = nullptr;
    aclTensor *dht = nullptr;
    aclTensor *dx = nullptr;
    aclTensor *dw = nullptr;
    aclTensor *db = nullptr;
    aclTensor *dh0 = nullptr;

    ret = CreateAclTensor(xHost, xShape, &xAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHost, xShape, &yAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHost, weightShape, &weightAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dyHost, xShape, &dyAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(initialStateHost, stateShape, &initialStateAddr, aclDataType::ACL_FLOAT, &initialState);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dhtHost, stateShape, &dhtAddr, aclDataType::ACL_FLOAT, &dht);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dxHost, dxShape, &dxAddr, aclDataType::ACL_FLOAT, &dx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dwHost, dwShape, &dwAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dbHost, dbShape, &dbAddr, aclDataType::ACL_FLOAT, &db);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dh0Host, dh0Shape, &dh0Addr, aclDataType::ACL_FLOAT, &dh0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnCausalConv1dBwdGetWorkspaceSize(x, y, weight, dy, initialState, dht, nullptr,
                                               activation, inputLayout, dx, dw, db, dh0,
                                               &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCausalConv1dBwd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBwd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(weight);
    aclDestroyTensor(dy);
    aclDestroyTensor(initialState);
    aclDestroyTensor(dht);
    aclDestroyTensor(dx);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);
    aclDestroyTensor(dh0);
    aclrtFree(xAddr);
    aclrtFree(yAddr);
    aclrtFree(weightAddr);
    aclrtFree(dyAddr);
    aclrtFree(initialStateAddr);
    aclrtFree(dhtAddr);
    aclrtFree(dxAddr);
    aclrtFree(dwAddr);
    aclrtFree(dbAddr);
    aclrtFree(dh0Addr);
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

## 变长序列调用说明

`TND`/`NTD` layout 下需要创建 `aclIntArray` 作为 `queryStartLocOptional`：

```cpp
std::vector<int64_t> queryStartLocHost = {0, 7, 72, 202};
aclIntArray *queryStartLoc = aclCreateIntArray(queryStartLocHost.data(), queryStartLocHost.size());
char inputLayout[] = "TND";

ret = aclnnCausalConv1dBwdGetWorkspaceSize(x, y, weight, dy, initialState, dht, queryStartLoc,
                                           activation, inputLayout, dx, dw, db, dh0,
                                           &workspaceSize, &executor);

aclDestroyIntArray(queryStartLoc);
```

## 常见问题

### Q1：为什么 `dwOutOptional` 和 `dbOutOptional` 仍在内部使用 `FLOAT` 累加？

`dw` 和 `db` 需要跨 batch、序列长度和 block 进行累加，kernel 内使用 `FLOAT` 可以降低低精度输入在累积阶段的精度损失；归约完成后再转换为输入数据类型写回。

### Q2：`BNSD`/`NTD` layout 下为什么 `dxOut` 不是同样的物理 layout？

`inputLayoutOptional` 只描述前向输出 `yOptional` 及其梯度 `dy` 的物理排布。`x` 和 `dxOut` 都保持逻辑 layout：固定长度为 `[B, T, D]`，变长为 `[totalTokens, D]`。

### Q3：什么时候必须提供 `queryStartLocOptional`？

当 `inputLayoutOptional` 为 `TND` 或 `NTD` 时必须提供。`queryStartLocOptional` 用于描述变长 batch 中每条序列在 `totalTokens` 维度上的起止位置。

### Q4：什么时候必须提供 `yOptional`？

当 `activation=1` 或 `activation=2` 时必须提供。SiLU/Swish 反向需要前向预激活输出 `yOptional` 计算有效梯度；`activation=0` 时可传 `nullptr`。

### Q5：`initialStateOptional` 和 `dhtOptional` 的作用是什么？

`initialStateOptional` 表示前向计算进入当前序列前的卷积历史状态，会参与序列开头位置的 `dw` 和 `dh0` 计算；`dhtOptional` 表示最终卷积状态的梯度，会反传到序列尾部的 `dx`。
