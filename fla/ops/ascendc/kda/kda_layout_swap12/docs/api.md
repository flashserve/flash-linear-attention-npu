# KdaLayoutSwap12 API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.kda_layout_swap12` | 支持 |
| aclnn | `aclnnKdaLayoutSwap12GetWorkspaceSize` / `aclnnKdaLayoutSwap12` | 支持 |
| Ascend C `<<<>>>` | `kda_layout_swap12<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `-` | 未实现 |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[D_0,D_1,D_2,...]，rank>=3` | FP16/BF16/FP32 | ND | 连续化后参与重排 |
| `dependency` | 可选 | `任意 tensor` | 任意 | ND | 仅调度依赖，不读取值 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `y` | `rank3: [D_1,D_0,D_2]；rank>=4: [D_0,D_2,D_1,...]` | 与 x 相同 | 连续转置结果 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `无` | - | `-` | - | 无公开属性 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnKdaLayoutSwap12GetWorkspaceSize(
const aclTensor *x,
const aclTensor *dependencyOptional,
const aclTensor *y,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnKdaLayoutSwap12(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成校验并创建 executor；第二段在传入 stream 上异步执行。输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
int32_t deviceId = 0;
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
aclrtStream stream = nullptr;
ACL_CHECK(aclrtCreateStream(&stream));

// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnKdaLayoutSwap12GetWorkspaceSize(
    x, dependency, y, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnKdaLayoutSwap12(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/kda_layout_swap12/routes/test_aclnn_kda_layout_swap12.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
kda_layout_swap12(x, *, dependency=None)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import kda_layout_swap12

B, T, H_v, K = 2, 129, 4, 128
x = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
y = kda_layout_swap12(x)
torch.npu.synchronize()
assert y.shape == (B, H_v, T, K)
torch.testing.assert_close(y.cpu(), x.permute(0, 2, 1, 3).contiguous().cpu())
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
kda_layout_swap12<<<blockDim, nullptr, stream>>>(x, dependency, y, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/kda_layout_swap12/routes/test_direct_kda_layout_swap12.cpp`。

## 6. `torch.ops.npu` API（可选）

当前未注册 `torch.ops.npu` 入口；调用方使用 `fla_npu.ops.ascendc`。

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- x rank 必须至少为 3；y rank、dtype 及交换后的每一维必须与 x 精确对应。
- 输入在 L2 中先 contiguous；当前 API 总是创建连续 y。
- dependency 只表达执行顺序，不提供数据、dtype 或 shape 语义，不得依赖其值改变输出。
- aclnn/Python 通过 executor 输入依赖排序；`<<<>>>` 直调不读取 dependency，调用者必须在同一 stream 上先发射其生产者。
- 所有维度必须为正；host 在进入 tiling 前拒绝空维度，kernel 的 usedCoreNum 至少为 1。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| x 或 y 为空 | ACLNN_ERR_PARAM_INVALID（CheckParams 外层映射） |
| x rank 小于 3、存在空维或 dtype 不在 FP16/BF16/FP32 | ACLNN_ERR_PARAM_INVALID |
| y 的 shape/dtype 不符合维 1/2 交换契约 | ACLNN_ERR_PARAM_INVALID |
| executor 创建、contiguous、内部 op 或 kernel 执行失败 | ACLNN_ERR_INNER/内部错误码 |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/kda_layout_swap12.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
