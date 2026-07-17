# CausalConv1d API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.causal_conv1d` | 支持 |
| aclnn | `aclnnCausalConv1dGetWorkspaceSize` / `aclnnCausalConv1d` | 支持 |
| Ascend C `<<<>>>` | `causal_conv1d<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_causal_conv1d` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,T,D] 或 [T,D]` | FP16/BF16 | BSH/TH | 输入序列 |
| `weight` | 必选 | `[W,D]` | FP16/BF16 | ND | depthwise 卷积权重 |
| `bias` | 可选 | `[D]` | FP16/BF16 | ND | 偏置 |
| `conv_states` | 必选/可变 | `[D_s,L_s,D]` | FP16/BF16 | ND | 历史输入状态，原地更新 |
| `query_start_loc` | 可选 | `[B+1]` | INT64 | ND | varlen 序列边界 |
| `cache_indices` | 可选 | `[B]` | INT64 | ND | 序列到状态槽的映射 |
| `initial_state_mode` | 可选 | `[B]` | INT64 | ND | 是否使用已有初始状态 |
| `num_accepted_tokens` | 可选 | `[B]` | INT64 | ND | 投机解码接受数 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `y` | `与 x 对应；head_num>0 时为 BNSD/NTD` | 与 x 一致 | 卷积输出 |
| `conv_states` | `[D_s,L_s,D]` | 与输入一致 | 原地更新后的状态 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `activation_mode` | int | `0` | 0=无激活，1=SiLU |
| `pad_slot_id` | int | `-1` | 跳过的缓存槽 |
| `run_mode` | int | `0` | 0=forward，1=update |
| `head_num` | int | `0` | forward 输出拆分 head；0 保持 BSH/TH |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(/* 参数见本页公共参数表 */, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnCausalConv1dGetWorkspaceSize(
    x, weight, bias, convStates, nullptr, nullptr, nullptr, nullptr, activationMode, padSlotId, runMode, headNum, y, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnCausalConv1d(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/causal_conv1d/routes/test_aclnn_causal_conv1d.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
causal_conv1d(x, weight, bias=None, conv_states=None, *, query_start_loc=None, cache_indices=None, initial_state_mode=None, num_accepted_tokens=None, activation_mode=0, pad_slot_id=-1, run_mode=0, head_num=0)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import causal_conv1d

B, T, D, W = 2, 64, 128, 3
x = torch.randn(B, T, D, device="npu", dtype=torch.bfloat16)
weight = torch.randn(W, D, device="npu", dtype=torch.bfloat16)
bias = torch.randn(D, device="npu", dtype=torch.bfloat16)
state = torch.zeros(B, W, D, device="npu", dtype=torch.bfloat16)
y = causal_conv1d(x, weight, bias, state, activation_mode=1, run_mode=0)
torch.npu.synchronize()
assert y.shape == x.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
causal_conv1d<runModeKey, widthKey, fnPlanKey><<<blockDim, nullptr, stream>>>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/causal_conv1d/routes/test_direct_causal_conv1d.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, T, D, W = 2, 64, 128, 3
x = torch.randn(B, T, D, device="npu", dtype=torch.bfloat16)
weight = torch.randn(W, D, device="npu", dtype=torch.bfloat16)
bias = torch.randn(D, device="npu", dtype=torch.bfloat16)
state = torch.zeros(B, W, D, device="npu", dtype=torch.bfloat16)
y = torch.ops.npu.npu_causal_conv1d(x, weight, bias, state, activation_mode=1, run_mode=0)
torch.npu.synchronize()
assert y.shape == x.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- 卷积宽度 W 仅支持 2/3/4，特征维 D 必须为 16 的倍数。
- activation_mode 仅支持 0/1，run_mode 仅支持 0/1。
- head_num>0 仅用于 forward，且必须整除 D；拆分后的 D/head_num 仍须为 16 的倍数。
- conv_states 的 L_s 至少为 W-1；rank-3 投机 update 还要求 L_s >= (W-1)+(T-1)。
- num_accepted_tokens 仅在 run_mode=1 且 W=4 时实现，值域为每条逻辑序列的 [0,token_count]。
- conv_states 为可变输入；非连续状态仅在 CANN >= 9.1 支持。
- rank-2 forward 必须提供 query_start_loc；query_start_loc、cache_indices、initial_state_mode 和接受数的长度必须与逻辑序列数一致。
- initial_state_mode 仅用于 run_mode=0 且元素只能为 0/1；cache_indices 只能选择有效状态槽或 pad_slot_id。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| x/weight/conv_states/y、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID |
| W 不在 2/3/4、D 未按 16 对齐、shape 或 dtype 不一致 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| rank-2 forward 缺 query_start_loc，或元数据长度/值域非法 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| num_accepted_tokens 用于非 update、W!=4 或超出 token 数 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| head_num 用于 update、不能整除 D 或拆分维未按 16 对齐 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/causal_conv1d.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、dense/varlen forward、decode/update、投机接受 token 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
