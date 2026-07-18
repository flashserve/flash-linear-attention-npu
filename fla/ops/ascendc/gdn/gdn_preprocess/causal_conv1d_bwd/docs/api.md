# CausalConv1dBwd API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.causal_conv1d_bwd` | 支持 |
| aclnn | `aclnnCausalConv1dBwdGetWorkspaceSize` / `aclnnCausalConv1dBwd` | 支持 |
| Ascend C `<<<>>>` | `causal_conv1d_bwd<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_causal_conv1d_bwd` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,T,D] 或 [T,D]` | FP32/FP16/BF16 | 逻辑 BSH/TH | 前向输入 |
| `y` | 条件可选 | `由 input_layout 决定` | 与 x 一致 | BSH/BSND/BNSD/TND/NTD | 预激活；activation=1/2 必选 |
| `weight` | 必选 | `[W,D]` | 与 x 一致 | ND | 卷积权重 |
| `dy` | 必选 | `与 y 同形` | 与 x 一致 | 由 input_layout 决定 | 上游梯度 |
| `initial_state` | 可选 | `[B,W,D]` | 与 x 一致 | ND | 前向初始状态 |
| `dht` | 可选 | `[B,W,D]` | 与 x 一致 | ND | 末状态梯度 |
| `query_start_loc` | TND/NTD 必选 | `[B+1]` | INT64 | ND | 变长序列边界 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dx` | `与逻辑 x 同形` | 与 x 一致 | 输入梯度 |
| `dw` | `[W,D]` | 与 weight 一致 | 权重梯度 |
| `db` | `[D]` | 与 weight 一致 | 偏置梯度 |
| `dh0` | `[B,W,D]` | 与 x 一致 | 初始状态梯度 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `activation` | int | `0` | `{0, 1, 2}` | 0=无激活，1=SiLU，2=Swish（同 SiLU） |
| `input_layout` | str | `BSND` | `{"BSH", "BSND", "BNSD", "TND", "NTD"}` | 输入输出布局 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnCausalConv1dBwdGetWorkspaceSize(/* 参数见本页公共参数表 */, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnCausalConv1dBwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnCausalConv1dBwdGetWorkspaceSize(
    x, y, weight, dy, initialState, dht, nullptr, activation, "BNSD", dx, dw, db, dh0, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnCausalConv1dBwd(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/causal_conv1d_bwd/routes/test_aclnn_causal_conv1d_bwd.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
causal_conv1d_bwd(x, y, weight, dy, initial_state=None, dht=None, *, query_start_loc=None, activation=0, input_layout='BSND')
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import causal_conv1d_bwd

B, T, D, W = 2, 64, 128, 3
x = torch.randn(B, T, D, device="npu", dtype=torch.float16)
weight = torch.randn(W, D, device="npu", dtype=torch.float16)
y = torch.randn(B, T, 4, D // 4, device="npu", dtype=torch.float16)
dy = torch.randn_like(y)
dx, dw, db, dh0 = causal_conv1d_bwd(x, y, weight, dy, activation=1, input_layout="BNSD")
torch.npu.synchronize()
assert dx.shape == x.shape and dw.shape == weight.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
causal_conv1d_bwd<<<blockDim, nullptr, stream>>>(x, y, weight, dy, initial_state, dht, query_start_loc, dx, dw, db, dh0, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/causal_conv1d_bwd/routes/test_direct_causal_conv1d_bwd.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, T, D, W = 2, 64, 128, 3
x = torch.randn(B, T, D, device="npu", dtype=torch.float16)
weight = torch.randn(W, D, device="npu", dtype=torch.float16)
y = torch.randn(B, T, 4, D // 4, device="npu", dtype=torch.float16)
dy = torch.randn_like(y)
dx, dw, db, dh0 = torch.ops.npu.npu_causal_conv1d_bwd(x, y, weight, dy, activation=1, input_layout="BNSD")
torch.npu.synchronize()
assert dx.shape == x.shape and dw.shape == weight.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- weight 必须 `[W,D]` 且 W 仅支持 2/3/4；D 或拆分后的 V 必须为 16 的倍数。
- activation=1/2 时 y 必须提供且与 dy 同 layout/shape。
- TND/NTD 必须提供合法 query_start_loc；累计长度非递减、首项为 0、末项为总 T，且总 token 数必须大于 0。
- initial_state/dht/dh0 的逻辑 shape 为 `[B,W,D]`。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| x/weight/dy/dx/dw/db/dh0、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID |
| activation 不在 0/1/2，或启用激活时 y 缺失/shape 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| layout 不在 BSH/BSND/BNSD/TND/NTD，或物理 rank/shape 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| W 不在 2/3/4、D/V 未按 16 对齐、浮点 dtype 不一致 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| 变长序列缺 query_start_loc 或累计长度非法 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/causal_conv1d_bwd.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长/变长序列、无激活/SiLU、可选初末状态 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
