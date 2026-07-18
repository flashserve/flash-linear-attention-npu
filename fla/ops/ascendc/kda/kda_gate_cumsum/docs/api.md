# KdaGateCumsum API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.kda_gate_cumsum` | 支持 |
| aclnn | `aclnnKdaGateCumsumGetWorkspaceSize` / `aclnnKdaGateCumsum` | 支持 |
| Ascend C `<<<>>>` | `kda_gate_cumsum<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_kda_gate_cumsum` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `g` | 必选 | `[B,T,H_v,K]/[B,H_v,T,K]/[T,H_v,K]/[H_v,T,K]` | FP16/BF16/FP32 | BSND/BNSD/TND/NTD | step gate 或 raw gate |
| `A_log` | 条件必选 | `[H_v]` | FP32 | ND | use_gate_in_kernel=true 时必选 |
| `dt_bias` | 可选 | `[H_v*K] 或 [H_v,K]` | FP32 | ND | safe-gate 偏置 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `gk` | `与 g 相同` | FP32 | chunk 内 log2 累积 gate |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | `{32, 64, 128}` | chunk 长度 |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 是否把 g 当作 raw gate |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 模式下必须为 true，非 raw gate 模式下必须为 false |
| `lower_bound` | double | `-5.0` | `[-5, 0)` | safe gate 下限 |
| `layout` | str | `BSND` | `{"BSND", "BNSD", "TND", "NTD"}` | 仅接受大写取值 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnKdaGateCumsumGetWorkspaceSize(
const aclTensor *g,
const aclTensor *aLogOptional,
const aclTensor *dtBiasOptional,
const aclIntArray *cuSeqlensOptional,
int64_t chunkSize,
bool useGateInKernel,
bool safeGate,
double lowerBound,
const char *layout,
const aclTensor *gkOut,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnKdaGateCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnKdaGateCumsumGetWorkspaceSize(
    g, aLog, dtBias, cuSeqlens, chunkSize, useGateInKernel, safeGate, lowerBound, layout, gk, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnKdaGateCumsum(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/kda_gate_cumsum/routes/test_aclnn_kda_gate_cumsum.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
kda_gate_cumsum(g, chunk_size, *, A_log=None, dt_bias=None, cu_seqlens=None, use_gate_in_kernel=False, safe_gate=False, lower_bound=None, layout='BSND')
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import kda_gate_cumsum

B, T, H_v, K, C = 1, 128, 4, 128, 64
raw = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
dt_bias = torch.randn(H_v, K, device="npu", dtype=torch.float32)
gk = kda_gate_cumsum(raw, C, A_log=A_log, dt_bias=dt_bias,
             use_gate_in_kernel=True, safe_gate=True,
             lower_bound=-5.0, layout="BSND")
torch.npu.synchronize()
assert gk.shape == raw.shape and gk.dtype == torch.float32
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
kda_gate_cumsum<<<blockDim, nullptr, stream>>>(g, aLog, dtBias, cuSeqlens, gk, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/kda_gate_cumsum/routes/test_direct_kda_gate_cumsum.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, T, H_v, K, C = 1, 128, 4, 128, 64
raw = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
dt_bias = torch.randn(H_v, K, device="npu", dtype=torch.float32)
gk = torch.ops.npu.npu_kda_gate_cumsum(raw, C, A_log=A_log, dt_bias=dt_bias,
             use_gate_in_kernel=True, safe_gate=True,
             lower_bound=-5.0, layout="BSND")
torch.npu.synchronize()
assert gk.shape == raw.shape and gk.dtype == torch.float32
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- K<=256，chunk_size 仅支持 32/64/128。
- use_gate_in_kernel=true 时 A_log 必须为 [H_v]、safe_gate 必须 true，dt_bias 若存在须为 [H_v*K] 或 [H_v,K]。
- lower_bound 仅支持 [-5,0)；use_gate_in_kernel=false 时 safe_gate 必须 false。
- use_gate_in_kernel=false 时 A_log 与 dt_bias 必须为空，避免未消费输入在不同通路产生歧义。
- rank4 变长序列物理 B 必须为 1；cu_seqlens 首项为 0、非递减且末项等于 T。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| rank/shape/dtype/layout、chunk_size、lower_bound 或模式组合非法 | ACLNN_ERR_PARAM_INVALID |
| step-gate 模式仍传入 A_log/dt_bias | ACLNN_ERR_PARAM_INVALID |
| Python 输入不是 NPU tensor 或 runtime/op_api 未加载 | RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/kda_gate_cumsum.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
