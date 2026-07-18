# <op_name> API 与调用示例

<!-- 每个算子只维护这一份 API 文档。先选择 ascendc 或 triton 实现类型，只保留对应章节；替换占位符，删除未实现的可选接口。“提示”和“示例”仅用于展示填写粒度，完成文档后应删除。 -->

## 1. API 总览

| 通路 | API/入口 | 支持情况 | 说明 |
| --- | --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.<op_name>` / `fla_npu.ops.triton.<op_name>` | 必须，二选一 | 必须与实现类型一致 |
| aclnn | `aclnn<OpName>GetWorkspaceSize`、`aclnn<OpName>` | Ascend C 算子必须 | Triton 算子删除本行 |
| Ascend C `<<<>>>` | 直接发射 | Ascend C 算子必须 | Triton 算子删除本行 |
| legacy | `torch.ops.npu.<op_name>` | 可选 | 未实现时删除本行及对应章节 |

> **填写示例：** `chunk_bwd_dv_local` 是 Ascend C 算子，因此填写 `fla_npu.ops.ascendc.chunk_bwd_dv_local`、`aclnnChunkBwdDvLocal*`、Ascend C `<<<>>>` 直调，以及可选的 `torch.ops.npu.npu_chunk_bwd_dv_local`；不填写 `fla_npu.ops.triton`。

## 2. 公共参数与约束

本文不重复定义 Shape 符号。参数 Shape 和接口语义统一引用[算子 README 的 Shape 变量说明附录](../README.md#shape-symbols)，并与所属模型的权威符号表保持一致。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Format/Layout | 取值范围 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `<input>` | 必选 | `<shape>` | `<dtype>` | `<format>` | `<range>` | `<description>` |

> **填写示例：** `chunk_bwd_dv_local` 应逐行列出 `q`、`k`、`dO`、`g`、`g_gamma`、`A`、`cu_seqlens`、`chunk_indices`。Shape 写为 `q/k=[B,H_k,T,K]`、`dO=[B,H_v,T,V]`、`g=[B,H_v,T]`，并说明 `H_v % H_k == 0`；不得在 Shape 中直接写固定维度值。

### 2.2 输出

| 名称 | Shape | Dtype | Format/Layout | 说明 |
| --- | --- | --- | --- | --- |
| `<output>` | `<shape>` | `<dtype>` | `<format>` | `<description>` |

> **填写示例：** `chunk_bwd_dv_local` 的 `out/dV` 写为 `[B,H_v,T,V]`；同时说明输出 dtype、format、非连续输出和无效区域语义。`V` 的固定支持值统一写入“已知限制”。

### 2.3 属性

| 名称 | 必选/可选 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `<attr>` | `<required>` | `<type>` | `<default>` | `<range>` | `<description>` |

> **填写示例：** `chunk_bwd_dv_local` 的 `scale` 和 `chunk_size` 均为必选属性。`chunk_size` 应在“取值范围”中完整列出 `{64, 128}`；`scale` 的推荐值写入“说明”。枚举或离散属性的合法值不得只写入“已知限制”，也不要把“推荐值”和“硬约束”混写。

## 3. aclnn API（仅 Ascend C 算子）

Triton 算子删除本章节。

### 3.1 接口签名

```cpp
aclnnStatus aclnn<OpName>GetWorkspaceSize(
    const aclTensor *input,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnn<OpName>(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

按实际接口补全参数，说明 workspace、executor、stream、异步执行、生命周期、返回值和每类错误码的触发条件。

> **接口示例：以 `chunk_bwd_dv_local` 为例：**
>
> ```cpp
> aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
>     const aclTensor *q,
>     const aclTensor *k,
>     const aclTensor *dO,
>     const aclTensor *g,
>     const aclTensor *gGammaOptional,
>     const aclTensor *aOptional,
>     const aclIntArray *cuSeqlensOptional,
>     const aclIntArray *chunkIndicesOptional,
>     double scale,
>     int64_t chunkSize,
>     const aclTensor *out,
>     uint64_t *workspaceSize,
>     aclOpExecutor **executor);
> ```

### 3.2 调用示例

```cpp
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_bwd_dv_local.h"

#define CHECK_ACL(expr)                                                                                \
    do {                                                                                               \
        const auto status = (expr);                                                                    \
        if (status != ACL_SUCCESS) {                                                                    \
            std::fprintf(stderr, "%s failed, error code: %d\\n", #expr, static_cast<int>(status));      \
            return static_cast<int>(status);                                                           \
        }                                                                                              \
    } while (0)

namespace {

size_t GetElementCount(const std::vector<int64_t> &shape)
{
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

class AclRuntime {
public:
    int Init(int32_t deviceId)
    {
        deviceId_ = deviceId;
        CHECK_ACL(aclInit(nullptr));
        aclInitialized_ = true;
        CHECK_ACL(aclrtSetDevice(deviceId_));
        deviceSet_ = true;
        CHECK_ACL(aclrtCreateStream(&stream_));
        return ACL_SUCCESS;
    }

    ~AclRuntime()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
        }
        if (aclInitialized_) {
            (void)aclFinalize();
        }
    }

    aclrtStream GetStream() const
    {
        return stream_;
    }

private:
    int32_t deviceId_ = 0;
    aclrtStream stream_ = nullptr;
    bool aclInitialized_ = false;
    bool deviceSet_ = false;
};

struct DeviceBuffer {
    void *data = nullptr;
    size_t size = 0;

    ~DeviceBuffer()
    {
        if (data != nullptr) {
            (void)aclrtFree(data);
        }
    }

    int Allocate(size_t byteSize)
    {
        size = byteSize;
        if (size == 0) {
            return ACL_SUCCESS;
        }
        return aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST);
    }
};

struct TensorHandle {
    aclTensor *value = nullptr;

    ~TensorHandle()
    {
        if (value != nullptr) {
            aclDestroyTensor(value);
        }
    }
};

int CreateFp16Tensor(const std::vector<int64_t> &shape, uint16_t fillValue,
                     DeviceBuffer &buffer, TensorHandle &tensor)
{
    const size_t elementCount = GetElementCount(shape);
    const size_t byteSize = elementCount * sizeof(uint16_t);
    CHECK_ACL(buffer.Allocate(byteSize));

    const std::vector<uint16_t> hostData(elementCount, fillValue);
    CHECK_ACL(aclrtMemcpy(buffer.data, buffer.size, hostData.data(), byteSize,
                         ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    tensor.value = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0,
                                   ACL_FORMAT_ND, shape.data(), shape.size(), buffer.data);
    if (tensor.value == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\\n");
        return 1;
    }
    return ACL_SUCCESS;
}

}  // namespace

int main()
{
    // 1. 初始化 ACL、device 和 stream。deviceId 按实际环境设置。
    AclRuntime runtime;
    CHECK_ACL(runtime.Init(/*deviceId=*/0));

    // 2. 构造定长场景输入和输出；示例数据类型为 FP16。
    constexpr int64_t B = 1;
    constexpr int64_t H_k = 2;
    constexpr int64_t H_v = 4;
    constexpr int64_t T = 128;
    constexpr int64_t K = 128;
    constexpr int64_t V = 128;
    constexpr int64_t chunkSize = 64;
    const double scale = 1.0 / std::sqrt(static_cast<double>(K));

    const std::vector<int64_t> qShape = {B, H_k, T, K};
    const std::vector<int64_t> kShape = {B, H_k, T, K};
    const std::vector<int64_t> dOShape = {B, H_v, T, V};
    const std::vector<int64_t> gShape = {B, H_v, T};
    const std::vector<int64_t> outShape = {B, H_v, T, V};

    DeviceBuffer qBuffer;
    DeviceBuffer kBuffer;
    DeviceBuffer dOBuffer;
    DeviceBuffer gBuffer;
    DeviceBuffer outBuffer;
    DeviceBuffer workspaceBuffer;
    TensorHandle q;
    TensorHandle k;
    TensorHandle dO;
    TensorHandle g;
    TensorHandle out;

    // 0x3C00 是 FP16 的 1.0，0x0000 是 FP16 的 0.0。
    CHECK_ACL(CreateFp16Tensor(qShape, 0x3C00, qBuffer, q));
    CHECK_ACL(CreateFp16Tensor(kShape, 0x3C00, kBuffer, k));
    CHECK_ACL(CreateFp16Tensor(dOShape, 0x3C00, dOBuffer, dO));
    CHECK_ACL(CreateFp16Tensor(gShape, 0x0000, gBuffer, g));
    CHECK_ACL(CreateFp16Tensor(outShape, 0x0000, outBuffer, out));

    // 3. 第一段接口查询 workspace；定长场景的四个可选输入均传空。
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnChunkBwdDvLocalGetWorkspaceSize(
        q.value,
        k.value,
        dO.value,
        g.value,
        /*gGammaOptional=*/nullptr,
        /*aOptional=*/nullptr,
        /*cuSeqlensOptional=*/nullptr,
        /*chunkIndicesOptional=*/nullptr,
        scale,
        chunkSize,
        out.value,
        &workspaceSize,
        &executor));

    // 4. 按第一段接口返回值分配 workspace，并调用第二段执行接口。
    CHECK_ACL(workspaceBuffer.Allocate(static_cast<size_t>(workspaceSize)));
    CHECK_ACL(aclnnChunkBwdDvLocal(workspaceBuffer.data, workspaceSize, executor,
                                   runtime.GetStream()));

    // 5. 同步 stream 后回拷输出。实际算子文档可在此补充结果校验。
    CHECK_ACL(aclrtSynchronizeStream(runtime.GetStream()));
    std::vector<uint16_t> hostOutput(GetElementCount(outShape));
    CHECK_ACL(aclrtMemcpy(hostOutput.data(), hostOutput.size() * sizeof(uint16_t),
                         outBuffer.data, outBuffer.size, ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf("chunk_bwd_dv_local finished, first FP16 raw value: 0x%04x\\n",
                static_cast<unsigned int>(hostOutput.front()));

    // Tensor、device buffer、workspace、stream、device 和 ACL 由局部对象析构释放。
    return ACL_SUCCESS;
}
```

> **模板填写要求：** 每个算子的 API 文档都必须保留一份与实际接口签名一致的完整 aclnn 最小示例，覆盖 ACL 初始化、stream 创建、tensor 构造、`GetWorkspaceSize`、workspace 分配、执行、同步、输出回拷和资源释放。不得只引用仓内其他文件，也不得用伪代码或文字步骤替代。以上代码以 `chunk_bwd_dv_local` 定长场景为例；填写其他算子时，应替换头文件、shape、dtype、属性、可选输入和两段式接口名称，并从参数表“取值范围”和“已知限制”选择合法取值。

## 4. `fla_npu.ops.ascendc` API（仅 Ascend C 算子）

Triton 算子删除本章节。

### 4.1 接口签名

```python
def op_name(input, *, attr=None):
    ...
```

说明参数、默认值、可选输入、输出形式、stream 和异步保活约定。

> **接口示例：** `chunk_bwd_dv_local` 的公开签名应与 schema 对齐：`chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None)`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import op_name

# 构造最小有效输入。
input_tensor = torch.empty(<shape>, device="npu", dtype=torch.float16)
output = op_name(input_tensor)
torch.npu.synchronize()

# 添加必要的 shape、dtype 或精度检查。
```

> **调用示例：以 `chunk_bwd_dv_local` 为例：**
>
> ```python
> import math
> import torch
> from fla_npu.ops.ascendc import chunk_bwd_dv_local
>
> def run_example(B, H_k, H_v, T, K, V, chunk_size):
>     q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
>     k = torch.randn_like(q)
>     d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
>     g = torch.randn(B, H_v, T, device="npu", dtype=torch.float32)
>
>     d_v = chunk_bwd_dv_local(
>         q, k, d_o, g,
>         scale=1.0 / math.sqrt(K),
>         chunk_size=chunk_size,
>         g_gamma=None,
>         A=None,
>         cu_seqlens=None,
>         chunk_indices=None,
>     )
>     assert d_v.shape == (B, H_v, T, V)
>     return d_v
> ```

## 5. `fla_npu.ops.triton` API（仅 Triton 算子）

Ascend C 算子删除本章节。旧 Triton 实现如仅作为 Ascend C 替换前的性能基线，应写入设计文档或性能报告，不作为该算子的公开 API。

### 5.1 接口签名

```python
def op_name(input, *, attr=None):
    ...
```

### 5.2 调用示例

```python
import torch
from fla_npu.ops.triton import op_name

input_tensor = torch.empty(<shape>, device="npu", dtype=torch.float16)
output = op_name(input_tensor)
torch.npu.synchronize()
```

> **提示：** `chunk_bwd_dv_local` 已选择 Ascend C 实现，因此其正式 API 文档应删除本章节。若某算子选择 Triton 实现，则保留本章节，并用 `fla_npu.ops.triton.<op_name>` 覆盖该算子的主调用示例。

## 6. Ascend C `<<<>>>` 直调（仅 Ascend C 算子）

Triton 算子删除本章节。

说明直调参数顺序、tiling data、block dim、workspace、stream、平台特化要求，以及 fixed/varlen 时可选参数的空指针处理。

### 6.1 调用示例

```cpp
// 填写 tiling data、block dim、workspace、stream 和算子参数准备过程。
<op><<<blockDim, l2ctrl, stream>>>(
    input,
    output,
    workspace,
    tiling);

// 同步 stream，检查返回码并验证结果。
```

> **调用示例说明：** `chunk_bwd_dv_local` 直调时应展示一个受支持的模板实例，依次传入 `q/k/d_o/g`、四个可选输入、`d_v`、workspace 和 tiling。模板参数涉及的固定维度和属性枚举应分别与“已知限制”和属性表“取值范围”保持一致；`blockDim` 和 tiling data 必须来自对应 host tiling 结果，不能写成无依据常量。

## 7. `torch.ops.npu` API（可选）

未实现该入口时删除本章节。

### 7.1 接口签名

```text
torch.ops.npu.<op_name>(...)
```

### 7.2 调用示例

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()
output = torch.ops.npu.<op_name>(...)
```

> **调用示例：以 `chunk_bwd_dv_local` 为例：**
>
> ```python
> import fla_npu
> import torch
>
> fla_npu.load_legacy_torch_ops()
> d_v = torch.ops.npu.npu_chunk_bwd_dv_local(
>     q, k, d_o, g,
>     scale=scale,
>     chunk_size=chunk_size,
>     g_gamma=None,
>     A=None,
>     cu_seqlens=None,
>     chunk_indices=None,
> )
> ```

## 8. 平台支持

| 平台 | SOC | 支持情况 | 限制 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 支持 | `<limitation>` |
| A3 | `ascend910_93` | 支持 | `<limitation>` |
| A5 | `ascend950` | 支持 | `<limitation>` |

> **填写示例：** `chunk_bwd_dv_local` 应分别说明 A2/A3 的公共 kernel 路径和 A5 `arch35/` 特化路径，并给出各平台是否支持 fixed/varlen、全部受支持 `V` 和 FP32 gate。

## 9. 已知限制

- `<fixed dimension or cross-parameter limitation>`
- `<unsupported optional input or mode>`

> **填写示例：** `chunk_bwd_dv_local` 在这里集中说明 `K` 仅支持 128、`V` 仅支持 128/256、varlen 仅支持 `B=1`，以及 `g_gamma/A` 当前必须为空。`chunk_size` 的合法值 `{64, 128}` 已写入属性表“取值范围”；只有平台差异或跨参数组合约束需要在本节补充。输入输出表中的 Shape 仍只写符号变量。

## 10. 异常与返回码

| 条件 | aclnn 返回码/异常 | 说明 |
| --- | --- | --- |
| `<invalid condition>` | `<return code>` | `<message and handling>` |

> **填写示例：** `chunk_bwd_dv_local` 应为“已知限制”中的每项约束列出对应返回码和错误信息，并补充 `H_v % H_k != 0`、varlen 两个索引只传一个等组合错误。

## 11. 文档自检

- [ ] API 签名与 `aclnn_*.h`、schema 和 Python 导出一致，`<<<>>>` 参数顺序与实现一致。
- [ ] 所有必选、可选、默认值、shape、dtype、format 和平台约束均有说明。
- [ ] 枚举或离散属性的全部合法值已在“取值范围”列完整列出，未只写入“已知限制”。
- [ ] 已按实现类型保留 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton`，未同时把两者声明为主入口。
- [ ] Ascend C 算子的 aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 示例完整可执行；Triton 算子的 `fla_npu.ops.triton` 示例完整可执行。
- [ ] 实现 `torch.ops.npu` 时已提供显式加载示例；未实现时已删除对应章节。
- [ ] 错误码与代码拦截、日志文本和负向测试一致。
