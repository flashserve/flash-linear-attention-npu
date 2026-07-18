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
// 填写完整的最小可执行示例：初始化 ACL、创建 stream、构造 tensor、
// 调用 GetWorkspaceSize、分配 workspace、执行算子、同步并释放资源。
```

> **调用示例说明：以 `chunk_bwd_dv_local` 定长场景为例：** 使用 `q/k=[B,H_k,T,K]`、`dO/out=[B,H_v,T,V]`、`g=[B,H_v,T]`，从“已知限制”选择合法的 `K`、`V`，并从属性表“取值范围”选择合法的 `chunkSize`；`gGammaOptional`、`aOptional`、`cuSeqlensOptional` 和 `chunkIndicesOptional` 传空。先调用 `aclnnChunkBwdDvLocalGetWorkspaceSize`，按返回大小分配 workspace，再调用 `aclnnChunkBwdDvLocal` 并同步 stream。

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
