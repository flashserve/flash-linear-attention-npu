# SolveTri API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.solve_tri` | 支持 |
| aclnn | `aclnnSolveTriGetWorkspaceSize` / `aclnnSolveTri` | 支持 |
| Ascend C `<<<>>>` | `solve_tri<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_solve_tri` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,H_v,T,C]、[B,T,H_v,C] 或 [T,H_v,C]` | FP16/BF16 | BHTD/BSND/TND | 严格下三角 A 的行存储 |
| `cu_seqlens` | TND 必选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | TND 必选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `x_out` | `与 x 相同` | 与 x 相同 | (I+A) 的 chunk-wise 逆 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `layout` | str | `bsnd` | 仅 bhtd/bsnd/tnd，小写 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnSolveTriGetWorkspaceSize(/* 参数见本页公共参数表 */, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSolveTri(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnSolveTriGetWorkspaceSize(
    x, nullptr, nullptr, "bsnd", out, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnSolveTri(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/solve_tri/routes/test_aclnn_solve_tri.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout='bsnd')
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import solve_tri

B, T, H, C = 1, 128, 4, 64
x = torch.randn(B, T, H, C, device="npu", dtype=torch.float16)
row = torch.arange(C, device="npu").view(1, 1, 1, C)
pos = torch.arange(T, device="npu").view(1, T, 1, 1) % C
x = torch.where(row < pos, x * 0.01, torch.zeros_like(x))
y = solve_tri(x, layout="bsnd")
torch.npu.synchronize()
assert y.shape == x.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
solve_tri<<<blockDim, nullptr, stream>>>(x, cu_seqlens, chunk_indices, x_out, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/solve_tri/routes/test_direct_solve_tri.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, T, H, C = 1, 128, 4, 64
x = torch.randn(B, T, H, C, device="npu", dtype=torch.float16)
row = torch.arange(C, device="npu").view(1, 1, 1, C)
pos = torch.arange(T, device="npu").view(1, T, 1, 1) % C
x = torch.where(row < pos, x * 0.01, torch.zeros_like(x))
y = torch.ops.npu.npu_solve_tri(x, layout="bsnd")
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

- 矩阵阶/最后一维 C 支持 16/32/64/128。
- layout 仅支持小写 `bhtd`、`bsnd`、`tnd`；TND 必须提供两个 varlen 索引，dense layout 不接受 varlen 索引。
- 输入必须表示严格下三角 A；对角线由算子加单位阵。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| x/xOut/layout、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR |
| layout 不是小写 bhtd/bsnd/tnd，或 layout 与 rank 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| TND 缺少任一 varlen 索引，或 dense layout 携带 varlen 索引 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| C 非 16/32/64/128、xOut shape/dtype 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/solve_tri.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、dense BHTD/BSND 与 varlen TND，支持尾块 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
