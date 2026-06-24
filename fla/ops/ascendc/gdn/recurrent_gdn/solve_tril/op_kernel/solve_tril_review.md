# solve_tril 编译问题修复报告

编译命令: `bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu`

> **注意**：用户原始命令为 `--vender_name=fla_npu` (typo)，已修正为 `--vendor_name=fla_npu`。

---

## 问题 1：`gawk` 未安装（环境问题）

**错误信息**:
```
build.sh: line 1871: gawk: command not found
```

**原因**: 构建脚本末尾使用 `gawk` 添加时间戳 (`| gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'`)，但系统中只有 `mawk`。通过测试确认 `mawk` 也支持 `strftime`：

```
$ echo "test" | mawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
[2026-06-24 04:09:18] test
```

**解决方案**: 在 CANN bin 目录创建软链接：
```bash
ln -sf /usr/bin/mawk /home/w00933206/Ascend/cann-9.0.0.beta2/cann-9.0.0-beta.2/bin/gawk
```

---

## 问题 2：指针成员访问错误（`solve_tril.h:82-88`）

**错误信息**:
```
error: member reference type 'const SolveTrilTilingData *' is a pointer;
       did you mean to use '->'?
```

**原因**: `Init()` 方法的参数 `tilingData` 是指针类型 `const SolveTrilTilingData *`，但成员访问使用了 `.` 运算符而非 `->`。

**修复**（7 处 `batchSize / seqLength / numHead / chunkSize / chunkNumInSeq / chunkNumTotal / mode`）:
```cpp
// Before:
batch_size = tilingData.batchSize;

// After:
batch_size = tilingData->batchSize;
```

---

## 问题 3：`Duplicate` 模板类型冲突（`solve_tril.h:136-138, 148`）

**错误信息**:
```
error: no matching function for call to 'Duplicate'
note: candidate template ignored: deduced conflicting types for parameter 'T' ('float' vs. 'half')
```

**原因**: `SolveTrilKernel<InDtype, OutDtype>` 中 `InDtype` 可能实例化为 `float` (如 `DTYPE_X=float`)。但代码中硬编码了 `half(0)` 作为标量值。`Duplicate` 的模板参数 `T` 需同时从 `LocalTensor<T>` (推导为 `float`) 和标量参数 (推导为 `half`) 推导，导致类型冲突。

查看 AscendC API (`kernel_operator_vec_duplicate_intf.h`):
```cpp
// Level 2 - 简单广播:
template <typename T>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count);

// Level 0 - 带 mask:
template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride);
```

**附带问题**: Level 2 的 `count` 参数类型是 `int32_t`，而 `chunkElems` 是 `uint64_t`，也需显式转换。

**修复**:
```cpp
// Before:
Duplicate(ub_I, half(0), chunkElems);
Duplicate(ub_Zero, half(0), chunkElems);
Duplicate(ub_A, half(0), chunkElems);
Duplicate(ub_I[UB_DIAG_I_OFF], half(1.0f), diagMask, 1, 1, 1);

// After — 使用模板参数替代硬编码类型:
Duplicate(ub_I, (InDtype)0, (int32_t)chunkElems);
Duplicate(ub_Zero, (InDtype)0, (int32_t)chunkElems);
Duplicate(ub_A, (InDtype)0, (int32_t)chunkElems);
Duplicate(ub_I[UB_DIAG_I_OFF], (InDtype)1.0f, diagMask, 1, 1, 1);
```

---

## 问题 4：`Sub` 缺少 `count` 参数（`solve_tril.h:216`）

**错误信息**:
```
error: no matching function for call to 'Sub'
```

**原因**: AscendC 的 `Sub` API (Level 2) 需要 4 个参数 `Sub(dst, src0, src1, count)`，但代码只传了 3 个。

查看 AscendC API (`kernel_operator_vec_binary_intf.h`):
```cpp
template <typename T>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);
```

**修复**:
```cpp
// Before:
AscendC::Sub(ub_A, ub_I, ub_A);

// After:
AscendC::Sub(ub_A, ub_I, ub_A, (int32_t)(chunk_size_actual * chunk_size_actual));
```

---

## 问题 5：Kernel 入口函数结构错误（`solve_tril.cpp`）

**错误信息**:
```
error: no matching function for call to 'solve_tril_1_tilingkey'
error: no matching function for call to 'solve_tril_0_tilingkey'
```

**原因分析**:

原始 kernel 入口使用了 `extern "C"` + `TILING_KEY_IS(1)` + `KERNEL_TASK_TYPE(1, ...)` 模式，存在三个问题：

1. **`extern "C"` 与 tiling key 机制冲突**: `extern "C"` 抑制了 C++ name mangling，而 `KERNEL_TASK_TYPE(1, ...)` 需要正确的符号名来生成 tiling key 变体函数（如 `solve_tril_1_tilingkey_aic`、`solve_tril_1_tilingkey_aiv` 等）。

2. **`GET_TILING_DATA` 在 `TILING_KEY_IS` 块外调用**: 导致 tiling key 0 编译路径无法正确注册 tiling data 类型，产生 `solve_tril_0_tilingkey` 错误。

3. **缺少 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 分发**: `KERNEL_TYPE_MIX_AIC_1_2` 模式下，AIC (Cube) 和 AIV (Vector) 核分别执行，需通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏分发。

**根本原因定位**: 通过分析 AscendC 源码 (`kernel_utils_macros.h`):
```cpp
#define KERNEL_TASK_TYPE(key, value)     ENABLE_FEATURE_FOR_COMPILE(key, value)
#define KERNEL_TASK_TYPE_DEFAULT(value)  ENABLE_FEATURE_FOR_COMPILE(default, value)
#define TILING_KEY_IS(k)  (TILING_KEY_VAR == (k))
```

以及查看成功案例 `chunk_bwd_dqkwg.cpp` 的完整结构，确认正确链路为：
1. 移除 `extern "C"`，恢复 C++ name mangling
2. `GET_TILING_DATA` 放到 `TILING_KEY_IS(1)` 块内部（在 `KERNEL_TASK_TYPE` 之后）
3. 添加 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 核类型分发
4. `AscendCUtils::SetOverflow(1)` 必须在 kernel 开头调用

**解决方案**: 参考同项目中成功编译的 `chunk_bwd_dqkwg.cpp` + `prepare_wy_repr_bwd_da.cpp`（均使用 `TILING_KEY_IS`）:

```cpp
// After（修复后）:
__global__ __aicore__ void solve_tril(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendCUtils::SetOverflow(1);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);
        if ASCEND_IS_AIC {
            SolveTrilKernel<DTYPE_X, DTYPE_X> op;
            op.Init(x, cu_seqlens, chunk_indices, out, workspace, &tilingData);
            op.Process();
        }
        if ASCEND_IS_AIV {
            SolveTrilKernel<DTYPE_X, DTYPE_X> op;
            op.Init(x, cu_seqlens, chunk_indices, out, workspace, &tilingData);
            op.Process();
        }
    }
    return;
}
```

**`Init` 方法签名**: 使用 `const SolveTrilTilingData *tilingData` 指针。`SolveTrilTilingData` 类型由 AscendC 构建系统从 host 端 `REGISTER_TILING_DATA_CLASS(SolveTril, SolveTrilTilingData)` 自动生成的扁平结构体，无需手写。

**扩展方式**: 后续可通过添加 `TILING_KEY_IS(2) { KERNEL_TASK_TYPE(2, ...); ... }` 等分支来扩展不同模板，tiling key 值在 host 端 `solve_tril_tiling.cpp` 中通过 `context->SetTilingKey(N)` 设置。

**关键对比 — `TILING_KEY_IS` vs `KERNEL_TASK_TYPE_DEFAULT`**:
| 方式 | 适合场景 | tiling key 管理 |
|------|----------|----------------|
| `TILING_KEY_IS(1) { KERNEL_TASK_TYPE(1, ...) }` | 需要根据 tiling key 分发不同模板实现 | 由 host 端 `SetTilingKey(N)` 控制 |
| `KERNEL_TASK_TYPE_DEFAULT(...)` | 单一路径，不需要 tiling key 区分 | 无需管理 |

---

## 问题 6：Kernel 端 tiling data 结构体冲突（`solve_tril_tiling_data.h`）

**错误信息**:
```
error: redefinition of 'SolveTrilTilingData'
```

**原因**: 手写的 kernel 端 `struct SolveTrilTilingData` (在 `op_kernel/solve_tril_tiling_data.h`) 与 AscendC 构建系统从 host 端 `REGISTER_TILING_DATA_CLASS(SolveTril, SolveTrilTilingData)` 自动生成的结构体同名冲突。

**根本原因**: AscendC 编译流程会从 host 的 `BEGIN_TILING_DATA_DEF(SolveTrilTilingData)` 定义自动生成 kernel 端可用的扁平结构体（public 成员可直接访问）。当 kernel 源文件目录下存在同名手写结构体时，会产生 `redefinition` 错误。参考 `chunk_fwd_o` — 它没有手写 kernel 端 tiling data 头文件，完全依赖构建系统自动生成。

**解决方案**: 删除手写的 `solve_tril_tiling_data.h` (该文件在 git status 中已标记为删除 `D`)。

---

## 问题 7：Operator 定义缺少可选输入（`solve_tril_def.cpp`）

**错误信息**:
```
The dtype size of input[0] of op SolveTril is 0.
Error: ops prepare build failed.
```

**原因分析**: Op 定义 (`solve_tril_def.cpp`) 只声明了 1 个输入 `x` 和 1 个输出 `y`，但 kernel 函数需要 6 个 `GM_ADDR` 参数：`x, cu_seqlens, chunk_indices, out, workspace, tiling`。生成的 kernel wrapper 只传 4 个参数 (`x, out, workspace, tiling`)，导致参数不匹配。

此外还需注意：
- `DataType` 数组大小必须跨所有 input/output 一致（每个位置对应一个编译变体）。本 op 有 2 个变体 (FLOAT16 / FLOAT)。
- `ParamType(OPTIONAL)` 用于可选输入。
- 不要使用 `ValueDepend(OPTIONAL)`，因为它会触发框架添加 INT64 等额外编译变体（与 kernel 实际使用的 `int32_t` 不匹配）。

**修复**: 参考同项目中 `chunk_bwd_dv_local_def.cpp` 的 OPTIONAL 输入模式:

```cpp
// Before:
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})    // 2 variants
    ...;
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    ...;

// After — 添加可选输入，DataType 数组大小保持一致：
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    ...;

this->Input("cu_seqlens")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32, ge::DT_INT32})       // 2 variants, 类型一致
    ...;

this->Input("chunk_indices")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_INT32, ge::DT_INT32})       // 2 variants, 类型一致
    ...;

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    ...;
```

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `op_kernel/solve_tril.h` | (1) `tilingData.X` → `tilingData->X` (2) `half(0)` → `(InDtype)0` (3) `Sub` 加 `count` 参数 (4) `Init` 保持 `const SolveTrilTilingData *tilingData` 签名 |
| `op_kernel/solve_tril.cpp` | 移除 `extern "C"`；将 `GET_TILING_DATA` 移入 `TILING_KEY_IS(1)` 块内 (`KERNEL_TASK_TYPE` 之后)；添加 `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分发 |
| `op_kernel/solve_tril_tiling_data.h` | **删除** — 与 AscendC 自动生成的结构体冲突 |
| `op_host/solve_tril_def.cpp` | 添加 `cu_seqlens` / `chunk_indices` 为 OPTIONAL 输入 (含正确的 `DataType` 数组)；`DynamicCompileStaticFlag(false)` |

---

## 最终结果

```bash
$ bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu -j4
...
[100%] Built target protoc
Self-extractable archive "fla-npu-fla_npu_linux-aarch64.run" successfully created.
CPack: - package: /home/w00933206/ops/flash-linear-attention-npu/build/fla-npu-fla_npu_linux-aarch64.run generated.
```

编译成功，0 错误，产物 `fla-npu-fla_npu_linux-aarch64.run` 已生成于 `build/` 和 `build_out/` 目录。
