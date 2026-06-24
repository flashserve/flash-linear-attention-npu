# chunk_bwd_dv_local regbase改写报告

本文档记录 `chunk_bwd_dv_local` 算子在 A5/arch35 分支上的 regbase 改写方式。整体目标是与 `prepare_wy_repr_bwd_da` 的改写策略保持一致：原有通用实现不动，在 kernel 侧和 host tiling 侧新增 arch35 分支，A5 编译时走 regbase 版本。

## 1. 改写范围

本次改写涉及以下位置：

- `op_kernel/chunk_bwd_dv_local.cpp`
  - 根据 `__CCE_AICORE__ == 310` 选择 arch35 头文件。
  - 非 A5 平台仍走原有 kernel 实现。
- `op_kernel/arch35/`
  - 新增 A5 分支目录。
  - `chunk_bwd_dv_local_struct.h`、`chunk_bwd_dv_local_common.h`、`chunk_bwd_dv_local_cube.h` 以 wrapper 方式复用原实现。
  - `chunk_bwd_dv_local_vector.h` 放置 regbase 改写后的 vector 侧实现。
- `op_host/chunk_bwd_dv_local_tiling.cpp`
  - 根据 `PlatformAscendC::GetCurNpuArch()` 判断 `DAV_3510`。
  - A5 平台转发到 `ChunkBwdDvLocalTilingA5`。
- `op_host/op_tiling/arch35/`
  - 新增 `chunk_bwd_dv_local_tiling_a5.h/.cpp`。
  - 复用原 tiling data 和 tiling processor，保持 tiling key、workspace 和 blockDim 计算逻辑与原算子一致。

## 2. Kernel侧改写思路

原 vector 侧使用 `TBuf` 申请多块中间 UB：

- `gFp32TBuf`
- `gFactorTBuf`
- `brcbTBuf`
- `maskTBuf`
- `zeroFp32TBuf`
- `kqFp32TBuf`

A5 regbase 改写后，中间计算不再落 UB buffer，而是放在 `RegTensor` 中完成。vector 侧只保留必要的输入输出队列：

- `gTQueIn`：读取当前 chunk 的完整 `g`
- `gHalfTQueIn`：读取当前 vector task 负责的行对应的 `g`
- `kqTQueIn`：读取 cube 写入 workspace 的 KQ
- `kqTQueOut`：写回 gated KQ

这样可以减少 UB 中间 buffer 的占用，同时让核心逐元素计算进入 `__simd_vf__` 函数，由 regbase 指令完成。

## 3. Vector侧数据流

A5 vector 侧仍保持原有 cube/vector 协议：

1. cube 侧把 KQ 写入 workspace。
2. vector 侧按 head/group/chunk 维度等待 cube 完成信号。
3. vector 侧从 GM/workspace 拷贝：
   - 当前 chunk 的完整 `g`
   - 当前 vector task 对应行的 `g`
   - 当前 vector task 对应行的 KQ
4. `ProcessGatedKqComputerVF` 在寄存器中计算 gated KQ。
5. vector 侧把结果写回 workspace。
6. vector 侧设置同步 flag，通知 cube 继续后续计算。

计算公式保持与原实现一致：

```text
gFactor = exp(min(gAll[col] - gRow[row], 0)) * scale
kqOut[row, col] = kqIn[row, col] * gFactor
```

同时保留两个置零条件：

- 下三角无效区：`col < row`
- tail 无效区：`col >= validColSize`

## 4. regbase VF实现要点

`ProcessGatedKqComputerVF` 采用模板参数承载 chunk size：

```cpp
template <uint16_t N_SIZE>
__simd_vf__ inline void ProcessGatedKqComputerVF(...);
```

调用处根据 tiling 中的 `chunkSize` 分发：

```cpp
if (strategy.chunkSize == 64) {
    ProcessGatedKqComputerVF<64>(...);
} else {
    ProcessGatedKqComputerVF<128>(...);
}
```

这样做有两个原因：

- `chunk_bwd_dv_local` 当前只支持 `chunkSize = 64/128`。
- A5 后端对 `LoadIn/StoreAlign` 的 UB 地址表达式较敏感，地址步长使用运行时变量时容易触发 `VAG's non-zero args must be hoisted`。模板化后，行步长可以使用编译期常量。

最终实现中采用如下模式：

```cpp
constexpr uint32_t ONE_ELE_NUM = N_SIZE;
uint32_t qkvMaskLen = N_SIZE;
uint32_t gMaskLen = N_SIZE;
uint32_t storeMaskLen = N_SIZE;

MaskReg maskQkv = UpdateMask<QKVT>(qkvMaskLen);
MaskReg maskG = UpdateMask<GT>(gMaskLen);
MaskReg maskStore = UpdateMask<QKVT>(storeMaskLen);

LoadIn<QKVT, false>(kqInReg, kqIn + ONE_ELE_NUM * rowIdx);
StoreAlign((__ubuf__ QKVT *&)kqOut + ONE_ELE_NUM * rowIdx, kqOutReg, maskStore);
```

这里有一个容易踩坑的点：`UpdateMask<T>` 的参数需要是可写的 `uint32_t&`，不能直接传 `constexpr` 或 const 临时量。因此 mask 长度使用普通 `uint32_t` 变量，UB 地址步长使用 `constexpr`。

## 5. V核流水优化

初版 regbase VF 的行循环是逐行串行执行：

```text
for i in [0, m):
    Load row i
    Execute row i
    Store row i
```

这种写法在每一行内部形成 `Load -> Execute -> Store` 的 Z 字形路径，当前行的计算必须等当前行 load 完成，下一行 load 又要等当前行 store 后才开始，V 核流水中间容易出现 bubble。

本次优化将行循环改为 W 形流水：

```text
Load row 0
for i in [0, m - 1):
    Execute row i
    Load row i + 1
    Store row i
Execute row m - 1
Store row m - 1
```

核心变化是提前把第 0 行读入 `RegTensor`，主循环只处理前 `m - 1` 行。每轮循环中先计算当前行，再加载下一行，最后 store 当前行。最后一行已经在主循环最后一轮被加载到寄存器，因此尾段只需要执行计算和 store。

实现上没有在主循环内增加 `if (rowIdx + 1 < mSize)` 这类判断，而是把逻辑拆成两个阶段：

- 主流水阶段：处理前 `m - 1` 行，并把下一行提前加载到寄存器。
- 尾行阶段：处理最后一行的计算和写回。

这样可以规避 VF 内分支判断带来的性能下降，同时压缩 load 和 execute 之间的等待 bubble。这里没有把尾行阶段拆成独立 C++ `__simd_vf__` 函数，是因为独立函数无法复用主流水阶段已经加载到寄存器中的最后一行；若物理拆成两个 VF 函数，最后一行通常需要重新从 UB load，一部分收益会被抵消。

对应代码结构为：

```cpp
LoadIn<QKVT, false>(kqInReg, kqIn);
LoadIn<GT, true>(gRowReg, gRowIn);

for (uint16_t rowIdx = 0; rowIdx < mSize - 1; ++rowIdx) {
    // Execute rowIdx
    ...
    LoadIn<QKVT, false>(kqInReg, kqIn + ONE_ELE_NUM * (rowIdx + 1));
    LoadIn<GT, true>(gRowReg, gRowIn + rowIdx + 1);
    StoreAlign((__ubuf__ QKVT *&)kqOut + ONE_ELE_NUM * rowIdx, kqOutReg, maskStore);
}

uint16_t rowIdx = mSize - 1;
// Execute and store the preloaded tail row.
...
StoreAlign((__ubuf__ QKVT *&)kqOut + ONE_ELE_NUM * rowIdx, kqOutReg, maskStore);
```

这类优化适用于按行独立计算、行间只存在顺序遍历关系、下一行数据可以提前加载且不会破坏当前行计算语义的 VF。迁移到其他算子时，需要重点确认：

- 当前行计算不依赖当前行 store 完成后的结果。
- 下一行 load 不会覆盖当前行后续计算还要使用的寄存器。
- tail 行必须单独收尾，避免主循环中加入动态判断。
- 如果最后一行已经被主循环提前加载进 Reg，尾段应尽量复用该 Reg，避免重复 Load。

## 6. Host侧改写思路

host 侧不能只依赖 kernel 编译宏，还需要在 tiling 入口显式分发 A5：

```cpp
const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
    ChunkBwdDvLocalTilingA5 chunkBwdDvLocalTilingA5;
    ...
    return ge::GRAPH_SUCCESS;
}
```

A5 tiling 类复用原有 `ChunkBwdDvLocalTilingData` 和 `ChunkBwdDvLocalTilingProcessor`，并保持以下契约不变：

- tiling key 仍由 strategy、Q dtype、G dtype、V size 等信息组成。
- blockDim 仍取 `min(chunkNumForT * b, coreNum)`。
- workspace 仍按原 head buffer 数和 chunk 矩阵大小申请。
- `SetScheduleMode(1)` 保持不变。

这样可以保证 A5 分支只替换 vector 执行方式，不改变算子对上层 graph、tiling key 和 workspace 协议的可见行为。

## 7. 经验总结

1. regbase 改写不只改 kernel 侧。
   - kernel 侧需要新增 arch35 include 分支。
   - host 侧也需要新增 arch35 tiling 分支，否则 A5 平台不会稳定进入对应实现。

2. 优先保持原算子的外部契约不变。
   - tiling key、workspace 大小、blockDim、同步 flag 和 workspace slot 布局尽量沿用原实现。
   - 这样可以把风险集中在 vector 计算实现内部。

3. regbase 适合替换中间 UB 计算。
   - 原来落在 `TBuf` 中的中间张量，可以改成 `RegTensor`。
   - GM 到 UB、UB 到 GM 的搬运仍放在 `__aicore__` 函数中处理。
   - `__simd_vf__` 函数只做寄存器计算。

4. A5 后端对地址表达式和 mask 参数有额外约束。
   - `UpdateMask<T>` 需要可写的 `uint32_t&`。
   - `LoadIn/StoreAlign` 的 UB 地址偏移尽量避免运行时步长。
   - 对固定枚举值的维度，例如 `chunkSize = 64/128`，优先使用模板分发，把步长固化成编译期常量。

5. 动态行号建议用寄存器递推。
   - 行循环中通过 `Adds(rowIdxReg, rowIdxReg, 1.0f, maskFull32)` 更新行号。
   - 避免在 VF 内反复构造与循环变量相关的动态 `Duplicate`。

6. mask 逻辑要逐项对齐原实现。
   - 下三角 mask、tail mask、dtype cast、scale 乘法都需要逐项保留。
   - regbase 改写后的结果应只改变实现路径，不改变数学语义。

7. V 核优化优先做流水重排。
   - 逐行 `Load -> Execute -> Store` 容易形成 Z 字形串行路径。
   - 可以先加载第 0 行，在主循环中执行当前行、预加载下一行、写回当前行。
   - 尾行单独处理，避免主循环内引入判断指令。

## 8. 验证结果

已执行 A5 kernel 编译验证，包含 W 形流水优化后的 arch35 vector 实现：

```bash
bash build.sh --opkernel --soc=ascend950 --ops=chunk_bwd_dv_local -p /usr/local/Ascend/cann-9.0.0-beta.2 -j1
```

编译通过，生成 `ChunkBwdDvLocal` 的 A5 kernel 目标。编译过程中 CANN/CATLASS 有若干 warning，但无 fatal error，最终 `ops_transformer_kernel` 构建成功。
