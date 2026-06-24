# prepare_wy_repr_bwd_da regbase改写梳理

本文档梳理 `prepare_wy_repr_bwd_da` 在 A5/arch35 分支中的 regbase 改写方式，覆盖 kernel 侧和 host tiling 侧的配套变化。代码范围如下：

- kernel 原实现：`op_kernel/prepare_wy_repr_bwd_da_*.h`
- kernel A5 实现：`op_kernel/arch35/prepare_wy_repr_bwd_da_*.h`
- host 原 tiling：`op_host/op_tiling/prepare_wy_repr_bwd_da_tiling.*`
- host A5 tiling：`op_host/op_tiling/arch35/prepare_wy_repr_bwd_da_tiling_a5.*`

## 1. 整体改写思路

原 vector 侧主要使用 AscendC 高阶向量 API：

- `TQue` 负责 GM 和 UB 之间搬入搬出。
- `TBuf<VECCALC>` 分配大量 fp32 中间缓存。
- `Cast`、`Brcb`、`Mul`、`Add`、`Sub`、`Select`、`Exp` 等 API 在 UB tensor 上完成计算。

A5 arch35 分支引入 regbase 后，整体分工变成：

- `__aicore__` 函数保留任务调度、GM/UB 搬运、队列管理和跨核同步。
- `__simd_vf__` 函数负责纯计算，输入输出是 `__ubuf__` 指针。
- 计算中间值尽量放在寄存器 `RegTensor` 中，不再为每一步 fp32 中间结果分配 UB `TBuf`。
- 原来依赖 UB mask/brcb tensor 的逻辑，改为寄存器广播、`Arange + Compare + Select` 或 `UpdateMask`。

因此 regbase 改写不是简单替换 vector API，而是重新划分了“搬运”和“计算”的边界；host 侧 tiling 也必须同步调整 UB 预算、tiling 数据结构和 cube/vector 协同参数。

## 2. A5 分支入口

### 2.1 host 侧按架构选择 tiling

`op_host/op_tiling/prepare_wy_repr_bwd_da_tiling.cpp` 中仍注册同一个 `Tiling4PrepareWyReprBwdDa`。进入 tiling 后先做通用输入维度检查，然后通过平台架构分流：

```cpp
if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
    PrepareWyReprBwdDaTilingA5 prepareWyReprBwdDaTilingA5;
    prepareWyReprBwdDaTilingA5.SetTiling(context);
    return ge::GRAPH_SUCCESS;
}
```

A5 专用 tiling 实现在 `op_host/op_tiling/arch35/prepare_wy_repr_bwd_da_tiling_a5.cpp`，这使原 tiling 和 A5 tiling 可以并存。

### 2.2 kernel 侧按编译宏包含 arch35 实现

`op_kernel/prepare_wy_repr_bwd_da.cpp` 中通过 `__CCE_AICORE__ == 310` 切到 arch35 文件：

```cpp
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/prepare_wy_repr_bwd_da_common.h"
#include "arch35/prepare_wy_repr_bwd_da_cube.h"
#include "arch35/prepare_wy_repr_bwd_da_vector.h"
#include "arch35/prepare_wy_repr_bwd_da_tiling_data_apt.h"
#else
#include "prepare_wy_repr_bwd_da_common.h"
#include "prepare_wy_repr_bwd_da_cube.h"
#include "prepare_wy_repr_bwd_da_vector.h"
#endif
```

A5 分支还显式注册了新的 tiling 数据结构：

```cpp
REGISTER_TILING_DEFAULT(PrepareWyReprBwdDaTilingDataA5);
GET_TILING_DATA(tilingData, tiling);
```

这要求 host 侧写入的 raw tiling 数据布局必须与 kernel 侧 `PrepareWyReprBwdDaTilingDataA5` 完全一致。

## 3. tiling 数据结构变化

原实现使用 `BEGIN_TILING_DATA_DEF(PrepareWyReprBwdDaTilingData)` 生成 tiling 类，字段包含：

- `B, HV, HK, T, K, V`
- `chunkSize, chunkNum`
- `rowNumKBetaG, rowNumVBeta, rowNumMDuDw, rowNumG`
- `isVariable`

A5 分支新增 `op_kernel/arch35/prepare_wy_repr_bwd_da_tiling_data_apt.h`，直接定义 packed struct：

```cpp
#pragma pack(push, 8)
struct PrepareWyReprBwdDaTilingDataA5 {
    int64_t B = 0;
    int64_t H = 0;
    int64_t T = 0;
    int64_t K = 0;
    int64_t V = 0;
    int64_t chunkSize = 0;
    int64_t chunkNum = 0;
    int64_t rowNumKBetaG = 0;
    int64_t rowNumVBeta = 0;
    int64_t rowNumMDuDw = 0;
    int64_t rowNumG = 0;
    int64_t isVariable = 0;
    int64_t gCVNum = 0;
};
#pragma pack(pop)
```

关键变化：

- `HV/HK` 合并为 `H`，A5 分支要求相关输入 head 维一致，kernel 侧不再做 GVA 的 `HV/HK` remap。
- 新增 `gCVNum`，用于最后一个 dA6 阶段的 cube/vector UB buffer 轮转深度。
- host 侧不再调用 `tiling.SaveToBuffer()`，而是直接 `memcpy_s` 这个 struct，并设置 `SetDataSize(sizeof(tiling_))`。

经验：如果后续算子只想做 regbase 性能改写，不应无意改变原有 shape 语义。本算子 A5 分支同时收敛了 `HV/HK` 逻辑；其他算子若仍需支持 GVA，需要把对应 head ratio 或 offset 逻辑继续保留到 A5 tiling/kernel 中。

## 4. host 侧 UB 预算如何配套改写

regbase 后，许多 fp32 中间 tensor 不再常驻 UB，而是在 `RegTensor` 中完成。host 侧 `rowNum*` 的计算必须重算，否则会过度保守或错误估算 UB。

### 4.1 原实现 UB 预算特点

原 `SetRowNum*` 会把以下内容都计入 UB：

- 输入/输出 TQue buffer。
- fp32 cast 缓存，如 `kFp32Buf`、`vFp32Buf`、`mduFp32Buf`、`gFactorTBuf`。
- broadcast 缓存，如 `betaFp32BrcbBuf`、`brcbTBuf`。
- mask 和 zero 缓存，如 `maskTBuf`、`zeroFp32TBuf`。

例如 `SetRowNumG` 原来要预算 `gFp32Buf`、`gAllFp32Buf`、`gFactorTBuf`、`brcbTBuf`、`dA6Fp32Buf`、`maskTBuf`、`zeroFp32TBuf` 等。

### 4.2 A5 regbase 预算方式

A5 tiling 中对应函数改名为：

- `SetRowNumKBetaGRegbase`
- `SetRowNumVBetaRegbase`
- `SetRowNumMDuDwRegbase`
- `SetRowNumGRegbase`

预算只保留搬运 buffer 和必要的输出 buffer：

| 阶段 | A5 侧主要 UB 预算 |
| --- | --- |
| `KBetaG` | `kInQue`、`betaInQue`、`gInQue`、`kBetaGOutQue` |
| `VBeta` | `vInQue`、`betaInQue`、`vBetaOutQue` |
| `MDuDw` | `mduInQue`、`mdwInQue`、`mduwOutQue` |
| `G` | `gIn`、`gAllIn`、dA6 CV buffer、dA out buffer |

以 `SetRowNumVBetaRegbase` 为例，原来的 `vFp32Buf + betaFp32Buf + betaFp32BrcbBuf` 都消失了：

```cpp
useUbSize += 2 * rowNum * V * sizeofKType;       // vInQue
useUbSize += 2 * rowNum * sizeofBetaType;        // betaInQue
useUbSize += 2 * rowNum * V * sizeofKType;       // vBetaOutQue
```

`SetRowNumGRegbase` 还根据剩余 UB 计算 `gCVNum`：

```cpp
uint64_t gCVNum = 2;
...
gCVNum += (ubSize - useUbSize) / (rowNum * chunkSize * sizeofKType);
gCVNum = std::min(gCVNum, MAX_CUBE_VEC_SYNC_NUM);
tiling_.rowNumG = rowNum;
tiling_.gCVNum = gCVNum;
```

这里 `rowNumG` 不只是 vector 每次处理的行数，也是 cube 侧每个 dA6 UB buffer 的行数；`gCVNum` 是 cube/vector 双方共享的 dA6 buffer 个数。

### 4.3 workspace 也要使用 A5 tiling 字段

原实现 workspace size 使用 `HV`：

```cpp
2 * B * HV * T * (chunkSize + maxKV)
```

A5 分支使用 `H`：

```cpp
2 * tiling_.B * tiling_.H * tiling_.T * (tiling_.chunkSize + maxKV)
```

kernel vector/cube 侧的 workspace offset 也同步改成：

```cpp
workspace + B * H * T * BT
```

经验：tiling 结构字段变化后，host workspace、kernel GlobalTensor offset、cube 参数和 vector 参数都要一起对齐，不能只改计算函数。

## 5. kernel common/cube 侧配套修改

虽然 regbase 的主体在 vector 侧，但本算子的最后阶段需要 cube/vector 直接协同，因此 cube 侧也做了重要修改。

### 5.1 common 常量和 chunk offset

`op_kernel/arch35/prepare_wy_repr_bwd_da_common.h` 新增：

- `FLAG_ID_MAX = 16`
- `MAX_CUBE_VEC_SYNC_NUM = 5`
- `UB_STAGES = 2`
- `SYNC_AIV_AIC_FLAG_BEGIN = 1`
- `SYNC_AIC_AIV_FLAG_BEGIN = 6`

`GetChunkOffset` 的 head 参数从 `HV` 改为 `H`，固定长度场景下 offset 使用 `bIdx * H * T`。

### 5.2 cube 架构和 dA6 输出方式变化

A5 cube 分支主要变化：

- `CATLASS_ARCH` 从 `2201` 改为 `3510`。
- `ArchTag` 从 `Arch::AtlasA2` 改为 `Arch::Ascend950`。
- matmul 实现切到 `Common::MmadPingpong` 和 `Common::BlockMmadTla`。
- 第四个 matmul `dA_6 = A.T @ dA_5` 不再直接写 GM，而是写到 UB list 中，再与 vector 侧同步消费。

原实现 dA6：

```cpp
auto tensorDA6 = tla::MakeTensor(gmDA6, params.layoutDA6, Arch::PositionGM{});
blockMmadDA6(tensorBlockDA5T, tensorBlockA, tensorBlockDA6, actualBlockShape);
Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
```

A5 实现：

```cpp
AscendC::LocalTensor<ElementDA6> ubDA6List[MAX_CUBE_VEC_SYNC_NUM];
...
blockMmadDA6(tensorBlockDA5T, tensorBlockA, tensorBlockDA6List,
             actualBlockShape, params.da6VecRow,
             beginSubBlockIdx,
             SYNC_AIV_AIC_FLAG_BEGIN,
             SYNC_AIC_AIV_FLAG_BEGIN,
             ubListId, 2, params.da6CVNum);
```

其中：

- `da6VecRow = tiling.rowNumG`
- `da6CVNum = tiling.gCVNum`
- cube 侧按 `da6VecRow * BT * sizeof(ElementDA6)` 切分 UB list。
- vector 侧等待 `SYNC_AIC_AIV_FLAG_BEGIN + cvListId` 后读取对应 dA6 UB buffer。
- vector 消费完成后设置 `SYNC_AIV_AIC_FLAG_BEGIN + cvListId`，让 cube 复用该 buffer。

经验：如果 regbase 改写同时把中间结果从 GM 改成 UB 直传，host tiling 必须给出双方一致的 buffer 行数和 buffer 个数；同步 flag 范围、list id 轮转和尾部 drain 都要成对设计。

## 6. vector 侧 regbase 改写模式

### 6.1 头文件和命名空间

A5 vector 文件新增：

```cpp
#include "prepare_wy_repr_bwd_da_tiling_data_apt.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC::MicroAPI;
```

类内新增大量 `__simd_vf__` 计算函数，例如：

- `ProcessKBetaGComputerVFOneLineOneCol`
- `ProcessKBetaGComputerVFMutiLineOneCol`
- `ProcessKBetaGComputerVFTwoCol`
- `ProcessVBetaComputerVFOneLineOneCol`
- `ProcessMDuDwComputerVFMutiLineOneCol`
- `ProcessGComputerVFMutiLineOneCol`

### 6.2 删除 VECCALC TBuf

原 vector 类中有大量 `TBuf<VECCALC>`：

```cpp
TBuf<VECCALC> vFp32Buf;
TBuf<VECCALC> kFp32Buf;
TBuf<VECCALC> betaFp32Buf;
TBuf<VECCALC> betaFp32BrcbBuf;
TBuf<VECCALC> mduFp32Buf;
...
TBuf<VECCALC> maskTBuf;
TBuf<VECCALC> zeroFp32TBuf;
```

A5 分支基本删除这些计算临时区，只保留输入输出队列：

```cpp
TQue<VECIN, 1> kInQue;
TQue<VECIN, 1> vInQue;
TQue<VECIN, 1> gInQue;
TQue<VECIN, 1> betaInQue;
TQue<VECIN, 1> mduInQue;
TQue<VECIN, 1> mdwInQue;

TQue<VECOUT, 1> kBetaGOutQue;
TQue<VECOUT, 1> vBetaOutQue;
TQue<VECOUT, 1> mduwOutQue;
```

`ProcessG` 更特殊：它不用 `TQue` 管 dA6，而是直接从 `Arch::Resource<Arch::Ascend950>::ubBuf` 分配 dA6 CV buffers 和双 stage 的 g/gAll/out buffers。

### 6.3 aicore 调度到 simd_vf 计算

每个阶段的 `__aicore__` 函数仍负责：

1. `InitBuffer`
2. `DataCopy` / `DataCopyPad`
3. `AllocTensor` / `EnQue` / `DeQue`
4. 取 UB 物理地址
5. 调用 `__simd_vf__` 计算函数
6. `DataCopy` 输出

典型调用方式：

```cpp
auto outAddr = reinterpret_cast<uint64_t>(tensorOut.GetPhyAddr());
auto inAddr = reinterpret_cast<uint64_t>(tensorIn.GetPhyAddr());
ProcessVBetaComputerVFOneLineOneCol(
    (__ubuf__ kType*)outAddr,
    (__ubuf__ kType*)inAddr,
    (__ubuf__ betaType*)betaInAddr,
    curRowNum, V);
```

经验：regbase 函数不要直接操作 `LocalTensor`，而是操作 `__ubuf__` 指针；aicore 层负责把队列 tensor 的物理地址传进去。

### 6.4 按列宽和行数选择计算分支

A5 通过 `VECTOR_REG_WIDTH / sizeof(kType)` 判断一行是否能放进一个 VF 宽度：

```cpp
uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
if (K <= eleKNumPerVf) {
    if (curRowNum == 1) {
        OneLineOneCol(...);
    } else {
        MutiLineOneCol(...);
    }
} else {
    TwoCol(...);
}
```

含义：

- `OneLineOneCol`：一行且列数不超过一个 VF。
- `MutiLineOneCol`：多行，每行不超过一个 VF，使用 `PRELOAD_NUM` 做两行流水。
- `TwoCol`：一行需要两个 VF 才能覆盖列方向，按一行两段处理。

在 `MDuDw` 和 `G` 阶段，列方向是 `BT`。本算子 `chunkSize` 为 64 或 128，通常一行可以由一个 VF 覆盖，因此只实现 one-col 分支。

### 6.5 dtype 与 fp32 计算

原实现是先把 UB tensor cast 到 fp32 buffer，再计算，最后 cast 回 `kType`。regbase 改为寄存器级：

- `RegTensor<kType>` / `RegTensor<betaType>` 承载输入。
- `RegTensor<float>` 承载计算中间结果。
- `CastHalf2Float<kType>` 把半精度向量拆成 fp32 zero/one 两个寄存器。
- `HalfOrFloat2Float<betaType>` 处理 `betaType` 可能为 half 或 float 的情况。
- `CastFloat2Half<kType>` 写回输出类型。

常见模式：

```cpp
RegTensor<kType> xReg;
RegTensor<float> xFp32ZeroReg, xFp32OneReg;
MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();

LoadIn<kType, false>(xReg, xPtr);
CastHalf2Float<kType>(xFp32ZeroReg, xFp32OneReg, xReg, maskFull16);
...
CastFloat2Half<kType>(outReg, outFp32ZeroReg, outFp32OneReg, maskFull32);
StoreAlign(outPtr, outReg, maskStore);
```

### 6.6 broadcast 和 mask 的替代方式

原实现中 `Brcb` 用于把 `beta/g` 按行 broadcast 到矩阵列方向。regbase 中常见替代方式是：

- 用 `LoadIn<betaType, true>` 读取标量或短向量。
- 转成 fp32 后，在 `MulFloatTwoReg` 中同一个标量寄存器同时作用到 zero/one 两个 fp32 寄存器。

原实现中 `maskTBuf` 预先生成三角 mask，再用 `Select`。regbase 中改为：

```cpp
Arange(colIdxReg, 0);
CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);
Duplicate(rowIdxReg, static_cast<float>(startRow));
CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect,
                                  colIdxFP32ZeroReg, colIdxFP32OneReg,
                                  rowIdxReg, rowIdxReg, maskFull32);
SelectTwoReg(resultZero, resultOne, zeroReg, zeroReg,
             resultZero, resultOne, maskZeroSelect, maskOneSelect);
```

这样三角 mask 不再占 UB，行号通过 `rowIdxReg` 每处理一行递增。

## 7. 四个 vector 阶段的具体改写

### 7.1 ProcessKBetaG

计算语义：

```text
kBetaG = k * beta * exp(g)
```

原流程：

1. `g -> fp32`
2. `Exp(g)`
3. `beta -> fp32`
4. `k -> fp32`
5. `beta * exp(g)`
6. `Brcb` 到列方向
7. `k * broadcast(beta * exp(g))`
8. cast 回 `kType`

A5 流程：

- aicore 层只搬入 `k/beta/g`，输出 `kBetaGOut`。
- simd_vf 层直接：
  - `LoadIn` beta、g、k
  - `HalfOrFloat2Float` beta/g
  - `Exp(g)`
  - `Mul(beta, exp(g))`
  - `CastHalf2Float(k)`
  - `MulFloatTwoReg(k, betaG)`
  - `CastFloat2Half` 并 `StoreAlign`
- 输出仍写到 `workSpace2Tensor`，供 cube 侧 `dw @ kbg.T` 使用。

该阶段支持 `OneLineOneCol`、`MutiLineOneCol` 和 `TwoCol`，用于覆盖 `K` 是否超过 VF 宽度的情况。

### 7.2 ProcessVBeta

计算语义：

```text
vBeta = v * beta
```

原流程依赖 `vFp32Buf`、`betaFp32Buf`、`betaFp32BrcbBuf`。A5 中这些 UB 临时区全部删除，计算与 `KBetaG` 类似，只是没有 `g` 和 `exp`：

```cpp
HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, ...);
CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, ...);
MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg,
               vFP32ZeroReg, vFP32OneReg,
               betaBrcbFP32Reg, betaBrcbFP32Reg, ...);
```

输出同样写到 `workSpace2Tensor`，供 cube 侧 `du @ vb.T` 使用。

### 7.3 ProcessMDuDw

计算语义：

```text
dA4 = tril(dA2 + dA1, diagonal=-1)
```

原流程：

1. dA2/dA1 cast 到 fp32 UB。
2. `Add`。
3. 使用预生成三角 `maskTBuf` 做 `Select`。
4. cast 回 `kType`。

A5 流程：

- 输入只保留 `mduInQue`、`mdwInQue`，输出 `mduwOutQue`。
- regbase 函数中用 `Arange` 生成列索引，用 `rowIdxReg = startRow` 表示当前真实行号。
- `CompareTwoReg<CMPMODE::GE>(colIdx, rowIdx)` 对 `col >= row` 的位置置零，保留严格下三角。
- 不再需要 `maskTBuf` 和 `zeroFp32TBuf`。

经验：三角 mask 类逻辑非常适合 regbase 化，尤其当 mask 可由行列索引直接推导时，可以节省一块 `BT * BT / 8` 的 UB。

### 7.4 ProcessG

计算语义：

```text
gFactor = exp(min(gAll - g, 0))
dA = upper_strict(-dA6 * gFactor)
```

原流程：

1. cube 先把 dA6 写到 GM。
2. vector 从 GM 搬入 dA6。
3. gAll/g/dA6 cast 到 fp32 UB。
4. 通过 `Copy + Brcb + Sub + Mins + Exp` 生成 `gFactor`。
5. `-dA6 * gFactor`。
6. 使用 `maskTBuf` 做上三角选择。
7. cast 后写 GM。

A5 流程变化最大：

- cube 不再把 dA6 先写 GM，而是写到 UB list。
- vector 使用 `tensorDA6InList[MAX_CUBE_VEC_SYNC_NUM]` 直接消费 cube 写好的 UB dA6。
- g/gAll/out 使用 `UB_STAGES = 2` 双 stage buffer，并用 MTE2/V/MTE3 事件控制搬运与计算。
- regbase 计算中：
  - `LoadAlign` 或 `LoadIn` 读取 gAll/g/dA6。
  - `SubFloatTwoReg(gAll, g)`。
  - `MinsFloatTwoReg(..., 0.0)`。
  - `ExpFloatTwoReg`。
  - `MulFloatTwoReg(-dA6, gFactor)`。
  - `CompareTwoReg<CMPMODE::LE>(colIdx, rowIdx)` 对 `col <= row` 置零，保留严格上三角。

同步流程：

1. vector 初始化时先对每个 CV buffer 设置 `SYNC_AIV_AIC_FLAG_BEGIN + i`，表示 buffer 可被 cube 使用。
2. cube 写完某个 dA6 UB buffer 后设置 `SYNC_AIC_AIV_FLAG_BEGIN + cvListId`。
3. vector 等待该 flag 后计算。
4. vector 消费完成后再设置 `SYNC_AIV_AIC_FLAG_BEGIN + cvListId`，允许 cube 复用。
5. 末尾 cube 侧等待所有 `SYNC_AIV_AIC` flag，避免提前退出。

经验：当一个阶段从“GM 中间结果”改为“UB 直传”时，regbase 改写会牵动 cube、vector、host tiling 三侧。此类改写收益大，但最容易出现死锁、list id 错位或 UB 预算不一致。

## 8. 迁移其他算子的建议流程

后续给其他算子做 regbase 改写时，可以按下面顺序推进。

1. 先拆分架构分支

在 host tiling 和 kernel include 层保留原实现，新增 `arch35` 分支。先保证 A5 路径能独立选择 tiling struct 和 kernel 文件。

2. 明确 tiling 数据契约

决定 A5 是否复用原 tiling 数据结构。如果字段、布局或新增同步参数发生变化，建议像本算子一样定义独立 packed struct，并确保 host `memcpy_s` 与 kernel `REGISTER_TILING_DEFAULT/GET_TILING_DATA` 完全匹配。

3. 盘点 vector 计算中的 UB 临时区

逐个列出 `TBuf<VECCALC>` 的用途：

- dtype cast 临时区
- broadcast 临时区
- mask 临时区
- fp32 结果临时区
- 常量 zero/one 临时区

能用 `RegTensor`、`LoadIn<..., true>`、`Arange`、`Compare`、`Select`、`UpdateMask` 替代的，都从 UB 预算中删除。

4. 重算 host rowNum

不要沿用原 `SetRowNum*`。regbase 后 rowNum 由新的 UB 常驻对象决定：输入队列、输出队列、必要的 stage buffer、cube/vector 共享 buffer。rowNum 计算和 kernel `InitBuffer/GetBufferByByte` 必须逐项对应。

5. 把计算函数下沉到 `__simd_vf__`

aicore 层保留搬运，simd_vf 层只接收 `__ubuf__` 指针和必要 shape 参数。典型参数包括：

- 输出 UB 指针
- 输入 UB 指针
- `mSize` / `nSize`
- `startRow`
- `lastLoopCnt`
- 尾块有效列数或有效元素数

6. 按 VF 宽度设计分支

至少考虑：

- 一行能放入一个 VF。
- 多行流水，通常配合 `PRELOAD_NUM`。
- 一行超过一个 VF，需要 two-col 或更多列分片。
- 末尾行数不足 `PRELOAD_NUM` 的 tail。

7. 用寄存器方式重写 broadcast/mask

广播优先考虑 scalar load 或短向量 load，不再生成 brcb UB tensor。规则 mask 优先考虑 `Arange + Compare + Select`，不再生成 UB mask tensor。

8. 处理 dtype 和尾块 mask

half/bfloat16/float 的 cast 路径要分别确认。`StoreAlign` 和部分列 `CastHalf2Float` 要使用 `UpdateMask`，否则最后一个 chunk 或列数小于 VF 宽度时容易写脏数据。

9. 如果涉及 cube/vector 协同，先设计同步协议

明确：

- buffer 个数由哪个 tiling 字段给出。
- 每个 buffer 的字节数如何计算。
- cube 等哪个 flag，vector 等哪个 flag。
- list id 如何轮转。
- 尾部未被当前 subblock 消费的 task 如何释放 flag。
- 退出前是否需要 drain 所有 flag。

10. 做覆盖测试

至少覆盖：

- `chunkSize = 64/128`
- `curChunkSize < chunkSize` 的尾块
- `curRowNum = 1`
- `curRowNum` 非 `PRELOAD_NUM` 整数倍
- `K/V` 小于、等于、大于 `VECTOR_REG_WIDTH / sizeof(kType)`
- half 和 float beta
- 固定长度与变长 `cu_seqlens/chunk_indices`
- 多 head、多 batch、多 subblock

## 9. 本算子改写的主要经验

- regbase 改写的核心收益来自减少 fp32 中间 UB 占用，把 cast、broadcast、mask、elementwise 计算搬到寄存器。
- host tiling 必须和 kernel 计算方式一起改；否则 rowNum、workspace、UB buffer 个数都会和实际执行不匹配。
- 对规则型 mask，不要继续分配 `maskTBuf`，用寄存器行列索引生成 mask 更自然。
- 对 broadcast，不要机械寻找 `Brcb` 的 regbase 版本，可以直接让标量寄存器参与向量乘法。
- `ProcessG` 这类 cube/vector 串联阶段可以进一步避免 GM 中间结果，但必须把 `rowNumG/gCVNum` 作为 host-kernel-cube-vector 的共同契约。
- A5 分支中的 `H`、workspace offset、`GetChunkOffset`、shape check 是一组联动修改；后续迁移时要警惕字段改名导致的 offset 漏改。
- `__simd_vf__` 函数中 tail、mask、load/store 对齐是风险最高的部分。建议每个分支都单独构造边界 case 验证。

