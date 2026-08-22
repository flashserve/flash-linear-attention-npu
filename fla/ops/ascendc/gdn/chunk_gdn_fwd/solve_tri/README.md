# SolveTri 算子说明

`SolveTri` 是用于计算下三角矩阵求逆的自定义算子，主要应用于 Gated Delta Rule 线性注意力机制中的 chunk-wise 矩阵求逆操作。

---

## 1. 算子功能

计算 $(I + A)^{-1}$，其中 $A$ 是严格下三角矩阵（对角线为0）。

$$Y = (I + A)^{-1}$$

该算子支持四种数据布局：
- **BSND**: `[Batch, T, Head, chunkSize]`，单 chunk 内数据不连续（行步长 = H×BT）
- **BNSD**: `[Batch, Head, T, chunkSize]`，单 chunk 内数据连续（行步长 = BT，BSND 的转置）
- **TND**: `[total_T, Head, chunkSize]`，变长序列模式，单 chunk 内数据不连续
- **NTD**: `[Head, total_T, chunkSize]`，变长序列模式，单 chunk 内数据连续（TND 的转置）

> BNSD/NTD 由于单 chunk 内数据连续，DataCopy 可使用 blockCount=1 实现连续搬运，效率高于 BSND/TND。

---

## 2. 接口定义

### 2.1 ACLNN 接口

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnSolveTriGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *cuSeqlens,
    const aclIntArray *chunkIndices,
    const char *layout,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子计算
aclnnStatus aclnnSolveTri(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 2.2 PyTorch 接口

```python
torch.ops.npu.npu_solve_tri(
    x: Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    layout: str = "bsnd"
) -> Tensor
```

---

## 3. 输入参数

| 参数 | 数据类型 | 是否必须 | 描述 |
|------|----------|----------|------|
| x | FLOAT16/BFLOAT16 | 是 | 输入下三角矩阵 |
| cu_seqlens | INT64 | TND 模式必须 | 累积序列长度 |
| chunk_indices | INT64 | TND 模式必须 | chunk 索引数组 |
| layout | string | 否 | 数据布局，支持 "bsnd"、"bnsd"、"tnd"、"ntd"，默认 "bsnd" |

---

## 4. 输入约束

1. **数据类型**：输入 x 仅支持 FLOAT16 和 BFLOAT16
2. **chunkSize**：最后一维（矩阵大小）仅支持 64 或 128
   - 在 Atlas A2（910 机器）上：
     - `chunkSize=64`：高精度分支，全程使用 FP32 计算
     - `chunkSize=128`：低精度分支，中间计算会 cast 成 FP16 或 BF16，且需满足 `H * chunkSize * 16 + 16 < 65536`
   - 在 Ascend 950 系列上：64 和 128 均正常计算
3. **数据布局**：
   - `bsnd`: 输入 shape 为 `[B, S, H, chunkSize]`，单 chunk 内数据不连续
   - `tnd`: 输入 shape 为 `[total_T, H, chunkSize]`，需配合 cu_seqlens 和 chunk_indices 使用，单 chunk 内数据不连续
4. **变长模式**：当 layout 为 "tnd" 时，cu_seqlens 和 chunk_indices 必须提供，数据类型为 INT64

---

## 5. 输出参数

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| xOut | FLOAT16/BFLOAT16 | 输出矩阵，shape 与输入一致 |

---

## 6. 算子实现

### 6.1 算法原理

使用 **MCH (Matrix Chain Halving) + MBH (Matrix Block Halving)** 算法高效计算下三角矩阵的逆：

1. 将矩阵分块为 $2 \times 2$ 块矩阵
2. 利用下三角矩阵的结构特性递归求解
3. 通过 AIC 核执行 CUBE 矩阵乘法，AIV 核生成辅助矩阵

## 7. 目录结构

```
solve_tri/
├── docs/
│   └── aclnnSolveTri.md
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_solve_tri.cpp
│   │   ├── aclnn_solve_tri.h
│   │   └── solve_tri.cpp
│   ├── solve_tri_def.cpp
│   ├── solve_tri_tiling.cpp
│   ├── solve_tri_tiling.h
│   └── CMakeLists.txt
├── op_kernel/
│   ├── arch35/
│   │   ├── mem.h
│   │   ├── solve_tri_ascend950.h
│   │   └── solve_tri_ascend950_common.h
│   ├── solve_tri.cpp
│   ├── solve_tri_common.h
│   ├── solve_tri_cube.h
│   └── solve_tri_vector.h
├── test/
│   └── test.py
├── CMakeLists.txt
└── README.md
```
