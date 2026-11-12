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

`bhtd` 是 ACLNN 中 `bnsd` 的兼容别名。实际性能取决于设备和输入形状。

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
| cu_seqlens | INT64 | TND/NTD 模式必须 | 累积序列长度 |
| chunk_indices | INT64 | TND/NTD 模式必须 | 展平的 `(seq_id, chunk_id)` 对 |
| layout | string | 否 | 数据布局，支持 "bsnd"、"bnsd"、"tnd"、"ntd"，默认 "bsnd" |

---

## 4. 输入约束

1. **数据类型**：输入 x 仅支持 FLOAT16 和 BFLOAT16
2. **chunkSize**：最后一维支持 16、32、64 和 128；批次、head 和 token 维度均须为正。
3. **变长模式**：`tnd` 和 `ntd` 必须同时提供 `cu_seqlens` 和 `chunk_indices`；定长布局不得提供这两个参数。`cu_seqlens` 从 0 开始、以 `total_T` 结束且非递减，相邻相等表示空序列；目前不支持所有序列均为空。
4. **chunk 索引**：展平数组中的每对为原始序列编号和该序列内从 0 开始的 chunk 编号，每个非空 chunk 必须恰好出现一次。各对可按任意顺序排列。尾部不足一个 chunk 时只计算有效矩阵区域。
5. **输出**：ACLNN 的 `xOut` 必须与 `x` 的 shape 和 dtype 一致且连续。Python 包装层返回新分配的同 shape、同 dtype 输出。

完整接口约束参见 [docs/api.md](docs/api.md)，实现概述参见 [docs/design.md](docs/design.md)。

本次新增的 shape、dtype、layout、输出及变长元数据校验和结构化错误仅用于 A2/A3。A5 保留 `546859c7` 的原有检查行为，调用方仍须满足其原有输入约束；不能将 A2/A3 的非法输入拦截结果视作 A5 的保证。

---

## 5. 输出参数

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| xOut | FLOAT16/BFLOAT16 | 输出矩阵，shape 与输入一致 |

---

## 6. 算子实现

### 6.1 算法原理

A2/A3 的 chunk 64 路径使用 AIC/AIV 协同求解，chunk 16/32/128 使用 FP32 Vector 前向替代。A5 按 chunk size 分发到对应实现。FP32 是内部计算或中间存储精度，不表示公开接口接受 FP32 输入。

新增 workspace 计算和 kernel 优化仅用于 A2/A3。A5 的 workspace 计算、kernel 入口及 `arch35` 实现保持上述基线行为；本次不对其原有 workspace 容量作新的验证结论。注册配置按平台分开，A2/A3 使用 `PrecisionReduceFlag(false)`，A5 保留基线的 `true`；该字段控制混合精度类型选择，不改变 kernel 内部的 FP32 运算。

## 7. 目录结构

```
solve_tri/
├── docs/
│   ├── aclnnSolveTri.md
│   ├── api.md
│   └── design.md
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
│   ├── solve_tri_fp32.h
│   ├── solve_tri_vector.h
│   └── solve_tri_vector_fp32.h
├── test/
│   └── test.py
├── CMakeLists.txt
└── README.md
```
