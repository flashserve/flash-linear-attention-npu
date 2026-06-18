# PrepareWyReprBwdDa 算子说明

`PrepareWyReprBwdDa` 是用于线性注意力（Linear Attention）或相关变体（如 Delta Rule）中 WY 表示（WY Representation）反向传播过程的自定义算子。该算子根据前向输入的 Key、Value、Beta、A 矩阵以及反向传入的 dw、du 梯度和 Gate，计算并输出针对 A 矩阵的梯度 `dA`。

该算子支持 **GVA（Grouped Value Attention）** 特性，允许 Key 头数（HK）与 Value 头数（HV）不等，即 `HV = group_size × HK`（group_size ≥ 1），同时支持 `Vdim = 256` 的较大 Value 维度。

---

## 1. 算子功能

在分块序列模型的反向传播过程中，计算以下张量的梯度：

- **dA**：矩阵 A 的梯度。A 矩阵通常是在分块线性注意力中用于构建线性算子核的中间表示。

### 1.1 GVA 特性

算子支持 GVA（Grouped Value Attention）场景，核心约束为：

- `HV` 必须是 `HK` 的整数倍，即 `HV = group_size × HK`（group_size ≥ 1）
- 当 `HK == HV` 时，退化为传统模式，完全兼容旧行为
- 当 `HK != HV` 时，Key 张量按 group_size 广播到 Value 头维度参与计算

### 1.2 Vdim 支持

算子支持 `Vdim = 128` 和 `Vdim = 256` 两种 Value 维度，`Kdim` 固定为 `128`。

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnPrepareWyReprBwdDaGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnPrepareWyReprBwdDa` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnPrepareWyReprBwdDaGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dAOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnPrepareWyReprBwdDa(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream
);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `k` | 输入 | 必选 | Key 输入张量 | 参与反向计算；接口执行前会先转为连续内存；头数为 HK | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value 输入张量 | 参与反向计算；接口执行前会先转为连续内存；头数为 HV | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `beta` | 输入 | 必选 | Beta 权重张量 | 参与反向计算；通常为 FP32 | `FLOAT16`、`BFLOAT16`、 `FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `a` | 输入 | 必选 | 前向输出 A 矩阵 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, BT]` | 支持 |
| `dw` | 输入 | 必选 | Weight 梯度输入 | 对应 w 分支的梯度输入 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, K]` | 支持 |
| `du` | 输入 | 必选 | U 分支梯度输入 | 对应 u 分支的梯度输入 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 门控张量 | 参与反向计算；通常为 FP32 | `FLOAT16`、`BFLOAT16`、 `FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列累计长度 | 变长模式输入，形状为 `[N+1]` | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 扁平化的一维数组 `[num_chunks * 2]` | `INT64` | `ND` | 1 维 | - |

*注：`BT` 维度通常等于 `chunkSize`。*

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `chunkSize` | 输入 | 接口侧必传 | 分块大小 | 对应 A 矩阵的最后一个维度 | `int64_t` | 常用 `64` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dAOut` | 输出 | A 矩阵梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, BT]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器 | `aclOpExecutor*` | - | - | - |

---

## 4. 调用约束与执行语义

### 4.1 形状与模式约束

- `k` 与 `dw` 的形状K 维一致：`[B, HK, T, K]` 与 `[B, HV, T, K]`。
- `v` 与 `du` 的形状必须完全一致：`[B, HV, T, V]`。
- `a` 与 `dAOut` 的形状必须完全一致：`[B, HV, T, BT]`。
- `beta` 与 `g` 的形状一致：`[B, HV, T]`。
- 当 `cuSeqlensOptional` 存在时，开启变长模式，此时 `B` 必须为 `1`。

### 4.2 GVA 约束

- `HV` 必须是 `HK` 的整数倍，即 `HV = group_size × HK`（group_size ≥ 1）。
- `k` 的头数为 `HK`，其余张量（`v`、`beta`、`a`、`dw`、`du`、`g`）的头数为 `HV`。
- `Kdim` 固定为 `128`，`Vdim` 支持 `128` 或 `256`。

### 4.3 连续性
- 算子内部会调用 `Contiguous` 接口。若输入 Tensor 已经连续，则可避免额外的内存拷贝开销。

---

## 5. Torch 测试调用示例

该算子可通过 PyTorch 接口直接调用，底层的两阶段接口（workspace + executor）已被封装，无需手动处理。
### 5.1 定长场景
```python
import torch
import torch_npu

device = "npu:0"

# 基本参数
B, H, T, K, V = 1, 4, 2048, 128, 128
chunk_size = 64
BT = chunk_size

# 构造输入
k = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, T, V, device=device, dtype=torch.float16)
beta = torch.randn(B, H, T, device=device, dtype=torch.float32)
a = torch.randn(B, H, T, BT, device=device, dtype=torch.float16)
dw = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
du = torch.randn(B, H, T, V, device=device, dtype=torch.float16)
# 构造满足约束的 g：负数且沿 T 单调递减
base = torch.rand(B, H, T, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

# 调用算子
dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
    k, v, beta, a, dw, du, g,
    chunk_size=chunk_size,
    cu_seqlens=None,
    chunk_indices=None
)

print(dA.shape) # Expected: [1, 4, 2048, 64]
```

### 5.2 变长场景
```python
import torch
import torch_npu

device = "npu:0"

# B 必须为 1
B, H, K, V = 1, 4, 128, 128
chunk_size = 64
BT = chunk_size

# 变长序列: [64, 128] -> 总长 192
cu_seqlens = torch.tensor([0, 64, 192], device=device, dtype=torch.int64)
total_len = cu_seqlens[-1].item()

# 构造 chunk_indices (flattened: [seq_id, chunk_id, ...])
# seq 0 (len 64): 1 chunk -> [0, 0]
# seq 1 (len 128): 2 chunks -> [1, 0, 1, 1]
chunk_indices = torch.tensor([0, 0, 1, 0, 1, 1], device=device, dtype=torch.int64)

k = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)
beta = torch.randn(B, H, total_len, device=device, dtype=torch.float32)
a = torch.randn(B, H, total_len, BT, device=device, dtype=torch.float16)
dw = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
du = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)
# 构造满足约束的 g：负数且沿 T 单调递减
base = torch.rand(B, H, T, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
    k, v, beta, a, dw, du, g,
    chunk_size=chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices
)

print(dA.shape)
```

---

## 6. 目录结构

```text
prepare_wy_repr_bwd_da/
├── examples/
│   └── test_aclnn_prepare_wy_repr_bwd_da.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_prepare_wy_repr_bwd_da.cpp
│   │   ├── aclnn_prepare_wy_repr_bwd_da.h
│   │   ├── prepare_wy_repr_bwd_da.cpp
│   │   └── prepare_wy_repr_bwd_da.h
│   ├── op_tiling/
│   │   ├── prepare_wy_repr_bwd_da_tiling.cpp
│   │   └── prepare_wy_repr_bwd_da_tiling.h
│   ├── CMakeLists.txt
│   └── prepare_wy_repr_bwd_da_def.cpp
├── op_kernel/
│   ├── prepare_wy_repr_bwd_da_common.h
│   ├── prepare_wy_repr_bwd_da_cube.h
│   ├── prepare_wy_repr_bwd_da_vector.h
│   └── prepare_wy_repr_bwd_da.cpp
├── test/
│   ├── test_da.py                 # 精度测试脚本
│   ├── test_da_performance.py     # 性能测试脚本
│   ├── test_da_cases.json         # 测试用例配置（JSON 格式）
│   ├── gen_perf_report.py         # 性能报告生成脚本
│   ├── run_da.sh                  # 统一启动脚本
│   ├── perf_report.csv            # 性能报告输出（运行后生成）
│   └── prof_output/               # msprof 采集输出目录（运行后生成）
├── CMakeLists.txt
├── README.md
└── run.sh
```

---

## 7. 测试使用说明

### 7.1 测试用例配置

测试用例统一在 `test/test_da_cases.json` 中配置，采用 JSON 数组格式。每个用例包含以下字段：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `name` | string | 用例名称 |
| `description` | string | 用例描述 |
| `enabled` | bool | 是否启用该用例，`true` 为执行，`false` 为跳过 |
| `B` | int | batch size |
| `query_head` | int | Key 头数（HK） |
| `value_head` | int | Value 头数（HV），需满足 `HV = group_size × HK` |
| `T` | int | 序列长度 |
| `Kdim` | int | Key 维度，固定为 128 |
| `Vdim` | int | Value 维度，支持 128 或 256 |
| `chunk_size` | int | 分块大小 |
| `dtype` | string | Key/Value 数据类型，`fp16` 或 `bf16` |
| `gtype` | string | Gate 数据类型，`fp16`、`bf16` 或 `fp32` |
| `varlen` | bool | 是否变长模式，`true` 为变长，`false` 为定长 |
| `mean_len` | int | 变长模式下 cu_seqlens 列表长度（仅 `varlen: true` 时有效） |

**用例启用方式**：修改 `test_da_cases.json` 中对应用例的 `"enabled"` 字段为 `true` 或 `false`，即可控制是否执行该用例。

### 7.2 run_da.sh 使用方法

`test/run_da.sh` 是统一的测试启动脚本，支持精度测试和性能测试两种模式。

```bash
cd test
# 精度测试（默认 device 0）
bash run_da.sh --precision

# 性能测试（自动 msprof 采集并生成性能报告）
bash run_da.sh --performance

# 指定 NPU device
bash run_da.sh --precision --device 0
bash run_da.sh --performance --device 1

# 指定自定义用例文件
bash run_da.sh --precision --json /path/to/custom_cases.json
```

**参数说明**：

| 参数 | 说明 |
| :--- | :--- |
| `--precision` | 运行精度测试（`test_da.py`） |
| `--performance` | 运行性能测试（`test_da_performance.py`），自动使用 `msprof` 采集性能数据 |
| `--json <path>` | 指定测试用例 JSON 文件路径，默认为 `test_da_cases.json` |
| `--device <id>` | 指定 NPU device id，默认为 4 |

**性能测试输出**：性能测试运行结束后，会在 `test/` 目录下生成 `perf_report.csv` 性能报告，包含每个用例的 shape 信息、数据类型、定长/变长标识和算子耗时。

