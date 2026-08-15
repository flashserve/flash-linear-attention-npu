# ChunkScaledDotKkt 算子说明

`ChunkScaledDotKkt` 用于 Gated Delta Rule 线性注意力中的 chunk-wise WY 表示构建。当前实现覆盖 `gk is None` 的定长和变长序列场景，输入和输出均采用 head-first 排布。

## 1. 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 2. 算子功能

对每个 batch、value head 和 chunk，计算带 gate 衰减和 beta 缩放的严格下三角 `K @ K^T`。GVA 场景下，`g`/`beta` 按 value head 维 `Hv` 输入，输出 `A` 与 value head 维 `Hv` 对齐；`K` 按 `hk = hv / (Hv / Hk)` 复用：

```text
hv in [0, Hv)
hk = hv / (Hv / Hk)
```

$$
A_{b,hv,t,c} =
\begin{cases}
\beta_{b,hv,t} \cdot \exp(\mathrm{clip}(g_{b,hv,t} - g_{b,hv,s}, -50, 50)) \cdot
\sum_d k_{b,hk,t,d} k_{b,hk,s,d}, & c < r \\
0, & c \ge r
\end{cases}
$$

其中 `BT=chunk_size`，`r=t mod BT`，`s=t-r+c`，`hv` 为 value head。输出只保留每个 chunk 内的严格下三角列，最后一维长度固定为 `BT`。

## 3. 接口定义

### 3.1 ACLNN 接口

```cpp
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    aclTensor *A,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 3.2 PyTorch 接口

```python
torch.ops.npu.npu_chunk_scaled_dot_kkt(
    k: Tensor,
    g: Tensor,
    beta: Tensor,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
    chunk_size: int = 64,
) -> Tensor
```

## 4. 输入参数

| 参数 | 数据类型 | Shape | 是否必须 | 描述 |
| -- | -- | -- | -- | -- |
| k | FLOAT16/BF16 | `[B,Hk,T,K]` | 是 | key 张量，head-first 排布 |
| g | FLOAT | `[B,Hv,T]` | 是 | chunk 内 cumulative gate，head-first 排布 |
| beta | FLOAT | `[B,Hv,T]` | 是 | 每个 token/value head 的缩放系数，head-first 排布 |
| cu_seqlens | INT64 | `[N+1]` | 否 | 变长序列累计长度；定长时不传 |
| chunk_indices | INT64 | `[2*num_chunks]` 或 `[num_chunks,2]` | 否 | 变长 chunk 元数据，按 `[seq_id, chunk_id]` 成对存放 |
| chunk_size | INT | - | 否 | chunk 大小，默认 64 |

## 5. 输出参数

| 输出 | 数据类型 | Shape | 描述 |
| -- | -- | -- | -- |
| A | FLOAT | `[B,Hv,T,chunk_size]` | WY 表示中的严格下三角块 |

## 6. 输入约束

1. `k` 必须为 4D，shape 为 `[B,Hk,T,K]`。
2. `g` 和 `beta` 必须为 3D，shape 为 `[B,Hv,T]`，并且 `B/T` 与 `k` 一致。
3. GVA 要求 `Hv % Hk == 0`；`Hk == Hv` 时退化为普通 MHA。KKT 输出 head 维为 `Hv`。
4. `chunk_size` 仅支持 `16`、`32`、`64`、`128`。
5. `cu_seqlens` 和 `chunk_indices` 必须同时传入或同时省略；省略时走定长 dense 序列。
6. `chunk_indices` 表示 `[seq_id, chunk_id]`，`chunk_id` 从 0 开始。
7. 当前版本不支持 `gk` 分支。
8. 输入建议传入 contiguous Tensor；PyTorch wrapper 会在调用 ACLNN 前执行 contiguous。

## 7. 目录结构

```text
chunk_scaled_dot_kkt/
├── docs/
│   └── aclnnChunkScaledDotKkt.md
├── op_host/
│   ├── chunk_scaled_dot_kkt_def.cpp
│   ├── chunk_scaled_dot_kkt_infershape.cpp
│   ├── chunk_scaled_dot_kkt_tiling.cpp
│   ├── chunk_scaled_dot_kkt_tiling.h
│   └── CMakeLists.txt
├── op_kernel/
│   ├── chunk_scaled_dot_kkt.cpp
│   ├── chunk_scaled_dot_kkt.h
│   └── chunk_scaled_dot_kkt_tiling_key.h
├── test/
│   └── test.py
├── CMakeLists.txt
└── README.md
```

## 8. 验证说明

算子目录测试入口：

```bash
python3 fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/test/test.py
```

torch 插件侧测试入口：

```bash
cd torch_custom/fla_npu/test
python3 test_npu_chunk_scaled_dot_kkt.py
```
