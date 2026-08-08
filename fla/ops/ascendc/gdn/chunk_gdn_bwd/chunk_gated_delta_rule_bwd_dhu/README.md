# ChunkGatedDeltaRuleBwdDhu 算子说明
`ChunkGatedDeltaRuleBwdDhu` 是一个用于分块门控 delta 规则（Chunk Gated Delta Rule）反向传播过程中的自定义算子。该算子沿时间轴反向扫描，根据前向激活值及上游梯度，计算隐藏状态梯度（dh）、初始隐藏状态梯度（dh0）以及更新后的 Value 梯度（dv2）。

---

## 1. 算子功能

在分块序列模型的反向传播阶段，对每个 chunk 执行反向隐藏状态递推，计算以下张量：

- **dh**：各 chunk 起始时刻的隐藏状态梯度（用于 WY 表示的反向传播）
- **dh0**：初始隐藏状态 `h0` 的梯度（当 `h0` 被提供时有意义）
- **dv2**：融合了隐藏状态贡献后的 Value 梯度（`dv` + 来自 `dh` 的贡献）

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnChunkGatedDeltaRuleBwdDhu` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w,
    const aclTensor *dO, const aclTensor *dv,
    const aclTensor *gOptional, const aclTensor *gkOptional,
    const aclTensor *h0Optional, const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize,
    const aclTensor *dhOut, const aclTensor *dh0Out, const aclTensor *dv2Out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(
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
|---|---|---|---|---|---|---|---|---|
| `q` | 输入 | 必选 | Query 输入张量 | 参与反向递推 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `k` | 输入 | 必选 | Key 输入张量 | 参与 dv2 计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `w` | 输入 | 必选 | Weight（衰减权重）输入张量 | 参与隐藏状态更新 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, K]` | 支持 |
| `dO` | 输入 | 必选 | 前向输出 `o` 的梯度张量 | 即上游输出梯度 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `dv` | 输入 | 必选 | Value 的上游梯度张量 | 将与来自 `dh` 的贡献叠加后输出为 `dv2` | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `gOptional` | 输入 | 可选 | Gate 张量 | 对隐藏状态递推施加指数门控 `exp(g)` | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `gkOptional` | 输入 | 可选 | Key-wise Gate 张量 | 对每个 Key 维度施加额外门控 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, K]` | 支持 |
| `h0Optional` | 输入 | 可选 | 初始隐藏状态张量 | 提供时参与递推初始化 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, K, V]` | 支持 |
| `dhtOptional` | 输入 | 可选 | 末尾隐藏状态的梯度张量 | 反向递推的起始梯度 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, K, V]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | 变长模式输入，形状为 `[N+1]` | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 变长模式输入，扁平化存储 `[seqIdx0, chunkIdx0, ...]`，长度为 `2 * numChunks` | `INT64` | `ND` | 1 维 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `scale` | 输入 | 可选属性，接口侧必传 | 缩放系数 | 推荐设置为 `1 / sqrt(K)` | `double` | 建议按 `1 / sqrt(K)` 设置 |
| `chunkSize` | 输入 | 可选属性，接口侧必传 | 分块大小 | 默认值为 `64`，仅支持 `64` 或 `128` | `int64_t` | 仅支持 `64` / `128` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `dhOut` | 输出 | 各 chunk 起始时刻的隐藏状态梯度 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, NT, K, V]` | 支持 |
| `dh0Out` | 输出 | 初始隐藏状态 `h0` 的梯度（仅当 `h0Optional` 非空时有意义）| `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, K, V]` | 支持 |
| `dv2Out` | 输出 | 融合了隐藏状态贡献后的 Value 梯度 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

> **注**：`NT` 为 chunk 总数，定长模式下 `NT = ceil(T / chunkSize)`；变长模式下 `NT` 由 `chunkIndices` 推导。

### 3.4 形状与约束

- `q`、`k` 的形状必须为 `[B, HK, T, K]`，二者完全同形。
- `w` 的形状必须为 `[B, HV, T, K]`，head 维与 value 侧对齐。
- `dO`、`dv` 的形状必须为 `[B, HV, T, V]`。
- `q` 与 `dO`/`dv` 的 `B`、`T` 必须一致，head 数允许不同（GVA）。
- `gOptional` 的形状必须为 `[B, HV, T]`（若提供）。
- `gkOptional` 的形状必须为 `[B, HV, T, K]`（若提供）。
- `h0Optional`、`dhtOptional` 的形状必须为 `[B, HV, K, V]`（若提供）。
- `dhOut` 的形状必须为 `[B, HV, NT, K, V]`。
- **GVA 约束**：`HV % HK == 0`；读 `q`/`k` 时使用 `hq = hv / (HV / HK)`，读/写 `w`/`dO`/`dv`/`g`/`dh`/`dv2` 使用 value head 索引 `hv`。
- 当前实现要求 `K ≤ 128`。
- 当前实现要求 `V ≤ 256`（Cube tile 原生按 `V` 上限 256 设计，**无**按 V 维切换的 TilingKey）。
- **TilingKey**：`g` 与 `q` 同 dtype 时为 Key=1，`g` 为 FP32 时为 Key=2（与 V 维无关）。
- `chunkSize` 当前仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 须同时提供，且仅支持 `B = 1`。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 必须同时提供或同时省略
  - 同时出现时启用变长模式（varlen）
  - 变长模式仅支持 `B = 1`

- `gOptional`：
  - 数据类型可以为 `FLOAT16`、`BFLOAT16` 或 `FLOAT`
  - 数据类型需与 `q` 类型一致，或为 `FLOAT`（FP32）

---

### 4.2 形状约束（强约束）

必须满足以下条件：

- `q, k`: `[B, HK, T, K]`
- `w`: `[B, HV, T, K]`
- `dO, dv, dv2Out`: `[B, HV, T, V]`
- `gOptional`: `[B, HV, T]`
- `gkOptional`: `[B, HV, T, K]`（若提供）
- `h0Optional, dhtOptional, dhOut`: head 维为 `HV`
- `HV % HK == 0`（GVA）

额外限制：

- `K ≤ 128`
- `V ≤ 256`（支持 `V = 256`）
- `chunkSize ∈ {64, 128}`

---

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- `numChunks` 由 `chunkIndices` 的长度推导（`len(chunkIndices) / 2`）
- 当前实现仅支持：

```text
B = 1
```

---

### 4.4 数值语义

算子对 chunk 进行**反向时间扫描**（从最后一个 chunk 到第一个 chunk），每个 chunk 内执行：

```text
# dv2：叠加来自隐藏状态的 Value 梯度
b_dv  = k_chunk @ b_dh                        # 从当前 dh 产生的 dv 贡献
if g:
    b_dv *= exp(g_last - g_chunk)[:, None]     # 门控衰减
dv2_chunk = b_dv + dv_chunk                   # 与上游 dv 叠加

# dh 存储（在更新前记录当前 chunk 的 dh）
dh[:, :, i_t] = b_dh

# 反向递推更新 b_dh（传递给上一 chunk）
if g:
    b_dh *= exp(g_last)
term1 = (q_chunk * exp(g_chunk))^T @ dO_chunk * scale
term2 = w_chunk^T @ dv2_chunk
b_dh = b_dh + term1 - term2
```

- `scale`：必须显式传入，推荐设置为 `1 / sqrt(K)`。

---

## 5. Torch 测试调用示例

### 5.1 固定长度模式

```python
import torch
import torch_npu
import math

def test_chunk_gated_delta_rule_bwd_dhu_fix():
    # 参数设置（GVA 示例：HK=2, HV=4, V=256）
    B, HK, HV, T, K, V = 2, 2, 4, 256, 128, 256
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.float16

    # 构造输入
    q   = torch.rand(B, HK, T, K, dtype=dtype).to(device)
    k   = torch.rand(B, HK, T, K, dtype=dtype).to(device)
    w   = torch.rand(B, HV, T, K, dtype=dtype).to(device)
    d_o = torch.rand(B, HV, T, V, dtype=dtype).to(device)
    dv  = torch.rand(B, HV, T, V, dtype=dtype).to(device)
    g   = torch.rand(B, HV, T,    dtype=torch.float32).to(device)

    # 调用算子（固定长度模式，启用 gate g）
    dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv,
        scale=scale,
        chunk_size=chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False
    )

    NT = (T + chunk_size - 1) // chunk_size
    print("dh   shape:", dh.shape)    # [B, HV, NT, K, V]
    print("dv2  shape:", dv2.shape)   # [B, HV, T, V]
    assert dh.shape  == (B, HV, NT, K, V)
    assert dv2.shape == (B, HV, T, V)
    print("Fix-length Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_bwd_dhu_fix()
```

### 5.2 变长模式

```python
import torch
import torch_npu
import math
import random

def prepare_cu_seqlens(cu_seqlens_len: int, total_length: int):
    """生成随机 cu_seqlens，首元素为 0，末元素为 total_length"""
    batchsize = cu_seqlens_len - 1
    remaining = total_length
    seq_lengths = []
    for i in range(batchsize - 1):
        min_len = 1
        max_len = remaining - (batchsize - 1 - i)
        if max_len < min_len:
            max_len = min_len
        seq_len = random.randint(min_len, max_len)
        seq_lengths.append(seq_len)
        remaining -= seq_len
    seq_lengths.append(remaining)
    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    return cu_seqlens

def prepare_chunk_indices(cu_seqlens, chunk_size: int):
    """根据 cu_seqlens 生成扁平化 chunk_indices"""
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices

def test_chunk_gated_delta_rule_bwd_dhu_varlen():
    # 参数设置（变长模式要求 B = 1）
    B, HK, HV, T, K, V = 1, 2, 4, 512, 128, 256
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.float16

    # 生成变长序列信息（将 T 个 token 随机切分为 4 段）
    cu_seqlens    = prepare_cu_seqlens(cu_seqlens_len=5, total_length=T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices) // 2

    # 构造输入
    q   = torch.rand(B, HK, T, K, dtype=dtype).to(device)
    k   = torch.rand(B, HK, T, K, dtype=dtype).to(device)
    w   = torch.rand(B, HV, T, K, dtype=dtype).to(device)
    d_o = torch.rand(B, HV, T, V, dtype=dtype).to(device)
    dv  = torch.rand(B, HV, T, V, dtype=dtype).to(device)
    g   = torch.rand(B, HV, T,    dtype=torch.float32).to(device)

    # 调用算子（变长模式，启用 gate g）
    dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv,
        scale=scale,
        chunk_size=chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False
    )

    print("dh   shape:", dh.shape)    # [B, HV, NT, K, V]
    print("dv2  shape:", dv2.shape)   # [B, HV, T, V]
    assert dh.shape  == (B, HV, NT, K, V)
    assert dv2.shape == (B, HV, T, V)
    print("Variable-length Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_bwd_dhu_varlen()
```

---

## 6. 目录结构

```text
chunk_gated_delta_rule_bwd_dhu/
├── examples/
│   └── test_aclnn_chunk_gated_delta_rule_bwd_dhu.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_chunk_gated_delta_rule_bwd_dhu.cpp
│   │   ├── aclnn_chunk_gated_delta_rule_bwd_dhu.h
│   │   ├── chunk_gated_delta_rule_bwd_dhu.cpp
│   │   └── chunk_gated_delta_rule_bwd_dhu.h
│   ├── op_tiling/
│   │   ├── chunk_gated_delta_rule_bwd_dhu_tiling.cpp
│   │   └── chunk_gated_delta_rule_bwd_dhu_tiling.h
│   ├── chunk_gated_delta_rule_bwd_dhu_def.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── chunk_gated_delta_rule_bwd_dhu_base.h
│   ├── chunk_gated_delta_rule_bwd_dhu_cube.h
│   ├── chunk_gated_delta_rule_bwd_dhu_vec.h
│   └── chunk_gated_delta_rule_bwd_dhu.cpp
└── test/
    └── test_chunk_gated_delta_rule_bwd_dhu.py
```
