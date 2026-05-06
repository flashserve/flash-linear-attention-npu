# RecurrentGatedDeltaRule 算子说明
`RecurrentGatedDeltaRule` 是用于循环门控 Delta 规则（Recurrent Gated Delta Rule，RGDR）的自定义算子，主要应用于线性注意力机制的推理场景。该算子在每个时间步根据当前输入 $q_t, k_t, v_t$ 和上一隐藏状态 $S_{t-1}$，计算当前输出 $o_t$ 并更新隐藏状态 $S_t$。

---

## 1. 算子功能

在每个时间步 $t$，网络根据输入计算注意力输出并更新隐藏状态：

- **out**：当前时间步注意力输出
- **state**：更新后的隐藏状态（原地更新）

计算公式为：

$$S_t := \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} + \beta_t \cdot (v_t - \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} k_t) k_t^T$$

$$o_t := \frac{S_t q_t}{\sqrt{d_k}}$$

其中：

| 符号 | 含义 | 形状 |
|------|------|------|
| $S_{t-1}, S_t$ | 隐藏状态矩阵 | $\mathbb{R}^{d_v \times d_k}$ |
| $q_t, k_t$ | Query / Key 向量 | $\mathbb{R}^{d_k}$ |
| $v_t$ | Value 向量 | $\mathbb{R}^{d_v}$ |
| $\alpha_t = e^{g_t}$ | 标量门控衰减系数 | $\mathbb{R}$ |
| $\alpha_{kt} = e^{gk_t}$ | 逐维门控衰减系数 | $\mathbb{R}^{d_k}$ |
| $\beta_t$ | 更新门控系数 | $\mathbb{R}$ |
| $o_t$ | 当前时间步输出 | $\mathbb{R}^{d_v}$ |

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnRecurrentGatedDeltaRule` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *numAcceptedTokens,
    float scaleValue,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnRecurrentGatedDeltaRule(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

令 $B$ 为 batch size，$T = \sum_i L_i$ 为累积序列长度，$N_k$ 为 key 头数，$N_v$ 为 value 头数，$D_k$ 为 key 维度，$D_v$ 为 value 维度，BlockNum 为状态块总数。

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|---|
| `query` | 输入 | 必选 | 公式中的 $q$；不支持空 Tensor | - | `BFLOAT16` | `ND` | `(T, Nk, Dk)` | 支持 |
| `key` | 输入 | 必选 | 公式中的 $k$；不支持空 Tensor | - | `BFLOAT16` | `ND` | `(T, Nk, Dk)` | 支持 |
| `value` | 输入 | 必选 | 公式中的 $v$；不支持空 Tensor | - | `BFLOAT16` | `ND` | `(T, Nv, Dv)` | 支持 |
| `beta` | 输入 | 必选 | 公式中的 $\beta$；不支持空 Tensor | - | `BFLOAT16` | `ND` | `(T, Nv)` | 支持 |
| `stateRef` | 输入&输出 | 必选 | 状态矩阵 $S$，算子执行后原地更新；不支持空 Tensor | **不支持非连续 Tensor** | `BFLOAT16` | `ND` | `(BlockNum, Nv, Dv, Dk)` | 不支持 |
| `actualSeqLengths` | 输入 | 必选 | 序列长度；不支持空 Tensor，首元素代表无效序列长度（即不参与计算的序列长度），其余 $B$ 个元素代表各 batch 的有效序列长度 | 第 1 至第 $B$ 个元素之和等于 $T$ | `INT32` | `ND` | `(B+1,)` | 支持 |
| `ssmStateIndices` | 输入 | 必选 | 输入序列到状态矩阵的映射索引，`state[ssmStateIndices[i]]` 表示第 $i$ 个 token 对应的状态块；不支持空 Tensor | 取值范围 `[0, BlockNum)` | `INT32` | `ND` | `(T,)` | 支持 |
| `g` | 输入 | 可选 | 标量衰减系数 $\alpha_t = e^g$ | 传 `nullptr` 时等价于全 0（$\alpha_t = 1$，即无标量衰减） | `FLOAT32` | `ND` | `(T, Nv)` | 支持 |
| `gk` | 输入 | 可选 | 逐维衰减系数 $\alpha_{kt} = e^{gk}$ | 传 `nullptr` 时等价于全 0（$\alpha_{kt} = \mathbf{1}$，即无逐维衰减） | `FLOAT32` | `ND` | `(T, Nv, Dk)` | 支持 |
| `numAcceptedTokens` | 输入 | 可选 | 每个序列接受的 token 数量 | 传 `nullptr` 时默认全部接受 | `INT32` | `ND` | `(B,)` | 支持 |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `scaleValue` | 输入 | 可选（默认 `1.0`） | Query 的缩放因子 | 推荐按 `1 / sqrt(Dk)` 设置 | `float` | 推荐 `1 / sqrt(Dk)` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `out` | 输出 | 公式中的 $o$，当前时间步注意力输出 | `BFLOAT16` | `ND` | `(T, Nv, Dv)` | 支持 |
| `stateRef` | 输入&输出 | 更新后的隐藏状态矩阵（原地更新） | `BFLOAT16` | `ND` | `(BlockNum, Nv, Dv, Dk)` | 不支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `query`、`key` 的形状为 `(T, Nk, Dk)`。
- `value` 的形状为 `(T, Nv, Dv)`。
- `beta` 的形状为 `(T, Nv)`。
- `stateRef` 的形状为 `(BlockNum, Nv, Dv, Dk)`，不支持非连续 Tensor。
- `actualSeqLengths` 为长度 $B+1$ 的一维 INT32 张量，首元素代表无效序列长度（不参与计算），第 1 至第 $B$ 个元素代表各 batch 的有效序列长度，其元素之和等于 $T$。
- `ssmStateIndices` 为长度 $T$ 的一维 INT32 张量，取值范围 `[0, BlockNum)`。
- 当前仅支持 `BFLOAT16` 精度（query/key/value/beta/stateRef/out）。
- 每个序列的有效 token 数 $L_i$（即 `actualSeqLengths[i+1]`，$i \in [0, B)$）须满足 $L_i \le 8$。

### 3.5 补充说明

- 输入张量 `query/key/value/beta/g` 支持非连续 Tensor 输入。
- `stateRef` 为原地输入输出，**不支持非连续 Tensor**。
- `gk` 当前版本暂不支持，须传 `None`。

---

## 4. 调用约束与执行语义

### 4.1 形状约束（强约束）

必须满足以下条件：

- `query, key`: `(T, Nk, Dk)`
- `value`: `(T, Nv, Dv)`
- `beta`: `(T, Nv)`
- `stateRef`: `(BlockNum, Nv, Dv, Dk)`
- `actualSeqLengths`: `(B+1,)`，首元素为无效序列长度（不参与计算），其余 `B` 个元素为各 batch 的有效序列长度，其之和等于 `T`
- `ssmStateIndices`: `(T,)`，取值范围 `[0, BlockNum)`

---

### 4.2 数值语义

- `g` 传 `None` 时：$\alpha_t = e^0 = 1$，即无标量衰减
- `gk` 当前版本暂不支持，须传 `None`
- `scaleValue` 推荐设置为：

```text
1 / sqrt(Dk)
```

---

## 5. Torch 测试调用示例

### 5.1 定长场景（每序列处理相同数量 token）

```python
import torch
import torch_npu
import math

def test_recurrent_gated_delta_rule_fixed():
    B, mtp = 4, 2           # 4 个 batch，每序列 2 个 token
    T = B * mtp             # 累积 token 数
    Nk, Nv, Dk, Dv = 4, 8, 128, 128
    BlockNum = T            # 每个 token 对应一个独立状态块
    scale = 1.0 / math.sqrt(Dk)
    device = "npu:0"

    query = torch.randn(T, Nk, Dk, dtype=torch.bfloat16).to(device)
    key   = torch.randn(T, Nk, Dk, dtype=torch.bfloat16).to(device)
    value = torch.randn(T, Nv, Dv, dtype=torch.bfloat16).to(device)
    state = torch.randn(BlockNum, Nv, Dv, Dk, dtype=torch.bfloat16).to(device)
    beta  = torch.rand(T, Nv, dtype=torch.bfloat16).to(device)
    g     = torch.randn(T, Nv, dtype=torch.float32).to(device)

    # actual_seq_lengths: shape (B+1,)，首元素为无效序列长度（不参与计算），其余 B 个元素为各 batch 的有效序列长度（总和须等于 T）
    actual_seq_lengths = torch.tensor([0] + [mtp] * B, dtype=torch.int32).to(device)
    # ssm_state_indices: 第 i 个 token 使用 state[ssm_state_indices[i]]
    ssm_state_indices  = torch.arange(T, dtype=torch.int32).to(device)

    out = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state,
        beta=beta,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,   # None = 每序列接受 1 个 token（默认）
        g=g,
        gk=None                     # 当前版本暂不支持，须传 None
    )

    assert out.shape == (T, Nv, Dv)
    print("out shape:", out.shape)   # (8, 8, 128)
    # state 已被原地更新

if __name__ == "__main__":
    test_recurrent_gated_delta_rule_fixed()
```

### 5.2 变长场景（投机推理，各序列接受不同数量 token）

变长场景下，不同序列接受的 token 数量不同（常见于投机推理中 draft 模型的结果验证）。

```python
import torch
import torch_npu
import math

def test_recurrent_gated_delta_rule_varlen():
    B, mtp = 4, 2
    T = B * mtp
    Nk, Nv, Dk, Dv = 4, 8, 128, 128
    BlockNum = T
    scale = 1.0 / math.sqrt(Dk)
    device = "npu:0"

    query = torch.randn(T, Nk, Dk, dtype=torch.bfloat16).to(device)
    key   = torch.randn(T, Nk, Dk, dtype=torch.bfloat16).to(device)
    value = torch.randn(T, Nv, Dv, dtype=torch.bfloat16).to(device)
    state = torch.randn(BlockNum, Nv, Dv, Dk, dtype=torch.bfloat16).to(device)
    beta  = torch.rand(T, Nv, dtype=torch.bfloat16).to(device)
    g     = torch.randn(T, Nv, dtype=torch.float32).to(device)

    # actual_seq_lengths: shape (B+1,)，首元素为无效序列长度（不参与计算），其余 B 个元素为各 batch 的有效序列长度（总和须等于 T）
    actual_seq_lengths = torch.tensor([0] + [mtp] * B, dtype=torch.int32).to(device)
    ssm_state_indices  = torch.arange(T, dtype=torch.int32).to(device)

    # 各序列实际接受的 token 数不同（投机推理验证结果）
    # 需满足 num_accepted_tokens[i] <= actual_seq_lengths[i] <= 8
    num_accepted_tokens = torch.tensor([2, 1, 2, 1], dtype=torch.int32).to(device)

    out = torch_npu.npu_recurrent_gated_delta_rule(
        query, key, value, state,
        beta=beta,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        g=g,
        gk=None
    )

    assert out.shape == (T, Nv, Dv)
    print("out shape:", out.shape)   # (8, 8, 128)

if __name__ == "__main__":
    test_recurrent_gated_delta_rule_varlen()
```

---

## 6. 目录结构

```text
recurrent_gated_delta_rule/
├── docs/
│   └── aclnnRecurrentGatedDeltaRule.md
├── examples/
│   └── test_aclnn_recurrent_gated_delta_rule.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_recurrent_gated_delta_rule.cpp
│   │   ├── aclnn_recurrent_gated_delta_rule.h
│   │   ├── recurrent_gated_delta_rule.cpp
│   │   └── recurrent_gated_delta_rule.h
│   ├── recurrent_gated_delta_rule_def.cpp
│   ├── recurrent_gated_delta_rule_infershape.cpp
│   ├── recurrent_gated_delta_rule_tiling.cpp
│   ├── recurrent_gated_delta_rule_tiling.h
│   └── CMakeLists.txt
└── op_kernel/
    ├── recurrent_gated_delta_rule_tiling_data.h
    ├── recurrent_gated_delta_rule.h
    └── recurrent_gated_delta_rule.cpp
```
