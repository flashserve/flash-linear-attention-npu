# CausalConv1dBwd

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR/Ascend 950DT</term>       |   √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |   √  |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |   √  |
| <term>Atlas 200I/500 A2 推理产品</term>          |   ×  |
| <term>Atlas 推理系列产品</term>                    |   ×  |
| <term>Atlas 训练系列产品</term>                    |   ×  |

## 功能说明

- 算子功能：完成因果一维卷积（Causal Conv1d）的反向传播计算，输出输入梯度 `dx`、权重梯度 `dw`、偏置梯度 `db` 和初始状态梯度 `dh0`。
- 支持固定长度和变长序列：
  - 固定长度：`BSND`/`BSH`、`BNSD`。
  - 变长序列：`TND`、`NTD`，通过 `queryStartLoc` 描述每条序列的起止位置。
- 支持输出梯度 layout 转换：`inputLayout` 只影响前向输出 `y` 及其梯度 `dy` 的物理排布，`x`、`dx` 始终保持逻辑 layout。

## 计算公式

令卷积核宽度为 $W$，特征维度为 $D$，前向预激活输出为 $y$，上游梯度为 $dy$。当 `activation=0` 时：

$$
g_t = dy_t
$$

当 `activation=1` 或 `activation=2` 时，按 SiLU/Swish 反向计算有效梯度：

$$
\sigma(y_t)=\frac{1}{1+e^{-y_t}}
$$

$$
g_t = dy_t \cdot \sigma(y_t) \cdot (1 + y_t \cdot (1 - \sigma(y_t)))
$$

反向传播计算：

$$
dx_t = \sum_{i=0}^{W-1} g_{t+i} \cdot weight_{W-1-i}
$$

$$
dw_{W-1-i} = \sum_{b,t} g_{b,t+i} \cdot x_{b,t}
$$

$$
db = \sum_{b,t} g_{b,t}
$$

当 `initial_state` 存在时，序列开头依赖的历史状态会参与 `dw` 和 `dh0` 计算；当 `dht` 存在时，最终卷积状态梯度会反传到序列尾部的 `dx`。

## 算子输入输出

### 输入

| 参数名 | 数据类型 | Shape | 是否必选 | 描述 |
| :-- | :-- | :-- | :--: | :-- |
| x | FLOAT/FLOAT16/BF16 | 固定长度 `[B, T, D]`；变长 `[totalTokens, D]` | 必选 | 前向输入，始终使用逻辑 layout。 |
| y | FLOAT/FLOAT16/BF16 | 由 `inputLayout` 决定 | 可选 | 前向预激活输出。`activation=1/2` 时必须提供。 |
| weight | FLOAT/FLOAT16/BF16 | `[W, D]` | 必选 | 卷积核权重。 |
| dy | FLOAT/FLOAT16/BF16 | 与 `y` 相同 | 必选 | 前向输出梯度。 |
| initial_state | FLOAT/FLOAT16/BF16 | `[B, W, D]` | 可选 | 初始卷积状态，与正向算子状态 layout 一致。 |
| dht | FLOAT/FLOAT16/BF16 | `[B, W, D]` | 可选 | 最终卷积状态梯度。 |
| queryStartLoc | INT64 | `[B+1]` | 可选 | 变长序列起止位置。`TND`/`NTD` 下必须提供。 |

### 输出

| 参数名 | 数据类型 | Shape | 是否必选 | 描述 |
| :-- | :-- | :-- | :--: | :-- |
| dx | FLOAT/FLOAT16/BF16 | 固定长度 `[B, T, D]`；变长 `[totalTokens, D]` | 必选 | 输入梯度，始终按逻辑 layout 输出。 |
| dw | FLOAT/FLOAT16/BF16 | `[W, D]` | 可选 | 权重梯度，数据类型与 `weight` 相同。 |
| db | FLOAT/FLOAT16/BF16 | `[D]` | 可选 | 偏置梯度，数据类型与输入相同。 |
| dh0 | FLOAT/FLOAT16/BF16 | `[B, W, D]` | 可选 | 初始状态梯度。 |

### 属性

| 属性名 | 类型 | 默认值 | 取值 | 描述 |
| :-- | :-- | :--: | :-- | :-- |
| activation | int64 | 0 | 0, 1, 2 | 0：无激活；1：SiLU；2：Swish，当前与 SiLU 等价。 |
| inputLayout | string | BSND | BSND, BSH, TND, BNSD, NTD | 输入 `y/dy` 的物理 layout。 |

## Layout 与 Shape

| inputLayout | `x` shape | `y/dy` shape | `dx` 输出 shape | `queryStartLoc` |
| :-- | :-- | :-- | :-- | :-- |
| `BSND` / `BSH` | `[B, T, D]` | `[B, T, D]` | `[B, T, D]` | 可不传 |
| `BNSD` | `[B, T, D]` | `[B, N, T, Dh]`，`D=N*Dh` | `[B, T, D]` | 可不传 |
| `TND` | `[totalTokens, D]` | `[totalTokens, D]` | `[totalTokens, D]` | 必须传 |
| `NTD` | `[totalTokens, D]` | `[N, totalTokens, Dh]`，`D=N*Dh` | `[totalTokens, D]` | 必须传 |

变长序列下，`queryStartLoc` 必须满足：

$$
queryStartLoc[0]=0
$$

$$
queryStartLoc[B]=totalTokens
$$

且 `queryStartLoc[i+1] >= queryStartLoc[i]`。

## 约束说明

- `x`、`y`、`weight`、`dy`、`initial_state`、`dht` 的数据类型必须一致。
- `dx`、`dh0` 的数据类型必须与 `x` 一致。
- `dw`、`db` 在 kernel 内部使用 `FLOAT` 累加，写回时转换为输入数据类型。
- `weight` 必须为二维 tensor，shape 为 `[W, D]`。
- `y` 和 `dy` 必须采用相同的物理 layout 和 shape；`x` 始终采用逻辑 layout。
- `initial_state`、`dht`、`dh0` 的逻辑 shape 为 `[B, W, D]`，与正向算子的状态 layout 一致。
- `BSND`/`TND` layout 下逻辑特征维 `D` 必须为 16 的倍数；`BNSD`/`NTD` layout 下最后一维 `Dh` 必须为 16 的倍数，逻辑特征维为 `D=N*Dh`。
- 不支持空序列，固定长度场景下 `T > 0`，变长场景下 `totalTokens > 0`。
- `activation=1` 或 `activation=2` 时必须提供 `y`。
- `TND`/`NTD` layout 下必须提供 `queryStartLoc`。

## 调用说明

| 调用方式 | 样例代码 | API 文档 |
| :-- | :-- | :-- |
| aclnn 接口 | [test_aclnn_causal_conv1d_bwd.cpp](./examples/test_aclnn_causal_conv1d_bwd.cpp) | [aclnnCausalConv1dBwd](./docs/aclnnCausalConv1dBwd.md) |
