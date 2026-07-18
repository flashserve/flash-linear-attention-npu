# CausalConv1dBwd

## 1. 功能概述

CausalConv1d 的训练反向算子。它支持 dense/varlen 和五种公开 layout，根据 x/weight/dy 及可选预激活 y、initial_state、dht 计算 dx/dw/db/dh0。

## 2. 数学定义

若启用 SiLU/Swish，先得到预激活梯度：

```text
dz = dy                                      activation=0
dz = dy * (sigmoid(y)+y*sigmoid(y)*(1-sigmoid(y))) activation=1/2
dx[t-j,d] += dz[t,d] * weight[j,d]
dw[j,d]   += dz[t,d] * x_or_state[t-j,d]
db[d]     += sum_t dz[t,d]
```

序列起点以前的 dx 贡献累积到 `dh0`，末状态梯度 `dht` 通过状态滚动关系反传。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,T,D] 或 [T,D]` | FP32/FP16/BF16 | 逻辑 BSH/TH | 前向输入 |
| `y` | 条件可选 | `由 input_layout 决定` | 与 x 一致 | BSH/BSND/BNSD/TND/NTD | 预激活；activation=1/2 必选 |
| `weight` | 必选 | `[W,D]` | 与 x 一致 | ND | 卷积权重 |
| `dy` | 必选 | `与 y 同形` | 与 x 一致 | 由 input_layout 决定 | 上游梯度 |
| `initial_state` | 可选 | `[B,W,D]` | 与 x 一致 | ND | 前向初始状态 |
| `dht` | 可选 | `[B,W,D]` | 与 x 一致 | ND | 末状态梯度 |
| `query_start_loc` | TND/NTD 必选 | `[B+1]` | INT64 | ND | varlen 序列边界 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dx` | `与逻辑 x 同形` | 与 x 一致 | 输入梯度 |
| `dw` | `[W,D]` | 与 weight 一致 | 权重梯度 |
| `db` | `[D]` | 与 weight 一致 | 偏置梯度 |
| `dh0` | `[B,W,D]` | 与 x 一致 | 初始状态梯度 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `activation` | int | `0` | `{0, 1, 2}` | 0=无激活，1=SiLU，2=Swish（同 SiLU） |
| `input_layout` | str | `BSND` | `{"BSH", "BSND", "BNSD", "TND", "NTD"}` | 输入输出布局 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | FP32/FP16/BF16，同次调用所有浮点输入一致 |
| Format/Layout | x/dx 始终逻辑 BSH/TH；y/dy 按 BSH、BSND、BNSD、TND 或 NTD |
| 模式 | dense/varlen、无激活/SiLU、可选初末状态 |



## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.causal_conv1d_bwd` |
| aclnn | `aclnnCausalConv1dBwdGetWorkspaceSize` / `aclnnCausalConv1dBwd` |
| Ascend C `<<<>>>` | `causal_conv1d_bwd<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_causal_conv1d_bwd` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/causal_conv1d_bwd/accuracy/test_causal_conv1d_bwd.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/causal_conv1d_bwd.json`；覆盖 dense/varlen、无激活/SiLU、可选初末状态。
- 参考实现：`torch_autograd_causal_conv1d_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- weight 必须 `[W,D]` 且 W 仅支持 2/3/4；D 或拆分后的 V 必须为 16 的倍数。
- activation=1/2 时 y 必须提供且与 dy 同 layout/shape。
- TND/NTD 必须提供合法 query_start_loc；累计长度非递减、首项为 0、末项为总 T，且总 token 数必须大于 0。
- initial_state/dht/dh0 的逻辑 shape 为 `[B,W,D]`。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=causal_conv1d_bwd python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/causal_conv1d_bwd/accuracy/test_causal_conv1d_bwd.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/causal_conv1d_bwd/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；varlen 打包场景通常为 1 |
| `T` | dense 序列长度或 varlen 打包后的 token 总数 |
| `H_v` | Value/Output/State head 数 |
| `V` | Value/Output 单 head 特征维 |
| `D` | 不区分 Q/K 与 V 时使用的通道维 |
| `W` | 一维卷积核宽度 |
