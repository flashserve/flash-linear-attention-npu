# CausalConv1d

## 1. 功能概述

GDN 输入预处理的因果一维卷积。run_mode=0 执行定长/变长序列前向并维护卷积状态，run_mode=1 执行 decode/投机更新；可选 SiLU 激活和 head layout 转换。

## 2. 数学定义

对通道 d 和时间 t：

```text
z[t,d] = bias[d] + sum(j=0..W-1, weight[j,d] * x[t-j,d])
y[t,d] = z[t,d]                    activation_mode=0
y[t,d] = z[t,d] * sigmoid(z[t,d]) activation_mode=1
```

`t-j` 越过序列起点时读取该序列的 `conv_states`；update 模式只提交有效/已接受 token 并原地滚动状态。

## 3. 输入、输出和属性

本文使用的 Shape 符号统一引用[GDN 模型符号表](../../README.md#model-shape-symbols)，不在算子 README 中重复定义。

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,T,D] 或 [T,D]` | FP16/BF16 | BSH/TH | 输入序列 |
| `weight` | 必选 | `[W,D]` | FP16/BF16 | ND | depthwise 卷积权重 |
| `bias` | 可选 | `[D]` | FP16/BF16 | ND | 偏置 |
| `conv_states` | 必选/可变 | `[D_s,L_s,D]` | FP16/BF16 | ND | 历史输入状态，原地更新 |
| `query_start_loc` | 可选 | `[B+1]` | INT64 | ND | 变长序列边界 |
| `cache_indices` | 可选 | `[B]` | INT64 | ND | 序列到状态槽的映射 |
| `initial_state_mode` | 可选 | `[B]` | INT64 | ND | 是否使用已有初始状态 |
| `num_accepted_tokens` | 可选 | `[B]` | INT64 | ND | 投机解码接受数 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `y` | `与 x 对应；head_num>0 时为 BNSD/NTD` | 与 x 一致 | 卷积输出 |
| `conv_states` | `[D_s,L_s,D]` | 与输入一致 | 原地更新后的状态 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `activation_mode` | int | `0` | `{0, 1}` | 0=无激活，1=SiLU |
| `pad_slot_id` | int | `-1` | int64 可表示范围 | 与 `cache_indices` 中表示跳过的槽位值一致 |
| `run_mode` | int | `0` | `{0, 1}` | 0=forward，1=update |
| `head_num` | int | `0` | `>=0` | forward 输出拆分 head；0 保持 BSH/TH，正值还须满足整除约束 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | x/weight/bias/state/y 为 FP16/BF16；元数据 INT64 |
| Format/Layout | 输入 BSH/TH；head_num>0 时输出 BNSD/NTD |
| 模式 | 定长/变长序列 forward、decode/update、投机接受 token |



## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.causal_conv1d` |
| aclnn | `aclnnCausalConv1dGetWorkspaceSize` / `aclnnCausalConv1d` |
| Ascend C `<<<>>>` | `causal_conv1d<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_causal_conv1d` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/causal_conv1d/accuracy/test_causal_conv1d.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/causal_conv1d.json`；覆盖定长/变长序列 forward、decode/update、投机接受 token。
- 参考实现：`torch_depthwise_causal_conv1d_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- 卷积宽度 W 仅支持 2/3/4，特征维 D 必须为 16 的倍数。
- activation_mode 仅支持 0/1，run_mode 仅支持 0/1。
- head_num>0 仅用于 forward，且必须整除 D；拆分后的 D/head_num 仍须为 16 的倍数。
- conv_states 的 L_s 至少为 W-1；rank-3 投机 update 还要求 L_s >= (W-1)+(T-1)。
- num_accepted_tokens 仅在 run_mode=1 且 W=4 时实现，值域为每条逻辑序列的 [0,token_count]。
- conv_states 为可变输入；非连续状态仅在 CANN >= 9.1 支持。
- rank-2 forward 必须提供 query_start_loc；query_start_loc、cache_indices、initial_state_mode 和接受数的长度必须与逻辑序列数一致。
- initial_state_mode 仅用于 run_mode=0 且元素只能为 0/1；cache_indices 只能选择有效状态槽或 pad_slot_id。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=causal_conv1d python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/causal_conv1d/accuracy/test_causal_conv1d.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/causal_conv1d/routes/`，均使用同一份 JSON 规格。
