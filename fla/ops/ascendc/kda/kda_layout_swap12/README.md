# KdaLayoutSwap12

## 1. 功能概述

KDA 内部的连续布局转置算子。rank3 交换维 0/1，rank>=4 交换维 1/2，其余维保持顺序；可选 dependency 只建立 executor 调度依赖，不参与数值计算。

## 2. 数学定义

```text
rank(x) = 3: y[j,i,...]   = x[i,j,...]
rank(x) >= 4: y[b,j,i,...] = x[b,i,j,...]
```

这是精确的数据重排，不执行浮点算术。`dependency` 的 shape/dtype 不改变 y；它仅让 L2
图显式等待前序生产者，防止临时 tensor 生命周期被 executor 提前复用。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[D_0,D_1,D_2,...]，rank>=3` | FP16/BF16/FP32 | ND | 连续化后参与重排 |
| `dependency` | 可选 | `任意 tensor` | 任意 | ND | 仅调度依赖，不读取值 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `y` | `rank3: [D_1,D_0,D_2]；rank>=4: [D_0,D_2,D_1,...]` | 与 x 相同 | 连续转置结果 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `无` | - | `-` | - | 无公开属性 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | FP16/BF16/FP32，输入输出一致 |
| Format/Layout | ND；rank3 交换维 0/1，rank>=4 交换维 1/2 |
| 模式 | rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy |



## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.kda_layout_swap12` |
| aclnn | `aclnnKdaLayoutSwap12GetWorkspaceSize` / `aclnnKdaLayoutSwap12` |
| Ascend C `<<<>>>` | `kda_layout_swap12<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | 未实现 |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/kda_layout_swap12/accuracy/test_kda_layout_swap12.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/kda_layout_swap12.json`；覆盖 rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy。
- 参考实现：`torch_permute_contiguous_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- x rank 必须至少为 3；y rank、dtype 及交换后的每一维必须与 x 精确对应。
- 输入在 L2 中先 contiguous；当前 API 总是创建连续 y。
- dependency 只表达执行顺序，不提供数据、dtype 或 shape 语义，不得依赖其值改变输出。
- aclnn/Python 通过 executor 输入依赖排序；`<<<>>>` 直调不读取 dependency，调用者必须在同一 stream 上先发射其生产者。
- 所有维度必须为正；host 在进入 tiling 前拒绝空维度，kernel 的 usedCoreNum 至少为 1。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=kda_layout_swap12 python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/kda_layout_swap12/accuracy/test_kda_layout_swap12.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/kda_layout_swap12/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Kimi Delta Attention (KDA)
- 模型级符号表：[KDA 模型符号表](../README.md#model-shape-symbols)
- 符号表版本：`kda-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size |
| `T` | dense 序列长度或 varlen 打包 token 总数 |
| `H_v` | Value/Output head 数 |
| `K` | Query/Key 单 head 特征维 |
| `V` | Value 单 head 特征维 |
| `N_c` | 当前调用中的 chunk 总数 |
| `D_0` | rank3 的第一交换维；rank>=4 时为 batch 维 |
| `D_1` | 待交换的第一维 |
| `D_2` | 待交换的第二维 |
| `D_0` | rank3 的第一交换维；rank>=4 时为 batch 维 |
| `D_1` | 待交换的第一维 |
| `D_2` | 待交换的第二维 |
