# <op_name>

<!-- 将 <op_name>、<OpName> 等占位符替换为实际算子信息，删除不适用内容。Hint 仅用于展示填写粒度，完成文档后应删除。 -->

## 1. 功能概述

说明算子的功能、典型使用场景，以及它在上层算法或模型中的作用。

> **Hint：** 以 `chunk_bwd_dv_local` 为例，应说明它位于 Gated Delta Rule 反向过程，根据 `q`、`k`、`dO` 和门控 `g` 计算 Value 的本地梯度 `dV_local`，而不是只写“用于反向计算”。

## 2. 数学定义

给出计算公式，并解释公式中的符号、维度和输入输出关系。

> **Hint：** `chunk_bwd_dv_local` 可按三个阶段描述：先计算 `Ws = K @ Q^T`，再对 `Ws` 应用门控指数、上三角 mask 和逐元素乘，最后计算 `dV_local = Ws_gated @ dO`；同时说明 `H_v % H_k == 0` 和 `hRatio = H_v / H_k`。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Format/Layout | 取值范围 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `<input>` | 必选 | `<shape>` | `<dtype>` | `<format>` | `<range>` | `<description>` |

> **Hint：** `chunk_bwd_dv_local` 的必选输入写为 `q/k: [B,H_k,T,K]`、`dO: [B,H_v,T,V]`、`g: [B,H_v,T]`；`q/k/dO` 支持 FP16/BF16，`g` 还支持 FP32。Shape 中只使用符号变量，固定维度取值放到“已知限制”。`g_gamma`、`A`、`cu_seqlens`、`chunk_indices` 应分别写明是否实现及组合约束。

### 3.2 输出

| 名称 | Shape | Dtype | Format/Layout | 说明 |
| --- | --- | --- | --- | --- |
| `<output>` | `<shape>` | `<dtype>` | `<format>` | `<description>` |

> **Hint：** `chunk_bwd_dv_local` 输出写为 `dV: [B,H_v,T,V]`，dtype 与 `dO` 一致；`V` 的固定支持值写入“已知限制”，不写进 Shape。

### 3.3 属性

| 名称 | 必选/可选 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `<attr>` | `<required>` | `<type>` | `<default>` | `<range>` | `<description>` |

> **Hint：** `chunk_bwd_dv_local` 的 `scale` 为必选浮点属性，通常取 `1/sqrt(K)`；`chunk_size` 为必选整数属性，具体支持值写入“已知限制”。

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`） |
| Dtype | `<dtypes>` |
| Format/Layout | `<formats/layouts>` |
| Shape | `<shape range>` |
| fixed/varlen | `<support status>` |

说明状态输入输出、padding、无效区域、尾块、空输入和其他边界语义，以及各平台存在的限制或差异。

> **Hint：** `chunk_bwd_dv_local` 同时支持 fixed 和 varlen。正文说明 varlen 需要同时提供 `cu_seqlens` 和 `chunk_indices`，并说明 `T` 非 `chunk_size` 整数倍和 `hRatio>1` 的行为；固定取值和 batch 限制写入“已知限制”。

## 5. 调用入口

实现类型：`<ascendc|triton>`

| 入口 | 支持情况 | 说明 |
| --- | --- | --- |
| `fla_npu.ops.ascendc` / `fla_npu.ops.triton` | 必须，二选一 | 与实现类型一致的 Python 主入口 |
| aclnn | Ascend C 算子必须 | Triton 算子删除本行 |
| Ascend C `<<<>>>` | Ascend C 算子必须 | Triton 算子删除本行 |
| `torch.ops.npu` | 可选 | 未实现时删除本行 |

完整接口和调用示例见 [API 文档](docs/api.md)，实现方案见 [设计文档](docs/design.md)。

> **Hint：** `chunk_bwd_dv_local` 的实现类型为 `ascendc`，因此填写 aclnn 入口 `aclnnChunkBwdDvLocal*`、Python 入口 `fla_npu.ops.ascendc.chunk_bwd_dv_local`、Ascend C `<<<>>>` 直调，以及可选 legacy 入口 `torch.ops.npu.npu_chunk_bwd_dv_local`；不要再列 `fla_npu.ops.triton.chunk_bwd_dv_local`。

## 6. 精度与性能

- 精度标杆：`<reference>`
- 精度阈值：`<tolerance and rationale>`
- 性能目标：`<target shapes and target>`
- Triton 对比范围：`<benchmark matrix>`

> **Hint：** `chunk_bwd_dv_local` 的矩阵至少覆盖 fixed/varlen、支持的 Q/K/V dtype、支持的 gate dtype、全部 `chunk_size`、全部 `V` 和不同 `hRatio`。若 Ascend C 替换 Triton，应列出相同输入下的 profiling 结果。

## 7. 已知限制

- `<unsupported case or limitation>`

> **Hint：** 不要只写“部分场景不支持”。例如应明确写出 `K` 仅支持 128、`V` 仅支持 128/256、`chunk_size` 仅支持 64/128、varlen 仅支持 `B=1`，以及预留输入 `g_gamma`、`A` 当前必须为空。

## 8. 构建与验证

```bash
# 填写构建命令
```

```bash
# 填写 API 示例运行命令
```

```bash
# 填写测试命令
```

> **Hint：** 命令应覆盖 wheel/OPP 构建、`docs/api.md` 中各通路示例、`tests/op_cases/<op_name>.json` 主精度矩阵和性能测试；只写通用 `pytest` 而不说明测试文件或筛选方式不够。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

### 9.1 模型符号基线

- 模型/算法族：`<model_name>`
- 模型级符号表：[`<model_name>` 模型符号表](<model-symbol-table-link>)
- 符号表版本：`<version>`

本附录使用的模型级符号名称和语义必须与模型根目录 README 中的权威符号表一致。算子未使用的模型符号可以省略；算子特有符号可以追加，但不得复用已有符号表达不同语义。

> **Hint：** GDN 模型下的所有算子应统一 `B`、`H_k`、`H_v`、`T`、`K`、`V`、`C` 等符号的含义；KDA 模型可以维护自己的符号集合，但所有 KDA 算子必须引用同一份 KDA 模型符号表。若模型符号发生变化，应在同一个 PR 中同步模型根 README 和受影响算子 README。

### 9.2 本算子使用的符号

| 变量 | 语义 | 示例关联维度 |
| --- | --- | --- |
| `B` | Batch size | 所有主输入和输出的第 0 维 |
| `H_k` | Q/K head 数 | `q`、`k` 的 head 维 |
| `H_v` | dO/dV head 数 | `dO`、`g`、`dV` 的 head 维 |
| `T` | 序列长度或当前张量承载的 token 数 | 序列维 |
| `K` | Q/K 单 head 特征维度 | `q`、`k` 的最后一维 |
| `V` | Value 单 head 特征维度 | `dO`、`dV` 的最后一维 |
| `C` | Chunk size；文中属性名可写为 `chunk_size` | 分块计算粒度 |

> **Hint：** `chunk_bwd_dv_local` 的 shape 可统一写成 `q/k=[B,H_k,T,K]`、`dO/dV=[B,H_v,T,V]`、`g=[B,H_v,T]`，并在正文说明 `H_v` 与 `H_k` 的关系。
