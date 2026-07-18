# SolveTri

## 1. 功能概述

对每个 chunk 的严格下三角矩阵 A 计算 `(I+A)^-1`，用于 WY 表示求解。输入最后一维保存当前 token 行的 chunk 列，输出保持相同布局。

## 2. 数学定义

对每个 batch/head/chunk，取有效阶数 M 的严格下三角矩阵 A：

```text
Y = inverse(I_M + tril(A, diagonal=-1))
```

实现采用分块前代/三角逆；尾块只对有效 M 求逆，padding 列按接口约定写零。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | `[B,H_v,T,C]、[B,T,H_v,C] 或 [T,H_v,C]` | FP16/BF16 | BHTD/BSND/TND | 严格下三角 A 的行存储 |
| `cu_seqlens` | TND 必选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | TND 必选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `x_out` | `与 x 相同` | 与 x 相同 | (I+A) 的 chunk-wise 逆 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `bsnd` | `{"bhtd", "bsnd", "tnd"}` | 仅支持小写取值 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | FP16/BF16 |
| Format/Layout | BHTD `[B,H_v,T,C]`、BSND `[B,T,H_v,C]`、TND `[T,H_v,C]`；layout 字符串必须小写 |
| 模式 | dense BHTD/BSND 与变长序列 TND，支持尾块 |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.solve_tri` |
| aclnn | `aclnnSolveTriGetWorkspaceSize` / `aclnnSolveTri` |
| Ascend C `<<<>>>` | `solve_tri<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_solve_tri` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/solve_tri/accuracy/test_solve_tri.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/solve_tri.json`；覆盖 dense BHTD/BSND 与变长序列 TND，支持尾块。
- 参考实现：`torch_linalg_triangular_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 在相同 shape/dtype/layout、warmup 和迭代配置下对比 `fla_npu.ops.triton.solve_tril_npu`；Ascend C 必须更快，仓内主 example 已切换到 Ascend C。

## 7. 已知限制

- 矩阵阶/最后一维 C 支持 16/32/64/128。
- layout 仅支持小写 `bhtd`、`bsnd`、`tnd`；TND 必须提供两个变长序列索引，定长布局 不接受变长序列索引。
- 输入必须表示严格下三角 A；对角线由算子加单位阵。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=solve_tri python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/solve_tri/accuracy/test_solve_tri.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/solve_tri/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；变长序列打包场景通常为 1 |
| `N` | 变长序列的逻辑序列数 |
| `T` | 定长序列长度或变长序列打包后的 token 总数 |
| `H_v` | Value/Output/State head 数 |
| `C` | chunk_size |
| `N_c` | 当前调用中的 chunk 总数 |
