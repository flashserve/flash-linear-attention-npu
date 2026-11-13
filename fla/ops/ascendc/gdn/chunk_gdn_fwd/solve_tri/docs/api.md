# SolveTri 接口契约

本文记录当前独立 SolveTri 的公开参数与边界。调用方式见 [aclnnSolveTri.md](aclnnSolveTri.md)，内部实现见 [design.md](design.md)。

## 接口与计算

ACLNN 分两步调用：`aclnnSolveTriGetWorkspaceSize(x, cuSeqlens, chunkIndices, layout, xOut, workspaceSize, executor)` 返回 workspace 大小与执行器，`aclnnSolveTri(workspace, workspaceSize, executor, stream)` 在指定流执行。Python 稳定入口为 `fla_npu.ops.ascendc.npu_solve_tri(x, *, cu_seqlens=None, chunk_indices=None, layout="bsnd")`，返回新分配的输出。

按 batch、head、序列和 chunk 独立计算 `Y = (I + A)^-1`。输入 `A` 的对角线及上三角应为零；算子不验证矩阵内容。最后一个 chunk 不足 `C` 行时，按实际有效行数求解并写回有效的方阵区域；无效输出区域不作数值保证。

## 平台隔离

本次新增的参数及变长元数据校验、结构化错误、workspace 计算和 kernel 优化仅用于 A2/A3。A5 保留 `546859c7` 的原有 host 检查、workspace 计算和 kernel 行为；下述合法输入约束仍须满足，但不承诺 A5 对非法输入提供新增的拦截或错误文本。A5 原有 workspace 容量不属于本次验证结论。

注册配置使用独立的 `OpAICoreConfig`：A2/A3 的 `PrecisionReduceFlag` 为 `false`，A5 保留基线的 `true`。该字段属于混合精度类型选择策略，不是内部 FP32 运算开关。

## 输入和输出

| 参数 | 约束 |
| --- | --- |
| `x` | 必需的 Device tensor，dtype 为 FLOAT16 或 BFLOAT16，format 为 ND，最后一维 `C` 为 16、32、64 或 128。ACLNN 内部将非连续输入转为连续输入。 |
| `xOut` | ACLNN 必需的 Device tensor，shape 和 dtype 与 `x` 一致，必须连续。Python 包装层分配同 shape、同 dtype 的输出。 |
| `layout` | ACLNN 可使用 `bsnd`、`bnsd`、`bhtd`、`tnd`、`ntd`，缺省为 `bsnd`；`bhtd` 是 `bnsd` 的兼容别名。Python 入口使用 `bsnd`、`bnsd`、`tnd`、`ntd`。 |
| `cuSeqlens` / `cu_seqlens` | 变长时必需，INT64 数组，长度至少为 2；定长时必须为空。 |
| `chunkIndices` / `chunk_indices` | 变长时必需，INT64 数组，展平的 `(seq_id, chunk_id)` 对；定长时必须为空。 |
| `workspaceSize`, `executor` | `GetWorkspaceSize` 的输出指针，均不得为空。 |

| 布局 | `x` 的 shape |
| --- | --- |
| `bsnd` | `[B, T, H, C]` |
| `bnsd` / `bhtd` | `[B, H, T, C]` |
| `tnd` | `[total_T, H, C]` |
| `ntd` | `[H, total_T, C]` |

`B`、`T`、`H`、`total_T` 必须为正，定长布局为 4 维，变长布局为 3 维。支持的目标产品为 A2、A3 和 A5；具体构建与设备验证结果以对应测试记录为准。

## 变长元数据

`cuSeqlens[0] == 0`，末项为 `total_T`，其余项非递减。相邻相等表示空序列，空序列不产生 chunk；当前实现要求 `total_T > 0`，因此不支持全空输入。

序列 `i` 长度为 `L = cuSeqlens[i+1] - cuSeqlens[i]`，其 chunk 个数为 `L / C + (L % C != 0)`。`chunkIndices` 包含全部非空 chunk 各一次，`seq_id` 为 `cuSeqlens` 中的原始序列下标，`chunk_id` 从 0 开始；各对允许重排。示例：`cuSeqlens=[0, 0, 65]` 且 `C=64` 时，索引可为 `[1, 0, 1, 1]`，也可为 `[1, 1, 1, 0]`。

## 异常

A2/A3 的新增校验在调度计算前对必选指针为空或参数不满足约束返回对应的参数错误。A5 保留基线的检查范围和错误映射，不承诺新增校验行为。返回状态分类见 [aclnnSolveTri.md](aclnnSolveTri.md)。设备执行错误和异步错误由 ACL 运行时返回或报告。
