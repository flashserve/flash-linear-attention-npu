# OpName API

<!--
示例文件：fla/ops/ascendc/ops_classify/op_name/docs/api.md

注意事项：
  1. 本文是全部公开接口的唯一定义来源：Python 入口、aclnn L2 签名、返回码、可选输出语义都在这里。
  2. 可选输出必须写清"传 nullptr 时的语义"，以及"传与不传是否影响计算结果"（本算子：不影响，逐位一致）。
  3. 返回码章节要与代码里的 CHECK_COND 分支逐条对应，写清触发条件与实际报错内容。
  4. 布局规则要区分"输入由 layout 解释"和"输出固定布局"；不要把输出写成跟随 layout。
  5. 不新增独立的 aclnn*.md；本文件是唯一接口文档。
-->

## Python 入口

```python
from fla_npu.ops.ascendc import op_name

y, state, x_norm = op_name(
    x, g, a_log=None, initial_state=None,
    cu_seqlens=None, chunk_indices=None,
    layout="BSND", scale=1.0, chunk_size=64, epsilon=1e-6,
    return_saved=False,        # True 时请求 state 与 x_norm
)
```

`return_saved=False` 时 `state`、`x_norm` 返回 `None`，且结果与 `True` 时逐位一致，只有是否落公开 GM 的区别。

## aclnn L2

```cpp
aclnnStatus aclnnOpNameGetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon,
    const aclTensor *yOut,          // 必选
    const aclTensor *stateOut,      // nullptr = 本次不导出
    const aclTensor *xNormOut,      // nullptr = 本次不导出
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnOpName(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream);
```

输出指针组合只支持两档，其它组合返回 `ACLNN_ERR_PARAM_INVALID`：

| 组合 | 档位 | 行为 |
| --- | --- | --- |
| 只传 `yOut` | `none` | 只计算并写出 `y` |
| `yOut` + `stateOut` + `xNormOut` | `save` | 额外写出状态与归一化中间量 |

## 布局

1. `layout` 只解释 `x/g` 输入；`y` 固定为 BSND 或 TND。
2. `state` 固定 `[B,H,chunk_count,D]`，与 `layout` 无关。
3. `x_norm` 固定与 `x` 相同的逻辑布局，内部按 head-major 计算，L2 导出时按输入布局写出。

## 返回值

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | `x`、`g`、`yOut` 为空 |
| `ACLNN_ERR_PARAM_INVALID` | `D != 128`；`chunk_size` 非 64/128；`epsilon <= 0`；`score` 非正；`cu_seqlens` 首元素非 0 / 非单调 / 末元素与总 token 数不符；`chunk_indices` 未与 `cu_seqlens` 同时给出；输出指针组合不是 `none`/`save` |
| `ACLNN_ERR_INNER_NULLPTR` | 内部张量或 workspace 申请失败 |
