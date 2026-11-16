# DFX 拦截与报错检视 reference

依据：[`opbase include/op_common/log/log.h`](https://gitcode.com/cann/opbase/blob/master/include/op_common/log/log.h)。

## 1. 根因

上游 `CHECK_RET` 使用 `OP_LOGE_WITHOUT_REPORT`，只打印 `Check <condition> failed` 并返回状态；
上游 `CHECK_COND` 使用通用 `OP_LOGE`，上报的是通用内部错误，而不是参数级结构化消息。调用方最终常只
看到 `aclnnStatus=161002`，无法知道哪个输入、实际值和期望值不匹配。

public constraint 的拦截必须优先使用 `OP_LOGE_FOR_*`，以便同时完成：

1. 输出可读日志；
2. 通过 `REPORT_PREDEFINED_ERR_MSG` 把结构化键值传到上层；
3. 让负向测试通过 `aclGetRecentErrMsg()` 或异常文本稳定断言；
4. 让文档中的返回码和实际分支一一对应。

## 2. 宏选择

| 约束 | 首选宏 |
| --- | --- |
| 必选参数为空 | `OP_LOGE_WITH_INVALID_INPUT(entity, param)` |
| shape 整体不符 | `OP_LOGE_FOR_INVALID_SHAPE(entity, param, actual, expected)` |
| shape 仅为 dim/rank 不符 | `OP_LOGE_FOR_INVALID_SHAPEDIM` 或 `_WITH_REASON` |
| 元素数量不符 | `OP_LOGE_FOR_INVALID_SHAPESIZE` |
| format/layout 不符 | `OP_LOGE_FOR_INVALID_FORMAT` |
| dtype 不符 | `OP_LOGE_FOR_INVALID_DTYPE` |
| 属性或标量取值不符 | `OP_LOGE_FOR_INVALID_VALUE`；条件组合复杂时用 `_WITH_REASON` |
| list/tensor 数量不符 | `OP_LOGE_FOR_INVALID_LISTSIZE` / `INVALID_TENSORNUM` |
| 多个参数之间的约束 | `OP_LOGE_FOR_INVALID_*S_WITH_REASON` |
| 无法归类的参数错误 | `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON` |

`_WITH_REASON` 也必须带实际值或明确触发组合，不能把“参数非法”作为 reason。

## 3. 本 skill 的检视落点

本 PR 只交付检视 skill，不修改 aclnn、tiling、CMake 或公共头文件。对当前仓库执行检视时：

1. aclnn/tiling 中只要仍有 `CHECK_RET`、`CHECK_COND` 或通用 `OP_LOGE` 承担 public 参数约束，
   就应报告为 `DFX-MESSAGE` 风险，因为该路径无法稳定携带参数名、实际值、期望值和原因。
2. 已改用 typed `OP_LOGE_FOR_*` 或能够在调用点保留结构化 report 的分支，再核对实际值、期望值、
   返回码和负向用例。
3. 如果调用方确实需要在全仓统一兼容层中重定义 guard，应另提实现 PR，说明 CMake target、include
   顺序、旧 CANN fallback、A2/A3/A5 编译和负向测试；该实现不属于本 skill PR。

因此本 PR 的结论是“能检测并指出这类问题”，不是“已经整改仓内 aclnn/tiling 代码”。

## 4. 双向核对

对每个文档约束按五列表核对：

| 文档约束 | 代码拦截位置 | 结构化消息 | 返回码 | 负向用例 |
| --- | --- | --- | --- | --- |
| 例：varlen 仅支持 `B=1` | `aclnn_<op>.cpp` | `FLA_NPU_DFX_*` 或 `OP_LOGE_FOR_*` | `161002` | 实际值 B=2 |

同时从代码反向检查：每个 `OP_LOGE_FOR_*`/guard 是否能在 README 或 `docs/api.md` 找到同等语义。

平台差异类错误必须写明“当前平台类别 + 该平台期望值”，不能只写单个参数错误。模式推导类错误
要说明由哪些输入组合推导出当前模式。

## 5. 验证

1. host 参数校验必须在进入 tiling/kernel 前完成；
2. 负向用例同时断言返回码和错误消息；
3. executor 捕获异常后自行 PASS 时，必须核对异常文本/ACLNN 错误码；
4. `aclGetRecentErrMsg()` 可能被运行时读取并清空，测试应优先检查异常文本，再以全局缓冲为辅；
5. 只看到返回码没有消息，判 DFX 未通过；
6. 日志、PR、issue 不得出现内网地址、账号、绝对路径、日志路径或 token。

## 6. 常见问题

- `CHECK_COND` 已写具体格式串，但没有 report，调用方仍只得到返回码；
- 只给 tiling 返回 `GRAPH_FAILED`，没有说明哪个输入或组合失败；
- `expected` 写“合法值”但不列实际允许集合；
- 同一约束在 Python、L0、L2、tiling 重复实现且文案不同；
- kernel 内做参数拦截；kernel 无法返回结构化 aclnn 错误；
- 文档列了返回码，代码没有对应分支；
- 负向用例只匹配 `161002`，任何错误都能 PASS。
