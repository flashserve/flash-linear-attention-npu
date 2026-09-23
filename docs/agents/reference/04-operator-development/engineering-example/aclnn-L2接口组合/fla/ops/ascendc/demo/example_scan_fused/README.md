# ExampleScanFused 示例算子说明（形态 C：只有 L2）

<!--
示例文件：fla/ops/ascendc/demo/example_scan_fused/README.md

注意事项：
  1. 本算子没有 def、没有 op_kernel：它只把已有算子的 L0 拼成一个公开入口。
  2. 本文件必须写清三件事：组合了哪些算子、各自的输入输出如何衔接、什么条件下回落/报错。
  3. 输出布局规则以本文件为准；被组合算子的内部布局不在这里重复定义，只写"如何转换"。
  4. 支持范围要写死（dtype/layout/chunk_size/连续性），不支持时返回 ACLNN_ERR_PARAM_INVALID。
-->

## 组合关系

```text
ExampleScanFused(x, g, ...) =
    ExampleScan        （主实现，产出 y/state/x_norm）
      -> ExampleScanTail（尾部归一化，产出 tail）
```

两次调用在同一个 aclOpExecutor 内登记，一次 `GetWorkspaceSize` + 一次 launch 完成；
中间张量由 executor 内部持有，不对外暴露。

## 输入输出

| 参数 | 必选/可选 | 说明 |
| --- | --- | --- |
| `x`、`g` | 必选 | 与 `ExampleScan` 相同 |
| `a_log`、`initial_state`、`cu_seqlens`、`chunk_indices` | 可选 | 直接透传给 `ExampleScan` |
| `layout`、`scale`、`chunk_size`、`epsilon` | 属性 | 与 `ExampleScan` 相同 |
| `y` | 必选输出 | 来自 `ExampleScan` |
| `state`、`x_norm` | 可选输出 | 来自 `ExampleScan`；同时给出或同时为空 |
| `tail` | 可选输出 | 来自 `ExampleScanTail`，需要尾部归一化时给出 |

## 已知限制

1. `x` 必须为 BF16、`D=128`、`chunk_size=64`、公开输出连续；其它组合返回
   `ACLNN_ERR_PARAM_INVALID` 并提示改用 `example_scan` 单算子入口。
2. 本算子**不支持** `output_mode` 属性：档位由公开输出指针组合推导（`y` / `save` / `save+tail`）。
3. 依赖算子未编译时（只编本算子），组合入口会在 tiling/launch 阶段报错并指向缺失依赖；
   不要把这种情况静默降级成单算子路径。
4. 三个 SoC 共用同一入口；平台差异在被组合算子的 kernel 里，不在本层。
5. 依赖算子的配置与 kernel 产物必须随本算子一起构建与打包：过滤构建（只列本算子）时必须按
   `<算子>_depends` 展开依赖闭包，否则会在调用本入口时报 `aclnnStatus=561103` +
   `Config_Error(EZ1013): ... the JSON configuration file of operator ... cannot be found`。
   这是依赖配置缺失，不是数值问题；发布前要用一次"只列本算子"的过滤构建验证包自洽。
