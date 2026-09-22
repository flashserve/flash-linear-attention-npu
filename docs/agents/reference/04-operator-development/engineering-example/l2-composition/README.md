# 示例：组合入口 / 只有 L2 接口的算子

> 对应 [`../../engineering-structure.md`](../../engineering-structure.md) §5.2 与 §5.4。
> 真实蓝本：[`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd/`](../../../../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd)（形态 C）、
> [`fla/ops/ascendc/kda/chunk_kda_fwd/`](../../../../../../fla/ops/ascendc/kda/chunk_kda_fwd)（形态 B 的 V2 与 L0 组合）。

## 1. 两种子形态

| | 形态 B：给已有算子加 V2 入口 | 形态 C：只有 L2 的组合算子 |
| --- | --- | --- |
| 触发场景 | 已发布入口（V1）不能改，但需要新开关/新可选输出/新场景 | 整套计算能由其它算子的 L0 拼出来 |
| 本目录文件 | [`.../example_scan/op_host/op_api/aclnn_example_scan_v2.h`](fla/ops/ascendc/demo/example_scan/op_host/op_api/aclnn_example_scan_v2.h)（声明；实现追加在 V1 的 `aclnn_example_scan.cpp` 尾部） | [`.../example_scan_fused/`](fla/ops/ascendc/demo/example_scan_fused/) 整棵目录 |
| def | 沿用 V1 的 def，不动 | **没有 def**（没有新 kernel 就没有新原型） |
| op_kernel | 不动 | **没有 op_kernel** |
| CMakeLists | 不改（只新增一个头文件） | `op_host/CMakeLists.txt`：`add_op_to_compiled_list()` + `set(<算子>_depends "...")` + `add_modules_sources(...)` |
| 测试 | V2 场景用例 + V1 回退用例 | 组合入口的端到端用例；被组合算子各自的 ATK 覆盖表由它们维护 |

## 2. 必备件（形态 C）

```text
fla/ops/ascendc/<模块>/<算子>/
|-- README.md                     # 组合关系、支持范围、与回落入口的关系
`-- op_host/
    |-- CMakeLists.txt            # 只有三行：add_op_to_compiled_list / depends / add_modules_sources
    `-- op_api/
        |-- aclnn_<算子>.h        # 公开 L2 接口
        `-- aclnn_<算子>.cpp      # 组合实现（l0op:: 调用其它算子的 L0）
```

算子发现规则（`cmake/func.cmake` 的 `op_add_subdirectory`）：构建系统用
`GLOB_RECURSE fla/ops/ascendc/CMakeLists.txt` 找算子目录，算子名 = `op_host` 的上一级目录名，
并要求该目录下有 `op_host/CMakeLists.txt` 且内部调用过 `add_op_to_compiled_list()`。因此：

1. 形态 C 的算子**必须**有 `op_host/CMakeLists.txt`，否则整个目录不进构建（也不会报错）；
2. 打开 `ENABLE_TEST` 时，构建系统要求 `${OP_DIR}/tests/CMakeLists.txt` 存在，否则该算子被跳过；
   组合算子若不参与单测，仍需在该目录说明这一点，避免 CI 静默跳过；
3. 没有 `def` 时 `ACLNNTYPE aclnn_exclude` 依然要写，表示不为该算子生成 aclnn 原型。

## 3. 禁止项

1. 不要在组合算子里复制被组合算子的 kernel 或数学实现；
2. 不要在组合入口里新增"只在组合路径生效"的隐藏参数（用户看到两个接口却只对一个生效）；
3. 不要把可选性写进 def（形态 C 根本没有 def）；可选输出仍然只在 L2 用可空描述符表达；
4. 不要在 L2 里解释 autograd 重计算策略——由 Python/legacy 层决定传哪些输出指针；
5. 不要在缺依赖时静默降级：依赖没编译出来要在 tiling/launch 阶段明确报错，并指向缺失依赖算子。

## 4. 组合入口的 L2 只做四件事

1. 入参校验（支持范围、dtype/layout、元数据单调性）；
2. 按组合顺序调用被组合算子的 L0（同一个 executor，一次 `GetWorkspaceSize` 登记全部任务）；
3. 把内部张量按公开输出布局拼接（`l0op::ViewCopy` / `ReuseOrAlloc`），必要时做一次明确的 layout 转换；
4. 把 workspace 总量交给调用方（`uniqueExecutor->GetWorkspaceSize()`）。

细节见本目录两个实际文件里的注意事项：
[`aclnn_example_scan_fused.h`](fla/ops/ascendc/demo/example_scan_fused/op_host/op_api/aclnn_example_scan_fused.h)、
[`aclnn_example_scan_fused.cpp`](fla/ops/ascendc/demo/example_scan_fused/op_host/op_api/aclnn_example_scan_fused.cpp)。
