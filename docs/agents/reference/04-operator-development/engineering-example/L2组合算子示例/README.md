# 示例：组合入口 / 只有 L2 接口的算子

> 对应 [`../../engineering-structure.md`](../../engineering-structure.md) §5.2 与 §5.4。
> 真实蓝本：[`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd/`](../../../../../../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd)（形态 C）、
> [`fla/ops/ascendc/kda/chunk_kda_fwd/`](../../../../../../fla/ops/ascendc/kda/chunk_kda_fwd)（形态 B 的 V2 与 L0 组合）。

## 1. 两种子形态

| | 形态 B：给已有算子加 V2 入口 | 形态 C：只有 L2 的组合算子 |
| --- | --- | --- |
| 触发场景 | 已发布入口（V1）不能改，但需要新开关/新可选输出/新场景 | 整套计算能由其它算子的 L0 拼出来 |
| 本目录文件 | [`.../op_name/op_host/op_api/aclnn_op_name_v2.h`](fla/ops/ascendc/ops_classify/op_name/op_host/op_api/aclnn_op_name_v2.h)（声明；实现追加在 V1 的 `aclnn_op_name.cpp` 尾部） | [`.../op_name_fused/`](fla/ops/ascendc/ops_classify/op_name_fused/) 整棵目录 |
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

torch_custom/fla_npu/                  # 交付件布局与 torch_custom/fla_npu/README.md §1.1 一致
|-- csrc/src/stable_<算子>.cpp        # 新建：组合算子同样要 schema + 适配（一算子一文件）
|-- csrc/src/stable_ops.cpp           # 改：include 一行 + m.def/m.impl 两行
`-- fla_npu/ops/ascendc/
    |-- _stable.py                    # 改：真签名 wrapper（校验与默认值；场景选择不在这层）
    `-- __init__.py                   # 改：_ASCENDC_OPS 加一行 public 名
```

一个容易写错的点：**V1/V2 的场景选择在 C++ 适配层的 `run_` 里**，不在 Python wrapper 里
（仓内真实例子：`csrc/src/stable_chunk_kda_fwd.cpp` 内部在 `aclnnChunkKdaFwdV2` 与 `aclnnChunkKdaFwd`
之间分支）。Python wrapper 只做"schema 表达不了的参数校验 + 默认值补齐 + 透传"。

ATK 侧组合算子不需要自己的 TilingKey 覆盖表（没有自己的 kernel），但仍要有端到端用例：至少一条
走组合路径、一条走回落路径，并覆盖缺依赖时报错可定位的场景（见 §4）。

算子发现规则（`cmake/func.cmake` 的 `op_add_subdirectory`）：构建系统用
`GLOB_RECURSE fla/ops/ascendc/CMakeLists.txt` 找算子目录，算子名 = `op_host` 的上一级目录名，
并要求该目录下有 `op_host/CMakeLists.txt` 且内部调用过 `add_op_to_compiled_list()`。因此：

1. 形态 C 的算子**必须**有 `op_host/CMakeLists.txt`，否则整个目录不进构建（也不会报错）；
2. 测试不放算子目录：组合算子的用例补在仓库根 `tests/atk/<算子>/` 下；构建脚本里的 `ENABLE_TEST`
   探测逻辑以仓库实际约定为准，不要为了通过探测而在算子目录里加回 `tests/`；
3. 没有 `def` 时 `ACLNNTYPE aclnn_exclude` 依然要写，表示不为该算子生成 aclnn 原型。

## 3. 禁止项

1. 不要在组合算子里复制被组合算子的 kernel 或数学实现；
2. 不要在组合入口里新增"只在组合路径生效"的隐藏参数（用户看到两个接口却只对一个生效）；
3. 不要把可选性写进 def（形态 C 根本没有 def）；可选输出仍然只在 L2 用可空描述符表达；
4. 不要在 L2 里解释 autograd 重计算策略——由 Python/legacy 层决定传哪些输出指针；
5. 不要在缺依赖时静默降级：依赖没编译出来要在 tiling/launch 阶段明确报错，并指向缺失依赖算子。

## 4. 依赖闭包与过滤构建（最容易被漏掉的一条）

组合入口的依赖要在**两个地方**同时生效，缺任一处都会"编译、安装都成功，跑到组合入口才失败"：

| 侧 | 位置 | 作用 |
| --- | --- | --- |
| 构建侧 | 本算子 `op_host/CMakeLists.txt` 里按 `<算子>_depends` 展开的 `foreach` + `add_subdirectory` | 过滤构建（`FLA_NPU_OPS` / `--ops` 只列主算子）时，把依赖算子的 `op_host`（def/config/tiling）带进本次构建 |
| 打包侧 | `cmake/custom_build.cmake` 按同一份 `${<算子>_depends}` 安装依赖算子的 `op_kernel` 产物 | 让依赖算子的 JSON 配置与 kernel 产物一起进包并注册 |

漏掉的症状固定为运行期：`aclnnStatus=561103` 且
`Config_Error(EZ1013): ... the JSON configuration file of operator aclnn<Op>_0_<Dep> cannot be found`、
`AclOpKernelInit failed, opType: <Dep>`。这是依赖配置缺失，**不是算子数值或 kernel 问题**，排查时不要先看精度。
真实案例：`chunk_kda_fwd` 的 V2 组合入口缺少 `chunk_fwd_h` 的配置（Issue #695，由 #705 系列合入修复）。

验证要求：改动组合入口或 `<算子>_depends` 后，做一次"只列主算子"的过滤构建，并跑一条走组合路径的
端到端用例；全量构建通过**不能**证明过滤构建自洽（该缺陷在全量构建下不出现）。

## 4. 组合入口的 L2 只做四件事

1. 入参校验（支持范围、dtype/layout、元数据单调性）；
2. 按组合顺序调用被组合算子的 L0（同一个 executor，一次 `GetWorkspaceSize` 登记全部任务）；
3. 把内部张量按公开输出布局拼接（`l0op::ViewCopy` / `ReuseOrAlloc`），必要时做一次明确的 layout 转换；
4. 把 workspace 总量交给调用方（`uniqueExecutor->GetWorkspaceSize()`）。

细节见本目录两个实际文件里的注意事项：
[`aclnn_op_name_fused.h`](fla/ops/ascendc/ops_classify/op_name_fused/op_host/op_api/aclnn_op_name_fused.h)、
[`aclnn_op_name_fused.cpp`](fla/ops/ascendc/ops_classify/op_name_fused/op_host/op_api/aclnn_op_name_fused.cpp)。
