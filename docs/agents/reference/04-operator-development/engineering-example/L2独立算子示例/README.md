# 示例算子（形态 A：独立实现；`op_name` 为算子名占位符）

> 本目录把 [`../../engineering-structure.md`](../../engineering-structure.md) 落成实际文件。
> 路径里的 `ops_classify` 是算子类别占位符（实际替换为 `fla/ops/ascendc/` 下的真实类别，如 `gdn`），
> `op_name` 是算子名占位符：下面这个算子是**虚构**的，语义取"chunk 内扫描 + 可选保存中间量"，
> 只展示工程结构，代码不可编译，也不表示任何真实算子。
>
> 示例语义（便于理解字段命名）：
>
> - 输入：`x[B,H,T,D]`（BF16/FP16）、`g[B,H,T]`（FP32/BF16 门控）、`a_log`（可选）、
>   `initial_state`（可选）、`cu_seqlens`/`chunk_indices`（可选，变长）；
> - 输出：`y`（必选）、`state`（chunk 末状态，可选导出）、`x_norm`（反向需要的归一化中间量，可选导出）；
> - 档位：`none`（只要 `y`）/ `save`（额外导出 `state`、`x_norm`）；
> - 模板参数：`D_T_X`、`D_T_G`、`NORM_MODE`、`USE_STATE`、`OUTPUT_MODE`。

## 目录

```text
L2独立算子示例/                            # 形态 A 的示例根目录（英文小写连字符命名）
|-- README.md                          # 本文件
|-- fla/ops/ascendc/ops_classify/op_name/ # 算子工程本体（镜像真实路径）
|-- torch_custom/fla_npu/              # 调用层改动点
`-- tests/atk/op_name/            # 单算子看护资产索引
```

## 读法

1. 每个文件开头都有 `注意事项` 注释块：该文件**必须**满足的要求与常见错误；
2. 文件内代码是骨架，标 `// ...` 处按本算子补齐；
3. 建议顺序：`_def.cpp` → `_tiling.h` → `_output_mask.h` → `_tiling_processor.h` → `_tiling.cpp`
   → `_tiling_key.h` → `op_kernel/<算子>.cpp` → `arch22|arch35` → `op_api`（L0 → L2）
   → `torch_custom` → `tests/atk`。

## 这个示例刻意覆盖的规范点

| 规范点 | 示例中的体现 |
| --- | --- |
| 输出全部 `REQUIRED`，可选性只在 L2 | `_def.cpp` 的 3 个输出都是 `REQUIRED`；`aclnn_op_name.h` 用可空 `stateOut/xNormOut` |
| 档位由非空组合推导并显式校验 | `_output_mask.h` + `aclnn_op_name.cpp` 的 `ResolveOutputMode` |
| 模板化 TilingKey 必需 | `_tiling_key.h` 的 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` + `_tiling.cpp` 的 `GET_TPL_TILING_KEY` |
| `arch22` 与 `arch35` 都是平台实现目录 | `op_kernel/arch22/`、`op_kernel/arch35/`；`op_host/arch22/`、`op_host/arch35/` 下的 `*_tiling_impl.h` |
| TilingKey 不编码平台 | key 只由 dtype 与模式决定；平台在 kernel 入口按 `__CCE_AICORE__` 选择 |
| 三层看护 | `tests/ut/op_name/`、`tests/atk/op_name/`、调用层门禁（算子目录里不放测试） |
