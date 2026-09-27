# 示例算子（形态 A：独立实现；`op_name` 为算子名占位符）

> 本目录把 [`../../engineering-structure.md`](../../engineering-structure.md) 落成实际文件。
> 路径里的 `ops_classify` 是算子类别占位符（实际替换为 `fla/ops/ascendc/` 下的真实类别，如 `gdn`），
> `op_name` 是算子名占位符：下面这个算子是**虚构**的，语义取"chunk 内扫描 + 可选保存中间量"，
> 只展示工程结构，代码不可编译，也不表示任何真实算子。
>
> 示例语义（便于理解字段命名）：
>
> - 输入：`x[B,T,H,D]`（BSND，BF16/FP16；varlen 为 `x[T,H,D]`）、`g` 与 `x` 少一维（FP32/BF16 门控）、`a_log`（可选）、
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
`-- tests/atk/op_name/                 # 单算子看护资产（落在仓库根 tests/ 下）
    |-- README.md                      # 输入限制、约束清单、三类映射、验收记录
    |-- atk_op_name.json / _perf.json / _mss.json
    |-- op_name.yaml                   # 参数空间：真实 inputs / attr
    |-- gen_op_name.py                 # 参数修正（只改 dtype/shape/attr 到合法范围）
    `-- executor_op_name.py            # 执行插件：真实 inputs/attr → 真实 outputs
```

## 读法

1. 每个文件开头都有 `注意事项` 注释块：该文件**必须**满足的要求与常见错误；
2. 文件内代码是骨架，标 `// ...` 处按本算子补齐；
3. 建议顺序：`_def.cpp` → `_tiling.h` → `_output_mask.h` → `_tiling_processor.h` → `_tiling.cpp`
   → `op_kernel/_tiling_key.h` → `op_kernel/arch22|arch35/{_struct.h, _cube.h, _vec.h}`
   → `op_kernel/<算子>.cpp`（薄入口）→ `op_api`（L0 → L2）→ `torch_custom` → `tests/atk`；
   kernel 结构以 `chunk_gated_delta_rule_bwd_finalize` 为样板（见规范 §4.4）。

## 这个示例刻意覆盖的规范点

| 规范点 | 示例中的体现 |
| --- | --- |
| 输出全部 `REQUIRED`，可选性只在 L2 | `_def.cpp` 的 3 个输出都是 `REQUIRED`；`aclnn_op_name.h` 用可空 `stateOut/xNormOut` |
| 档位由非空组合推导并显式校验 | `_output_mask.h` + `aclnn_op_name.cpp` 的 `ResolveOutputMode` |
| 模板化 TilingKey 必需 | `_tiling_key.h` 的 `ASCENDC_TPL_ARGS_DECL`/`ASCENDC_TPL_SEL` + `_tiling.cpp` 的 `GET_TPL_TILING_KEY` |
| `arch22` 与 `arch35` 都是平台实现目录 | `op_kernel/arch22/`、`op_kernel/arch35/`；`op_host/op_tiling/arch22/`、`op_host/op_tiling/arch35/` 下的 `*_tiling_impl.h` |
| TilingKey 不编码平台 | key 只由 dtype 与模式决定；平台在 kernel 入口按 `__CCE_AICORE__` 选择 |
| 看护位置 | `tests/atk/op_name/`（单算子）+ 调用层门禁；算子目录里不放测试 |
| ATK 用例用真实 inputs/attr/outputs | `op_name.yaml` 声明 `x/g/a_log/initial_state/cu_seqlens/chunk_indices/layout/chunk_size/scale/epsilon/return_saved`；固化 JSON 不含 `low_precision_marker`/`fp32_marker`/`case_spec` 等占位参数 |

## kernel 结构怎么读（与 `chunk_gated_delta_rule_bwd_finalize` 对齐）

### 占位符与符号替换规则

示例路径里的占位符是"可替换标签"，不是真实名字；复制到新算子时必须按表替换。

| 位置 | 示例值 | 替换规则 | 真实例 |
| --- | --- | --- | --- |
| 类别目录 | `fla/ops/ascendc/ops_classify/` | 换成真实类别目录名（snake_case） | `gdn`、`kda` |
| 算子目录 / 文件名 | `op_name` | 换成真实算子名（snake_case） | `chunk_fwd_h` |
| 命名空间 | `OpsClassify` | 类别目录名的 PascalCase，**kernel 侧**（archXX 实现、common.h、tiling_key、入口）共用一个 | `gdn` → `GDN` |
| 符号前缀 | `OP_NAME_`、`OpName` | 换成算子名的大写/驼峰形式 | `CHUNK_FWD_H_`、`ChunkFwdH` |
| 头文件保护宏 | `OP_NAME_VEC_ARCH35_H` | 同样带算子名与 arch，避免跨算子撞名 | `CHUNK_FWD_H_VEC_ARCH35_H` |
| TPL 档位 token | `OP_NAME_TPL_BF16` | 是宏，host / kernel / tiling_key 都直接写宏名，不加命名空间限定 | `TPL_BF16` |

注意别把 kernel 的类别命名空间与 host 框架命名空间混为一谈：`ops`（`_def.cpp` 的注册命名空间）、
`optiling`（tiling）、`l0op`（L0 内部接口）是框架/层次的保留命名空间；类别的 PascalCase 命名空间
只用于 kernel 侧的结构体、文件作用域函数与 `ASCENDC_TPL_*` 声明块。host 侧不重复声明类别命名空间，
两边靠"TPL 宏 token 取值一致 + TilingData 字段逐一对应"对齐：host 用 `BEGIN_TILING_DATA_DEF`
声明并按 op 名注册（`REGISTER_TILING_DATA_CLASS(OpName, OpNameTilingData)`），kernel 用
`GET_TILING_DATA_WITH_STRUCT` 按结构体解析同一块内存，字段顺序/类型/数量任一不同就是 ABI 不一致。

只给"薄入口 + 一个类"不足以复制样板算子的可读性。示例的
`op_kernel/arch22|arch35/op_name_vec.h`、`op_name_cube.h` 都按固定四层写，复制到新算子时保持顺序。
两条硬约束：**函数不放进类/结构体**（结构体只放数据，行为全部是文件作用域 `inline` 函数）；
**arch35 的向量计算必须是 VF 融合函数**（`__simd_vf__ inline` + `AscendC::MicroAPI`，
arch22 无 VF 通路，用同名函数 + 普通向量指令，差异写在平台差异表里）。

| 层 | 看什么 | 示例位置 |
| --- | --- | --- |
| ① 文件头三张表 | Stage 表（谁生产/谁消费/怎么同步）、UB 或 L1+L0 布局表（偏移/大小/内容/生命周期）、同步协议表（flag 名/方向/背压） | `archXX/op_name_vec.h` 顶部注释块 |
| ② Stage 计算函数 | `StageNVf(...)`：arch35 是 `__simd_vf__ inline` + MicroAPI 的 VF 融合（`RegTensor`/`MaskReg`）；arch22 同名函数用普通向量指令。按 Stage 号顺序排列，只碰 UB | 结构体之前 |
| ③ 数据结构体 | 只放数据：GM 张量、UB/L1/L0 张量、事件 id、只读状态与游标；模板别名（dtype/档位）也在这里 | `struct OpNameVectorContext` / `OpNameCubeContext` |
| ④ 行为层（结构体外） | `InitOpNameVector`（接线 + buffer 划分 + 事件预置）→ `ProcessOpNameVector`（任务主循环 + 调 Stage + 收尾）→ `StageN...`（输入/输出/复用/同步四行注释）→ `CloseAndReleaseEvents` | 结构体之后，全部文件作用域 `inline` |

配套的现状说明：

- `op_kernel/op_name.cpp` 保持薄入口：只做 dtype traits、tiling 解析、workspace 区域命名（同一行写生命周期）、
  AIC/AIV 分派；入口只构造 `Context` 结构体并调用 `Init<角色>`/`Process<角色>`，不出现 Stage 细节。
- `op_kernel/op_name_common.h` 放平台无关资产：具名同步 flag、`ChunkInfo` 与 `GetChunkInfo`、
  `GetWorkspaceChunkOffset`、`Min`；`archXX/op_name_struct.h` 放 TilingData、平台资源常量与
  UB/L1/L0 布局常量（与文件头布局表逐行对应）。
- 事件生命周期固定：`Init<角色>` 里按 slot `AllocEventID` 并 `SetFlag` 开首轮，`Process<角色>` 末尾
  统一闭环后 `ReleaseEventID`；示例的 `CloseAndReleaseEvents()` 就是这一段。
- 编译路径开关用按场景命名的 `FLA_TORCH_EXTENSION_INLINE_BUILD`（算子源码被 torch 扩展内联编译时
  由构建侧定义，此时跳过 host tiling 框架头与 `__global__` 入口），不用泛化负向宏 `TORCH_MODE`：
  后者名字会被误读成"torch_custom 构建"，且 `#ifndef` 让默认分支依赖一个仓内常规构建不出现的符号。
