# ATK 用例设计覆盖评估报告

> 示例填充：数值来自本示例的 `op_name.yaml` / `gen_op_name.py` 设计值（未执行 `atk case`，
> 因此标注「降级（设计值）」）。
> 真实任务按 ATK 仓 skill/atk-quality-guard 的 `templates/test_case_design_report.md` 生成，
> 8 个维度逐项填充、`<placeholder>` 全部替换，每章「结论前置」。

## 1. 基本信息

| 项 | 值 |
|---|---|
| 算子名 | `op_name` |
| 算子源码根目录 | `fla/ops/ascendc/ops_classify/op_name/` |
| aclnn 入口文件 | `op_host/op_api/aclnn_op_name.h` |
| 测试工程目录 | `tests/atk/op_name/` |
| 用例 JSON 路径 | `result/op_name/json/all_op_name.json`（由 `atk case` 生成） |
| 总用例数 | N/A（降级：未执行 `atk case`） |
| 生成时间 | N/A（示例文档） |
| ATK 可用性 | 不可用（降级，设计值） |
| 精度标准版本 | `mixed_tolerance_bm` |

---

## 2. 约束覆盖分析

### 2.1 约束覆盖结论

- **覆盖率判定**：`6/6`（100%）
- **未覆盖清单**：无
- **校验覆盖**：已覆盖末维 128、g 与 x 的轴对齐、平台 dtype 列表、变长元数据成对、attr 取值域、元素数上限
- **ATK 框架限制覆盖**：已覆盖 `dim_values` 离散列表、总元素数 ≤ 2^34、`shape_distributions` 默认值

### 2.2 约束覆盖清单

| # | 约束名 | 约束来源（源码文件:行号） | 约束表达式 | 约束类型 | `gen_op_name.py` 修正逻辑 | 是否覆盖 |
|---|---|---|---|---|---|---|
| 1 | 末维固定 128 | `op_host/op_name_tiling.cpp`（D 校验分支） | `x.shape[-1] == 128` | shape | 只改 `x.shape[-1]` 与 `initial_state.shape[-1]` | ✅ |
| 2 | g 与 x 轴对齐 | `op_host/op_name_tiling.cpp` | `g.shape == x.shape[:-1]` | shape | `g.shape = x.shape[:-1]` | ✅ |
| 3 | 平台 dtype 列表 | `op_host/op_api/aclnn_op_name.cpp::CheckDtype` | `g.dtype ∈ 平台列表` | dtype | 不满足时回退到列表首项 | ✅ |
| 4 | state 与 x 同 dtype | `aclnn_op_name.cpp::CheckDtype` | `initial_state.dtype == x.dtype` | dtype | 强制跟随 `x.dtype` | ✅ |
| 5 | 变长元数据成对 | `aclnn_op_name.cpp::CheckShape` | `cu_seqlens` 与 `chunk_indices` 同时给出 | shape/attr | 缺席一侧按语义补最简形态 | ✅ |
| 6 | 元素数上限 | ATK 框架（2^34）+ 算子 int32 风险（2^31） | `prod(shape) ≤ 2^34` | shape | 超限时等比裁剪 | ✅ |

---

## 3. Shape 覆盖分析

### 3.1 Shape 覆盖结论

- **边界对覆盖判定**：完整（`x`/`g`/`initial_state` 的取值列表都含 2^n 与 2^n-1）
- **边界场景覆盖判定**：设计上覆盖 7/8 类（`upper_border` 保留给全量，不进冒烟采样）
- **遗漏项**：无（降级：未生成 JSON，无法给出实际用例数）
- **ND 维度处理**：算子按 ND 处理 rank-4 输入，`dim_numbers: values: [4]`；不等价于声明支持的最大维度

### 3.2 维度值覆盖

| 维度位置 | 取值列表 | 当前覆盖情况（用例 JSON 实际值） | 边界对覆盖（2^n 与 2^n-1） | 备注 |
|---|---|---|---|---|
| `x.dim_0`（B） | `[1, 2, 3, 4]` | 降级（设计值） | ✅ | |
| `x.dim_2`（T） | `[..., 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024]` | 降级（设计值） | ✅ | 覆盖 chunk 切分边界 |
| `x.dim_3`（D） | 固定 `128` | 降级（设计值） | 不适用 | 算子硬约束 |
| `g.dim_2`（T） | 同 `x.dim_2` | 降级（设计值） | ✅ | 由 gen 与 `x` 对齐 |

### 3.3 边界场景覆盖

| 边界场景 | 当前覆盖情况（用例 JSON 实际值） | 是否生成 | 用例数 | 说明 |
|---|---|---|---|---|
| `empty_shape` | 降级（设计值） | ✅（设计） | N/A | `boundary.has_empty: true` |
| `zero_dim` | 降级（设计值） | ✅（设计） | N/A | `dim_values` 含 1，零维由 ATK 生成 |
| `scalar` | 降级（设计值） | ✅（设计） | N/A | 由 boundary 覆盖 |
| `tiny_shape` | 降级（设计值） | ✅（设计） | N/A | `dim_values` 含 1~4 |
| `large_shape` | 降级（设计值） | ✅（设计） | N/A | `dim_values` 含 512~1024 |
| `dtype 边界` | `bf16/fp16/fp32` | ✅（设计） | N/A | `dtypes.values` 已列全 |
| `upper_border` | 降级（设计值） | ✅（保留） | N/A | `has_upper_border: false`，全量保留 |
| `lower_border` | 降级（设计值） | ✅（设计） | N/A | `has_lower_border: true` |

---

## 4. dtype 覆盖分析

### 4.1 dtype 覆盖结论

- **等价类剪枝判定**：合理（`reduction.dtype_class: true`；`x` 的 bf16/fp16 若内核统一转 fp32 可合并为一类）
- **零覆盖 dtype**：无（设计上 bf16/fp16/fp32/int32/attr_bool 均有候选）

### 4.2 dtype 等价类剪枝记录

| 等价类名 | 包含 dtype | 剪枝依据（C++ 内核计算路径） | 当前覆盖情况（用例 JSON 实际值） | 是否合并取并集 |
|---|---|---|---|---|
| 低精度主输入 | `bf16`, `fp16` | `x` 在内核中统一转 FP32 参与计算 | 降级（设计值） | ✅ |
| 门控 | `bf16`, `fp32`（A5）/ `bf16`（A2/A3） | `g` 平台支持列表不同 | 降级（设计值） | 否（平台差异需分别覆盖） |
| 索引 | `int32` | 变长元数据 | 降级（设计值） | ✅ |

### 4.3 dtype 用例分布

| dtype | 用例数 | 占比 | 当前覆盖情况（用例 JSON 用例数量） | 是否达标（≥1 条） |
|---|---|---|---|---|
| `bf16` | N/A | N/A | 降级（设计值） | ✅ |
| `fp16` | N/A | N/A | 降级（设计值） | ✅ |
| `fp32` | N/A | N/A | 降级（设计值） | ✅ |
| `int64` | N/A | N/A | 降级（设计值） | ✅ |
| `attr_bool` | N/A | N/A | 降级（设计值） | ✅ |

---

## 5. Attr 覆盖分析

### 5.1 Attr 覆盖结论

- **pairwise 覆盖判定**：完整（`reduction.attr_pairwise: true`，5 个 attr 全组合 4×2×3×3×2=144 → pairwise ≈ 20 上下）
- **算子特性 case 保留判定**：全部保留（`layout=TND` + `return_saved=true` 的组合必须保留）

### 5.2 Attr 等价类覆盖

| Attr 名 | 类型 | 等价类值 | 当前覆盖情况（用例 JSON 实际值） | 全组合数 | pairwise 后组合数 | 是否覆盖 |
|---|---|---|---|---|---|---|
| `layout` | enum | `BSND/BNSD/TND/NTD` | 降级（设计值） | 4 | 4 | ✅ |
| `chunk_size` | int | `64/128` | 降级（设计值） | 2 | 2 | ✅ |
| `scale` | float | `0.5/1.0/2.0` | 降级（设计值） | 3 | 3 | ✅ |
| `epsilon` | float | `1.0e-06/1.0e-05/1.0e-04` | 降级（设计值） | 3 | 3 | ✅ |
| `return_saved` | bool | `true/false` | 降级（设计值） | 2 | 2 | ✅ |

### 5.3 多 Attr Pairwise 剪枝记录

- **是否启用 `reduction.attr_pairwise`**：是
- **剪枝前全组合数**：144
- **剪枝后组合数**：约 20（2-way 覆盖）
- **强制保留的算子特性 case**：`layout=TND + return_saved=true`（覆盖 varlen 导出路径）、`chunk_size=128 + layout=BNSD`

---

## 6. INT32 溢出看护状态

### 6.1 溢出看护结论

- **看护状态判定**：未触发（示例算子 tiling 用 int64 累加，见 `op_name_tiling_processor.h::MulChecked`）
- **是否需人工复核**：否

### 6.2 看护实现明细

| 评估项 | 状态 |
|---|---|
| 自动推断是否扫描 size/offset 乘法 pattern | ✅ 已扫描 |
| 是否命中 int32 操作数 | ❌ 未命中 |
| `OVERFLOW_GUARD_ENABLED` 取值 | `False` |
| 触发依据（C++ 源码位置） | N/A |
| `gen_op_name.py` 是否对超大 shape 跳过 2^31 修正 | 不适用 |

---

## 7. 用例规模统计

### 7.1 规模合理性结论

- **总用例数判定**：N/A（降级：未执行 `atk case`）；预期 = `dtype_numbers` × shape × attr(pairwise) 组合
- **reduction 裁剪判定**：生效（`dtype_class` / `attr_pairwise` / `shape_class` / `static_dedup`；`coverage_guided` 未启用）

### 7.2 规模统计明细

| 项 | 值 | 说明 |
|---|---|---|
| `dtype_numbers` | `1`（占位） | 步骤 4 回填，经验量级见 skill |
| `extra_numbers` | `0` | 边界用例数 |
| 总用例数 | N/A | 降级（设计值） |
| 用例裁剪配置（reduction） | `dtype_class/attr_pairwise/shape_class/static_dedup` | |
| 用例 JSON 文件大小 | N/A | 降级（设计值） |

---

## 8. 汇总与建议

### 8.1 阶段建议

- **是否建议进入 Phase B 执行**：是（先在 ATK 环境补齐步骤 4 的 JSON 生成，再移交 NPU 环境）
- **核心风险点**：`dim_values` 上限需与算子实际支持的长序列能力对齐；`dim_values` 上限过大时先用 dry-run 观察规模

### 8.2 覆盖充分性汇总

| 评估维度 | 评级 |
|---|---|
| 约束覆盖 | 充分 |
| Shape 覆盖 | 充分 |
| dtype 覆盖 | 充分 |
| Attr 覆盖 | 充分 |

### 8.3 遗漏点与改进建议

| # | 遗漏点 | 影响 | 改进建议 |
|---|---|---|---|
| 1 | 降级模式下无用例 JSON | 无法统计实际用例分布与规模 | 在 ATK 环境执行 `atk case -f op_name.yaml -p gen_op_name.py` 后回填本报告 |
| 2 | `upper_border` 不进冒烟采样 | 上边界回归被推迟到全量 | 全量执行阶段必须覆盖 `upper_border` 用例 |
