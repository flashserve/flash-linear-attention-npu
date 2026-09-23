<!--
示例文件：tests/README.md（仓库根目录的 tests/）

规则：**所有测试用例集中在根目录 `tests/` 下，算子目录里不放测试。**

- 算子目录（`fla/ops/ascendc/<类别>/<算子>/`）只放实现、`docs/` 与构建配置；不建 `op_host/tests/`、
  `op_kernel/tests/`，也不在算子目录下放用例数据与一次性脚本；历史算子里残留的这类目录属于旧写法。
- 根 `tests/` 下按已有规划分目录，单算子看护只有 `tests/atk/<算子>/` 一处：

```text
tests/
`-- atk/<op_name>/    # 单算子看护：精度 / 性能 / 确定性 / mssanitizer（见 tests/atk/README.md）
```

- 不要引入未在 `tests/atk/README.md` 中登记的测试子目录（例如自造 `ut/`、`op_cases/` 之类），
  也不要把用例散落到算子目录里；需要新的测试层次时先在 `tests/atk/README.md` 登记再落地。
- 不要提交测试产物：`atk_output/`、`result/`、xlsx、profiling/sanitizer 日志、`__pycache__`。
-->

# tests（示例根目录）

| 目录 | 职责 | 示例位置 |
| --- | --- | --- |
| `atk/<op_name>/` | 精度、性能、确定性、内存检测；三份 JSON + yaml + gen + executor + README | [`atk/op_name/`](atk/op_name/) |

算子目录里没有测试：`op_host/` 只保留被测实现与 header-only 头文件，测试所需的输入、标杆与用例都在
`tests/atk/<算子>/` 内组织（见 [`atk/op_name/README.md`](atk/op_name/README.md)）。

tiling 分支与档位的静态保护放在实现侧（`op_host/op_name_output_mask.h` 的 `static_assert`），
运行期覆盖证据与 TilingKey 覆盖表按 `tests/atk/README.md` 的「TilingKey 覆盖交付」维护。
