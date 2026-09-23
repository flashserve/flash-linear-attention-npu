<!--
示例文件：tests/README.md（仓库根目录的 tests/）

规则：**所有测试用例集中在根目录 `tests/` 下，算子目录里不放测试。**

- 算子目录（`fla/ops/ascendc/<类别>/<算子>/`）只放实现、`docs/` 与构建配置；不建 `op_host/tests/`、
  `op_kernel/tests/`，也不在算子目录下放一次性脚本与用例数据；历史算子里残留的 tests 目录属于旧写法。
- 根 `tests/` 按测试类型分目录，再按算子分：

```text
tests/
|-- atk/<op_name>/          # 单算子看护：精度 / 性能 / 确定性 / mssanitizer（见 tests/atk/README.md）
`-- ut/<op_name>/           # host / tiling 单测：档位推导、分核、workspace、溢出保护（无 NPU 也能跑）
```

- 新增或修改测试前先读 `tests/atk/README.md` 与 `docs/agents/05-算子测试.md`；用例设计来源与执行入口
  以仓库当前约定为准，不要自造第二套。
- 不要提交测试产物：`atk_output/`、`result/`、xlsx、profiling/sanitizer 日志、`__pycache__`。
-->

# tests（示例根目录）

| 目录 | 职责 | 示例位置 |
| --- | --- | --- |
| `atk/<op_name>/` | 精度、性能、确定性、内存检测；三份 JSON + yaml + gen + executor + README | [`atk/op_name/`](atk/op_name/) |
| `ut/<op_name>/` | host / tiling 单测：档位与掩码映射、分核余数、workspace 计算、溢出保护 | [`ut/op_name/`](ut/op_name/) |

算子侧只保留被测试引用的实现关系：`op_host/op_name_tiling_processor.h` 保持 header-only，供
`tests/ut/<op_name>/` 直接 include；测试目录的构建/执行入口按仓库当前的 UT 构建方式接入
（本示例给出 `ut/<op_name>/CMakeLists.txt`），不再通过算子目录下的 `add_subdirectory(tests)` 触发。

正式看护资产见 [`atk/op_name/README.md`](atk/op_name/README.md)。
