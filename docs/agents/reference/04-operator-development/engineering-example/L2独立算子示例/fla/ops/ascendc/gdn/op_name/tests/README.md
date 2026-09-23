<!--
示例文件：fla/ops/ascendc/gdn/op_name/tests/README.md

注意事项：
  1. 本目录只放"算子自带"的脚本/数据（例如开发期的对齐脚本、临时的输入生成器）；
     正式看护资产放 tests/atk/<算子>/，不要在这里另起一套。
  2. ENABLE_TEST 打开时，构建系统要求 `${OP_DIR}/tests/CMakeLists.txt` 存在，否则该算子会被跳过；
     需要 host 单测时把 CMakeLists 放这里的上级 op_host/tests/，不要与本目录混用。
  3. 不要提交测试产物：atk_output/、result/、xlsx、profiling/sanitizer 日志、__pycache__。
  4. 一次性脚本用完即删或移入对应 issue 的复现目录，不留在算子目录里。
-->

# op_name 自带脚本

| 文件 | 用途 | 是否长期保留 |
| --- | --- | --- |
| `gen_scan_cases.py`（示例，按需新建） | 开发期输入生成/对齐，正式用例由 `tests/atk/op_name/gen_op_name.py` 负责 | 否 |

正式看护见 [`../../../../../../tests/atk/op_name/README.md`](../../../../../../tests/atk/op_name/README.md)。
