/**
 * 示例追加片段：torch_custom/fla_npu/csrc/src/stable_ops.cpp
 *
 * 注意事项：
 *   1. 本文件是**追加**内容，不是新文件：stable_ops.cpp 是唯一的编译单元，
 *      每个算子文件里定义的名字只有被它 #include 进来才参与编译。
 *   2. include 顺序有约定：先 common（如 stable_example_scan_common.cpp），再按算子名字母序；
 *      漏 include 的适配文件在运行期表现为 "dispatcher 找不到实现"，
 *      stable_coverage.py 会先报 "never compiled"。
 *   3. 注册只有两行：m.def(kSchema_<op>); 与 m.impl("<op>", &boxed_adapter<run_<op>>);
 *      算子名必须与 schema、Python wrapper、公开名一致。
 *   4. 只在这里注册，不要在别处另起一套注册表；`_stream_probe` 是调试入口，不需要为新算子改动。
 *   5. 新增算子后重跑 tools/stable_coverage.py、tools/op_abi_parity.py、tests/test_stable_gates.py。
 */

// ---- 1) include 列表：common 在前，其余按算子名排序 -------------------------
#include "stable_causal_conv1d_common.cpp"   // 共享 helper 在前（示例）
#include "stable_example_scan.cpp"           // 新增：本算子的 kSchema_ + run_
// ... 其余算子按名字排序 ...

// ---- 2) 注册：在 STABLE_TORCH_LIBRARY 的 m 上，与相邻算子并列 -----------------
namespace fla_npu_stable {

// ... 已有注册 ...

m.def(kSchema_example_scan);
m.impl("npu_example_scan",
       &fla_npu_stable::stable::boxed_adapter<run_npu_example_scan>);

} // namespace fla_npu_stable
